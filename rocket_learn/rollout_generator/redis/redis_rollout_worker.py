"""
Redis rollout worker — functional style.

Structure:
  Pure functions  — no I/O, no side effects, deterministic given inputs
  I/O functions   — clearly labelled, kept at the edges
  run()           — thin loop that wires pure + I/O together
  RedisRolloutWorker — config dataclass + entry point
"""

import itertools
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import sqlite3 as sql
from redis import Redis
from rlgym.api import RLGym
from tabulate import tabulate

import rocket_learn.utils.generate_episode
from rocket_learn.agent.pretrained_policy import HardcodedAgent
from rocket_learn.rollout_generator.redis.utils import (
    EXPERIENCE_PER_MODE, MAX_QUEUE, OPPONENT_MODELS, WORKER_IDS,
    _serialize, _unserialize_model, encode_buffers, get_rating,
    io_fetch_latest, io_push_rollout, parse_model_id,
)
from rocket_learn.utils.env_wrapper import RLGymV2GymWrapper
from rocket_learn.utils.util import probability_NvsM
from rlgym.rocket_league.state_mutators import MutatorSequence
from rlgym_tools.rocket_league.state_mutators.variable_team_size_mutator import VariableTeamSizeMutator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkerConfig:
    name:                 str
    device:               str
    past_version_prob:    float = 0.2
    evaluation_prob:      float = 0.01
    sigma_target:         float = 1.0
    dynamic_gm:           bool  = False
    team_size:            int   = 1
    render:               bool  = False
    send_gamestates:      bool  = True
    send_obs:             bool  = True
    full_team_evaluations:bool  = False
    live_progress:        bool  = True
    local_cache_name:     Optional[str] = None
    pretrained_agents:    Dict[Any, float] = field(default_factory=dict)
    scoreboard:           Any   = None


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def select_gamemode(mode_exp: Dict[str, int]) -> Tuple[int, int]:
    """Pick the gamemode with least experience. Ties broken uniformly."""
    min_exp = min(mode_exp.values())
    least   = [m for m, v in mode_exp.items() if v == min_exp]
    mode    = np.random.choice(least)
    b, o    = mode.split("v")
    return int(b), int(o)


def decide_n_old(
    evaluate: bool,
    n_agents: int,
    past_version_prob: float,
    pretrained_agents: Dict[Any, float],
) -> Tuple[int, Optional[Any]]:
    """
    Returns (n_old, pretrained_choice).
    Pure: takes a pre-drawn random float to stay testable.
    """
    if evaluate:
        return n_agents, None

    pretrained_total = sum(pretrained_agents.values())
    rand = np.random.random()

    if rand < past_version_prob:
        return np.random.randint(1, n_agents), None

    if rand < past_version_prob + pretrained_total:
        cumulative = past_version_prob
        for agent, prob in pretrained_agents.items():
            cumulative += prob
            if rand < cumulative:
                return np.random.randint(1, n_agents), agent

    return 0, None


def pick_eval_version_probs(
    sigmas: np.ndarray,
    sigma_target: float,
) -> Optional[np.ndarray]:
    """
    Returns probability array over versions for an eval game,
    or None if all sigmas are below target (fall back to latest-only).
    """
    probs = np.clip(sigmas - sigma_target, a_min=0, a_max=None)
    if probs.sum() == 0:
        if np.random.normal(0, sigma_target) > 1:
            return np.ones(len(sigmas)) / len(sigmas)
        return None
    return probs / probs.sum()


def pick_old_version_probs(
    values,
    target_rating,
    n_new: int,
    n_old: int,
    eval_version_idx: Optional[int],
) -> np.ndarray:
    """Win-probability weighting for selecting old opponents."""
    probs = np.zeros(len(values))
    for i, rating in enumerate(values):
        if eval_version_idx is not None and i == eval_version_idx:
            continue  # don't add more of the same agent in eval
        p = probability_NvsM([rating], [target_rating])
        probs[i] = (p * (1 - p)) ** ((n_new + n_old) // 2)
    s = probs.sum()
    return probs / s if s > 0 else probs


def pick_fair_matchup(
    versions: List,
    values,
    latest_rating,
    n_new: int,
) -> Tuple[List, List]:
    """
    From a list of chosen version indices, find the permutation with
    the fairest team split (AlphaStar-style p*(1-p) weighting).
    Returns (version_keys, ratings).
    """
    matchups, qualities = [], []
    for perm in itertools.permutations(versions):
        it_ratings = [latest_rating if v == -1 else values[v] for v in perm]
        mid = len(it_ratings) // 2
        p = probability_NvsM(it_ratings[:mid], it_ratings[mid:])
        if n_new == 0 and set(perm[:mid]) == set(perm[mid:]):
            p = 0
        matchups.append(perm)
        qualities.append(p * (1 - p))

    qualities = np.array(qualities)
    s = qualities.sum()
    if s == 0:
        return versions, [latest_rating] * len(versions)

    k = np.random.choice(len(matchups), p=qualities / s)
    chosen = matchups[k]
    chosen_ratings = [latest_rating if v == -1 else values[v] for v in chosen]
    return list(chosen), chosen_ratings


def build_version_table(
    versions: List,
    ratings: List,
    blue: int,
    orange: int,
    pretrained_choice: Optional[Any],
) -> str:
    """Pure: formats matchup as a display table string."""
    def fmt(v, r):
        if pretrained_choice is not None and v == "na":
            return [type(pretrained_choice).__name__, "N/A"]
        if v == "na":
            return ["Human", "N/A"]
        label = f"Latest ({-v})" if isinstance(v, int) and v < 0 else str(v)
        return [label, f"{r.mu:.2f}±{2 * r.sigma:.2f}"]

    info        = [fmt(v, r) for v, r in zip(versions, ratings)]
    blue_info   = info[:blue]
    orange_info = info[blue:]

    pad = abs(blue - orange)
    if blue < orange:
        blue_info   += [["", ""]] * pad
    elif orange < blue:
        orange_info += [["", ""]] * pad

    rows = [bv + ov for bv, ov in zip(blue_info, orange_info)]
    return tabulate(rows, headers=["Blue", "rating", "Orange", "rating"],
                    tablefmt="rounded_outline")


def format_post_stats(
    n_steps: int,
    total_steps: int,
    result: int,
    goal_speed_kph: float,
) -> str:
    sign  = "+" if result > 0 else ""
    stats = (f"Rollout finished after {n_steps} steps "
             f"({total_steps} total steps), result was {sign}{result}")
    if result != 0:
        stats += f", goal speed: {goal_speed_kph:.2f} kph"
    return stats


# ---------------------------------------------------------------------------
# I/O functions
# ---------------------------------------------------------------------------

def io_fetch_latest_model(redis: Redis, device: str, current_model_id: Optional[str]):
    latest = io_fetch_latest(redis)
    if latest is None or latest[0] == current_model_id:
        return current_model_id, None
    model_id, model_bytes = latest
    assert parse_model_id(model_id)
    return model_id, _unserialize_model(model_bytes, device)


def build_rollout_payload(model_id: str, rollout_data, versions: List, uuid: str, name: str,
                          result: int, send_obs: bool, send_gamestates: bool) -> Tuple:
    return (model_id, rollout_data, versions, uuid, name, result, send_obs, send_gamestates, True)


def io_read_mode_experience(redis: Redis) -> Dict[str, int]:
    return {m.decode(): int(v) for m, v in redis.hgetall(EXPERIENCE_PER_MODE).items()}


def io_increment_mode_experience(redis: Redis, mode: str, exp: int) -> None:
    redis.hincrby(EXPERIENCE_PER_MODE, mode, exp)


def io_init_mode_experience(redis: Redis, team_size: int) -> None:
    for i in range(1, team_size + 1):
        redis.hsetnx(EXPERIENCE_PER_MODE, f"{i}v{i}", 0)


def io_get_ratings(redis: Redis, gamemode: str, model_id: Optional[str]):
    return get_rating(gamemode, model_id, redis)


def io_build_past_model_loader(redis: Redis, device: str, local_cache_name: Optional[str]):
    """
    Returns a cached loader function: model_id -> model.
    Encapsulates SQLite caching so callers stay pure.
    """
    db = None
    if local_cache_name:
        db = sql.connect(f"redis-model-cache-{local_cache_name}.db")
        db.execute("""
            CREATE TABLE IF NOT EXISTS MODELS (
                id TEXT PRIMARY KEY,
                parameters BLOB NOT NULL
            )
        """)

    @lru_cache(maxsize=8)
    def load(model_id: str):
        assert parse_model_id(model_id)
        if db is not None:
            rows = db.execute(
                "SELECT parameters FROM MODELS WHERE id = ?", (model_id,)
            ).fetchall()
            if rows:
                return _unserialize_model(rows[0][0], device)
            bytestream = redis.hget(OPPONENT_MODELS, model_id)
            if bytestream is None:
                return None
            db.execute("INSERT INTO MODELS (id, parameters) VALUES (?, ?)",
                       (model_id, bytestream))
            db.commit()
            return _unserialize_model(bytestream, device)
        bytestream = redis.hget(OPPONENT_MODELS, model_id)
        return None if bytestream is None else _unserialize_model(bytestream, device)

    return load


def io_build_env(rlgym_env: RLGym, team_size_mutator: VariableTeamSizeMutator,
                 default_team_size: int) -> RLGymV2GymWrapper:
    team_size_mutator.mode_weights = {(default_team_size, default_team_size): 1}
    rlgym_env.state_mutator = MutatorSequence(team_size_mutator, rlgym_env.state_mutator)
    return RLGymV2GymWrapper(rlgym_env)


# ---------------------------------------------------------------------------
# Matchup builder — thin I/O wrapper around pure logic
# ---------------------------------------------------------------------------

def build_matchup(
    redis: Redis,
    current_agent,
    load_past_model,
    n_agents: int,
    latest_model_id: str,
    evaluate: bool,
    past_version_prob: float,
    pretrained_agents: Dict,
    sigma_target: float,
    full_team_evaluations: bool,
) -> Optional[Tuple[List, List, List, bool]]:
    """
    Returns (agents, versions, ratings, is_evaluate).
    Versions use latest_model_id for latest agent slots.
    """
    per_team = n_agents // 2
    gamemode = f"{per_team}v{per_team}"

    n_old, pretrained_choice = decide_n_old(
        evaluate, n_agents, past_version_prob, pretrained_agents
    )
    n_new = n_agents - n_old

    latest_key = f"{latest_model_id}-stochastic"

    # All-latest game
    if n_old == 0:
        rating  = io_get_ratings(redis, gamemode, latest_key)
        if rating is None:
            return None
        ratings = [rating] * n_new
        agents   = [current_agent] * n_new
        has_latest = True
        return agents, [latest_model_id] * n_new, ratings, not has_latest

    all_ratings = io_get_ratings(redis, gamemode, None)
    if latest_key not in all_ratings:
        return None
    latest_rating = all_ratings[latest_key]
    keys   = list(all_ratings.keys())
    values = list(all_ratings.values())

    is_eval = n_new == 0

    if pretrained_choice is not None:
        slot_versions = [-1] * n_new + ["na"] * n_old
        np.random.shuffle(slot_versions)
    elif is_eval:
        sigmas = np.array([r.sigma for r in values])
        probs  = pick_eval_version_probs(sigmas, sigma_target)
        if probs is None:
            # All sigmas below target — pick uniformly, still run eval
            probs = np.ones(len(values)) / len(values)

        chosen_idx = int(np.random.choice(len(keys), p=probs))
        chosen_key = keys[chosen_idx]
        if full_team_evaluations:
            eval_slots = [chosen_key] * per_team
        else:
            eval_slots = [chosen_key]

        # Fill remaining slots — duplicate chosen version if not enough others
        n_extra = n_old - len(eval_slots)
        if n_extra > 0:
            extra_probs = pick_old_version_probs(values, values[chosen_idx],
                                                 n_new, n_old, chosen_idx)
            if extra_probs.sum() == 0:
                extra_probs = np.ones(len(values)) / len(values)
            extra_slots = np.random.choice(len(extra_probs), size=n_extra,
                                           p=extra_probs, replace=True).tolist()
            slot_versions = eval_slots + [keys[i] for i in extra_slots]
        else:
            slot_versions = eval_slots
    else:
        if n_new == 0:
            n_new, n_old = n_old, 0
        old_probs  = pick_old_version_probs(values, latest_rating, n_new, n_old, None)
        old_chosen = np.random.choice(len(old_probs), size=n_old,
                                      p=old_probs, replace=True).tolist()
        slot_versions = [-1] * n_new + [keys[i] for i in old_chosen]

    # Find fairest permutation
    idx_versions = [
        -1 if v == -1 else (keys.index(v) if v in keys else v)
        for v in slot_versions
    ]
    fair_versions, fair_ratings = pick_fair_matchup(
        idx_versions, values, latest_rating, n_new
    )

    # Resolve agents
    agents = []
    resolved_versions = []
    for v in fair_versions:
        if v == -1:
            agents.append(current_agent)
            resolved_versions.append(latest_model_id)
        elif pretrained_choice is not None and v == "na":
            agents.append(pretrained_choice)
            resolved_versions.append("na")
        else:
            key    = keys[v] if isinstance(v, int) else v
            base   = "-".join(key.split("-")[:-1])
            model  = load_past_model(base)
            if model is None:
                return None
            model.deterministic = key.endswith("deterministic")
            agents.append(model)
            resolved_versions.append(key)

    has_latest  = latest_model_id in resolved_versions
    is_evaluate = not has_latest
    return agents, resolved_versions, fair_ratings, is_evaluate


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(config: WorkerConfig, env: RLGym) -> None:
    redis = config._redis  # injected by RedisRolloutWorker

    load_past_model   = io_build_past_model_loader(redis, config.device, config.local_cache_name)
    team_size_mutator = VariableTeamSizeMutator(mode_weights={})
    wrapped_env       = io_build_env(env, team_size_mutator, config.team_size)

    current_model_id: Optional[str] = None
    current_agent                   = None
    total_steps                     = 0

    while True:
        # --- fetch latest model if updated ---
        current_model_id, new_agent = io_fetch_latest_model(
            redis, config.device, current_model_id
        )
        if new_agent is not None:
            current_agent = new_agent
            if config.dynamic_gm:
                io_init_mode_experience(redis, config.team_size)
        if current_agent is None:
            time.sleep(1)
            continue

        # --- pick gamemode ---
        if config.dynamic_gm:
            mode_exp     = io_read_mode_experience(redis)
            blue, orange = select_gamemode(mode_exp)
        else:
            blue = orange = config.team_size

        team_size_mutator.mode_weights = {(blue, orange): 1}

        # --- build matchup ---
        evaluate = np.random.random() < config.evaluation_prob
        matchup = build_matchup(
            redis          = redis,
            current_agent  = current_agent,
            load_past_model= load_past_model,
            n_agents       = blue + orange,
            latest_model_id = current_model_id,
            evaluate       = evaluate,
            past_version_prob   = config.past_version_prob,
            pretrained_agents   = config.pretrained_agents,
            sigma_target        = config.sigma_target,
            full_team_evaluations = config.full_team_evaluations,
        )
        if matchup is None:
            continue
        agents, versions, ratings, evaluate = matchup

        pretrained_choice = next(
            (a for a in agents if isinstance(a, HardcodedAgent)), None
        )
        table = build_version_table(versions, ratings, blue, orange, pretrained_choice)

        # --- run episode ---
        if evaluate:
            print("EVALUATION GAME\n" + table)
            result = rocket_learn.utils.generate_episode.generate_episode(
                wrapped_env, agents,
                evaluate=True,
                scoreboard=config.scoreboard,
                progress=config.live_progress,
                render=config.render,
            )
            print(f"Evaluation finished, goal differential: {result}\n")
            continue
        else:
            print("ROLLOUT\n" + table)
            try:
                rollouts, result = rocket_learn.utils.generate_episode.generate_episode(
                    wrapped_env, agents,
                    evaluate=False,
                    scoreboard=config.scoreboard,
                    progress=config.live_progress,
                    render=config.render,
                )
            except EnvironmentError:
                wrapped_env.attempt_recovery()
                continue

            if len(rollouts[0].observations) <= 1:
                print(" ** Rollout Generation Error: Restarting **\n")
                continue

            n_steps     = len(rollouts[0].observations)
            episode_exp = n_steps * len(rollouts)
            total_steps += episode_exp

            if config.dynamic_gm:
                io_increment_mode_experience(redis, f"{blue}v{orange}", episode_exp)

            goal_speed = (np.linalg.norm(rollouts[0].infos[-2]["state"].ball.linear_velocity)
                          * 0.036)
            print(format_post_stats(n_steps, total_steps, result, goal_speed) + "\n")

        # --- send to learner ---
        rollout_data  = encode_buffers(rollouts,
                                       return_obs=config.send_obs,
                                       return_rewards=True,
                                       return_states=config.send_gamestates)
        assert current_model_id in versions
        rollout_bytes = _serialize(build_rollout_payload(
            current_model_id, rollout_data, versions, config._uuid, config.name,
            result, config.send_obs, config.send_gamestates))
        if io_push_rollout(redis, rollout_bytes) >= MAX_QUEUE:
            print("Had to limit rollouts. Learner may have crashed or is overloaded")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

class RedisRolloutWorker:
    def __init__(self, redis: Redis, name: str, env: RLGym, device: str,
                 past_version_prob=0.2, evaluation_prob=0.01, sigma_target=1.0,
                 dynamic_gm=False, team_size=1, render=False,
                 send_gamestates=True, send_obs=True, scoreboard=None,
                 pretrained_agents=None, local_cache_name=None,
                 full_team_evaluations=False, live_progress=True):

        assert send_gamestates or send_obs, "Must send at least one of obs or states"
        assert team_size > 0, "team_size must be >= 1"

        self._env   = env
        self._redis = redis
        self._uuid  = str(uuid4())

        redis.rpush(WORKER_IDS, self._uuid)
        print(f"Started worker {self._uuid} on host "
              f"{redis.connection_pool.connection_kwargs.get('host')} "
              f"under name {name}")

        self._config = WorkerConfig(
            name=name, device=device,
            past_version_prob=past_version_prob,
            evaluation_prob=evaluation_prob,
            sigma_target=sigma_target,
            dynamic_gm=dynamic_gm,
            team_size=team_size,
            render=render,
            send_gamestates=send_gamestates,
            send_obs=send_obs,
            full_team_evaluations=full_team_evaluations,
            live_progress=live_progress,
            local_cache_name=local_cache_name,
            pretrained_agents=pretrained_agents or {},
            scoreboard=scoreboard,
        )
        # Inject redis + uuid into config (frozen, so use object.__setattr__)
        object.__setattr__(self._config, "_redis", redis)
        object.__setattr__(self._config, "_uuid",  self._uuid)

    def run(self):
        run(self._config, self._env)

