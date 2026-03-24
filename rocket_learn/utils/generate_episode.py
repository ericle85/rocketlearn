import time
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from rocket_learn.agent.policy import Policy
from rocket_learn.agent.pretrained_policy import HardcodedAgent
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.utils.env_wrapper import RLGymV2GymWrapper
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator
from rlgym_tools.rocket_league.state_mutators.random_scoreboard_mutator import RandomScoreboardMutator

from rlgym.rocket_league.api import GameState
# --- Pure helpers ---

def _make_latest_policy_indices(policies):
    return [0 if isinstance(p, HardcodedAgent) else 1 for p in policies]


def _compute_actions_shared(policies, observations):
    """All policies are the same non-hardcoded policy — batch compute."""
    policy = policies[0]
    if isinstance(observations[0], tuple):
        obs = tuple(
            np.concatenate([o[i] for o in observations], axis=0)
            for i in range(len(observations[0]))
        )
    else:
        obs = observations
    dist = policy.get_action_distribution(obs)
    action_indices = policy.sample_action(dist)
    log_probs = policy.log_prob(dist, action_indices)
    actions = policy.env_compatible(action_indices)
    return (
        list(action_indices.cpu().numpy()),
        list(actions),
        list(log_probs.cpu().numpy()),
    )


def _compute_actions_per_agent(policies, observations, last_state):
    """Mixed policies — compute per agent."""
    all_indices, all_actions, all_log_probs = [], [], []
    for index, (policy, obs) in enumerate(zip(policies, observations)):
        if isinstance(policy, HardcodedAgent):
            actions = policy.act(last_state, index)
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)
            all_indices.append(None)
            all_actions.append(actions)
            all_log_probs.append(None)
        elif isinstance(policy, Policy):
            dist = policy.get_action_distribution(obs)
            action_indices = policy.sample_action(dist)[0]
            log_probs = policy.log_prob(dist, action_indices).item()
            actions = policy.env_compatible(action_indices)
            all_indices.append(action_indices.cpu().numpy())
            all_actions.append(actions)
            all_log_probs.append(log_probs)
        else:
            raise TypeError(f"Unknown policy type: {type(policy)}")
    return all_indices, all_actions, all_log_probs


def _compute_actions(policies, observations, last_state):
    if (
        not isinstance(policies[0], HardcodedAgent)
        and all(p == policies[0] for p in policies)
    ):
        return _compute_actions_shared(policies, observations)
    return _compute_actions_per_agent(policies, observations, last_state)


def _filter_by_latest(items, latest_policy_indices):
    return [item for i, item in enumerate(items) if latest_policy_indices[i] == 1]


def _update_score(b, o, state : GameState, last_goal_scored):
    goal_scored = state.goal_scored
    if goal_scored and not last_goal_scored:
        if state.scoring_team == 0:
            b += 1
        else:
            o += 1
    return b, o, goal_scored


def _build_eval_env(env : RLGymV2GymWrapper):
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.reward_functions.goal_reward import GoalReward
    from rlgym.rocket_league.done_conditions import NoTouchTimeoutCondition, AnyCondition
    from rlgym_tools.rocket_league.state_mutators.game_mutator import GameMutator
    from rocket_learn.utils.eval_game_utils import GameCondition, ScoreboardInitMutator, KickoffOnGoalEngine

    saved = {
        "state_mutator": env.rlgym_env.state_mutator,
        "reward_fn": env.rlgym_env.reward_fn,
        "truncation_cond": env.rlgym_env.truncation_cond,
        "termination_cond": env.rlgym_env.termination_cond,
        "transition_engine": env.rlgym_env.transition_engine,
    }

    env.rlgym_env.transition_engine = KickoffOnGoalEngine(RocketSimEngine())
    env.rlgym_env.state_mutator = MutatorSequence(
        FixedTeamSizeMutator(),
        ScoreboardInitMutator(),
        GameMutator(),
    )
    env.rlgym_env.termination_cond = AnyCondition(
        GameCondition(
            seconds_per_goal_forfeit=10 * len(env.rlgym_env.agents),
            max_overtime_seconds=300,
        ),
    )
    env.rlgym_env.truncation_cond = NoTouchTimeoutCondition(timeout_seconds=60)
    env.rlgym_env.reward_fn = GoalReward()

    return saved


def _restore_env(env : RLGymV2GymWrapper, saved):
    env.rlgym_env.transition_engine = saved["transition_engine"]
    env.rlgym_env.state_mutator = saved["state_mutator"]
    env.rlgym_env.termination_cond = saved["termination_cond"]
    env.rlgym_env.truncation_cond = saved["truncation_cond"]
    env.rlgym_env.reward_fn = saved["reward_fn"]


# --- Main ---

TICK_SKIP = 8  # physics ticks per step
PHYSICS_HZ = 120


def generate_episode(
    env: RLGymV2GymWrapper,
    policies,
    evaluate=False,
    scoreboard=None,
    progress=False,
    render=False,
) -> tuple[list[ExperienceBuffer], int]:
    """
    Create experience buffer data by interacting with the environment.

    Args:
        render: If True, call env.render() each step and throttle to real-time.
    """
    pbar = tqdm(unit=" steps") if progress else None
    tick_duration = TICK_SKIP / PHYSICS_HZ  # seconds per step at real time

    saved_eval = None
    saved_scoreboard_mutator = None

    if evaluate:
        saved_eval = _build_eval_env(env)

    if scoreboard is not None:
        saved_scoreboard_mutator = env.rlgym_env.state_mutator
        if not evaluate:
            env.rlgym_env.state_mutator = MutatorSequence(
                RandomScoreboardMutator(),
                env.rlgym_env.state_mutator,
            )
    observations = env.reset()
    last_state = env.rlgym_env.state

    latest_policy_indices = _make_latest_policy_indices(policies)
    rollouts = [ExperienceBuffer(infos=[]) for _ in range(sum(latest_policy_indices))]
    b = o = 0
    last_goal_scored = False
    result = 0

    # Real-time throttle state
    rt_start = time.perf_counter()
    next_tick_time = rt_start + tick_duration

    with torch.no_grad():
        while True:
            all_indices, all_actions, all_log_probs = _compute_actions(
                policies, observations, last_state
            )

            stacked_actions = np.vstack(all_actions)
            old_obs = observations
            observations, rewards, terminated, truncated, info = env.step(stacked_actions)

            if len(policies) <= 1:
                observations, rewards = [observations], [rewards]

            filtered_obs = _filter_by_latest(old_obs, latest_policy_indices)
            filtered_indices = _filter_by_latest(all_indices, latest_policy_indices)
            filtered_rewards = _filter_by_latest(rewards, latest_policy_indices)
            filtered_log_probs = _filter_by_latest(all_log_probs, latest_policy_indices)

            terminated_list = [terminated] * len(filtered_obs)
            truncated_list = [truncated] * len(filtered_obs)

            assert len(filtered_obs) == len(filtered_indices)
            assert len(filtered_obs) == len(filtered_rewards)
            assert len(filtered_obs) == len(filtered_log_probs)
            assert len(filtered_obs) == len(rollouts)

            if not evaluate:
                step_info = {**info, "blue_score": b, "orange_score": o}
                for exp_buf, obs, act, rew, term, trunc, log_prob in zip(
                    rollouts,
                    filtered_obs,
                    filtered_indices,
                    filtered_rewards,
                    terminated_list,
                    truncated_list,
                    filtered_log_probs,
                ):
                    exp_buf.add_step(obs, act, rew, term, trunc, log_prob, step_info)

            # Render + real-time throttle
            if render or pbar is not None:
                env.render()

            if render:
                current_time = time.perf_counter()
                wait = next_tick_time - current_time
                if wait > 0:
                    time.sleep(wait)
                next_tick_time += tick_duration

            if pbar is not None:
                pbar.update()
                igt = pbar.n * TICK_SKIP / PHYSICS_HZ
                prog_str = f"{igt // 60:02.0f}:{igt % 60:02.0f} IGT"
                if evaluate:
                    prog_str += f", BLUE {b} - {o} ORANGE"
                pbar.set_postfix_str(prog_str)

            b, o, last_goal_scored = _update_score(
                b, o, env.rlgym_env.state, last_goal_scored
            )

            if terminated or truncated:
                break

            last_state = info["state"]

    if scoreboard is not None:
        env.rlgym_env.state_mutator = saved_scoreboard_mutator

    if evaluate:
        _restore_env(env, saved_eval)
        return result

    if pbar is not None:
        pbar.close()

    return rollouts, result
