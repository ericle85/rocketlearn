import os
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from redis import Redis

import rocket_learn.rollout_generator.redis.redis_rollout_generator as generator_module
import rocket_learn.rollout_generator.redis.utils as utils
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.rollout_generator.redis.redis_rollout_worker import (
    build_matchup, build_rollout_payload, io_build_past_model_loader, io_fetch_latest_model,
)

Model = Tuple[int, int]
NO_MODEL: Model = (-1, -1)
BATCH_SIZE = 2


@dataclass(frozen=True)
class Spec:
    latest: Model
    pool: frozenset
    queue: tuple
    wstep: tuple
    wmodel: tuple
    wopp: tuple
    lstep: str
    lmodel: Model
    lbatch: tuple


def setat(t: tuple, i: int, v) -> tuple:
    return t[:i] + (v,) + t[i + 1:]


def spec_init(n_workers: int) -> Spec:
    first = (0, 0)
    return Spec(first, frozenset({first}), (), ("fetchModel",) * n_workers, (NO_MODEL,) * n_workers,
                (NO_MODEL,) * n_workers, "collectRollouts", first, ())


def fetch_model(s: Spec, w: int) -> List[Spec]:
    return [replace(s, wmodel=setat(s.wmodel, w, s.latest), wstep=setat(s.wstep, w, "pickOpponent"))] \
        if s.wstep[w] == "fetchModel" else []


def pick_opponent(s: Spec, w: int) -> List[Spec]:
    return [replace(s, wopp=setat(s.wopp, w, o), wstep=setat(s.wstep, w, "playAndPush")) for o in s.pool] \
        if s.wstep[w] == "pickOpponent" and s.wmodel[w] in s.pool else []


def abandon_dead_run(s: Spec, w: int) -> List[Spec]:
    return [replace(s, wstep=setat(s.wstep, w, "fetchModel"))] \
        if s.wstep[w] == "pickOpponent" and s.wmodel[w] not in s.pool else []


def play_and_push(s: Spec, w: int, max_queue: int) -> List[Spec]:
    pushed = s.queue + ((s.wmodel[w], s.wopp[w]),)
    return [replace(s, queue=pushed[1:] if len(pushed) > max_queue else pushed,
                    wstep=setat(s.wstep, w, "fetchModel"))] if s.wstep[w] == "playAndPush" else []


def worker_crash(s: Spec, w: int) -> List[Spec]:
    return [replace(s, wstep=setat(s.wstep, w, "fetchModel"), wmodel=setat(s.wmodel, w, NO_MODEL),
                    wopp=setat(s.wopp, w, NO_MODEL))]


def pop_rollout(s: Spec) -> List[Spec]:
    if s.lstep != "collectRollouts" or not s.queue:
        return []
    head = s.queue[0]
    batch = s.lbatch if head[0] != s.lmodel else s.lbatch + (head,)
    return [replace(s, queue=s.queue[1:], lbatch=batch,
                    lstep="trainAndPublish" if len(batch) >= BATCH_SIZE else "collectRollouts")]


def train_and_publish(s: Spec) -> List[Spec]:
    trained = (s.lmodel[0], s.lmodel[1] + 1)
    return [replace(s, lmodel=trained, latest=trained, pool=s.pool | {trained}, lbatch=(),
                    lstep="collectRollouts")] if s.lstep == "trainAndPublish" else []


def learner_restart(s: Spec) -> List[Spec]:
    fresh = (s.lmodel[0] + 1, 0)
    return [replace(s, lmodel=fresh, latest=fresh, pool=frozenset({fresh}), queue=(), lbatch=(),
                    lstep="collectRollouts")]


def next_states(s: Spec, label: str, w: Optional[int], max_queue: int) -> List[Spec]:
    return {
        "FetchModel": lambda: fetch_model(s, w),
        "PickOpponent": lambda: pick_opponent(s, w),
        "AbandonDeadRun": lambda: abandon_dead_run(s, w),
        "PlayAndPush": lambda: play_and_push(s, w, max_queue),
        "WorkerCrashAndRestart": lambda: worker_crash(s, w),
        "PopRollout": lambda: pop_rollout(s),
        "TrainAndPublish": lambda: train_and_publish(s),
        "LearnerCrashAndStartNewRun": lambda: learner_restart(s),
    }[label]()


def trains_only_fresh_rollouts(s: Spec) -> bool:
    return all(r[0] == s.lmodel for r in s.lbatch)


def workers_never_ahead_of_learner(s: Spec) -> bool:
    return all(m[0] < s.lmodel[0] or (m[0] == s.lmodel[0] and m[1] <= s.lmodel[1]) for m in s.wmodel)


class Violation(AssertionError):
    pass


class Runs:
    def __init__(self):
        self.ids: Dict[str, int] = {}

    def model(self, model_id: Optional[str]) -> Model:
        if model_id is None:
            return NO_MODEL
        run, version = utils.parse_model_id(model_id)
        return self.ids.setdefault(run, len(self.ids)), version


def strip_quality_suffix(version: str) -> str:
    return version.rsplit("-", 1)[0] if version.endswith(("-stochastic", "-deterministic")) else version


def opponent_of(model_id: str, versions: List[str]) -> str:
    return next((strip_quality_suffix(v) for v in versions if strip_quality_suffix(v) != model_id), model_id)


@dataclass
class Worker:
    loader: Callable
    step: str = "fetchModel"
    model_id: Optional[str] = None
    agent: object = None
    opponent_id: Optional[str] = None
    versions: Optional[List[str]] = None


def rollout_bytes(model_id: str, versions: List[str], marker: float) -> bytes:
    buffer = ExperienceBuffer(observations=[np.zeros(2)] * 2, actions=[0, 1], rewards=[marker, marker],
                              terminated=[False, True], truncated=[False, False], log_probs=[0.0, 0.0])
    data = utils.encode_buffers([buffer], return_obs=True, return_rewards=True, return_states=False)
    return utils._serialize(build_rollout_payload(model_id, data, versions, "uuid", "name", 0, True, False))


class HookedRedis(Redis):
    def __init__(self, before: Callable[[], None], **kwargs):
        super().__init__(**kwargs)
        self.before = before

    def execute_command(self, *args, **kwargs):
        self.before()
        return super().execute_command(*args, **kwargs)

    def pipeline(self, transaction=True, shard_hint=None):
        pipe = super().pipeline(transaction, shard_hint)
        execute = pipe.execute
        pipe.execute = lambda *a, **k: (self.before(), execute(*a, **k))[1]
        return pipe


class TraceChecker:
    def __init__(self, port: int, n_workers: int, seed: int, interleave_prob: float, past_version_prob: float):
        self.rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.observer = Redis(port=port)
        self.interleave_prob = interleave_prob
        self.past_version_prob = past_version_prob
        self.depth = 0
        self.runs = Runs()
        self.markers: Dict[float, Tuple[str, str]] = {}
        self.trace: List[str] = []
        self.redis = [HookedRedis(self.interleave_hook(i), port=port) for i in range(n_workers + 1)]
        self.workers = [self.fresh_worker(w) for w in range(n_workers)]
        self.depth = 1
        self.learner = RedisRolloutGenerator(self.redis[n_workers], torch.nn.Linear(2, 2), pop_timeout=0.01)
        self.depth = 0
        self.states = [spec_init(n_workers)]
        self.check("Init", self.observe())

    def fresh_worker(self, w: int) -> Worker:
        return Worker(loader=io_build_past_model_loader(self.redis[w], "cpu", None))

    def interleave_hook(self, actor: int) -> Callable[[], None]:
        def hook():
            if self.depth == 0 and self.rng.random() < self.interleave_prob:
                self.depth += 1
                others = [w for w in range(len(self.workers)) if w != actor]
                if others:
                    self.worker_step(self.rng.choice(others))
                self.depth -= 1
        return hook

    def observe(self) -> Dict:
        latest = utils.io_fetch_latest(self.observer)
        queue = [utils._unserialize(b) for b in self.observer.lrange(utils.ROLLOUTS, 0, -1)]
        return dict(
            latest=self.runs.model(latest[0]),
            pool=frozenset(self.runs.model(k.decode()) for k in self.observer.hkeys(utils.OPPONENT_MODELS)),
            queue=tuple((self.runs.model(p[0]), self.runs.model(opponent_of(p[0], p[2]))) for p in queue),
            wstep=tuple(w.step for w in self.workers),
            wmodel=tuple(self.runs.model(w.model_id) for w in self.workers),
            wopp=tuple(self.runs.model(w.opponent_id) for w in self.workers),
            lmodel=self.runs.model(self.learner.model_id),
        )

    def check(self, label: str, obs: Dict, w: Optional[int] = None, extra: Callable[[Spec], bool] = lambda s: True):
        self.trace.append(label if w is None else f"{label}(w{w})")
        candidates = self.states if label == "Init" else [
            n for s in self.states for n in next_states(s, label, w, utils.MAX_QUEUE)]
        matched = [s for s in candidates if all(getattr(s, k) == v for k, v in obs.items()) and extra(s)]
        if not matched:
            raise Violation(f"step {len(self.trace)} {self.trace[-1]} not allowed by spec\n"
                            f"observed: {obs}\nspec candidates: {candidates}\ntrace: {self.trace}")
        bad = [s for s in matched if not (trains_only_fresh_rollouts(s) and workers_never_ahead_of_learner(s))]
        if bad:
            raise Violation(f"invariant broken after {self.trace[-1]}: {bad}\ntrace: {self.trace}")
        self.states = list(dict.fromkeys(matched))

    def run_action(self, label: str, w: Optional[int], action: Callable[[], Optional[str]]):
        try:
            label = action() or label
        except Violation:
            raise
        except Exception as e:
            raise Violation(f"code crashed taking spec step {label}{'' if w is None else f'(w{w})'}: "
                            f"{type(e).__name__}: {e}\ntrace: {self.trace}") from e
        self.check(label, self.observe(), w)

    def worker_step(self, w: int):
        worker = self.workers[w]
        if self.rng.random() < 0.05:
            self.workers[w] = self.fresh_worker(w)
            return self.check("WorkerCrashAndRestart", self.observe(), w)
        redis = self.redis[w]

        def fetch():
            model_id, agent = io_fetch_latest_model(redis, "cpu", worker.model_id)
            worker.model_id, worker.agent = model_id, agent or worker.agent
            worker.step = "pickOpponent"

        def pick():
            matchup = build_matchup(
                redis=redis, current_agent=worker.agent, load_past_model=worker.loader, n_agents=2,
                latest_model_id=worker.model_id, evaluate=False, past_version_prob=self.past_version_prob,
                pretrained_agents={}, sigma_target=1.0, full_team_evaluations=False)
            if matchup is None:
                worker.step = "fetchModel"
                return "AbandonDeadRun"
            versions = matchup[1]
            worker.versions, worker.opponent_id = versions, opponent_of(worker.model_id, versions)
            worker.step = "playAndPush"

        def push():
            marker = float(len(self.markers) + 1)
            self.markers[marker] = (worker.model_id, worker.opponent_id)
            utils.io_push_rollout(redis, rollout_bytes(worker.model_id, worker.versions, marker))
            worker.step = "fetchModel"

        label, action = {"fetchModel": ("FetchModel", fetch), "pickOpponent": ("PickOpponent", pick),
                         "playAndPush": ("PlayAndPush", push)}[worker.step]
        self.run_action(label, w, action)

    def learner_step(self):
        if self.rng.random() < 0.1:
            def restart():
                self.learner = RedisRolloutGenerator(self.learner.redis, torch.nn.Linear(2, 2), pop_timeout=0.01)
            return self.run_action("LearnerCrashAndStartNewRun", None, restart)

        real_pop = generator_module.io_pop_rollout

        def pop(redis, timeout):
            popped = real_pop(redis, timeout)
            if popped is not None:
                self.check("PopRollout", self.observe())
            elif self.depth == 0:
                self.depth += 1
                self.worker_step(self.rng.randrange(len(self.workers)))
                self.depth -= 1
            return popped

        generator_module.io_pop_rollout = pop
        try:
            buffers = self.learner.collect(BATCH_SIZE)
        finally:
            generator_module.io_pop_rollout = real_pop
        got = [self.markers[m] for m in dict.fromkeys(float(b.rewards[0]) for b in buffers)]
        batch = tuple((self.runs.model(m), self.runs.model(o)) for m, o in got)
        matched = [s for s in self.states if s.lstep == "trainAndPublish" and s.lbatch == batch]
        if not matched:
            raise Violation(f"collect returned {batch}, spec batches: {[(s.lstep, s.lbatch) for s in self.states]}"
                            f"\ntrace: {self.trace}")
        self.states = matched
        self.run_action("TrainAndPublish", None, lambda: self.learner.publish(torch.nn.Linear(2, 2)) and None)

    def run(self, steps: int):
        for _ in range(steps):
            actor = self.rng.randrange(len(self.workers) + 1)
            if actor == len(self.workers):
                self.learner_step()
            else:
                self.worker_step(actor)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_redis(port):
    proc = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
                            stdout=subprocess.DEVNULL)
    redis = Redis(port=port)
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            redis.ping()
            return proc
        except Exception:
            time.sleep(0.05)
    proc.kill()
    raise RuntimeError("redis-server did not start")


def test_traces_refine_spec(port, seeds=range(20), steps=150, interleave_prob=0.3, past_version_prob=0.5):
    utils.MAX_QUEUE = 3
    failures = []
    for seed in seeds:
        try:
            TraceChecker(port, n_workers=2, seed=seed, interleave_prob=interleave_prob,
                         past_version_prob=past_version_prob).run(steps)
        except Violation as v:
            failures.append(f"seed {seed}: {v}")
    assert not failures, "\n\n".join(failures)


if __name__ == "__main__":
    port = free_port()
    proc = start_redis(port)
    try:
        test_traces_refine_spec(port)
        print("all traces refine tla/Training.tla")
    finally:
        proc.kill()
