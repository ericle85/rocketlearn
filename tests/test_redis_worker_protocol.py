import os
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from redis import Redis

from rocket_learn.rollout_generator.redis.utils import (
    _serialize, _serialize_model, _unserialize, io_publish_model, io_start_new_run,
    make_model_id, parse_model_id,
)
from rocket_learn.rollout_generator.redis.redis_rollout_worker import (
    build_matchup, build_rollout_payload, io_build_past_model_loader,
    io_fetch_latest_model,
)


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
            return proc, redis
        except Exception:
            time.sleep(0.05)
    proc.kill()
    raise RuntimeError("redis-server did not start")


def model_bytes(seed):
    torch.manual_seed(seed)
    return _serialize_model(torch.nn.Linear(1, 1))


def test_fetch_tag_and_payload(redis):
    first_id = io_start_new_run(redis, model_bytes(0))
    run_id, version = parse_model_id(first_id)
    assert version == 0

    model_id, agent = io_fetch_latest_model(redis, "cpu", None)
    assert model_id == first_id and agent is not None
    assert io_fetch_latest_model(redis, "cpu", model_id) == (first_id, None)

    second_id = make_model_id(run_id, 1)
    io_publish_model(redis, second_id, model_bytes(1), ("1v1",))
    model_id, agent = io_fetch_latest_model(redis, "cpu", model_id)
    assert model_id == second_id and agent is not None

    agents, versions, _, is_eval = build_matchup(
        redis=redis, current_agent=agent, load_past_model=None, n_agents=2,
        latest_model_id=model_id, evaluate=False, past_version_prob=0.0,
        pretrained_agents={}, sigma_target=1.0, full_team_evaluations=False,
    )
    assert versions == [second_id, second_id] and agents == [agent, agent] and not is_eval

    payload = _unserialize(_serialize(build_rollout_payload(
        model_id, [], versions, "uuid", "name", 0, True, False)))
    assert payload[0] == second_id
    assert list(payload[1:]) == [[], versions, "uuid", "name", 0, True, False, True]


def test_cache_keyed_by_full_model_id(redis):
    old_id = io_start_new_run(redis, model_bytes(0))
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            old_weight = io_build_past_model_loader(redis, "cpu", "t")(old_id).weight.item()
            new_id = io_start_new_run(redis, model_bytes(1))
            assert parse_model_id(new_id)[1] == parse_model_id(old_id)[1]
            new_weight = io_build_past_model_loader(redis, "cpu", "t")(new_id).weight.item()
            assert new_weight != old_weight
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    proc, redis = start_redis(free_port())
    try:
        test_fetch_tag_and_payload(redis)
        test_cache_keyed_by_full_model_id(redis)
        print("worker protocol tests passed")
    finally:
        proc.kill()
