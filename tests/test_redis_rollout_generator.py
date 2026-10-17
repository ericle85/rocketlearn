import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from redis import Redis

from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.rollout_generator.redis.utils import (
    LATEST_RATING_ID, QUALITIES, ROLLOUTS, _serialize, encode_buffers,
    io_fetch_latest, io_push_rollout, make_model_id, parse_model_id,
)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rollout(model_id, marker):
    buffer = ExperienceBuffer(observations=[np.zeros(2)] * 2, actions=[0, 1], rewards=[marker, marker],
                              terminated=[False, True], truncated=[False, False], log_probs=[0.0, 0.0])
    data = encode_buffers([buffer], return_obs=True, return_rewards=True, return_states=False)
    return _serialize((model_id, data, [0], "uuid", "name", 0, True, False, True))


def markers(buffers):
    return [float(b.rewards[0]) for b in buffers]


def test_generator(redis):
    redis.rpush(ROLLOUTS, b"leftover-from-old-run")
    generator = RedisRolloutGenerator(redis, torch.nn.Linear(2, 2), team_size=2, pop_timeout=0.1)
    assert redis.llen(ROLLOUTS) == 0

    v0 = generator.model_id
    run_id, version = parse_model_id(v0)
    assert version == 0 and io_fetch_latest(redis)[0] == v0
    assert redis.hexists(QUALITIES.format("2v2"), f"{v0}-deterministic")

    v1 = generator.publish(torch.nn.Linear(2, 2))
    assert v1 == make_model_id(run_id, 1) and io_fetch_latest(redis)[0] == v1
    assert redis.get(LATEST_RATING_ID).decode() == v1
    assert redis.hexists(QUALITIES.format("1v1"), f"{v1}-stochastic")

    io_push_rollout(redis, rollout(v0, 1.0))
    io_push_rollout(redis, rollout(make_model_id("otherrun", 1), 2.0))
    io_push_rollout(redis, rollout(v1, 3.0))
    io_push_rollout(redis, rollout(v1, 4.0))
    assert markers(generator.collect(2)) == [3.0, 4.0]
    assert redis.llen(ROLLOUTS) == 0

    v2 = generator.publish(torch.nn.Linear(2, 2))
    io_push_rollout(redis, rollout(v1, 5.0))
    io_push_rollout(redis, rollout(v2, 6.0))
    assert markers(generator.collect(1)) == [6.0]


if __name__ == "__main__":
    port = free_port()
    server = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
                              stdout=subprocess.DEVNULL)
    try:
        time.sleep(0.5)
        test_generator(Redis(port=port))
        print("test_redis_rollout_generator ok")
    finally:
        server.kill()
