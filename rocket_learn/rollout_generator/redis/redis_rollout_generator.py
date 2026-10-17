from itertools import islice
from typing import Iterator, List

from redis import Redis

from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.redis.utils import (
    LATEST_RATING_ID, _serialize_model, _unserialize, decode_buffers,
    io_pop_rollout, io_publish_model, io_start_new_run, make_model_id, parse_model_id,
)


def _decode(payload) -> List[ExperienceBuffer]:
    _, rollout_data, versions, _, _, _, has_obs, has_states, has_rewards = payload
    return decode_buffers(rollout_data, versions, has_obs, has_states, has_rewards)[0]


class RedisRolloutGenerator:
    def __init__(self, redis: Redis, model, team_size: int = 1, pop_timeout: float = 1.0):
        self.redis = redis
        self.pop_timeout = pop_timeout
        self.gamemodes = tuple(f"{i}v{i}" for i in range(1, team_size + 1))
        self.model_id = io_start_new_run(redis, _serialize_model(model), self.gamemodes)

    def _payloads(self) -> Iterator:
        while True:
            popped = io_pop_rollout(self.redis, self.pop_timeout)
            if popped is not None:
                yield _unserialize(popped)

    def collect(self, batch_size: int) -> List[ExperienceBuffer]:
        batch = list(islice((p for p in self._payloads() if p[0] == self.model_id), batch_size))
        assert all(p[0] == self.model_id for p in batch), "TrainsOnlyFreshRollouts violated"
        return [buffer for payload in batch for buffer in _decode(payload)]

    def publish(self, model) -> str:
        assert self.redis.get(LATEST_RATING_ID).decode() == self.model_id, "another writer published a model"
        run_id, version = parse_model_id(self.model_id)
        trained_id = make_model_id(run_id, version + 1)
        assert parse_model_id(trained_id) == (run_id, version + 1)
        io_publish_model(self.redis, trained_id, _serialize_model(model), self.gamemodes)
        self.model_id = trained_id
        return trained_id
