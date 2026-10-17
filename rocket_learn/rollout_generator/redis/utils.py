# Constants for consistent key lookup
import pickle
import zlib
from typing import List, Optional, Union, Dict, Tuple
from uuid import uuid4

import numpy as np
from redis import Redis
from trueskill import Rating
from rlgym.api import ObsBuilder, RewardFunction, ActionParser

from rocket_learn.experience_buffer import ExperienceBuffer
# from rocket_learn.utils.gamestate_encoding import encode_gamestate
import msgpack
import msgpack_numpy as m

QUALITIES = "qualities-{}"
N_UPDATES = "num-updates"
# SAVE_FREQ = "save-freq"
# MODEL_FREQ = "model-freq"

MODEL_LATEST = "model-latest"

ROLLOUTS = "rollout"
OPPONENT_MODELS = "opponent-models"
WORKER_IDS = "worker-ids"
CONTRIBUTORS = "contributors"
LATEST_RATING_ID = "latest-rating-id"
EXPERIENCE_PER_MODE = "experience-per-mode"
_ALL = (
    N_UPDATES, MODEL_LATEST, ROLLOUTS, OPPONENT_MODELS,
    WORKER_IDS, CONTRIBUTORS, LATEST_RATING_ID, EXPERIENCE_PER_MODE)

MAX_QUEUE = 1000

m.patch()


def make_model_id(run_id: str, version: int) -> str:
    return f"{run_id}-{version}"


def parse_model_id(model_id: str) -> Tuple[str, int]:
    run_id, version = model_id.rsplit("-", 1)
    return run_id, int(version)


def io_fetch_latest(redis: Redis) -> Optional[Tuple[str, bytes]]:
    blob = redis.get(MODEL_LATEST)
    return None if blob is None else tuple(_unserialize(blob))


def _queue_publish(pipe, model_id: str, model_bytes: bytes, gamemodes: Tuple[str, ...]):
    default = _serialize([Rating().mu, Rating().sigma])
    seeded = pipe.set(MODEL_LATEST, _serialize([model_id, model_bytes])) \
                 .hset(OPPONENT_MODELS, model_id, model_bytes) \
                 .set(LATEST_RATING_ID, model_id)
    for mode in gamemodes:
        for suffix in ("stochastic", "deterministic"):
            seeded = seeded.hsetnx(QUALITIES.format(mode), f"{model_id}-{suffix}", default)
    return seeded


def io_publish_model(redis: Redis, model_id: str, model_bytes: bytes,
                     gamemodes: Tuple[str, ...] = ()) -> None:
    _queue_publish(redis.pipeline(transaction=True), model_id, model_bytes, gamemodes).execute()


def io_start_new_run(redis: Redis, model_bytes: bytes, gamemodes: Tuple[str, ...] = ()) -> str:
    model_id = make_model_id(uuid4().hex, 0)
    _queue_publish(redis.pipeline(transaction=True).flushdb(), model_id, model_bytes, gamemodes).execute()
    return model_id


def io_push_rollout(redis: Redis, rollout_bytes: bytes) -> int:
    n_items, _ = (redis.pipeline(transaction=True)
                  .rpush(ROLLOUTS, rollout_bytes)
                  .ltrim(ROLLOUTS, -MAX_QUEUE, -1)
                  .execute())
    return n_items


def io_pop_rollout(redis: Redis, timeout: float) -> Optional[bytes]:
    popped = redis.blpop([ROLLOUTS], timeout=timeout)
    return None if popped is None else popped[1]


# Helper methods for easier changing of byte conversion
def _serialize(obj):
    return zlib.compress(msgpack.packb(obj))


def _unserialize(obj):
    return msgpack.unpackb(zlib.decompress(obj))


def _serialize_model(mdl):
    device = next(mdl.parameters()).device  # Must be a better way right?
    mdl_bytes = pickle.dumps(mdl.cpu())
    mdl.to(device)
    return mdl_bytes


def _unserialize_model(buf, device: str):
    import torch
    agent = pickle.loads(buf)
    device_obj = torch.device(device)
    agent.to(device_obj)
    # Update the device attribute if it exists
    if hasattr(agent, 'device'):
        agent.device = device_obj
    return agent


def get_rating(gamemode: str, model_id: Optional[str], redis: Redis) -> Union[Rating, Dict[str, Rating]]:
    """
    Get the rating of a player.
    :param gamemode: The game mode to get the rating for.
    :param model_id: The id of the model.
    :param redis: The redis client.
    :return: The rating of the player.
    """
    quality_key = QUALITIES.format(gamemode)
    if model_id is None:  # Return all ratings
        return {
            k.decode("utf-8"): Rating(*_unserialize(v))
            for k, v in redis.hgetall(quality_key).items()
        }
    return Rating(*_unserialize(redis.hget(quality_key, model_id)))


def encode_buffers(buffers: List[ExperienceBuffer], return_obs=True, return_rewards=True, return_states=False):
    from rocket_learn.utils.gamestate_encoding import encode_gamestate
    res = []

    if return_states:
        states = [[encode_gamestate(info["state"], info.get("blue_score", 0), info.get("orange_score", 0)) for info in buffer.infos if "state" in info]
                  for buffer in buffers]
        res.append(states)

    if return_obs:
        observations = [buffer.observations for buffer in buffers]
        res.append(observations)

    if return_rewards:
        rewards = np.asarray([buffer.rewards for buffer in buffers])
        res.append(rewards)

    actions = np.asarray([buffer.actions for buffer in buffers])
    log_probs = np.asarray([buffer.log_probs for buffer in buffers])
    terminated = np.asarray([buffer.terminated for buffer in buffers])
    truncated = np.asarray([buffer.truncated for buffer in buffers])
    res.append(actions)
    res.append(log_probs)
    res.append(terminated)
    res.append(truncated)

    return res

# obs_build_factory is the obs builder
# act_parse_factory is the act parser
# rew_func_factory is the rew func
def decode_buffers(enc_buffers, versions, has_obs, has_states, has_rewards):
    assert has_obs, "Must have the obs"
    assert has_states or has_obs, "Must have at least one of obs or states"
    assert has_states or has_rewards, "Must have at least one of rewards or states"
    assert not has_obs or has_obs and has_rewards, "Must have both obs and rewards"  # TODO obs+no reward?

    i = 0
    if has_states:
        game_states = enc_buffers[i]
        if len(game_states) == 0:
            raise RuntimeError
        i += 1
    else:
        game_states = None
    if has_obs:
        obs = enc_buffers[i]
        i += 1
    else:
        obs = None
    if has_rewards:
        rewards = enc_buffers[i]
        i += 1
        # dones = np.zeros_like(rewards, dtype=bool)  # TODO: Support for dones?
        # if len(dones) > 0:
        #     dones[:, -1] = True
    else:
        rewards = None
        # dones = None
    actions = enc_buffers[i]
    i += 1
    log_probs = enc_buffers[i]
    i += 1
    terminated = enc_buffers[i]
    i += 1
    truncated = enc_buffers[i]
    i += 1

    if obs is None:
        # TODO: Fix GameState construction for RLGym v2, No batch OBS builder yet
        raise NotImplementedError("GameState reconstruction not yet implemented for RLGym v2")
    else:  # We have everything we need
        buffers = []
        for i in range(len(obs)):
            buffers.append(
                ExperienceBuffer(observations=obs[i],
                                 actions=actions[i],
                                 rewards=rewards[i],
                                 terminated=terminated[i],
                                 truncated=truncated[i],
                                 log_probs=log_probs[i])
            )
        return buffers, game_states
