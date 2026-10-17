import os

import torch
from redis import Redis

from rocket_learn.agent.discrete_ff import DiscreteFF
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]

OBS_SIZE = 172
N_ACTIONS = 90
LAYER_SIZES = (256, 256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEAM_SIZE = 1
BATCH_SIZE = 16


def placeholder_train(model, buffers):
    return model


if __name__ == "__main__":
    redis = Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
    model = DiscreteFF(input_shape=OBS_SIZE, n_actions=N_ACTIONS, layer_sizes=LAYER_SIZES, device=DEVICE)
    generator = RedisRolloutGenerator(redis, model, team_size=TEAM_SIZE)
    print(f"Started run with model {generator.model_id}")

    while True:
        buffers = generator.collect(BATCH_SIZE)
        model = placeholder_train(model, buffers)
        print(f"Trained on {len(buffers)} buffers, published {generator.publish(model)}")
