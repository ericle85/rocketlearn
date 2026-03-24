"""
Barebones learner: builds a fresh model and uploads it to Redis so workers can start collecting.
Run this once before starting workers, then replace with your real training loop.
"""

import os
import torch
from redis import Redis

from rocket_learn.agent.discrete_ff import DiscreteFF
from trueskill import Rating
from rocket_learn.rollout_generator.redis.utils import (
    _serialize_model, _serialize, MODEL_LATEST, VERSION_LATEST, N_UPDATES,
    LATEST_RATING_ID, QUALITIES, OPPONENT_MODELS
)

# --- Config (must match worker.py) ---
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]

OBS_SIZE = 172        # match your obs builder output size
N_ACTIONS = 90        # LookupTableAction default
LAYER_SIZES = (256, 256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEAM_SIZE = 1         # must match worker.py team_size
# -------------------------------------

if __name__ == "__main__":
    redis = Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
    redis.flushdb()  # WARNING: THIS CLEARS EVERYTHING IN THE REDIS DATABASE! BE CAREFUL!

    model = DiscreteFF(
        input_shape=OBS_SIZE,
        n_actions=N_ACTIONS,
        layer_sizes=LAYER_SIZES,
        device=DEVICE,
    )

    model_bytes = _serialize_model(model)
    redis.set(MODEL_LATEST, model_bytes)
    redis.set(VERSION_LATEST, 0)
    redis.set(N_UPDATES, 0)

    # Initialize rating for version 0
    gamemode = f"{TEAM_SIZE}v{TEAM_SIZE}"
    quality_key = QUALITIES.format(gamemode)
    rating_id = "0"
    redis.set(LATEST_RATING_ID, rating_id)
    default_rating = Rating()
    redis.hset(quality_key, f"{rating_id}-stochastic", _serialize([default_rating.mu, default_rating.sigma]))
    redis.hset(OPPONENT_MODELS, rating_id, model_bytes)

    print(f"Uploaded fresh model to Redis ({len(model_bytes) / 1024:.1f} KB)")
    print(f"Device: {DEVICE} | obs_size={OBS_SIZE} | n_actions={N_ACTIONS} | layers={LAYER_SIZES}")
    print("Workers can now start.")
