import os
from typing import Any
import numpy as np

from redis import Redis

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, KickoffMutator
from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import RocketSimVisRenderer
from rlgym.rocket_league import common_values
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward
from rlgym_tools.rocket_league.state_mutators.random_physics_mutator import RandomPhysicsMutator

from rewards.kickoff_reward import KickoffReward
from rewards.common_rewards import FaceBallReward
from rewards.common_rewards import StayOnGroundReward
from rewards.touch_condition import BallTouchedCondition
if __name__ == "__main__":
    """

    Starts up a rocket-learn worker process, which plays out a game, sends back game data to the 
    learner, and receives updated model parameters when available

    """

    # OPTIONAL ADDITION:
    # LIMIT TORCH THREADS TO 1 ON THE WORKERS TO LIMIT TOTAL RESOURCE USAGE
    # TRY WITH AND WITHOUT FOR YOUR SPECIFIC HARDWARE
    import torch

    torch.set_num_threads(1)

    device = "cpu"
    spawn_opponents = True
    team_size = 1
    action_repeat = 8
    no_touch_timeout_seconds = 10
    game_timeout_seconds = 10
    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds), BallTouchedCondition())

    reward_fn = KickoffReward()

    obs_builder = DefaultObs()

    state_mutator = MutatorSequence(KickoffMutator()) # DO NOT INCLUDE TEAMSIZE MUTATOR YET
    
    # BUILD THE ROCKET LEAGUE MATCH THAT WILL BE USED FOR TRAINING
    # -ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER
    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer(),
    )

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    r = Redis(host="127.0.0.1", password=os.environ["REDIS_PASSWORD"])

    # LAUNCH ROCKET LEAGUE AND BEGIN TRAINING
    # -past_version_prob SPECIFIES HOW OFTEN OLD VERSIONS WILL BE RANDOMLY SELECTED AND TRAINED AGAINST
    RedisRolloutWorker(r, "example", env, device,
                       past_version_prob=.2,
                       evaluation_prob=0.01,
                       sigma_target=2,
                       dynamic_gm=False,
                       team_size=team_size,
                       send_obs=True,
                       streamer_mode=False,
                       send_gamestates=False,
                       local_cache_name=None).run()
