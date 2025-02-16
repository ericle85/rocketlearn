from typing import Any
import numpy

from redis import Redis

from rlgym_sim.envs import Match
from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_sim.utils.reward_functions.default_reward import DefaultReward
from rlgym_sim.utils.state_setters.default_state import DefaultState
from rlgym_sim.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym_sim.utils.action_parsers.discrete_act import DiscreteAction

from custom_obs import AdvancedObsPadder
from custom_action import ImmortalAction
from rlgym_sim.utils.action_parsers.lookup_act import LookupAction

from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker

from rlgym_sim.utils.reward_functions import CombinedReward
from custom_rewards import SpeedTowardBallReward, AirReward
from rlgym_sim.utils.reward_functions.common_rewards import VelocityReward,AlignBallGoal,SaveBoostReward,FaceBallReward, VelocityBallToGoalReward, \
        EventReward

from custom_rewards import PickUpBoostReward, SpeedTowardBallReward, AirReward, TouchBallRewardScaledByHitForce, JumpTouchReward,DribbleReward, DribbleToFlickReward, PossessionReward, PlayerOnWallReward, AerialDistanceReward, WavedashReward, GoalSpeedReward
from rlgym_sim.utils.reward_functions.common_rewards.zero_sum_reward import ZeroSumReward
from dotenv import load_dotenv
import os

load_dotenv() 
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]

# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    """

    Starts up a rocket-learn worker process, which plays out a game, sends back game data to the 
    learner, and receives updated model parameters when available

    """

    # OPTIONAL ADDITION:
    # LIMIT TORCH THREADS TO 1 ON THE WORKERS TO LIMIT TOTAL RESOURCE USAGE
    # TRY WITH AND WITHOUT FOR YOUR SPECIFIC HARDWARE
    import torch
    #defualt is 1
    torch.set_num_threads(1)

    reward_fn = CombinedReward.from_zipped(
        (EventReward(touch=1), 50), # Giant reward for actually hitting the ball
        (SpeedTowardBallReward(), 5), # Move towards the ball!
        (FaceBallReward(), 1), # Make sure we don't start driving backward at the ball
        (AirReward(), 0.15) # Make sure we don't forget how to jump
    )

    from rlgym_sim.utils.state_setters import DefaultState, RandomState
    #from rlgym_sim.utils.state_setters.dribbleSetter import DribbleState
    from custom_state_setters import WeightedSampleSetter, WallPracticeState, AirDrag

    state_setter = WeightedSampleSetter.from_zipped(
        #(DefaultState(), 0.2),
        # (WallPracticeState(), 0.2),
        # (DribbleState(False,False,True), .5),
        (RandomState(True,True,False), 0.2)
        # (AirDrag(),0.2)
    ) 

    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 25
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    # BUILD THE ROCKET LEAGUE MATCH THAT WILL USED FOR TRAINING
    # -ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER
    match = Match(
        reward_function=reward_fn,
        terminal_conditions=[TimeoutCondition(timeout_ticks),
                             GoalScoredCondition()],
        obs_builder=ExpandAdvancedObs(),
        action_parser=ImmortalAction(),
        state_setter=state_setter,
        team_size=1,
        spawn_opponents=True,
    )
    match._tick_skip = 8 # because rlgym_sim doesn't support tick_skip in the match class

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    r = Redis(host="127.0.0.1", password=REDIS_PASSWORD)

    # LAUNCH ROCKET LEAGUE AND BEGIN TRAINING
    # -past_version_prob SPECIFIES HOW OFTEN OLD VERSIONS WILL BE RANDOMLY SELECTED AND TRAINED AGAINST
    RedisRolloutWorker(r, "example", match,
                       past_version_prob=.2,
                       evaluation_prob=0.01,
                       sigma_target=2,
                       dynamic_gm=False,
                       send_obs=True,
                       streamer_mode=False,
                       send_gamestates=False,
                       force_paging=False,
                       auto_minimize=True,
                       local_cache_name="example_model_database",
                       live_progress=False).run()
