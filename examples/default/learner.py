import os
import wandb
import numpy
from typing import Any


import torch.jit
from torch.nn import Linear, Sequential, ReLU

from redis import Redis

from rlgym_sim.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_sim.utils.reward_functions.default_reward import DefaultReward
from rlgym_sim.utils.action_parsers.discrete_act import DiscreteAction
from custom_rewards import SpeedTowardBallReward, AirReward
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import VelocityReward,AlignBallGoal,SaveBoostReward,FaceBallReward, VelocityBallToGoalReward, \
        EventReward

from custom_action import ImmortalAction
from rlgym_sim.utils.action_parsers.lookup_act import LookupAction
from custom_obs import AdvancedObsPadder

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import SplitLayer

from rocket_learn.utils.stat_trackers.common_trackers import Touch, DistToBall,EpisodeLength, Speed, TimeoutRate, Boost, BehindBall, TouchHeight, AirTouch, AirTouchHeight, BallHeight, BallSpeed, CarOnGround, GoalSpeed, MaxGoalSpeed

# load dotenv
from dotenv import load_dotenv

load_dotenv()

WANDB_API_KEY = os.environ["WANDB_API_KEY"]
WANDB_USERNAME = "leeric85"
WANDB_PROJECT = os.environ["WANDB_PROJECT"]
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]

# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    """
    
    Starts up a rocket-learn learner process, which ingests incoming data, updates parameters
    based on results, and sends updated model parameters out to the workers
    
    """

    # ROCKET-LEARN USES WANDB WHICH REQUIRES A LOGIN TO USE. YOU CAN SET AN ENVIRONMENTAL VARIABLE
    # OR HARDCODE IT IF YOU ARE NOT SHARING YOUR SOURCE FILES
    wandb.login(key=WANDB_API_KEY)
    logger = wandb.init(project=WANDB_PROJECT, entity=WANDB_USERNAME)
    logger.name = "DEFAULT_LEARNER_EXAMPLE"

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    redis = Redis(password=REDIS_PASSWORD)

    # ** ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER **
    def obs():
        return ExpandAdvancedObs()

    def rew():
        return CombinedReward.from_zipped(
        (EventReward(touch=1), 50), # Giant reward for actually hitting the ball
        (SpeedTowardBallReward(), 5), # Move towards the ball!
        (FaceBallReward(), 1), # Make sure we don't start driving backward at the ball
        (AirReward(), 0.15) # Make sure we don't forget how to jump
    )

    def act():
        return ImmortalAction()
    
    stat_trackers = [
        Speed(),TimeoutRate(), Touch(), EpisodeLength(), Boost(), BehindBall(), TouchHeight(),
        DistToBall(), AirTouch(), AirTouchHeight(), BallHeight(), BallSpeed(), CarOnGround(),
        GoalSpeed(), MaxGoalSpeed(),
    ]



    # THE ROLLOUT GENERATOR CAPTURES INCOMING DATA THROUGH REDIS AND PASSES IT TO THE LEARNER.
    # -save_every SPECIFIES HOW OFTEN REDIS DATABASE IS BACKED UP TO DISK
    # -model_every SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS. THESE ARE USED FOR TRUESKILL
    # COMPARISON AND TRAINING AGAINST PREVIOUS VERSIONS
    # -clear DELETE REDIS ENTRIES WHEN STARTING UP (SET TO FALSE TO CONTINUE WITH OLD AGENTS)
    rollout_gen = RedisRolloutGenerator("demo-bot", redis,
                                        obs,
                                        rew,
                                        act,
                                        logger=logger,
                                        save_every=100,
                                        model_every=1000,
                                        clear=True,
                                        stat_trackers=stat_trackers)

    # ROCKET-LEARN EXPECTS A SET OF DISTRIBUTIONS FOR EACH ACTION FROM THE NETWORK, NOT
    # THE ACTIONS THEMSELVES. SEE network_setup.readme.txt FOR MORE INFORMATION
    split = (3, 3, 3, 3, 3, 2, 2, 2)
    total_output = sum(split)

    # TOTAL SIZE OF THE INPUT DATA
    state_dim = 107

    critic = Sequential(
        Linear(state_dim, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, 1)  # Output a single value (state value)
    )

    actor = DiscretePolicy(Sequential(
        Linear(state_dim, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, total_output),
        SplitLayer(splits=split)
    ), split)

    # CREATE THE OPTIMIZER
    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": 2e-4},
        {"params": critic.parameters(), "lr": 2e-4}
    ])

    # PPO REQUIRES AN ACTOR/CRITIC AGENT
    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    # INSTANTIATE THE PPO TRAINING ALGORITHM
    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=0.01,
        n_steps=100_000,
        batch_size=100_000,
        minibatch_size=100_000,
        epochs=20,
        gamma=599 / 600,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=1,
        max_grad_norm=0.5,
        logger=logger,
        device="cpu",
        zero_grads_with_none=True,
    )

    # BEGIN TRAINING. IT WILL CONTINUE UNTIL MANUALLY STOPPED
    # -iterations_per_save SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    # -save_dir SPECIFIES WHERE
    # file_path = r'/Users/ericle/rocket-learn-sim-master/checkpoint_save_directory/checkpoint.pt'
    #alg.load('/Users/ericle/rocket-learn-sim-master/checkpoint_save_directory/rocket-learn_latest/rocket-learn_latest\checkpoint.pt')
    alg.run(iterations_per_save=100, save_dir='/Users/ericle/rocket-learn-sim-master/checkpoint_save_directory/')
