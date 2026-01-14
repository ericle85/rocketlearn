import os
import wandb
import numpy as np
from typing import Any
from typing import List, Dict, Any, Tuple

import torch.jit
from torch.nn import Linear, Sequential, ReLU

from redis import Redis

from rlgym.rocket_league.obs_builders.default_obs import DefaultObs
from rlgym.api import ObsBuilder, AgentID
from rlgym.rocket_league.api import Car, GameState
from rlgym.rocket_league.reward_functions import TouchReward
from rlgym.rocket_league.action_parsers import LookupTableAction

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_ff import DiscreteFF
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import SplitLayer


# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandDefaultOBS(DefaultObs):
    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[
        AgentID, np.ndarray]:
        self._state = state
        obs = {}
        for agent in agents:
            obs[agent] = np.expand_dims(self._build_obs(agent, state, shared_info), 0)
        return obs


if __name__ == "__main__":
    """
    
    Starts up a rocket-learn learner process, which ingests incoming data, updates parameters
    based on results, and sends updated model parameters out to the workers
    
    """

    # ROCKET-LEARN USES WANDB WHICH REQUIRES A LOGIN TO USE. YOU CAN SET AN ENVIRONMENTAL VARIABLE
    # OR HARDCODE IT IF YOU ARE NOT SHARING YOUR SOURCE FILES
    wandb.login(key=os.environ["WANDB_API_KEY"])
    logger = wandb.init(project="demo", entity="leeric85")
    logger.name = "DEFAULT_LEARNER_EXAMPLE"

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    redis = Redis(password=os.environ["REDIS_PASSWORD"])

    # ** ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER **
    # NOT BEING USED CURRENTLY
    def obs():
        return DefaultObs()

    def rew():
        return TouchReward()

    def act():
        return LookupTableAction()


    # THE ROLLOUT GENERATOR CAPTURES INCOMING DATA THROUGH REDIS AND PASSES IT TO THE LEARNER.
    # -save_every SPECIFIES HOW OFTEN REDIS DATABASE IS BACKED UP TO DISK
    # -model_every SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS. THESE ARE USED FOR TRUESKILL
    # COMPARISON AND TRAINING AGAINST PREVIOUS VERSIONS
    # -clear DELETE REDIS ENTRIES WHEN STARTING UP (SET TO FALSE TO CONTINUE WITH OLD AGENTS)
    rollout_gen = RedisRolloutGenerator("demo-bot", redis, obs, rew, act,
                                        logger=logger,
                                        save_every=100,
                                        model_every=100,
                                        clear=True)

    # TOTAL SIZE OF THE INPUT DATA
    state_dim = 172
    n_actions = 90
    device = "cuda"

    critic = Sequential(
        Linear(state_dim, 256),  # Layer 1
        ReLU(),
        Linear(256, 256),        # Layer 2
        ReLU(),
        Linear(256, 256),        # Layer 3
        ReLU(),
        Linear(256, 1)           # Output layer
    )

    actor =  DiscreteFF(state_dim,n_actions,[256, 256, 256], device)

    # CREATE THE OPTIMIZER
    optim = torch.optim.AdamW([
        {"params": actor.parameters(), "lr": 1e-4},
        {"params": critic.parameters(), "lr": 1e-4}
    ])

    # PPO REQUIRES AN ACTOR/CRITIC AGENT
    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    # INSTANTIATE THE PPO TRAINING ALGORITHM
    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=0.01,
        n_steps=50_000,
        batch_size=50_000,
        minibatch_size=50_000,
        epochs=2,
        gamma=0.99,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=1,
        max_grad_norm=0.5,
        logger=logger,
        device=device,
    )

    # BEGIN TRAINING. IT WILL CONTINUE UNTIL MANUALLY STOPPED
    # -iterations_per_save SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    # -save_dir SPECIFIES WHERE
    alg.run(iterations_per_save=100, save_dir="checkpoint_save_directory")
