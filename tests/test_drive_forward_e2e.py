import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from rlgym.api import RLGym, ActionParser, StateMutator
from rlgym.rocket_league.action_parsers import RepeatAction
from rlgym.rocket_league.done_conditions import TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, KickoffMutator, FixedTeamSizeMutator
from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import RocketSimVisRenderer

from rocket_learn.agent.policy import Policy
from rocket_learn.utils.env_wrapper import RLGymV2GymWrapper
from rocket_learn.utils.generate_episode import generate_episode

FORWARD = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
IDLE = np.zeros(8, dtype=np.float32)
CAR_Y = slice(53, 54)
POS_COEF = 1 / 2300
STEP_SECONDS = 8 / 120


class RawAction(ActionParser):
    def get_action_space(self, agent):
        return "real", 8

    def reset(self, agents, initial_state, shared_info):
        pass

    def parse_actions(self, actions, state, shared_info):
        return {agent: np.asarray(action, dtype=np.float32) for agent, action in actions.items()}


class FacePositiveY(StateMutator):
    def apply(self, state, shared_info):
        car = state.cars["blue-0"]
        car.physics.position = np.array([0.0, -2000.0, 17.0], dtype=np.float32)
        car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
        car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        car.physics.euler_angles = np.array([0.0, np.pi / 2, 0.0], dtype=np.float32)
        state.ball.position = np.array([3000.0, 0.0, 93.0], dtype=np.float32)


class FixedActionPolicy(Policy):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, obs):
        pass

    def get_action_distribution(self, obs):
        return 1 if np.ndim(obs) == 1 else len(obs)

    def sample_action(self, distribution, deterministic=None):
        return torch.from_numpy(np.tile(self.action, (distribution, 1)))

    def log_prob(self, distribution, selected_action):
        return torch.zeros(distribution)

    def entropy(self, distribution, selected_action):
        return torch.zeros(distribution)

    def env_compatible(self, action):
        return action.cpu().numpy()


def make_env():
    return RLGymV2GymWrapper(RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(blue_size=1, orange_size=1), KickoffMutator(), FacePositiveY()),
        obs_builder=DefaultObs(),
        action_parser=RepeatAction(RawAction(), repeats=8),
        reward_fn=TouchReward(),
        termination_cond=TimeoutCondition(timeout_seconds=2),
        truncation_cond=TimeoutCondition(timeout_seconds=2),
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer(),
    ))


def test_drive_forward_y_increases():
    env = make_env()
    obs_shape = env.observation_space.shape
    rollouts, _ = generate_episode(env, [FixedActionPolicy(FORWARD), FixedActionPolicy(IDLE)], render=True)
    env.close()

    shapes = [np.shape(obs) for buf in rollouts for obs in buf.observations]
    assert all(s == obs_shape for s in shapes), shapes[:3]

    driver_ys, idle_ys = (np.array([obs[CAR_Y][0] for obs in buf.observations]) / POS_COEF for buf in rollouts)
    velocities = np.diff(driver_ys) / STEP_SECONDS
    print("\n".join(f"step {i:>2}  y {y:>8.1f}  v {v:>7.1f} uu/s" for i, (y, v) in enumerate(zip(driver_ys[1:], velocities), 1)))

    assert len(driver_ys) > 10
    assert np.isclose(driver_ys[0], -2000.0, atol=1.0), driver_ys[0]
    assert np.all(np.diff(driver_ys) > 0), driver_ys
    assert np.allclose(idle_ys, idle_ys[0], atol=1.0), idle_ys
    assert all(np.array_equal(a, FORWARD) for a in rollouts[0].actions)
    assert all(np.array_equal(a, IDLE) for a in rollouts[1].actions)


if __name__ == "__main__":
    test_drive_forward_y_increases()
    print("ok")
