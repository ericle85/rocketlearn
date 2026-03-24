"""
Two tests for generate_episode:

1. test_fake_data       — fully mocked env + policy; verifies ExperienceBuffer structure
2. test_forward_only    — real RocketSim env with a policy hardcoded to always drive forward;
                          prints per-step rollout so you can visually sanity-check it

Run:
    python tests/test_generate_episode.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.distributions import Categorical
from unittest.mock import MagicMock, patch

from rocket_learn.agent.policy import Policy
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.utils.generate_episode import generate_episode
from rocket_learn.utils.env_wrapper import RLGymV2GymWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_SIZE = 172
N_ACTIONS = 90


class ForwardOnlyPolicy(Policy):
    """
    Always outputs action index 16 = throttle=1, steer=0, boost=0, handbrake=0.
    Verified via LookupTableAction.make_lookup_table() enumeration:
        throttle=1, steer=0, boost=0, handbrake=0 → row 16
    """
    FORWARD_IDX = 16

    def __init__(self):
        super().__init__()

    def forward(self, obs):
        pass

    def get_action_distribution(self, obs):
        if isinstance(obs, list):
            obs = np.array(obs)
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        batch = obs.shape[0] if obs.dim() > 1 else 1
        # One-hot logits so argmax == FORWARD_IDX, but keep it stochastic-compatible
        logits = torch.full((batch, N_ACTIONS), -1e9)
        logits[:, self.FORWARD_IDX] = 0.0
        return Categorical(logits=logits)

    def sample_action(self, distribution, deterministic=None):
        batch = distribution.logits.shape[0]
        return torch.full((batch,), self.FORWARD_IDX, dtype=torch.long)

    def log_prob(self, distribution, selected_action):
        return distribution.log_prob(selected_action)

    def entropy(self, distribution, selected_action):
        return distribution.entropy()

    def env_compatible(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        return action


# ---------------------------------------------------------------------------
# Test 1: fake data
# ---------------------------------------------------------------------------

def _make_mock_env(n_steps=5, obs_size=OBS_SIZE, n_agents=1):
    """
    Returns a mock RLGymV2GymWrapper that runs for n_steps then terminates.
    Observations are random, rewards cycle 0..n_steps-1, state is a MagicMock.
    """
    env = MagicMock(spec=RLGymV2GymWrapper)

    obs = np.random.rand(n_agents, obs_size).astype(np.float32)
    env.reset.return_value = obs

    step_counter = {"i": 0}

    def fake_step(actions):
        i = step_counter["i"]
        step_counter["i"] += 1
        new_obs = np.random.rand(n_agents, obs_size).astype(np.float32)
        rewards = [float(i)] * n_agents
        terminated = i >= n_steps - 1
        truncated = False
        state = MagicMock()
        state.goal_scored = False
        info = {"state": state}
        return new_obs, rewards, terminated, truncated, info

    env.step.side_effect = fake_step

    # rlgym_env.state used for last_state on first call
    env.rlgym_env = MagicMock()
    env.rlgym_env.state = MagicMock()
    env.rlgym_env.state.goal_scored = False

    return env


def test_fake_data():
    print("\n=== test_fake_data ===")
    n_steps = 5
    env = _make_mock_env(n_steps=n_steps, n_agents=1)
    policy = ForwardOnlyPolicy()

    rollouts, result = generate_episode(env, [policy])

    assert len(rollouts) == 1, f"expected 1 rollout, got {len(rollouts)}"
    buf: ExperienceBuffer = rollouts[0]

    assert buf.size() == n_steps, f"expected {n_steps} steps, got {buf.size()}"
    assert len(buf.observations) == n_steps
    assert len(buf.actions) == n_steps
    assert len(buf.rewards) == n_steps
    assert len(buf.log_probs) == n_steps
    assert len(buf.terminated) == n_steps
    assert len(buf.truncated) == n_steps

    # Rewards should be 0,1,2,...,n_steps-1 (stored as lists per agent)
    expected_rewards = [[float(i)] for i in range(n_steps)]
    assert buf.rewards == expected_rewards, f"rewards mismatch: {buf.rewards}"

    # Every action should be FORWARD_IDX
    for i, act in enumerate(buf.actions):
        idx = int(act[0]) if hasattr(act, '__len__') else int(act)
        assert idx == ForwardOnlyPolicy.FORWARD_IDX, \
            f"step {i}: expected action {ForwardOnlyPolicy.FORWARD_IDX}, got {act}"

    # terminated only on last step
    assert buf.terminated[-1] == True
    assert all(t == False for t in buf.terminated[:-1])

    print(f"  steps      : {buf.size()}")
    print(f"  rewards    : {buf.rewards}")
    print(f"  actions    : {[int(a[0]) if hasattr(a, '__len__') else int(a) for a in buf.actions]}")
    print(f"  terminated : {buf.terminated}")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 2: forward-only with real RocketSim env
# ---------------------------------------------------------------------------

def test_forward_only():
    print("\n=== test_forward_only (real sim) ===")
    try:
        from rlgym.api import RLGym
        from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
        from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, AnyCondition
        from rlgym.rocket_league.obs_builders import DefaultObs
        from rlgym.rocket_league.reward_functions import TouchReward
        from rlgym.rocket_league.sim import RocketSimEngine
        from rlgym.rocket_league.state_mutators import MutatorSequence, KickoffMutator, FixedTeamSizeMutator
        from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import RocketSimVisRenderer
    except ImportError as e:
        print(f"  SKIPPED (import error: {e})")
        return

    env = RLGymV2GymWrapper(RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(blue_size=1, orange_size=1), KickoffMutator()),
        obs_builder=DefaultObs(),
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=TouchReward(),
        termination_cond=GoalCondition(),
        truncation_cond=NoTouchTimeoutCondition(timeout_seconds=5),
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer(udp_ip="100.90.35.22"),
    ))

    # 1v1 → 2 agents, both drive forward
    policy = ForwardOnlyPolicy()
    rollouts, result = generate_episode(env, [policy, policy], render = True)
    env.close()

    assert len(rollouts) == 2  # one rollout per agent in 1v1

    for agent_idx, buf in enumerate(rollouts):
        print(f"\n  --- Agent {agent_idx} ({buf.size()} steps) ---")
        print(f"  {'step':>4}  {'action':>6}  {'reward':>8}  {'term':>5}  {'trunc':>5}")
        print(f"  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*5}")
        for i in range(buf.size()):
            act = buf.actions[i]
            act_idx = int(act[0]) if hasattr(act, '__len__') else int(act)
            rew = buf.rewards[i]
            rew_val = rew[0] if hasattr(rew, '__len__') else rew
            print(f"  {i:>4}  {act_idx:>6}  {rew_val:>8.4f}  "
                  f"{str(buf.terminated[i]):>5}  {str(buf.truncated[i]):>5}")

        # All actions must be FORWARD_IDX
        for i, act in enumerate(buf.actions):
            idx = int(act[0]) if hasattr(act, '__len__') else int(act)
            assert idx == ForwardOnlyPolicy.FORWARD_IDX, \
                f"agent {agent_idx} step {i}: action {idx} != {ForwardOnlyPolicy.FORWARD_IDX}"

    print(f"\n  All steps used action {ForwardOnlyPolicy.FORWARD_IDX} (throttle=1, steer=0)")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_fake_data()
    test_forward_only()
    print("\nAll tests done.")
