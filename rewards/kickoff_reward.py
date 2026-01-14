from typing import List, Dict, Any

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM, BALL_RADIUS, CAR_MAX_SPEED

class FirstTouchReward(RewardFunction[AgentID, GameState, float]):
    """
        Gives a reward to the team who touches the ball first
    """

    def __init__(self):
        self.reward_given = False
        self.first_touch_team = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.reward_given = False
        self.first_touch_team = None

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:

        if not self.reward_given:
            for car in state.cars.values():
                if car.ball_touches > 0:
                    self.first_touch_team = car.team_num
                    self.reward_given = True
                    return {agent: self._get_reward(agent, state) for agent in agents}
        return {agent: 0 for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        if car.team_num == self.first_touch_team:
            return 1
        else:
            return -1
        
class KickoffReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards acceleration toward the ball and first touch during kickoff.
    Blue team gets positive rewards, orange team gets negative (zero-sum).
    """
    def __init__(self, win_w=10, accel_w=0.5, touch_w=1):
        self.win_w = win_w
        self.accel_w = accel_w
        self.touch_w = touch_w
        self.last_state = None
        self.current_state = None
        self.last_cars = {}  # Store previous car states by agent
        self.rewards = 0

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_state = None
        self.current_state = initial_state
        self.last_cars = {agent: initial_state.cars[agent] for agent in agents}
        self.rewards = 0

    def _maybe_update_rewards(self, agents: List[AgentID], state: GameState, final=False):
        if state == self.current_state:
            return
        self.last_state = self.current_state
        self.current_state = state
        self.rewards = 0

        for agent in agents:
            old_car = self.last_cars[agent]
            new_car = state.cars[agent]

            diff_vel = (new_car.physics.linear_velocity - old_car.physics.linear_velocity) / CAR_MAX_SPEED
            diff_pos = state.ball.position - new_car.physics.position
            norm_diff_pos = diff_pos / np.linalg.norm(diff_pos)
            rew = self.accel_w * np.dot(diff_vel, norm_diff_pos)

            # Only reward first touch (when ball was at center)
            if new_car.ball_touches > old_car.ball_touches and (self.last_state.ball.position[:2] == 0).all():
                rew += self.touch_w

            if new_car.team_num == BLUE_TEAM:
                self.rewards += rew
            else:
                self.rewards -= rew

        # Update last_cars for next step
        self.last_cars = {agent: state.cars[agent] for agent in agents}

        if final and abs(state.ball.position[1]) > BALL_RADIUS:
            self.rewards += self.win_w * (2 * (state.ball.position[1] > 0) - 1)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        self._maybe_update_rewards(agents, state, final=False)
        return {agent: self.rewards if state.cars[agent].team_num == BLUE_TEAM else -self.rewards
                for agent in agents}