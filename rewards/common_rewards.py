from typing import List, Dict, Any

import math
import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER, BALL_MAX_SPEED

class FaceBallReward(RewardFunction[AgentID, GameState, float]):
    """
        Gives a reward when facing the ball
    """
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]: 
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        pos_diff = state.ball.position - car.physics.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        return float(np.dot(car.physics.forward, norm_pos_diff))
    

class VBGReward(RewardFunction[AgentID, GameState, float]):
    """
        Gives a reward for ball velocity to the goal
    """

    def __init__(self, own_goal=False):
        self.own_goal = own_goal
        

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]: 
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        if car.team_num == BLUE_TEAM and not self.own_goal \
                or car.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_CENTER)
        else:
            objective = np.array(BLUE_GOAL_CENTER)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / BALL_MAX_SPEED
        return float(np.dot(norm_pos_diff, norm_vel))
