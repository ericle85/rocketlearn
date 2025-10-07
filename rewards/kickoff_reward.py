from typing import List, Dict, Any

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState

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