from typing import List, Dict, Any

from rlgym.api import DoneCondition, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import TICKS_PER_SECOND


class BallTouchedCondition(DoneCondition[AgentID, GameState]):

    def __init__(self):
        """
        Ends if a ball is touched
        """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        done = False
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches > 0:
                done = True
        
        return {agent: done for agent in agents}
