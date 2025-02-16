import numpy as np

from rlgym_sim.utils import math
from rlgym_sim.utils.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED, BALL_RADIUS
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.common_values import CEILING_Z, BACK_WALL_Y
import time
from typing import Optional

class StrongTouchReward(RewardFunction):
    def __init__(self, threshold = 500):
        super().__init__()
        self.prev_ball_velocity = None
        self.threshold = threshold

    def reset(self, initial_state: GameState):
        # Reset the previous ball velocity
        self.prev_ball_velocity = None


    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0
        
        # Check if the player touched the ball
        if player.ball_touched:
            current_ball_velocity = np.array(state.ball.linear_velocity)
            
            if self.prev_ball_velocity is not None:
                # Calculate the distance (change in velocity)
                velocity_change = np.linalg.norm(current_ball_velocity - self.prev_ball_velocity)
                
                if velocity_change < self.threshold:
                    return 0
                # Scale the reward based on the strength of the touch
                reward = np.sqrt(velocity_change)
            
            # Update the previous ball velocity
            self.prev_ball_velocity = current_ball_velocity

        return reward

class JumpTouchReward(RewardFunction):
    """
    a ball touch reward that only triggers when the agent's wheels aren't in contact with the floor
    adjust minimum ball height required for reward with 'min_height' as well as reward scaling with 'exp'
    """

    def __init__(self, min_height=92, exp=0.2):
        self.min_height = min_height
        self.exp = exp

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
            ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return ((state.ball.position[2] - 92) ** self.exp)-1

        return 0
        

KPH_TO_VEL = 250/9
class TouchBallRewardScaledByHitForce(RewardFunction):
    def __init__(self):
        super().__init__()
        self.max_hit_speed = 130 * KPH_TO_VEL
        self.last_ball_vel = None
        self.cur_ball_vel = None

    # game reset, after terminal condition
    def reset(self, initial_state: GameState):
        self.last_ball_vel = initial_state.ball.linear_velocity
        self.cur_ball_vel = initial_state.ball.linear_velocity

    # happens 
    def pre_step(self, state: GameState):
        self.last_ball_vel = self.cur_ball_vel
        self.cur_ball_vel = state.ball.linear_velocity

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            reward = np.linalg.norm(self.cur_ball_vel - self.last_ball_vel) / self.max_hit_speed
            return reward
        return 0    
    

class CutIntoBallReward(RewardFunction):
    def __init__(self, direction_change_threshold=0.4):
        super().__init__()
        self.direction_change_threshold = direction_change_threshold
        self.prev_velocity = None
        self.cur_velocity = None

    def reset(self, initial_state: GameState):
        self.prev_velocity = None
        self.cur_velocity = None

    def pre_step(self, state: GameState, player: PlayerData):
        self.prev_velocity = self.cur_velocity
        # Assuming we have access to player's velocity in the state
        self.cur_velocity = player.car_data.linear_velocity

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched and self.prev_velocity is not None:
            velocity_change = np.linalg.norm(self.cur_velocity - self.prev_velocity)
            
            if velocity_change > self.direction_change_threshold:
                return velocity_change
        
        return 0


class PlayerOnWallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return player.on_ground and player.car_data.position[2] > 300
    

class FlickReward(RewardFunction):
    def __init__(self, ball_speed_threshold=500, flip_acceleration_threshold=1000, distance_threshold=300, position_threshold=50):
        super().__init__()
        self.ball_speed_threshold = ball_speed_threshold
        self.flip_acceleration_threshold = flip_acceleration_threshold
        self.distance_threshold = distance_threshold
        self.position_threshold = position_threshold
        self.prev_ball_velocity = np.array([0, 0, 0])
        self.prev_car_orientation = np.array([0, 0, 0])

    def reset(self, initial_state: GameState):
        self.prev_ball_velocity = np.array([0, 0, 0])
        self.prev_car_orientation = np.array([0, 0, 0])

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0

        # Calculate the distance between the car and the ball
        car_position = player.car_data.position
        ball_position = state.ball.position
        distance_to_ball = np.linalg.norm(car_position - ball_position)

        # Calculate the velocity of the ball
        current_ball_velocity = np.array(state.ball.linear_velocity)
        ball_speed_increase = np.linalg.norm(current_ball_velocity - self.prev_ball_velocity)

        # Calculate the change in orientation (flip detection)
        current_car_orientation = player.car_data.rotation_mtx()[:, 0]  # forward direction of the car
        flip_acceleration = np.linalg.norm(current_car_orientation - self.prev_car_orientation)

        # Check if the ball is on top of the car (x and y positions are close, and z is higher for the ball)
        ball_on_top = (abs(car_position[0] - ball_position[0]) < self.position_threshold and
                       abs(car_position[1] - ball_position[1]) < self.position_threshold and
                       ball_position[2] > car_position[2])

        # Check if the conditions for a flick are met
        if (distance_to_ball < self.distance_threshold and
                ball_speed_increase > self.ball_speed_threshold and
                flip_acceleration > self.flip_acceleration_threshold and
                ball_on_top):
            reward = 1.0  # or scale as needed

        # Update previous states
        self.prev_ball_velocity = current_ball_velocity
        self.prev_car_orientation = current_car_orientation

        return reward


class PickUpBoostReward(RewardFunction):
    def init(self) -> None:
        super().init()
        self.last = 0

    def reset(self, initial_state: GameState) -> None:
        self.last = 0

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.car_data.position[2] < 2 * BALL_RADIUS:
            boost_diff = player.boost_amount - self.last
            self.last = player.boost_amount
            return -boost_diff

        return 0



RAMP_HEIGHT = 256

class AerialDistanceReward(RewardFunction):
    def __init__(self, height_scale: float, distance_scale: float):
        super().__init__()
        self.height_scale = height_scale
        self.distance_scale = distance_scale

        self.current_car: Optional[PlayerData] = None
        self.prev_state: Optional[GameState] = None
        self.ball_distance: float = 0
        self.car_distance: float = 0

    def reset(self, initial_state: GameState):
        self.current_car = None
        self.prev_state = initial_state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = 0
        is_current = self.current_car is not None and self.current_car.car_id == player.car_id
        # Test if player is on the ground
        if player.car_data.position[2] < RAMP_HEIGHT:
            if is_current:
                is_current = False
                self.current_car = None
        # First non ground touch detection
        elif player.ball_touched and not is_current:
            is_current = True
            self.ball_distance = 0
            self.car_distance = 0
            rew = self.height_scale * max(player.car_data.position[2] + state.ball.position[2] - 2 * RAMP_HEIGHT, 0)
        # Still off the ground after a touch, add distance and reward for more touches
        elif is_current:
            self.car_distance += np.linalg.norm(player.car_data.position - self.current_car.car_data.position)
            self.ball_distance += np.linalg.norm(state.ball.position - self.prev_state.ball.position)
            # Cash out on touches
            if player.ball_touched:
                rew = self.distance_scale * (self.car_distance + self.ball_distance)
                self.car_distance = 0
                self.ball_distance = 0

        if is_current:
            self.current_car = player  # Update to get latest physics info

        self.prev_state = state

        return rew / (2 * BACK_WALL_Y)
    

class DribbleReward(RewardFunction):
    def __init__(self):
        super().__init__()

        self.MIN_BALL_HEIGHT = 109.0
        self.MAX_BALL_HEIGHT = 180.0
        self.MAX_DISTANCE = 200.0
        self.SPEED_MATCH_FACTOR = 2.0
        self.BALL_RADIUS = BALL_RADIUS
        self.BALL_RADIUS_MULTIPLIER = 3

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action):
        # Close OP
        opponent_distances = [
            np.linalg.norm(opponent.car_data.position - state.ball.position)
            for opponent in state.players
            if opponent.team_num != player.team_num
        ]
        closest_opponent_distance = min(opponent_distances)

        # Distancia Check
        if closest_opponent_distance > self.BALL_RADIUS_MULTIPLIER * self.BALL_RADIUS:
            if player.on_ground and self.MIN_BALL_HEIGHT <= state.ball.position[2] <= self.MAX_BALL_HEIGHT \
                and np.linalg.norm(player.car_data.position - state.ball.position) < self.MAX_DISTANCE:
                
                player_speed = np.linalg.norm(player.car_data.linear_velocity)
                ball_speed = np.linalg.norm(state.ball.linear_velocity)
                speed_reward = (player_speed / CAR_MAX_SPEED) + self.SPEED_MATCH_FACTOR * (1 - abs(player_speed - ball_speed) / (player_speed + ball_speed))
                
                #print("Dribbling!")
                return speed_reward

        return 0.0  
    

class DribbleToFlickReward(RewardFunction):
    def __init__(self, flick_w=1.0, ticks_to_check=7):
        super().__init__()
        self.flick_w = flick_w
        self.ticks_to_check = ticks_to_check
        self.player_states = {}

    def reset(self, initial_state: GameState):
        # Reset states for all players
        self.player_states = {}

    def get_reward(self, player: PlayerData, state: GameState, previous_action):
        player_id = player.car_id  # Assuming car_id is a unique identifier for each player

        # Initialize player state if not already done
        if player_id not in self.player_states:
            self.player_states[player_id] = {
                "counter": 0,
                "prev_ball_velocity": None,
                "prev_has_jump": True,
                "prev_dribbling": False
            }

        # Retrieve player-specific state
        player_state = self.player_states[player_id]

        car_pos = player.car_data.position
        ball_pos = state.ball.position
        ball_velocity = state.ball.linear_velocity

        distance_to_ball = np.linalg.norm(car_pos - ball_pos)
        on_ground = player.on_ground
        has_jump = player.has_flip

        is_dribbling = distance_to_ball < 3 * BALL_RADIUS and ball_pos[2] > car_pos[2]

        reward = 0.0

        # Start counter after jump
        if player_state["prev_dribbling"] and not has_jump and player_state["prev_has_jump"] and not on_ground:
            player_state["counter"] = self.ticks_to_check
            player_state["prev_ball_velocity"] = ball_velocity

        # Check for flick after specified ticks
        if player_state["counter"] > 0:
            player_state["counter"] -= 1
            if player_state["counter"] == 0:
                # Calculate ball speed difference
                ball_speed_change = np.linalg.norm(ball_velocity) - np.linalg.norm(player_state["prev_ball_velocity"])
                if ball_speed_change > 3:
                    reward = self.flick_w * ball_speed_change / CAR_MAX_SPEED

        # Update states for next calculation
        player_state["prev_has_jump"] = has_jump
        player_state["prev_dribbling"] = is_dribbling

        return reward
        
class AirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array
        
        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0
        

class JumpTouchReward(RewardFunction):
    """
    a ball touch reward that only triggers when the agent's wheels aren't in contact with the floor
    adjust minimum ball height required for reward with 'min_height' as well as reward scaling with 'exp'
    """

    def __init__(self, min_height=92, exp=0.2):
        self.min_height = min_height
        self.exp = exp

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
            ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return ((state.ball.position[2] - 92) ** self.exp)-1

        return 0
    
class PlayerOnWallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return player.on_ground and player.car_data.position[2] > 300
    

class SpeedTowardBallReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Velocity of our player
        player_vel = player.car_data.linear_velocity
        
        # Difference in position between our player and the ball
        # When getting the change needed to reach B from A, we can use the formula: (B - A)
        pos_diff = (state.ball.position - player.car_data.position)
        
        # Determine the distance to the ball
        # The distance is just the length of pos_diff
        dist_to_ball = np.linalg.norm(pos_diff)
        
        # We will now normalize our pos_diff vector, so that it has a length/magnitude of 1
        # This will give us the direction to the ball, instead of the difference in position
        # Normalizing a vector can be done by dividing the vector by its length
        dir_to_ball = pos_diff / dist_to_ball

        # Use a dot product to determine how much of our velocity is in this direction
        # Note that this will go negative when we are going away from the ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)
        
        if speed_toward_ball > 0:
            # We are moving toward the ball at a speed of "speed_toward_ball"
            # The maximum speed we can move toward the ball is the maximum car speed
            # We want to return a reward from 0 to 1, so we need to divide our "speed_toward_ball" by the max player speed
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            # We are not moving toward the ball
            # Many good behaviors require moving away from the ball, so I highly recommend you don't punish moving away
            # We'll just not give any reward
            return 0
        
class ReddDribble(RewardFunction):
    def __init__(self):
        super().__init__()

        self.MIN_BALL_HEIGHT = 109.0
        self.MAX_BALL_HEIGHT = 180.0
        self.MAX_DISTANCE = 197.0
        self.SPEED_MATCH_FACTOR = 2.0

    def reset(self, initial_state : GameState):
        pass

    def get_reward(self, player : PlayerData, state : GameState, previous_action):
        if player.on_ground and state.ball.position[2] >= self.MIN_BALL_HEIGHT \
            and state.ball.position[2] <= self.MAX_BALL_HEIGHT and np.linalg.norm(player.car_data.position - state.ball.position) < self.MAX_DISTANCE:
            
            player_speed = np.linalg.norm(player.car_data.linear_velocity)
            ball_speed = np.linalg.norm(state.ball.linear_velocity)
            speed_reward = (player_speed / CAR_MAX_SPEED) + self.SPEED_MATCH_FACTOR * (1 - abs(player_speed - ball_speed) / (player_speed + ball_speed))
        
            return speed_reward

        return 0.0
    

class GoalSpeedReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.blue_score = 0
        self.orange_score = 0
        self.prev_ball_velocity = None

    def reset(self, initial_state: GameState):
        self.blue_score = initial_state.blue_score
        self.orange_score = initial_state.orange_score
        self.prev_ball_velocity = np.array(initial_state.ball.linear_velocity)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0

        # Check for score change
        if (player.team_num == 0 and state.blue_score > self.blue_score) or \
           (player.team_num == 1 and state.orange_score > self.orange_score):
            
            # Update scores
            self.blue_score = state.blue_score
            self.orange_score = state.orange_score

            # Calculate ball speed
            ball_speed = np.linalg.norm(state.ball.linear_velocity)

            # Calculate reward based on ball speed and placement
            reward = ball_speed

            # Optional: Add more conditions to reward based on goal placement

        # Update previous ball velocity
        self.prev_ball_velocity = np.array(state.ball.linear_velocity)

        return reward

class PossessionReward(RewardFunction):
    def __init__(self, max_distance=2300):
        """
        Initializes the PossessionReward.
        :param max_distance: Maximum possible distance to normalize the reward.
        """
        self.max_distance = max_distance

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """
        Calculate the reward based on the difference between the closest player's distance to the ball on each team.
        Returns a float between -1 and 1.
        """
        ball_pos = np.array(state.ball.position)

        # Initialize lists to hold distances of all players to the ball
        team_distances = []
        opponent_distances = []

        for p in state.players:
            p_pos = np.array(p.car_data.position)
            dist_to_ball = np.linalg.norm(p_pos - ball_pos)

            if p.team_num == player.team_num:
                # Append distance of player's team members
                team_distances.append(dist_to_ball)
            else:
                # Append distance of opponent team members
                opponent_distances.append(dist_to_ball)

        # Ensure there are players on both teams
        if not team_distances or not opponent_distances:
            return 0.0

        # Find the closest distance to the ball for each team
        min_team_distance = min(team_distances)
        min_opponent_distance = min(opponent_distances)

        # Calculate the difference in distances
        distance_difference = min_opponent_distance - min_team_distance

        # Normalize the difference to be within the range of -1 to 1
        normalized_reward = np.clip(distance_difference / self.max_distance, -1, 1)

        return normalized_reward
    

class LemTouchBallReward(RewardFunction):
    def __init__(self, aerial_weight=0):
        self.aerial_weight = aerial_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            if not player.on_ground and player.car_data.position[2] >= 256:
                height_reward = np.log1p(player.car_data.position[2] - 256)
                return height_reward
        return 0
    

class WavedashReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        if player.is_flipping and player.on_ground:
            reward=1
        return reward
        