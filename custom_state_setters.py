from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper, DefaultState, RandomState
from random import choices
from typing import Sequence, Union, Tuple
from rlgym_sim.utils.math import rand_vec3
import numpy as np
from numpy import random as rand
import random

from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z, GOAL_HEIGHT,\
    SIDE_WALL_X, BACK_WALL_Y, CAR_MAX_SPEED, CAR_MAX_ANG_VEL, BALL_RADIUS

class WeightedSampleSetter(StateSetter):
    """
    Samples StateSetters randomly according to their weights.

    :param state_setters: 1-D array-like of state-setters to be sampled from
    :param weights: 1-D array-like of the weights associated with each entry in state_setters
    """

    def __init__(self, state_setters: Sequence[StateSetter], weights: Sequence[float]):
        super().__init__()
        self.state_setters = state_setters
        self.weights = weights
        assert len(state_setters) == len(weights), \
            f"Length of state_setters should match the length of weights, " \
            f"instead lengths {len(state_setters)} and {len(weights)} were given respectively."

    @classmethod
    def from_zipped(
            cls,
            *setters_and_weights: Union[StateSetter, Tuple[RewardFunction, float]]
    ) -> "WeightedSampleSetter":
        """
        Alternate constructor which takes any number of either state setters, or (state setters, weight) tuples.
        :param setters_and_weights: a sequence of StateSetter or (StateSetter, weight) tuples
        """
        rewards = []
        weights = []
        for value in setters_and_weights:
            if isinstance(value, tuple):
                r, w = value
            else:
                r, w = value, 1.
            rewards.append(r)
            weights.append(w)
        return cls(tuple(rewards), tuple(weights))

    def reset(self, state_wrapper: StateWrapper):
        """
        Executes the reset of randomly sampled state-setter

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        choices(self.state_setters, weights=self.weights)[0].reset(state_wrapper)

BALL_RADIUS = 94
DEG_TO_RAD = 3.14159265 / 180

class WallPracticeState(StateSetter):

    def __init__(self, air_dribble_odds=1/3, backboard_roll_odds=1/3, side_high_odds=1/3):
        """
        WallPracticeState to setup wall practice
        """
        super().__init__()

        self.air_dribble_odds = air_dribble_odds
        self.backboard_roll_odds = backboard_roll_odds
        self.side_high_odds = side_high_odds

    def reset(self, state_wrapper: StateWrapper):
        choice_list = [0] * int(self.backboard_roll_odds * 100) + \
                      [1] * int(self.side_high_odds * 100) + \
                      [2] * int(self.air_dribble_odds * 100)
        scenario_pick = random.choice(choice_list)

        if scenario_pick == 0:
            self._short_goal_roll(state_wrapper)
        elif scenario_pick == 1:
            self._side_high_roll(state_wrapper)
        elif scenario_pick == 2:
            self._air_dribble_setup(state_wrapper)

    def _air_dribble_setup(self, state_wrapper):
        """
        A medium roll up a side wall with the car facing the roll path

        :param state_wrapper:
        """

        axis_inverter = 1 if random.randrange(2) == 1 else -1
        team_side = 0 if random.randrange(2) == 1 else 1
        team_inverter = 1 if team_side == 0 else -1

        #if only 1 play, team is always 0

        ball_x_pos = 3000 * axis_inverter
        ball_y_pos = random.randrange(7600) - 3800
        ball_z_pos = BALL_RADIUS
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (2000 + (random.randrange(1000) - 500)) * axis_inverter
        ball_y_vel = random.randrange(1000) * team_inverter
        ball_z_vel = 0
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        chosen_car = [car for car in state_wrapper.cars if car.team_num == team_side][0]
        #if randomly pick, chosen_car is from orange instead

        car_x_pos = 2500 * axis_inverter
        car_y_pos = ball_y_pos
        car_z_pos = 27

        yaw = 0 if axis_inverter == 1 else 180
        car_pitch_rot = 0 * DEG_TO_RAD
        car_yaw_rot = (yaw + (random.randrange(40) - 20)) * DEG_TO_RAD
        car_roll_rot = 0 * DEG_TO_RAD

        chosen_car.set_pos(car_x_pos, car_y_pos, car_z_pos)
        chosen_car.set_rot(car_pitch_rot, car_yaw_rot, car_roll_rot)
        chosen_car.boost = 100

        for car in state_wrapper.cars:
            if car is chosen_car:
                continue

            # set all other cars randomly in the field
            car.set_pos(random.randrange(2944) - 1472, random.randrange(3968) - 1984, 0)
            car.set_rot(0, (random.randrange(360) - 180) * (3.1415927 / 180), 0)

    def _side_high_roll(self, state_wrapper):
        """
        A high vertical roll up the side of the field

        :param state_wrapper:
        """
        sidepick = random.randrange(2)

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1


        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        ball_x_pos = 3000 * side_inverter
        ball_y_pos = random.randrange(1500) - 750
        ball_z_pos = BALL_RADIUS
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (2000 + random.randrange(1000) - 500) * side_inverter
        ball_y_vel = random.randrange(1500) - 750
        ball_z_vel = random.randrange(300)
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        wall_car_blue = [car for car in state_wrapper.cars if car.team_num == 0][0]

        #blue car setup
        blue_pitch_rot = 0 * DEG_TO_RAD
        blue_yaw_rot = 90 * DEG_TO_RAD
        blue_roll_rot = 90 * side_inverter * DEG_TO_RAD
        wall_car_blue.set_rot(blue_pitch_rot, blue_yaw_rot, blue_roll_rot)

        blue_x = 4096 * side_inverter
        blue_y = -2500 + (random.randrange(500) - 250)
        blue_z = 600 + (random.randrange(400) - 200)
        wall_car_blue.set_pos(blue_x, blue_y, blue_z)
        wall_car_blue.boost = 100

        #orange car setup
        wall_car_orange = None
        if len(state_wrapper.cars) > 1:
            wall_car_orange = [car for car in state_wrapper.cars if car.team_num == 1][0]
            # orange car setup
            orange_pitch_rot = 0 * DEG_TO_RAD
            orange_yaw_rot = -90 * DEG_TO_RAD
            orange_roll_rot = -90 * side_inverter * DEG_TO_RAD
            wall_car_orange.set_rot(orange_pitch_rot, orange_yaw_rot, orange_roll_rot)

            orange_x = 4096 * side_inverter
            orange_y = 2500 + (random.randrange(500) - 250)
            orange_z = 400 + (random.randrange(400) - 200)
            wall_car_orange.set_pos(orange_x, orange_y, orange_z)
            wall_car_orange.boost = 100

        for car in state_wrapper.cars:
            if len(state_wrapper.cars) == 1 or car is wall_car_orange or car is wall_car_blue:
                continue

            # set all other cars randomly in the field
            car.set_pos(random.randrange(2944) - 1472, random.randrange(3968) - 1984, 0)
            car.set_rot(0, (random.randrange(360) - 180) * (3.1415927/180), 0)

    def _short_goal_roll(self, state_wrapper):
        """
        A short roll across the backboard and down in front of the goal

        :param state_wrapper:
        :return:
        """

        if len(state_wrapper.cars) > 1:
            defense_team = random.randrange(2)
        else:
            defense_team = 0
        sidepick = random.randrange(2)

        defense_inverter = 1
        if defense_team == 0:
            # change side
            defense_inverter = -1

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1

        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        x_random = random.randrange(446)
        ball_x_pos = (-2850 + x_random) * side_inverter
        ball_y_pos = (5120 - BALL_RADIUS) * defense_inverter
        ball_z_pos = 1400 + random.randrange(400) - 200
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (1000 + random.randrange(400) - 200) * side_inverter
        ball_y_vel = 0
        ball_z_vel = 550
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)


        wall_car = [car for car in state_wrapper.cars if car.team_num == defense_team][0]

        wall_car_x = (2000 - random.randrange(500)) * side_inverter
        wall_car_y = 5120 * defense_inverter
        wall_car_z = 1000 + (random.randrange(500) - 500)
        wall_car.set_pos(wall_car_x, wall_car_y, wall_car_z)

        wall_pitch_rot = (0 if side_inverter == -1 else 180) * DEG_TO_RAD
        wall_yaw_rot = 0 * DEG_TO_RAD
        wall_roll_rot = -90 * defense_inverter * DEG_TO_RAD
        wall_car.set_rot(wall_pitch_rot, wall_yaw_rot, wall_roll_rot)
        wall_car.boost = 25

        if len(state_wrapper.cars) > 1:
            challenge_car = [car for car in state_wrapper.cars if car.team_num != defense_team][0]
            challenge_car.set_pos(0, 1000 * defense_inverter, 0)

            challenge_pitch_rot = 0 * DEG_TO_RAD
            challenge_yaw_rot = 90 * defense_inverter * DEG_TO_RAD
            challenge_roll_rot = 0 * DEG_TO_RAD
            challenge_car.set_rot(challenge_pitch_rot, challenge_yaw_rot, challenge_roll_rot)
            challenge_car.boost = 100

        for car in state_wrapper.cars:
            if len(state_wrapper.cars) == 1 or car is wall_car or car is challenge_car:
                continue

            car.set_pos(random.randrange(2944) - 1472, (-4500 + random.randrange(500) - 250) * defense_inverter, 0)
            car.set_rot(0, (random.randrange(360) - 180) * DEG_TO_RAD, 0)


class TeamSizeSetter(StateSetter):
    def __init__(self):
        super().__init__()
        self.default = DefaultState()
        self.count = 3

    def build_wrapper(self, max_team_size: int, spawn_opponents: bool) -> StateWrapper:
        wrapper = StateWrapper(blue_count=self.count, orange_count=self.count if spawn_opponents else 0)
        self.count -= 1
        if self.count == 0:
            self.count = 3

        return wrapper

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.blue_count = self.count
        state_wrapper.orange_count = self.count if state_wrapper.orange_count > 0 else 0
        # You can optionally call the default reset if needed
        self.default.reset(state_wrapper)  # Comment this line if you don't need default reset behavior
        self.count -= 1
        if self.count == 0:
            self.count = 3


class AirDrag(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        rng = np.random.default_rng()
        # Set up our desired spawn location and orientation for car0 - special one in air
        # don't at me about the magic numbers, just go with it.
        # blue should aim directly towards orange, and vice versa

        car_attack = state_wrapper.cars[0]
        car_defend = None
        for car_y in state_wrapper.cars:
            if car_y.team_num == ORANGE_TEAM:
                car_defend = car_y
        orange_fix = 1
        if rand.choice([0, 1]) and len(state_wrapper.cars) > 1:
            for car_i in state_wrapper.cars:
                if car_i.team_num == ORANGE_TEAM:
                    car_attack = car_i
                    car_defend = state_wrapper.cars[0]  # blue is always 0
                    orange_fix = -1
                    continue

        x_choice = rand.choice([0, 2]) - 1
        rand_x = x_choice * (rng.uniform(0, SIDE_WALL_X - 250))
        rand_y = rng.uniform(-BACK_WALL_Y + 1300, BACK_WALL_Y - 1300)
        rand_z = rng.uniform(300, 800)
        desired_car_pos = [rand_x, rand_y, rand_z]  # x, y, z
        desired_pitch = 20 * DEG_TO_RAD
        desired_yaw = 0  # 90 * DEG_TO_RAD
        desired_roll = 0  # 90 * x_choice * DEG_TO_RAD
        desired_rotation = [desired_pitch, desired_yaw, desired_roll]

        car_attack.set_pos(*desired_car_pos)
        car_attack.set_rot(*desired_rotation)
        car_attack.boost = 100

        car_attack.set_lin_vel(20 * x_choice, rng.uniform(800, 1200), 60)
        car_attack.set_ang_vel(0, 0, 0)

        # Now we will spawn the ball on top of the car matching the velocity
        ball_y: np.float32
        if rand_y < 0:
            ball_y = rand_y - 40
        else:
            ball_y = rand_y + 40
        state_wrapper.ball.set_pos(x=rand_x, y=ball_y + BALL_RADIUS / 2,
                                   z=rand_z + BALL_RADIUS / 2)
        state_wrapper.ball.set_lin_vel(20 * x_choice, rng.uniform(800, 1200), 20)
        state_wrapper.ball.set_ang_vel(0, 0, 0)

        # Loop over every car in the game, skipping 1 since we already did it
        for car in state_wrapper.cars:
            if car.id == car_attack.id:
                pass

            # put the defense car in front of net
            elif car.id == car_defend.id:
                car.set_pos(rng.uniform(-1600, 1600), orange_fix * rng.uniform(3800, 5000), 0)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33
                continue

            # rest of the cars are random
            else:
                car.set_pos(rng.uniform(-1472, 1472), rng.uniform(-1984, 1984), 0)
                car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
                car.boost = 0.33


