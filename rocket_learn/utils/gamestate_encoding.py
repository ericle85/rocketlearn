from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.api import Car
from rlgym.rocket_league.api import PhysicsObject

def _encode_physics(cd : PhysicsObject) -> list:
    return cd.position.tolist() + cd.quaternion.tolist() + cd.linear_velocity.tolist() + cd.angular_velocity.tolist()

def _encode_car(car_id, p: Car) -> list:
    return (
        [car_id, p.team_num]
        + _encode_physics(p.physics)
        + _encode_physics(p.inverted_physics)
        + [
            p.is_demoed,
            any(p.wheels_with_contact),
            p.ball_touches > 0,
            not p.has_jumped,
            not p.has_flipped and not p.has_double_jumped,
            p.boost_amount,
        ]
    )

def _encode_ball(bd: PhysicsObject) -> list:
    return bd.position.tolist() + bd.linear_velocity.tolist() + bd.angular_velocity.tolist()

def encode_gamestate(state: GameState, blue_score: int = 0, orange_score: int = 0) -> list:
    return (
        [0, blue_score, orange_score]
        + state.boost_pad_timers.tolist()
        + _encode_ball(state.ball)
        + _encode_ball(state.inverted_ball)
        + [val for car_id, p in state.cars.items() for val in _encode_car(car_id, p)]
    )


NUM_BOOST_PADS = 34  # rlgym v2: 34 boost pad locations
# car_id + team_num (2) + physics (13) + inverted_physics (13) + stats (6) = 34
PLAYER_INFO_LENGTH = 34

# Now some constants for easy and consistent querying of gamestate values
class StateConstants:
    DUMMY = 0
    BLUE_SCORE = 1
    ORANGE_SCORE = 2
    BOOST_PADS = slice(3, 3 + NUM_BOOST_PADS)
    BALL_POSITION = slice(BOOST_PADS.stop, BOOST_PADS.stop + 3)
    BALL_LINEAR_VELOCITY = slice(BALL_POSITION.stop, BALL_POSITION.stop + 3)
    BALL_ANGULAR_VELOCITY = slice(BALL_LINEAR_VELOCITY.stop, BALL_LINEAR_VELOCITY.stop + 3)

    PLAYERS = slice(BALL_ANGULAR_VELOCITY.stop + 9, None)  # Skip inverted ball data
    CAR_IDS = slice(0, None, PLAYER_INFO_LENGTH)
    TEAM_NUMS = slice(CAR_IDS.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_POS_X = slice(TEAM_NUMS.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_POS_Y = slice(CAR_POS_X.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_POS_Z = slice(CAR_POS_Y.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_QUAT_W = slice(CAR_POS_Z.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_QUAT_X = slice(CAR_QUAT_W.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_QUAT_Y = slice(CAR_QUAT_X.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_QUAT_Z = slice(CAR_QUAT_Y.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_LINEAR_VEL_X = slice(CAR_QUAT_Z.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_LINEAR_VEL_Y = slice(CAR_LINEAR_VEL_X.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_LINEAR_VEL_Z = slice(CAR_LINEAR_VEL_Y.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_ANGULAR_VEL_X = slice(CAR_LINEAR_VEL_Z.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_ANGULAR_VEL_Y = slice(CAR_ANGULAR_VEL_X.start + 1, None, PLAYER_INFO_LENGTH)
    CAR_ANGULAR_VEL_Z = slice(CAR_ANGULAR_VEL_Y.start + 1, None, PLAYER_INFO_LENGTH)
    IS_DEMOED = slice(CAR_ANGULAR_VEL_Z.start + 1 + 13, None, PLAYER_INFO_LENGTH)  # Skip inverted physics
    ON_GROUND = slice(IS_DEMOED.start + 1, None, PLAYER_INFO_LENGTH)
    BALL_TOUCHED = slice(ON_GROUND.start + 1, None, PLAYER_INFO_LENGTH)
    HAS_JUMP = slice(BALL_TOUCHED.start + 1, None, PLAYER_INFO_LENGTH)
    HAS_FLIP = slice(HAS_JUMP.start + 1, None, PLAYER_INFO_LENGTH)
    BOOST_AMOUNT = slice(HAS_FLIP.start + 1, None, PLAYER_INFO_LENGTH)
