import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from rocket_learn.utils.env_wrapper import RLGymV2GymWrapper


class CountingEnv:
    state = None
    action_spaces = {"a": (0, 90), "b": (0, 90)}
    observation_spaces = {"a": (0, 3), "b": (0, 3)}

    def reset(self):
        self.t = 0
        return {"a": np.zeros(3), "b": np.full(3, 100.0)}

    def step(self, actions):
        self.t += 1
        done = self.t == 4
        return (
            {"a": np.full(3, float(self.t)), "b": np.full(3, 100.0 + self.t)},
            {"a": 0.0, "b": 0.0},
            {"a": False, "b": False},
            {"a": done, "b": done},
        )


def test_stored_obs_are_not_aliased():
    env = RLGymV2GymWrapper(CountingEnv())
    obs = env.reset()
    stored = []
    truncated = False
    while not truncated:
        stored.append([row for row in obs])
        obs, _, _, truncated, _ = env.step(np.zeros(2))

    assert [s[0][0] for s in stored] == [0.0, 1.0, 2.0, 3.0]
    assert [s[1][0] for s in stored] == [100.0, 101.0, 102.0, 103.0]


if __name__ == "__main__":
    test_stored_obs_are_not_aliased()
    print("ok")
