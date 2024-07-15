import numpy as np
from gymnasium import Wrapper


class HumanFeedback(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._agent_location = None
        self._target_locations = None

    def reset(self):
        observation, info = super().reset()
        self._agent_location = observation
        self._target_locations = info["target_locations"]
        return observation, info

    def _get_target_dist(self, agent_location):
        return np.linalg.norm(agent_location - self._target_locations, ord=-np.inf)

    def _get_reward(self, previous_location, current_location):
        # print(f"distance from previous {previous_location}: ", self._get_target_dist(previous_location))
        # print(f"distance from new {current_location}: ", self._get_target_dist(current_location))
        return 1 if self._get_target_dist(previous_location) > self._get_target_dist(current_location) else 0

    def step(self, action):
        previous_location = self._agent_location
        observation, _, terminated, truncated, info = self.env.step(action)
        self._agent_location = observation
        current_location = self._agent_location

        reward = self._get_reward(previous_location, current_location)

        return observation, reward, terminated, truncated, info
