import numpy as np
import pygame

import gymnasium as gym

from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 32}

    def __init__(self, render_mode=None, size=5):
        self._agent_location = None
        self._target_locations = None
        self.size = size
        self.window_size = 512

        self.observation_space = spaces.Box(0, self.size - 1, shape=(2,), dtype=int)

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_locations, ord=1),
            "target_locations": self._target_locations
        }

    def reset(self, seed=None, options=None):
        # print("starting new episode")
        super().reset(seed=seed)

        self._target_locations = np.array([[0, 0],
                                           [self.size - 1, self.size - 1]])

        while self._agent_location is None or np.any(np.all(self._agent_location == self._target_locations, axis=1)):
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = np.any(np.all(self._agent_location == self._target_locations, axis=1))

        reward = 0 if terminated else -1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
            if self.clock is None:
                self.clock = pygame.time.Clock()

        # Make Canvas
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )

        # Paint targets
        for target_location in self._target_locations:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * target_location,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw lines between squares
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
