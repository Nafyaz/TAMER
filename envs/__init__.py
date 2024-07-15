from gymnasium.envs.registration import register
from envs.grid_world import GridWorldEnv

register(
     id="GridWorld-v0",
     entry_point="envs:GridWorldEnv",
)
