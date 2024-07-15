import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from algorithms.tamer import Tamer
from common.exp_manager import ExpManager
from envs.grid_world import GridWorldEnv
import matplotlib.pyplot as plt

from wrappers.human_feedback import HumanFeedback

ENV_SIZE = 8

if __name__ == "__main__":
    train_env = HumanFeedback(gym.make("GridWorld-v0", render_mode=None, size=ENV_SIZE))
    eval_env = TimeLimit(gym.make("GridWorld-v0", render_mode=None, size=ENV_SIZE), max_episode_steps=40)
    tamer = Tamer(size=ENV_SIZE, learning_rate=0.1)
    exp = ExpManager(train_env, eval_env, tamer)

    _, eval_rewards = exp.train(step_count=3000)

    plt.title("TAMER (8x8)")
    plt.xlabel("Training steps")
    plt.ylabel("Average evaluation reward per episode")
    plt.plot(eval_rewards)
    plt.show()

    print(tamer.model)