import numpy as np

from algorithms.base_algo import BaseAlgo
from policies.greedy import Greedy


class Tamer(BaseAlgo):
    def __init__(self, size, policy=Greedy, learning_rate=0.1, seed=42):
        self.size = size
        self.policy = policy
        self.learning_rate = learning_rate

        np.random.seed(seed)
        self.model = np.random.rand(size, size, 4)

    def predict(self, state):
        return self.policy.predict(self.model, state)

    def update_model(self, state, action, reward, next_state):
        _x, _y = state
        error = reward - self.model[_x, _y, action]
        self.model[_x, _y, action] += self.learning_rate*error
        # print("updated state: ", self.model[_x, _y, :], end="\n\n")

