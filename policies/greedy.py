import numpy as np

from policies.base_policy import BasePolicy


class Greedy(BasePolicy):
    @staticmethod
    def predict(model, state):
        _x, _y = state
        # print(f"choosing action for {_x}, {_y}: ", self.model[_x, _y, :])
        # print("chosen action: ", np.argmax(self.model[_x, _y, :]))
        return np.argmax(model[_x, _y, :])
