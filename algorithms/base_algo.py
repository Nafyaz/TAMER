from abc import abstractmethod


class BaseAlgo:
    @abstractmethod
    def predict(self, state):
        pass

    @abstractmethod
    def update_model(self, state, action, reward, next_state):
        pass
