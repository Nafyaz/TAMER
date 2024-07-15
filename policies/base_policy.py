from abc import abstractmethod


class BasePolicy:
    @staticmethod
    def predict(model, state):
        pass

