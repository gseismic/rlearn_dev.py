from .decay_strategy import DecayStrategy

class ConstantDecay(DecayStrategy):
    def __init__(self, epsilon_value):
        self.epsilon_value = epsilon_value

    def start(self):
        pass

    def step(self):
        pass

    def get_epsilon(self, epsilon_start, epsilon_end):
        return self.epsilon_value