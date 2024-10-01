from .decay_strategy import DecayStrategy

class ExponentialDecay(DecayStrategy):
    def __init__(self, decay_rate):
        self.decay_rate = decay_rate
        self.current_step = 0

    def start(self):
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_epsilon(self, epsilon_start, epsilon_end):
        return epsilon_end + (epsilon_start - epsilon_end) * (self.decay_rate ** self.current_step)