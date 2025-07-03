from .decay_strategy import DecayStrategy
import time

class TimeDecay(DecayStrategy):
    def __init__(self, decay_rate=None, half_life=None):
        self.decay_rate = decay_rate
        self.half_life = half_life
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def step(self):
        pass

    def get_epsilon(self, epsilon_start, epsilon_end):
        elapsed_time = time.time() - self.start_time
        if self.decay_rate:
            return epsilon_start * (1 - self.decay_rate * elapsed_time) + epsilon_end * (self.decay_rate * elapsed_time)
        elif self.half_life:
            return epsilon_end + (epsilon_start - epsilon_end) * (0.5 ** (elapsed_time / self.half_life))