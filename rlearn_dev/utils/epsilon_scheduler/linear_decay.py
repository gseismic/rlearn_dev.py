from .decay_strategy import DecayStrategy
import numpy as np

class LinearDecay(DecayStrategy):
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0

    def start(self):
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_epsilon(self, epsilon_start, epsilon_end):
        fraction = min(self.current_step / self.total_steps, 1.0)
        return epsilon_start - fraction * (epsilon_start - epsilon_end)
    
class LinearCycleDecay(DecayStrategy):
    def __init__(self, total_steps, cycle_epsilon=0.1, cycle_steps=1000):
        self.total_steps = total_steps
        self.cycle_epsilon = cycle_epsilon
        self.cycle_steps = cycle_steps
        self.current_step = 0

    def start(self):
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_epsilon(self, epsilon_start, epsilon_end):
        fraction = min(self.current_step / self.total_steps, 1.0)
        cycle_epsilon = self.cycle_epsilon * np.cos(2*np.pi * self.current_step/self.cycle_steps)
        linear_epsilon = epsilon_start - fraction * (epsilon_start - epsilon_end)
        epsilon = linear_epsilon + cycle_epsilon
        return np.clip(epsilon, epsilon_end, epsilon_start)