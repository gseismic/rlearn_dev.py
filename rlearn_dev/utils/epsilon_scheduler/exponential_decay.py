import numpy as np
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
    
class ExponentialCycleDecay(DecayStrategy):
    def __init__(self, decay_rate, cycle_epsilon=0.1, cycle_steps=1000):
        self.decay_rate = decay_rate
        self.cycle_epsilon = cycle_epsilon
        self.cycle_steps = cycle_steps
        self.current_step = 0

    def start(self):
        self.current_step = 0   
        
    def step(self):
        self.current_step += 1

    def get_epsilon(self, epsilon_start, epsilon_end):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (self.decay_rate ** self.current_step)
        cycle_epsilon = self.cycle_epsilon * np.cos(2*np.pi * self.current_step/self.cycle_steps)
        return np.clip(epsilon + cycle_epsilon, epsilon_end, epsilon_start)
