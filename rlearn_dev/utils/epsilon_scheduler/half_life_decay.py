import numpy as np
from .decay_strategy import DecayStrategy

class HalfLifeDecay(DecayStrategy):
    def __init__(self, half_life):
        self.half_life = half_life
        self.current_step = 0

    def start(self):
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_epsilon(self, epsilon_start, epsilon_end):
        # print(self.current_step, f'{self.current_step / self.half_life}, {epsilon_start=}, {epsilon_end=}')
        epsilon =  epsilon_end + (epsilon_start - epsilon_end) * (0.5 ** (self.current_step / self.half_life))
        # print(f'epsilon: {epsilon}')
        return epsilon
    
class HalfLifeCycleDecay(DecayStrategy):
    def __init__(self, half_life, cycle_epsilon=0.1, cycle_steps=1000):
        self.half_life = half_life
        self.cycle_epsilon = cycle_epsilon
        self.cycle_steps = cycle_steps
        self.current_step = 0

    def start(self):
        self.current_step = 0
    
    def step(self):
        self.current_step += 1

    def get_epsilon(self, epsilon_start, epsilon_end):
        eps1 =  epsilon_end + (epsilon_start - epsilon_end) * (0.5 ** (self.current_step / self.half_life))
        eps2 = self.cycle_epsilon * np.cos(2*np.pi * self.current_step/self.cycle_steps)
        epsilon = eps1 + eps2
        # print(f'{eps1=}, {eps2=}, {epsilon=}')
        return np.clip(epsilon, epsilon_end, epsilon_start)