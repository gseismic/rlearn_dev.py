from .decay_strategy import DecayStrategy

class StepDecay(DecayStrategy):
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0

    def start(self):
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_epsilon(self, epsilon_start, epsilon_end):
        if self.current_step < self.total_steps:
            return epsilon_start - (epsilon_start - epsilon_end) * (self.current_step / self.total_steps)
        return epsilon_end