from .decay_strategy import DecayStrategy

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
    
class LinearCosDecay(DecayStrategy):
    def __init__(self, total_steps, T=1000):
        self.total_steps = total_steps
        self.current_step = 0

    def start(self):
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_epsilon(self, epsilon_start, epsilon_end):
        fraction = min(self.current_step / self.total_steps, 1.0)
        return epsilon_start - fraction * (epsilon_start - epsilon_end)
