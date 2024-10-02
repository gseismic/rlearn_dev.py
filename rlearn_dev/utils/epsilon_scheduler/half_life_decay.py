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