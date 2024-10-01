import time
from .decay_strategy import DecayStrategy

class CustomDecay(DecayStrategy):
    def __init__(self, decay_function, *args, time_decay_function=None, **kwargs):
        self.decay_function = decay_function
        self.time_decay_function = time_decay_function
        self.args = args
        self.kwargs = kwargs
        self.current_step = 0
        self.start_time = None

    def start(self):
        self.current_step = 0
        self.start_time = time.time()

    def step(self):
        self.current_step += 1

    def get_epsilon(self, epsilon_start, epsilon_end):
        if self.time_decay_function:
            elapsed_time = time.time() - self.start_time
            return self.time_decay_function(elapsed_time, *self.args, **self.kwargs, epsilon_start=epsilon_start, epsilon_end=epsilon_end)
        return self.decay_function(self.current_step, *self.args, **self.kwargs, epsilon_start=epsilon_start, epsilon_end=epsilon_end)