from .linear_decay import LinearDecay
from .exponential_decay import ExponentialDecay
from .half_life_decay import HalfLifeDecay
from .time_decay import TimeDecay
from .step_decay import StepDecay
from .custom_decay import CustomDecay
import numpy as np

class EpsilonScheduler:
    registered_schedulers = {}

    @classmethod
    def register_scheduler(cls, name, scheduler_cls):
        cls.registered_schedulers[name] = scheduler_cls

    def __init__(self, epsilon_start, epsilon_end, scheduler_type='linear', **kwargs):
        assert epsilon_start > epsilon_end, 'epsilon_start must be greater than epsilon_end'
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        if scheduler_type in self.registered_schedulers:
            self.decay_strategy = self.registered_schedulers[scheduler_type](**kwargs)
        elif scheduler_type == 'custom_decay':
            decay_function = kwargs.get('decay_function')
            time_decay_function = kwargs.get('time_decay_function', None)
            if decay_function is None and time_decay_function is None:
                raise ValueError('Either custom decay function or time decay function must be provided.')
            self.decay_strategy = CustomDecay(decay_function, time_decay_function=time_decay_function, **kwargs)
        elif scheduler_type == 'linear':
            total_steps = kwargs.get('total_steps', 10000)
            self.decay_strategy = LinearDecay(total_steps)
        elif scheduler_type == 'exponential':
            decay_rate = kwargs.get('decay_rate', 0.01)
            self.decay_strategy = ExponentialDecay(decay_rate)
        elif scheduler_type == 'half_life':
            half_life = kwargs.get('half_life', 1000)
            self.decay_strategy = HalfLifeDecay(half_life)
        elif scheduler_type == 'time_decay':
            decay_rate = kwargs.get('decay_rate', None)
            half_life = kwargs.get('half_life', None)
            self.decay_strategy = TimeDecay(decay_rate, half_life)
        elif scheduler_type == 'step_decay':
            total_steps = kwargs.get('total_steps', 10000)
            self.decay_strategy = StepDecay(total_steps)
        else:
            raise ValueError(f'Unknown scheduler type: {scheduler_type}')
        self.decay_strategy.start()

    def step(self):
        self.decay_strategy.step()

    def get_epsilon(self):
        epsilon = self.decay_strategy.get_epsilon(self.epsilon_start, self.epsilon_end)
        return np.clip(epsilon, self.epsilon_end, self.epsilon_start)