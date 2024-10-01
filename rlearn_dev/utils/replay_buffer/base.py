
from abc import ABC, abstractmethod
from collections import namedtuple

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer(ABC):
    
    def __init__(self, capacity):
        self.capacity = capacity
    
    @abstractmethod
    def add(self, *args):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass
    
    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    @abstractmethod
    def __len__(self):
        pass


class BasePrioritizedReplayBuffer(ReplayBuffer):
    
    @abstractmethod
    def update_priority(self, *args):
        pass
