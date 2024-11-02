
from abc import ABC, abstractmethod
from collections import namedtuple

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class BaseReplayBuffer(ABC):
    
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
    
    def size(self):
        return len(self)

class BasePrioritizedReplayBuffer(BaseReplayBuffer):
    
    @abstractmethod
    def update_priorities(self, indices, td_errors):
        pass
