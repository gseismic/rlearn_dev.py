import random
from collections import deque
from .base import BaseReplayBuffer, Experience

class RandomReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.buffer = deque(maxlen=capacity)

    def add(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def state_dict(self):
        return {
            'buffer': list(self.buffer)
        }

    def load_state_dict(self, state_dict):
        self.buffer = deque(state_dict['buffer'], maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)
