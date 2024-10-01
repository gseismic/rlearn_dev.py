import numpy as np
import random
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.size = 0

    def add(self, experience):
        """Add experience to the buffer with maximum priority."""
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.size % self.capacity] = experience
        
        # Assign maximum priority
        if len(self.priorities) < self.capacity:
            self.priorities.append(1.0)
        else:
            self.priorities[self.size % self.capacity] = 1.0

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of experiences based on their priorities."""
        if self.size == 0:
            raise ValueError("Buffer is empty")

        # Calculate probabilities
        priorities = np.array(self.priorities)[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on calculated probabilities
        indices = np.random.choice(self.size, batch_size, p=probabilities)

        experiences = [self.buffer[idx] for idx in indices]
        weights = (self.size * probabilities[indices]) ** (-beta)  # Importance sampling weights
        weights /= weights.max() 

        return experiences, indices, weights

    def update_priorities(self, indices, priorities):
        """Update the priorities of the sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority