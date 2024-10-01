import numpy as np
from .base import BasePrioritizedReplayBuffer, Experience

class PrioritizedReplayBuffer(BasePrioritizedReplayBuffer):

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-5):
        super().__init__(capacity)
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.pos = 0
        self.tree_sum = 0
        self.max_priority = 1.0

    def add(self, *args):
        # Add new experience to the buffer | 将新经验添加到缓冲区
        experience = Experience(*args)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        # Update priorities | 更新优先级
        self.priorities[self.pos] = float(self.max_priority)
        self.tree_sum += self.max_priority ** self.alpha - (self.priorities[self.pos] ** self.alpha if len(self.buffer) == self.capacity else 0)
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        # Sample experiences based on priorities | 基于优先级采样经验
        buffer_len = len(self.buffer)
        if buffer_len == 0:
            return [], [], []

        # P(i) = p_i^α / Σ_k p_k^α
        priorities = self.priorities[:buffer_len] ** self.alpha
        probs = priorities / priorities.sum()  # Ensure sum of probabilities is 1

        indices = np.random.choice(buffer_len, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # weights: 样本重要性
        self.beta = min(1.0, self.beta + self.beta_increment)
        # w_i = (N * P(i))^(-β) / max_j w_j
        weights = (buffer_len * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        # Update priorities based on TD errors | 根据TD误差更新优先级
        for idx, td_error in zip(indices, td_errors):
            td_error_scalar = td_error.item() if hasattr(td_error, 'item') else td_error
            priority = float((abs(td_error_scalar) + self.epsilon) ** self.alpha)
            self.tree_sum += priority - self.priorities[idx] ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def state_dict(self):
        return {
            'buffer': list(self.buffer),
            'priorities': self.priorities,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_increment': self.beta_increment,
            'epsilon': self.epsilon,    
        }
    
    def load_state_dict(self, state_dict):
        self.buffer = state_dict['buffer']
        self.priorities = state_dict['priorities']
        self.alpha = state_dict['alpha']
        self.beta = state_dict['beta']
        self.beta_increment = state_dict['beta_increment']
        self.epsilon = state_dict['epsilon']

    def __len__(self):
        return len(self.buffer)