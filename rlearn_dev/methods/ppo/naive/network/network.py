import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np
from rlearn_dev.utils.nn.mlp import mlp

class Actor(nn.Module):
    def compute_distribution(self, observations):
        raise NotImplementedError

    def compute_log_probability(self, policy, actions):
        raise NotImplementedError

    def forward(self, observations, actions=None):
        policy = self.compute_distribution(observations)
        log_prob = None
        if actions is not None:
            log_prob = self.compute_log_probability(policy, actions)
        return policy, log_prob

class MLPCategoricalActor(Actor):
    def __init__(self, observation_dim, action_dim, hidden_layer_sizes, activation):
        super().__init__()
        self.logits_network = mlp([observation_dim] + list(hidden_layer_sizes) + [action_dim], activation)

    def compute_distribution(self, observations):
        logits = self.logits_network(observations)
        return Categorical(logits=logits)

    def compute_log_probability(self, policy, actions):
        return policy.log_prob(actions)

class MLPGaussianActor(Actor):
    def __init__(self, observation_dim, action_dim, hidden_layer_sizes, activation):
        super().__init__()
        initial_log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(initial_log_std))
        self.mean_network = mlp([observation_dim] + list(hidden_layer_sizes) + [action_dim], activation)

    def compute_distribution(self, observations):
        action_mean = self.mean_network(observations)
        action_std = torch.exp(self.log_std)
        return Normal(action_mean, action_std)

    def compute_log_probability(self, policy, actions):
        return policy.log_prob(actions).sum(axis=-1)

class MLPCritic(nn.Module):
    def __init__(self, observation_dim, hidden_layer_sizes, activation):
        super().__init__()
        self.value_network = mlp([observation_dim] + list(hidden_layer_sizes) + [1], activation)

    def forward(self, observations):
        return torch.squeeze(self.value_network(observations), -1)
