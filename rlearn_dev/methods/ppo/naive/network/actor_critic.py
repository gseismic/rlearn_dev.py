import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(state_dim).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(state_dim).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_value(self, x, action, compute_entropy=True):
        assert action is not None
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        entropy = probs.entropy() if compute_entropy else 0 
        return probs.log_prob(action), entropy, self.critic(x)
    
    def get_action_and_value(self, x, action=None,
                             compute_entropy=True, deterministic=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=1)
            else:
                action = probs.sample()
        entropy = probs.entropy() if compute_entropy else 0 
        return action, probs.log_prob(action), entropy, self.critic(x)
