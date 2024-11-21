import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from .utils import layer_init

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim if state_dim else (1,)
        self.action_dim = action_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.prod(self.state_dim), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.prod(self.state_dim), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.temperature = 1.0 # nn.Parameter(torch.tensor(1.0))

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None,
                             compute_entropy=True, deterministic=False):
        logits = self.actor(x) /self.temperature
        probs = Categorical(logits=logits)
        if action is None:
            if deterministic:
                assert logits.shape == (1, self.action_dim), 'only support single action'
                action = torch.argmax(logits, dim=1)
                assert logits.shape[0] == 1, 'only support single action'
            else:
                action = probs.sample()
        entropy = probs.entropy() if compute_entropy else None
        return action, probs.log_prob(action), entropy, self.critic(x)
