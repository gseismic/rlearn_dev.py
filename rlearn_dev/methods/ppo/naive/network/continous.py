import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from .utils import layer_init

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_space, rpo_alpha=0.0, scale_action=True):
        """
        Args:
            scale_action (bool): 是否裁剪动作
        """
        super().__init__()
        self.state_dim = state_dim if state_dim else (1,)
        self.action_dim = action_space.shape
        self.action_space = action_space
        self.rpo_alpha = rpo_alpha

        if self.action_space.high is not None and self.action_space.low is not None:
            self.action_low = torch.tensor(self.action_space.low, dtype=torch.float32)
            self.action_high = torch.tensor(self.action_space.high, dtype=torch.float32)
            self.scale_action = scale_action
        else:
            self.scale_action = False
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.prod(self.state_dim), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        if self.scale_action:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.prod(self.state_dim), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, np.prod(self.action_dim)), std=0.01),
                nn.Tanh(), # 确保动作在有效范围内
            )
        else:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.prod(self.state_dim), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                # 没有裁剪，初始均匀动作
                layer_init(nn.Linear(64, np.prod(self.action_dim)), std=0.01)
            )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_dim)))
        if self.scale_action:
            self.register_buffer(
                "action_scale", torch.tensor((self.action_space.high - self.action_space.low) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((self.action_space.high + self.action_space.low) / 2.0, dtype=torch.float32)
            )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, compute_entropy=True, deterministic=False):
        # from: https://docs.cleanrl.dev/rl-algorithms/rpo/#implementation-details
        action_mean = self.actor_mean(x)
        if self.scale_action:
            # print(f"action_mean: {action_mean}")
            action_mean = action_mean * self.action_scale + self.action_bias
            # print(f"action_mean: {action_mean}")
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = probs.sample()
        else: # new to RPO
            # sample again to add stochasticity, for the policy update
            device = action_mean.device
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
            
        entropy = probs.entropy().sum(1) if compute_entropy else None
        if self.scale_action:
            # TODO: ensure device
            device = action.device
            action = torch.clip(action, self.action_low.to(device), self.action_high.to(device))
        return action, probs.log_prob(action).sum(1), entropy, self.critic(x)
