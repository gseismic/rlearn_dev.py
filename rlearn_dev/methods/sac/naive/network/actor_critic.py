import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# 预测soft-Q值的网络 | network to predict soft-Q values
# TODO: 增加网络初始方法
# TODO: 增加dropout
class Critic(nn.Module):
    """Soft-Q value network"""
    # same with Critic in ddpg
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        assert state_dim > 0 and action_dim > 0 and hidden_dim > 0
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # state.shape: (batch_size, state_dim)
        # action.shape: (batch_size, action_dim)
        assert len(state.shape) == 2
        assert len(action.shape) == 2
        x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# same with cleanrl.sac_continuous_action.Actor
# more: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
class Actor(nn.Module):
    default_log_std_min = -5
    default_log_std_max = 2
    EPS = 1e-6
    def __init__(self, state_dim: int, action_dim: int,
                 action_high: np.ndarray, action_low: np.ndarray,
                 log_std_min: Optional[float] = None, log_std_max: Optional[float] = None,
                 hidden_dim: Optional[int] = None):
        super().__init__()
        assert state_dim > 0, "state_dim must be positive"
        assert action_dim > 0, "action_dim must be positive"
        assert np.all(action_high > action_low), "action_high must be greater than action_low"
        
        self.log_std_min = log_std_min if log_std_min is not None else self.default_log_std_min
        self.log_std_max = log_std_max if log_std_max is not None else self.default_log_std_max
        assert self.log_std_min < self.log_std_max, f"log_std_min: {self.log_std_min} must be less than log_std_max: {self.log_std_max}"
        hidden_dim = hidden_dim if hidden_dim is not None else 256
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        # 和naive.ddpg的actor一样, 只是多了log_std | same with naive.ddpg.Actor but with log_std
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        self.register_buffer(
            "action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        assert len(x.shape) == 2, f"Expected 2D input tensor, got shape {x.shape}"
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std) # range: (-1, 1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1) 
        return mean, log_std

    def get_action(self, x, *,
                   compute_log_prob: bool = True,
                   compute_mean: bool = False,
                   deterministic: bool = False):
        mean, log_std = self(x)
        std = log_std.exp()
        if deterministic is False: 
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
        else:
            y_t = mean
        action = y_t * self.action_scale + self.action_bias
        # ∫p_action(a)da = ∫p_y(y) * |dy/da| da = ∫p_x(x) * |dx/dy| * |dy/da| da
        # x_t (normal distribution) -> y_t = tanh(x_t) -> action = y_t * action_scale + action_bias
        # p_action(a) = p_y(y) * |dy/da| = p_x(x) * |dx/dy| * |dy/da|
        # a = y * action_scale + action_bias
        # dy/da = action_scale
        # dx/dy = 1 / (1 - y^2)
        # |dx/dy| = 1 / (1 - y^2)
        # p_action(a) = p_x(x) * 1 / (1 - y^2) * action_scale
        if compute_log_prob:
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.EPS)
            # 假设动作维度是独立的，联合概率就是各个维度概率的乘积，在对数空间就是求和 | assume action dimensions are independent, joint probability is the product of individual probabilities, in log space it's the sum
            # dim=1: 沿着第1维(动作维度)求和 | sum over action dimension
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None
        # mean 为没有进行重参数化的动作 | mean is the action without reparameterization
        if compute_mean:
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            mean = None
        return action, log_prob, mean
