import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from .base import BaseImitation

class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [256, 256]
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class BehaviorCloning(BaseImitation):
    """
    行为克隆算法实现
    
    Args:
        observation_space: 观察空间
        action_space: 动作空间
        expert_data: 专家数据
        learning_rate: 学习率
        hidden_dims: 隐藏层维度
        device: 运行设备
    """
    def __init__(
        self,
        observation_space: Dict[str, Any],
        action_space: Dict[str, Any],
        expert_data: Optional[Dict] = None,
        learning_rate: float = 1e-3,
        hidden_dims: list = [256, 256],
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            expert_data=expert_data,
            device=device,
            **kwargs
        )
        
        # 确定输入输出维度
        self.input_dim = self._get_observation_dim()
        self.output_dim = self._get_action_dim()
        
        # 初始化策略网络
        self.policy = PolicyNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
    def _get_observation_dim(self) -> int:
        """获取观察空间维度"""
        if isinstance(self.observation_space, dict):
            return sum(np.prod(space['shape']) for space in self.observation_space.values())
        return np.prod(self.observation_space['shape'])
    
    def _get_action_dim(self) -> int:
        """获取动作空间维度"""
        if isinstance(self.action_space, dict):
            return sum(np.prod(space['shape']) for space in self.action_space.values())
        return np.prod(self.action_space['shape'])
    
    def _preprocess_observation(
        self,
        observation: Union[np.ndarray, Dict]
    ) -> torch.Tensor:
        """预处理观察值"""
        if isinstance(observation, dict):
            # 将字典中的所有观察值展平并拼接
            obs = np.concatenate([
                obs_value.reshape(-1) for obs_value in observation.values()
            ])
        else:
            obs = observation.reshape(-1)
        
        return torch.FloatTensor(obs).to(self.device)
    
    def forward(
        self,
        observation: Union[np.ndarray, Dict],
        state: Optional[Any] = None
    ) -> Tuple[np.ndarray, None]:
        """模型推理"""
        self.policy.eval()
        with torch.no_grad():
            obs = self._preprocess_observation(observation)
            action = self.policy(obs)
            return action.cpu().numpy(), None
    
    def train_step(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """训练步骤"""
        self.policy.train()
        
        # 获取数据
        observations = torch.FloatTensor(batch['observations']).to(self.device)
        expert_actions = torch.FloatTensor(batch['actions']).to(self.device)
        
        # 前向传播
        predicted_actions = self.policy(observations)
        
        # 计算损失
        loss = self.criterion(predicted_actions, expert_actions)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._training_info = {
            'loss': loss.item(),
            'step': self._current_step
        }
        self._current_step += 1
        
        return self._training_info
    
    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self._current_step,
            'training_info': self._training_info
        }, path)
    
    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._current_step = checkpoint['training_step']
        self._training_info = checkpoint['training_info'] 