from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import gymnasium as gym
import copy

class BaseImitation(ABC):
    """
    模仿学习基类 | Base imitation class
    
    属性:
        observation_space: 观察空间
        action_space: 动作空间
        expert_data: 专家数据
        device: 运行设备 (cpu/cuda)
    """
    def __init__(
        self,
        agent_policy: Any,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        **kwargs
    ):
        self.agent_policy = agent_policy
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
    
    @staticmethod
    def generate_expert_data(
        env: gym.Env,
        agent: Any,
        deterministic: bool,
        num_episodes: int,
        max_episode_steps: Optional[int] = None,
        max_total_steps: Optional[int] = None,
        flatten: bool = True,
        output_reward: bool = False
    ) -> Dict[str, Any]:
        """生成专家数据"""
        states, actions, rewards = [], [], []
        total_steps = 0
        
        for _ in range(num_episodes):
            episode_states, episode_actions, episode_rewards = [], [], []
            episode_terminated, episode_truncated, episode_info = [], [], []
            
            state, _ = env.reset()
            # 深拷贝初始状态
            episode_states.append(copy.deepcopy(state))
            
            step = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # 为预测创建状态的深拷贝，避免agent可能的修改
                predict_state = copy.deepcopy(state)
                action = agent.predict(predict_state, deterministic=deterministic)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # 深拷贝所有数据
                episode_actions.append(copy.deepcopy(action))
                episode_rewards.append(reward)  # 数值类型无需深拷贝
                episode_states.append(copy.deepcopy(next_state))
                episode_terminated.append(terminated)
                episode_truncated.append(truncated)
                episode_info.append(copy.deepcopy(info))
                
                # 更新状态为next_state的深拷贝
                state = copy.deepcopy(next_state)
                step += 1
                
                if max_episode_steps is not None and step >= max_episode_steps:
                    break
                total_steps += 1
            
            if max_total_steps is not None and total_steps >= max_total_steps:
                break
                
            states.append(copy.deepcopy(episode_states))
            actions.append(copy.deepcopy(episode_actions))
            rewards.append(copy.deepcopy(episode_rewards))
            
        return {
            "observations": copy.deepcopy(states), 
            "actions": copy.deepcopy(actions), 
            "rewards": copy.deepcopy(rewards)
        }
        
    def reset(self) -> None:
        """重置模型状态"""
        self._current_step = 0
        self._training_info = {}
    
    def learn(self, expert_data) -> None:
        """学习专家数据"""
        pass
    
    @abstractmethod
    def predict(
        self,
        observation: Union[np.ndarray, Dict],
        deterministic: bool = True
    ) -> Any:
        """预测动作"""
        raise NotImplementedError
        
    @abstractmethod
    def train_step(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        raise NotImplementedError
        
    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError
        
    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError