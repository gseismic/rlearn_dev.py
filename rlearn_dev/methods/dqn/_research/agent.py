import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from cfgdict import Schema, Field
from ....utils.replay_buffer import (
    Experience,
    RandomReplayBuffer, PrioritizedReplayBuffer
)
from ....core.agent.naive import OnlineAgent
from ....nnblock.core.noisy_linear import DenseNoisyLinear, FactorizedNoisyLinear
from .network.dqn import DQN, DuelingDQN, C51Network

class DQNAgent_Main(OnlineAgent):
    """Online DQN Agent
    DQNAgent_Main is a class for training and evaluating a DQN agent.
    
    Notes:
        - Discrete action space Only | 仅支持离散动作空间
    
    reference:
        - https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
        - https://github.com/Curt-Park/rainbow-is-all-you-need/tree/master?tab=readme-ov-file 
        - https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/dqn.py
    """
    
    schema = Schema(
        learning_rate=Field(default=1e-3, type='float', gt=0),
        batch_size=Field(default=64, type='int', gt=0),
        gamma=Field(default=0.99, type='float', min=0, max=1),
        epsilon_start=Field(default=1.0, type='float', gt=0),
        epsilon_end=Field(default=0.01, type='float', gt=0),
        epsilon_decay=Field(default=0.995, type='float', gt=0, max=1),
        target_update_freq=Field(default=10, type='int', gt=0),
        memory_size=Field(default=10000, type='int', gt=0),
        dueling_dqn=Field(default=True, type='bool'),
        double_dqn=Field(default=True, type='bool'),
        prioritized_replay=Field(default=True, type='bool'),
        hidden_layers=Field(default=[128, 128], type='list', min_len=1),
        device=Field(default='cpu', type='str', choices=['cpu', 'cuda']),
        verbose_freq=Field(default=10, type='int', gt=0),
        use_grad_clip=Field(default=True, type='bool'),  # 新增：是否使用梯度裁剪
        max_grad_norm=Field(default=1.0, type='float', gt=0),  # 保留最大梯度范数参数
        use_noisy_net=Field(default=False, type='bool'),
        noisy_net_type=Field(default='dense', type='str', choices=['dense', 'factorized']),
        noisy_net_std_init=Field(default=0.5, type='float', gt=0),
        noisy_net_k=Field(default=1, type='int', gt=0),
        noise_decay=Field(default=0.99, type='float', gt=0, max=1),
        min_exploration_factor=Field(default=0.1, type='float', gt=0),
        algorithm=Field(default='dqn', type='str', choices=['dqn', 'c51']),  # 新增：算法选择
        num_atoms=Field(default=51, type='int', gt=0),  # 新增：C51 分布的原子数
        v_min=Field(default=-10.0, type='float'),  # 新增：C51 分布的最小值
        v_max=Field(default=10.0, type='float'),  # 新增: C51 分布的最大值
    )
    
    def initialize(self, *args, **kwargs):
        self.state_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = self.env.action_space.n
        self.epsilon = self.config['epsilon_start']
        self.update_steps = 0
        self.logger.info(f"DQNAgent_Main initialized with state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        self.use_grad_clip = self.config.get('use_grad_clip', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.logger.info(f"Gradient clipping: {'enabled' if self.use_grad_clip else 'disabled'}")
        if self.use_grad_clip:
            self.logger.info(f"Max gradient norm set to: {self.max_grad_norm}")
        self.use_noisy_net = self.config.get('use_noisy_net', False)
        self.noisy_net_type = self.config.get('noisy_net_type', 'dense')
        self.noisy_net_std_init = self.config.get('noisy_net_std_init', 0.5)
        self.noisy_net_k = self.config.get('noisy_net_k', 1)
        self.logger.info(f"Noisy Net: {'enabled' if self.use_noisy_net else 'disabled'}")
        if self.use_noisy_net:
            self.logger.info(f"Noisy Net Type: {self.noisy_net_type}")
            self.logger.info(f"Noisy Net Std Init: {self.noisy_net_std_init}")
            if self.noisy_net_type == 'factorized':
                self.logger.info(f"Noisy Net K: {self.noisy_net_k}")
        self.noise_decay = self.config.get('noise_decay', 0.99)
        self.min_exploration_factor = self.config.get('min_exploration_factor', 0.1)
        self.logger.info(f"Noise Decay: {self.noise_decay}")
        self.logger.info(f"Min Exploration Factor: {self.min_exploration_factor}")
        self.init_networks()
            
    def init_networks(self):
        if self.config['algorithm'] == 'c51':
            self.q_network = C51Network(
                self.state_dim, 
                self.action_dim, 
                self.config['num_atoms'], 
                self.config['v_min'], 
                self.config['v_max'],
                self.use_noisy_net, 
                self.noisy_net_type, 
                self.noisy_net_std_init, 
                self.noisy_net_k,
                min_exploration_factor=self.min_exploration_factor,
                noise_decay=self.noise_decay
            )
            self.target_network = C51Network(
                self.state_dim, 
                self.action_dim, 
                self.config['num_atoms'], 
                self.config['v_min'], 
                self.config['v_max'],
                self.use_noisy_net, 
                self.noisy_net_type, 
                self.noisy_net_std_init, 
                self.noisy_net_k,
                min_exploration_factor=self.min_exploration_factor,
                noise_decay=self.noise_decay
            )
            self.logger.info(f"C51 network initialized with state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        elif self.config['dueling_dqn']:
            self.q_network = DuelingDQN(
                self.state_dim, 
                self.action_dim, 
                self.use_noisy_net, 
                self.noisy_net_type, 
                self.noisy_net_std_init, 
                self.noisy_net_k,
                min_exploration_factor=self.min_exploration_factor,
                noise_decay=self.noise_decay
            )
            self.target_network = DuelingDQN(
                self.state_dim, 
                self.action_dim, 
                self.use_noisy_net, 
                self.noisy_net_type, 
                self.noisy_net_std_init, 
                self.noisy_net_k,
                min_exploration_factor=self.min_exploration_factor,
                noise_decay=self.noise_decay
            )
            self.logger.info(f"DuelingDQN initialized with state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        else:
            self.q_network = DQN(
                self.state_dim, 
                self.action_dim, 
                self.use_noisy_net, 
                self.noisy_net_type, 
                self.noisy_net_std_init, 
                self.noisy_net_k,
                min_exploration_factor=self.min_exploration_factor,
                noise_decay=self.noise_decay
            )
            self.target_network = DQN(
                self.state_dim, 
                self.action_dim, 
                self.use_noisy_net, 
                self.noisy_net_type, 
                self.noisy_net_std_init, 
                self.noisy_net_k,
                min_exploration_factor=self.min_exploration_factor,
                noise_decay=self.noise_decay
            )
            self.logger.info(f"DQN initialized with state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['learning_rate'])
        self.logger.info(f"Optimizer initialized with learning_rate: {self.config['learning_rate']}")
        
        if self.config['prioritized_replay']:   
            self.memory = PrioritizedReplayBuffer(self.config['memory_size'])
            self.logger.info(f"PrioritizedReplayBuffer initialized with memory_size: {self.config['memory_size']}")
        else:
            self.memory = RandomReplayBuffer(self.config['memory_size'])
            self.logger.info(f"RandomReplayBuffer initialized with memory_size: {self.config['memory_size']}")
        
    def select_action(self, state):
        """
        基于epsilon-greedy策略选择动作 | Select action based on epsilon-greedy policy
        Args:
            - state: 状态 | State
        """
        if not self.use_noisy_net and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if self.config['algorithm'] == 'c51':
                distribution = self.q_network(state)
                expected_values = (distribution * self.q_network.support).sum(2)
                return expected_values.argmax().item()
            else:
                q_values = self.q_network(state)
                return q_values.argmax().item()
    
    def update(self):
        if len(self.memory) < self.config['batch_size']:
            return
        
        if self.config['prioritized_replay']:
            experiences, indices, weights = self.memory.sample(self.config['batch_size'])
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = self.memory.sample(self.config['batch_size'])
            weights = None
        
        batch = Experience(*zip(*experiences)) # shape: (batch_size, )
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)  # shape: (batch_size, state_dim)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)  # shape: (batch_size, 1)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)  # shape: (batch_size, 1)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)  # shape: (batch_size, state_dim)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)  # shape: (batch_size, 1)
        
        if self.config['algorithm'] == 'c51':
            current_dist = self.q_network(state_batch)
            log_p = torch.log(current_dist[range(self.config['batch_size']), action_batch.squeeze()])
            
            with torch.no_grad():
                target_dist = self.target_network(next_state_batch)
                if self.config['double_dqn']:
                    next_actions = self.q_network(next_state_batch).sum(dim=2).argmax(dim=1)
                else:
                    next_actions = target_dist.sum(dim=2).argmax(dim=1)
                
                next_dist = target_dist[range(self.config['batch_size']), next_actions]
                
                t_z = reward_batch + (1 - done_batch) * self.config['gamma'] * self.q_network.support.unsqueeze(0)
                t_z = t_z.clamp(self.config['v_min'], self.config['v_max'])
                b = (t_z - self.config['v_min']) / ((self.config['v_max'] - self.config['v_min']) / (self.config['num_atoms'] - 1))
                l = b.floor().long()
                u = b.ceil().long()
                
                target_prob = torch.zeros_like(next_dist)
                offset = torch.linspace(0, (self.config['batch_size'] - 1) * self.config['num_atoms'], self.config['batch_size']).long().unsqueeze(1).expand(self.config['batch_size'], self.config['num_atoms']).to(self.device)
                
                target_prob.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
                target_prob.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
            
            loss = -(target_prob * log_p).sum(1)
            
            # 计算q_values用于优先经验回放
            q_values = (current_dist * self.q_network.support.unsqueeze(0).unsqueeze(0)).sum(2)
            expected_q_values = (target_prob * self.q_network.support.unsqueeze(0)).sum(1)
        else:   
            q_values = self.q_network(state_batch).gather(1, action_batch)
            
            with torch.no_grad():
                if self.config['double_dqn']:
                    next_actions = self.q_network(next_state_batch).argmax(dim=1, keepdim=True)
                    next_q_values = self.target_network(next_state_batch).gather(dim=1, index=next_actions)
                else:
                    next_q_values = self.target_network(next_state_batch).max(dim=1)[0].unsqueeze(1)
            
            expected_q_values = reward_batch + (1 - done_batch) * self.config['gamma'] * next_q_values
            
            loss = nn.MSELoss(reduction='none')(q_values, expected_q_values.detach())
        
        if weights is not None: 
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        if self.config['prioritized_replay']:
            if self.config['algorithm'] == 'c51':
                td_errors = (q_values.gather(1, action_batch).detach() - expected_q_values.unsqueeze(1)).abs().cpu().numpy()
            else:
                td_errors = (q_values.detach() - expected_q_values).abs().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)
        
        self.update_steps += 1
        if self.update_steps % self.config['target_update_freq'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if not self.use_noisy_net:
            self.epsilon = max(self.config['epsilon_end'], self.epsilon * self.config['epsilon_decay'])
        
        if self.use_noisy_net:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
            for module in self.q_network.modules():
                if isinstance(module, (DenseNoisyLinear, FactorizedNoisyLinear)):
                    module.step_update()
            for module in self.target_network.modules():
                if isinstance(module, (DenseNoisyLinear, FactorizedNoisyLinear)):
                    module.step_update()
    
    def learn(self, 
              num_episodes, 
              max_step_per_episode=None, 
              max_total_steps=None, 
              target_episode_reward=None, 
              target_window_avg_reward=None,
              target_window_length=None,
              seed=None):
        """
        Args: 
            - num_episodes: 训练的次数 | Number of episodes to train
            - max_step_per_episode: 每个episode的最大步数 | Maximum steps per episode
            - max_total_steps: 总的训练步数 | Total steps to train
            - target_reward: 目标奖励 | Target reward to achieve
            - seed: 随机种子 | Random seed
        Returns: None
        Notes: 
            - 当设置max_total_steps时，num_episodes和max_step_per_episode将失效
            - 当设置target_reward时，num_episodes和max_step_per_episode将失效
        """
        self.try_seed_all(seed)
        
        self.q_network.to(self.device)
        self.target_network.to(self.device) 
        
        self.monitor = RewardMonitor(
            max_step_per_episode=max_step_per_episode,
            max_total_steps=max_total_steps,
            target_episode_reward=target_episode_reward,
            target_window_avg_reward=target_window_avg_reward,
            target_window_length=target_window_length
        )
        
        should_stop = False
        for episode_idx in range(num_episodes):
            state, _ = self.env.reset() # do NOT use seed
            self.monitor.before_episode_start()
            
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.monitor.after_step_env(next_state, reward, terminated, truncated, info)
                
                done = terminated or truncated
                self.memory.add(state, action, reward, next_state, done)
                self.update()
                
                state = next_state
                exit_episode, exit_learning, (exit_learning_code, exit_learning_msg) = self.monitor.check_exit_conditions()
                
                if exit_learning:
                    if exit_learning_code == 0:
                        should_stop = True
                        break
                    elif exit_learning_code >= 1:
                        should_stop = True
                        break
                    else:
                        raise ValueError(f"Invalid exit learning code: {exit_learning_code}")
                
                if exit_episode:
                    break
                
            self.monitor.after_episode_end()            
            if should_stop or (episode_idx + 1) % self.config['verbose_freq'] == 0:
                self.logger.info(f"Episode {episode_idx+1}/{num_episodes}, Episode Reward: {self.monitor.episode_reward}")
                
            if should_stop:
                if exit_learning_code == 0:
                    self.logger.info(exit_learning_msg)
                elif exit_learning_code >= 1:
                    self.logger.warning(exit_learning_msg)
                else:
                    raise ValueError(f"Invalid exit learning code: {exit_learning_code}")
                break
            
            if episode_idx == num_episodes - 1:
                self.logger.warning(f"Reached the maximum number of episodes: {num_episodes}")
        
        exit_info = {
            "reward_list": self.monitor.all_episode_rewards
        }
        return exit_info
    
    def save(self, path):
        # Save the model | 保存模型
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
            'update_steps': self.update_steps
        }, path)
    
    def load(self, path):
        # Load the model | 加载模型
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self.init_networks()
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_steps = checkpoint['update_steps']
