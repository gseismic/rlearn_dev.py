import torch
import numpy as np
from pathlib import Path
from ....core.agent.main.online_agent import OnlineAgent
from ....utils.optimizer import get_optimizer_class
from ....utils.replay_buffer import RandomReplayBuffer
from .network.api import get_model
from torch.distributions import Categorical

class C51Agent(OnlineAgent):
    """
    C51Agent 类是基于 C51 算法的智能体，用于在强化学习环境中进行训练和决策。
    | C51Agent class is an agent based on the C51 algorithm, used for training and decision-making in reinforcement learning environments.
    
    Notes:
        - 目前仅能处理离散动作
    """

    def initialize(self):
        self.state_dim = np.prod(self.env.observation_space.shape) # TODO: encoder
        self.action_dim = self.env.action_space.n
        self.num_atoms = self.config.get('num_atoms', 51)
        self.v_min = self.config.get('v_min', -10)
        self.v_max = self.config.get('v_max', 10)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.z = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.gamma = self.config.get('gamma', 0.99)
        self.buffer_size = self.config.get('replay_buffer_capacity', 10000)
        self.replay_buffer = RandomReplayBuffer(capacity=self.buffer_size)
        self.batch_size = self.config.get('batch_size', 32)
        self.target_update_freq = self.config.get('target_update_freq', 10)
        self.policy_net = get_model(self.state_dim, self.action_dim, self.num_atoms,
                                    model_type=self.config['model_type'],
                                    model_kwargs=self.config['model_kwargs'])
    
        self.target_net = get_model(self.state_dim, self.action_dim, self.num_atoms,
                                    model_type=self.config['model_type'],
                                    model_kwargs=self.config['model_kwargs'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        optimizer_class = get_optimizer_class(self.config.get('optimizer'), 'adam')
        optimizer_kwargs = self.config.get('optimizer_kwargs', {'lr': 0.0003})
        self.optimizer = optimizer_class(self.policy_net.parameters(), **optimizer_kwargs)       
    
    def select_action(self, state, epsilon=None):
        if epsilon is not None and np.random.random() < epsilon:
            # or: return self.env.action_space.sample()
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)  # shape: (1, state_dim)
                distribution = self.policy_net(state)  # shape: (1, action_dim, num_atoms)
                self.z = self.z.to(distribution.device)  # 确保z在正确的设备上
                # 各动作的期望 Q(s, a) := E[Z(s, a)]
                expected_Q = (distribution * self.z.unsqueeze(0).unsqueeze(0)).sum(2)  # shape: (1, action_dim)
                # self.z.unsqueeze(0).unsqueeze(0) shape: (1, 1, num_atoms)
                return expected_Q.argmax(1).item()  # 返回一个标量值

    def step(self, state, action, reward, next_state, done, episode_steps, total_steps):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        state_batch = torch.FloatTensor(np.array(state_batch))
        action_batch = torch.LongTensor(np.array(action_batch))
        reward_batch = torch.FloatTensor(np.array(reward_batch))
        next_state_batch = torch.FloatTensor(np.array(next_state_batch))
        done_batch = torch.FloatTensor(np.array(done_batch))
        
        with torch.no_grad():
            next_distribution = self.target_net(next_state_batch)
            next_q = (next_distribution * self.z.unsqueeze(0).unsqueeze(0)).sum(2)
            next_action = next_q.argmax(1)
            next_distribution = next_distribution[range(self.batch_size), next_action]

            Tz = reward_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * self.gamma * self.z.unsqueeze(0)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            # u = (l == u).long() + u # in case of b == l == u

            target_distribution = torch.zeros_like(next_distribution)
            offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms)

            target_distribution.view(-1).index_add_(0, (l + offset).view(-1), (next_distribution * (u.float() - b)).view(-1))
            target_distribution.view(-1).index_add_(0, (u + offset).view(-1), (next_distribution * (b - l.float())).view(-1))

        current_distribution = self.policy_net(state_batch)
        current_distribution = current_distribution[range(self.batch_size), action_batch]

        loss = -(target_distribution * torch.log(current_distribution + 1e-8)).sum(1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def predict(self, state, deterministic=True, out_probs=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)  # shape: (1, state_dim)
            Q_distributions = self.policy_net(state)  # shape: (1, action_dim, num_atoms)
            self.z = self.z.to(Q_distributions.device)  # 确保z在正确的设备上
            # self.z.unsqueeze(0).unsqueeze(0) shape: (1, 1, num_atoms)
            expected_Qs = (Q_distributions * self.z.unsqueeze(0).unsqueeze(0)).sum(2)  # shape: (1, action_dim)
            
            info = {
                'Q_values': expected_Qs,
                'Q_distributions': Q_distributions,
            }
            if deterministic:
                action = expected_Qs.argmax(1).item()
                if out_probs:
                    probs = expected_Qs.softmax(dim=1).squeeze().tolist()
                    info['action_probs'] = probs
                    info['action_prob'] = probs[action]
            else:
                print(expected_Qs)
                probs = expected_Qs.softmax(dim=1).squeeze()
                dist = Categorical(probs=probs)
                action = dist.sample().item()
                action_prob = probs[action].item()
                print(f'probs: {probs}', f'action: {action}: {action_prob}')
                # action = np.random.choice(len(probs), p=probs) 
                # [9.969103848561645e-05, 0.9999003410339355], ValueError: probabilities do not sum to 1
                if out_probs:
                    info['action_probs'] = probs.tolist()
                    info['action_prob'] = action_prob
            
            return action, info
    
    def from_model_dict(self, model_dict):
        self.config.update(model_dict['config'])
        self.initialize()
        self.policy_net.load_state_dict(model_dict['policy_net'])
        self.target_net.load_state_dict(model_dict['target_net'])
        self.optimizer.load_state_dict(model_dict['optimizer'])
   
    def model_dict(self):
        # 只保存与模型推理和训练相关的核心参数
        # replay_buffer等信息可自行保存
        state = {
            'config': self.config,
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'replay_buffer': self.replay_buffer.state_dict(),
        }
        return state
    
    def load_model_dict(self, model_dict):
        self.config = model_dict['config']
        self.initialize()
        self.policy_net.load_state_dict(model_dict['policy_net'])
        self.optimizer.load_state_dict(model_dict['optimizer'])
    
    # def checkpoint_dict(self):
    #     return self.model_dict()
    
    # def load_checkpoint_dict(self, state_dict):
    #     return self.load_model_dict(state_dict)