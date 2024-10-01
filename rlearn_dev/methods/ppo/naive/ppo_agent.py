import torch
import numpy as np
from pathlib import Path
from ....core.agent.main.online_agent import OnlineAgent
from .network.api import get_model
from ....utils.optimizer import get_optimizer_class

class PPOAgent(OnlineAgent):
    """PPO强化学习算法
    
    """
    
    def __init__(self, env, config, logger=None, seed=None):
        super().__init__(env, config, logger=logger, seed=seed)
    
    def initialize(self):
        if 'gamma' not in self.config:
            raise ValueError("`gamma` is not set")
        self.gamma = self.config['gamma']
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.model = get_model(state_dim, action_dim, 
                               model_type=self.config.get('model_type', 'ActorCriticMLP'),
                               model_kwargs=self.config.get('model_kwargs', {}))
        
        optimizer_class = get_optimizer_class(self.config.get('optimizer'), 'adam') 
        optimizer_kwargs = self.config.get('optimizer_kwargs', {'lr': 0.0003})
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)
        
        self.clip_range = self.config.get('clip_range', 0.2)
        self.n_epochs = self.config.get('n_epochs', 10)
        self.batch_size = self.config.get('batch_size', 64)
        
        self.buffer = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def step(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
        if len(self.buffer) >= self.batch_size:
            self.update()
            self.buffer = []

    def update(self):
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算优势
        with torch.no_grad():
            _, old_values = self.model(states)
            _, next_values = self.model(next_states)
            
        advantages = rewards + self.gamma * next_values * (1 - dones) - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.n_epochs):
            action_probs, values = self.model(states)
            dist = torch.distributions.Categorical(action_probs)
            
            new_log_probs = dist.log_prob(actions)
            old_log_probs = dist.log_prob(actions).detach()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = torch.nn.functional.mse_loss(values.squeeze(), rewards + self.gamma * next_values * (1 - dones))
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=1).item()
        else:
            action = torch.multinomial(action_probs, 1).item()
        
        return action, None

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.config.update(checkpoint['config'])
        self.initialize()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])