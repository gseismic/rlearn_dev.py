import torch
import numpy as np
from pathlib import Path
from ...core.agent.main.online_agent import OnlineAgent
from ...utils.optimizer import get_optimizer_class
from .network.api import get_model

class QACAgent(OnlineAgent):
    """基于TD约束的actor-critic QAC算法
    
    actor网络: 行动决策
    critic网络: 价值判断, 用TD方法约束q(s_t, a_t)
    """
    
    def __init__(self, env, config, logger=None, seed=None):
        super().__init__(env, config, logger=logger, seed=seed)
    
    def initialize(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'Using device: {self.device}')
        if 'gamma' not in self.config:
            raise ValueError("`gamma` is not set")
        self.gamma = self.config['gamma']
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.model = get_model(state_dim, action_dim, 
                               model_type=self.config['model_type'],
                               model_kwargs=self.config['model_kwargs']).to(self.device)
        
        optimizer_class = get_optimizer_class(self.config.get('optimizer'), 'adam') 
        optimizer_kwargs = self.config.get('optimizer_kwargs', {'lr': 0.001})
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)

    def model_dict(self):
        state = {
            'config': self.config,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return state
    
    def load_model_dict(self, model_dict):
        self.config = model_dict['config']
        self.initialize()
        self.model.load_state_dict(model_dict['model'])
        self.optimizer.load_state_dict(model_dict['optimizer'])

    def select_action(self, state): 
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, _ = self.model(state_tensor)
        action = np.random.choice(len(action_probs.squeeze()), p=action_probs.cpu().detach().numpy().squeeze())
        return action

    def step(self, state, action, reward, next_state, done, *args, **kwargs):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([float(done)]).to(self.device)
        
        _, value = self.model(state_tensor)
        _, next_value = self.model(next_state_tensor)
        td_target = reward_tensor + (1 - done_tensor) * self.gamma * next_value
        td_error = td_target - value

        self.optimizer.zero_grad()
        action_probs, value = self.model(state_tensor)
        actor_loss = -torch.log(action_probs[0, action]) * td_error.detach()
        critic_loss = torch.nn.functional.mse_loss(value, td_target)
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()

        return td_error.item()

    def predict(self, state, deterministic=False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.model(state_tensor)
            if deterministic:
                action = action_probs.argmax().item()
            else:
                action = np.random.choice(len(action_probs.squeeze()), p=action_probs.cpu().detach().numpy().squeeze())
            info = {
                'action_probs': action_probs.cpu().squeeze().tolist(),
                'value': value.cpu().item()
            }
            return action, info
        