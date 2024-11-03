import torch
import torch.optim as optim
import numpy as np
from warnings import warn
from cfgdict import Schema, Field
from ....core.agent.naive.online_agent import OnlineAgent
from .network import get_policy_network

class MCPGAgent(OnlineAgent):
    """
    Monte Carlo Policy Gradient (MCPG) Agent [REINFORCE]
    """
    schema = Schema(
        learning_rate=Field(type='float', default=0.001, ge=0),
        gamma=Field(type='float', default=0.99, ge=0, le=1),
        normalize_return=Field(type='bool', default=True),
        eps=Field(type='float', default=1e-8, ge=0),
        clip_grad_norm=Field(type='float', default=1.0, gt=0),
    )
    
    def initialize(self, *args, **kwargs):
        self.policy = get_policy_network(self.env.observation_space, self.env.action_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config['learning_rate'])
        self.gamma = self.config['gamma']
        self.eps = self.config['eps']
        self.states, self.actions, self.rewards = [], [], []
    
    def before_episode(self, state, info, **kwargs):
        self.states, self.actions, self.rewards = [], [], []
    
    def select_action(self, state, 
                      *, episode_idx=None, 
                      global_step_idx=None, 
                      episode_step_idx=None):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1)
        return action.cpu().numpy().item()

    def step(self, state, action, 
             next_state, reward, done, truncated, info,
             *, episode_idx=None, 
             global_step_idx=None, 
             episode_step_idx=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def after_episode(self, 
                      episode_rewards=None, 
                      episode_total_reward=None,
                      episode_idx=None,
                      **kwargs):
        returns = self._calculate_returns(self.rewards)
        self._update_policy(self.states, self.actions, returns)
    
    def _calculate_returns(self, rewards):
        returns = []
        R = 0
        # rewards: list[float] length: T
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.append(R)
        returns = torch.tensor(returns[::-1])
        if self.config['normalize_return']:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
        return returns
    
    def _update_policy(self, states, actions, returns: torch.Tensor):
        self.optimizer.zero_grad()
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        
        probs = self.policy(states)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        loss = -(log_probs * returns).mean()
        
        loss.backward()
        if self.config['clip_grad_norm'] is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['clip_grad_norm'])
        self.optimizer.step()

    def model_dict(self):
        return {
            'config': self.config.to_dict(),
            'policy': self.policy.state_dict(),
        }
    
    def load_model_dict(self, model_dict):
        self.config = self.make_config(model_dict['config'], logger=self.logger)
        self.initialize()
        self.policy.load_state_dict(model_dict['policy'])
    
    def predict(self, state, deterministic=None):
        if deterministic is not None:
            warn(f'deterministic={deterministic} parameter is ignored in the current implementation')
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = self.policy(state)
        info = {}
        return action.detach().cpu().numpy(), info

