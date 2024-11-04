import torch
import numpy as np
from warnings import warn
from cfgdict import Schema, Field
from ....utils.env import is_discrete_space
from ....core.agent.naive.online_agent import OnlineAgent

class TabularQAgent(OnlineAgent):
    """
    Tabular Q-Learning Agent
    """
    schema = Schema(
        learning_rate=Field(type='float', default=0.01, ge=0),
        gamma=Field(type='float', default=0.99, ge=0, le=1),
        epsilon_start=Field(type='float', default=0.5, ge=0, le=1),
        epsilon_end=Field(type='float', default=0.01, ge=0, le=1),
        epsilon_decay=Field(type='float', default=0.99, ge=0, le=1),
    )
    
    def initialize(self, *args, **kwargs):
        assert is_discrete_space(self.env.observation_space), 'Q-Learning only supports discrete state space'
        assert is_discrete_space(self.env.action_space), 'Q-Learning only supports discrete action space'
        self.state_n = self.env.observation_space.n
        self.action_n = self.env.action_space.n
        self.gamma = self.config['gamma']
        self.learning_rate = self.config['learning_rate']
        self.epsilon = self.config['epsilon_start']
        self.Q_table = torch.zeros(self.state_n, self.action_n)
        # self.Q_table = torch.rand(self.state_n, self.action_n)
    
    def before_episode(self, state, info, **kwargs):
        pass
    
    def select_action(self, state, 
                      *, episode_idx=None,
                      global_step_idx=None, 
                      episode_step_idx=None):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_n)
        else:
            action = torch.argmax(self.Q_table[state]).item()
        return action

    def step(self, state, action, 
             next_state, reward, terminated, truncated, info,
             *, episode_idx=None, 
             global_step_idx=None, 
             episode_step_idx=None):
        q = self.Q_table[state, action]
        
        if terminated or truncated:
            q_hat = reward
        else:
            best_next_action = torch.argmax(self.Q_table[next_state])
            q_hat = reward + self.gamma * self.Q_table[next_state, best_next_action]
        
        td_error = q_hat - q
        self.Q_table[state, action] = q + self.learning_rate * td_error
    
    def after_episode(self, 
                      episode_rewards=None, 
                      episode_total_reward=None,
                      episode_idx=None,
                      **kwargs):
        self.epsilon = self.config['epsilon_end'] + (self.config['epsilon_start'] - self.config['epsilon_end']) * (self.config['epsilon_decay'] ** episode_idx)

    def model_dict(self):
        return {
            'config': self.config.to_dict(),
            'Q_table': self.Q_table
        }
    
    def load_model_dict(self, model_dict):
        self.config = self.make_config(model_dict['config'], logger=self.logger)
        self.initialize()
        self.Q_table = model_dict['Q_table']
    
    def predict(self, state, deterministic=None):
        if deterministic is not None:
            warn(f'deterministic={deterministic} parameter is ignored in the current implementation')
        action = torch.argmax(self.Q_table[state]).item()
        info = {}
        return action, info
