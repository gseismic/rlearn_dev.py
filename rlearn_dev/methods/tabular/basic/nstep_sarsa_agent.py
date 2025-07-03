import torch
import numpy as np
from warnings import warn
from collections import deque
from cfgdict import Schema, Field
from ....utils.env import is_discrete_space
from ....core.agent.naive.online_agent import OnlineAgent

class TabularNStepSarsaAgent(OnlineAgent):
    """
    Tabular N-Step Sarsa Agent
    """
    schema = Schema(
        n_step=Field(type='int', default=5, ge=1),
        use_strict_n_step=Field(type='bool', default=True),
        learning_rate=Field(type='float', default=0.01, ge=0),
        gamma=Field(type='float', default=0.99, ge=0, le=1),
        epsilon_start=Field(type='float', default=0.5, ge=0, le=1),
        epsilon_end=Field(type='float', default=0.01, ge=0, le=1),
        epsilon_decay=Field(type='float', default=0.99, ge=0, le=1),
        use_numpy=Field(type='bool', default=True),
    )
    
    def initialize(self, *args, **kwargs):
        assert is_discrete_space(self.env.observation_space), 'Q-Learning only supports discrete state space'
        assert is_discrete_space(self.env.action_space), 'Q-Learning only supports discrete action space'
        self.state_n = self.env.observation_space.n
        self.action_n = self.env.action_space.n
        self.gamma = self.config['gamma']
        self.learning_rate = self.config['learning_rate']
        self.epsilon = self.config['epsilon_start']
        self.n_step = self.config['n_step']
        self.use_strict_n_step = self.config['use_strict_n_step']
        self.use_numpy = self.config['use_numpy']
        if self.use_numpy:
            self.Q_table = np.zeros((self.state_n, self.action_n))
        else:
            self.Q_table = torch.zeros(self.state_n, self.action_n)
        # 
        # n_step = 3：
        # t=0: (s0, a0, r0)
        # t=1: (s1, a1, r1)
        # t=2: (s2, a2, r2)
        # t=3: (s3, a3) - compute Q(s3,a3)
        self.trajectory = deque(maxlen=self.n_step + 1)
    
    def before_episode(self, state, info, **kwargs):
        pass
    
    def select_action(self, state, 
                      *, episode_idx=None,
                      global_step_idx=None,
                      episode_step_idx=None):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_n)
        return np.argmax(self.Q_table[state]) if self.use_numpy else torch.argmax(self.Q_table[state]).item()

    def step(self, state, action, 
             next_state, reward, terminated, truncated, info,
             *, episode_idx=None, 
             global_step_idx=None, 
             episode_step_idx=None):
        self.trajectory.append((state, action, reward, next_state, terminated or truncated))
        
        if terminated or truncated:
            while self.trajectory:
                self.update(self.trajectory, is_terminal=True)
                self.trajectory.popleft()
        else:
            self.update(self.trajectory, is_terminal=False)
    
    def update(self, trajectory, is_terminal):
        current_n_step = len(trajectory)
        
        # 简化步数检查逻辑
        required_steps = self.n_step if is_terminal else self.n_step + 1
        if self.use_strict_n_step and current_n_step < required_steps:
            return False
        
        actual_n_step = min(current_n_step, self.n_step)
        G = sum((self.gamma ** i) * r for i, (_, _, r, _, _) in 
                enumerate(list(trajectory)[:actual_n_step]))
        
        state, action, _, _, _ = trajectory[0]
        
        if not is_terminal and len(trajectory) > actual_n_step:
            _, _, _, next_state, _ = trajectory[actual_n_step]
            next_action = self.select_action(next_state)
            bootstrap_value = (self.Q_table[next_state][next_action] if self.use_numpy 
                             else self.Q_table[next_state][next_action].item())
            G += (self.gamma ** actual_n_step) * bootstrap_value
        
        td_error = G - self.Q_table[state][action]
        self.Q_table[state][action] += self.learning_rate * td_error
        return True
    
    def after_episode(self, episode_idx=None, **kwargs):
        self.epsilon = self.config['epsilon_end'] + (self.config['epsilon_start'] - self.config['epsilon_end']) * (self.config['epsilon_decay'] ** episode_idx)

    def model_dict(self):
        return {'config': self.config.to_dict(), 'Q_table': self.Q_table}
    
    def load_model_dict(self, model_dict):
        self.config = self.make_config(model_dict['config'], logger=self.logger)
        self.initialize()
        self.Q_table = model_dict['Q_table']
    
    def predict(self, state, deterministic=None):
        deterministic = True if deterministic is None else deterministic
        state = torch.FloatTensor(np.array(state))
        action = self.select_action(state) if not deterministic else (
            np.argmax(self.Q_table[state]) if self.use_numpy else torch.argmax(self.Q_table[state]).item()
        )
        return action, {'q_values': self.Q_table[state], 'epsilon': self.epsilon}
