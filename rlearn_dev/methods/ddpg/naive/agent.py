import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from warnings import warn
from cfgdict import Schema, Field
from ....core.agent.naive.vector import OnlineAgent
from ....utils.replay_buffer.vector import RandomReplayBuffer
from ....utils.misc import polyak_update
from .network.api import get_actor, get_critic

# referece:  clearnrl, stable-baselines3
class DDPGAgent(OnlineAgent):
    """
    Deep Deterministic Policy Gradient
    """
    schema = Schema(
        actor_lr=Field(type='float', default=0.0003, ge=0),
        critic_lr=Field(type='float', default=0.0003, ge=0),
        gamma=Field(type='float', default=0.99, ge=0, le=1),
        buffer_size=Field(type='int', default=1_000, ge=0),
        batch_size=Field(type='int', default=256, ge=0),
        q_learning_starts=Field(type='int', default=1_000, ge=1),
        policy_learning_starts=Field(type='int', default=5_000, ge=1),
        policy_frequency=Field(type='int', default=2, ge=0),
        tau=Field(type='float', default=0.005, ge=0, le=1),
        exploration_noise=Field(type='float', default=0.1, ge=0, le=1),
        critic_grad_norm_clip=Field(type='float', default=1.0, ge=0),
        policy_grad_norm_clip=Field(type='float', default=1.0, ge=0),
    )
        
    def initialize(self):
        self.critic_grad_norm_clip = self.config['critic_grad_norm_clip']
        self.policy_grad_norm_clip = self.config['policy_grad_norm_clip']
        if self.config['q_learning_starts'] >= self.config['policy_learning_starts']:
            raise ValueError(f'q_learning_starts={self.config["q_learning_starts"]} must be less than policy_learning_starts={self.config["policy_learning_starts"]}')
        self.q_learning_starts = self.config['q_learning_starts']
        self.policy_learning_starts = self.config['policy_learning_starts']
        self.learning_starts = self.q_learning_starts
        self.policy_frequency = self.config['policy_frequency']
        self.actor = get_actor(self.state_space, self.action_space)
        self.critic = get_critic(self.state_space, self.action_space)
        self.target_actor = get_actor(self.state_space, self.action_space)
        self.target_critic = get_critic(self.state_space, self.action_space)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config['critic_lr'])
        
        self.replay_buffer = RandomReplayBuffer(max_samples=self.config['buffer_size'], 
                                                state_space=self.state_space,
                                                action_space=self.action_space,
                                                num_envs=self.num_envs,
                                                device=self.device, 
                                                state_dtype=np.float32,
                                                action_dtype=np.float32,
                                                reward_dtype=np.float32)
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']
        self._q_training_started = False
        self._policy_training_started = False
        self._update_target_networks(tau=1.0)

    def select_action(self, state, *, episode_step_idx=None, global_step_idx=None, **kwargs):
        if global_step_idx is not None and global_step_idx < self.learning_starts:
            action = np.array([self.action_space.sample() for _ in range(self.num_envs)])
        else:
            with torch.no_grad():
                action = self.actor(torch.FloatTensor(state).to(self.device))
                action += torch.normal(0, self.actor.action_scale * self.config['exploration_noise'])
                action = action.cpu().numpy().clip(self.action_space.low, self.action_space.high)
        return action.astype(self.action_space.dtype)
    
    def step(self, states, actions, next_states, rewards, terminates, truncates, infos, 
             *, global_step_idx=None, episode_step_idx=None, episode_idx=None, **kwargs):
        dones = np.logical_or(terminates, truncates)
        # states: (num_envs, *state_shape)
        # actions: (num_envs, *action_dim)
        # rewards: (num_envs,)
        # next_states: (num_envs, *state_shape)
        # dones: (num_envs,)
        self.replay_buffer.add(states, actions, rewards, next_states, dones, infos)
        # 允许运行buffer未满时更新网络 | Allow network updates when the buffer is not full
        if global_step_idx + 1 >= self.learning_starts:
            if not self._q_training_started:
                self._q_training_started = True
                self.logger.info('Q-network training started')
            self._update_networks(self.config['batch_size'], global_step_idx)

    def _update_networks(self, batch_size, global_step_idx):
        batch = self.replay_buffer.sample(batch_size, copy=True)
        # batch.states: (batch_size, *state_shape)
        states = batch.states
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_states
        dones = batch.dones

        # compute target q value
        with torch.no_grad():
            # sarsa-like
            actions_of_next_states = self.target_actor(next_states)
            values_of_next_states = self.target_critic(next_states, actions_of_next_states)
            td_target_q = rewards + (1 - dones) * self.gamma * values_of_next_states.view(-1)
            
        current_q = self.critic(states, actions).view(-1)
        critic_loss = nn.MSELoss()(current_q, td_target_q) # td_target_q.shape: (batch_size,)
        self.critic_optimizer.zero_grad()
        if self.critic_grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_norm_clip)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 策略网络更新比Q网络慢 | policy-network update slower than Q-network
        if (
            (global_step_idx >= self.policy_learning_starts)
            and ((global_step_idx + 1 - self.policy_learning_starts) % self.policy_frequency == 0)
        ):
            if not self._policy_training_started: 
                self._policy_training_started = True 
                self.logger.info('Policy-network training started') 
            actor_loss = - self.critic(states, self.actor(states)).mean() 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.policy_grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.policy_grad_norm_clip)
            self.actor_optimizer.step()
            self._update_target_networks()
    
    def _update_target_networks(self, tau=None):
        # polyak update: target_params = (1 - tau) * target_params + tau * params
        tau = self.tau if tau is None else tau
        polyak_update(self.actor.parameters(), self.target_actor.parameters(), tau)
        polyak_update(self.critic.parameters(), self.target_critic.parameters(), tau)
        
    def load_model_dict(self, model_dict):
        self.config = self.make_config(model_dict['config'])
        self.initialize()
        self.actor.load_state_dict(model_dict['actor'])
        self.critic.load_state_dict(model_dict['critic'])
        self._update_target_networks(tau=1.0) # copy from actor and critic

    def model_dict(self):
        return {
            'config': self.config.to_dict(), 
            'actor': self.actor.state_dict(), 
            'critic': self.critic.state_dict(), 
        }
        
    def predict(self, state, deterministic=None):
        if deterministic is not None:
            warn(f'DDPG is not a stochastic policy, deterministic={deterministic} is ignored')
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()
