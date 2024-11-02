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
class TD3Agent(OnlineAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient
    
    Note:
        - network直接使用的是ddpg的，没有做任何修改，和clearnrl的td3中的network不同
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
        policy_noise=Field(type='float', default=0.2, ge=0, le=1), # new compared to naive.DDPG
        noise_clip=Field(type='float', default=0.5, ge=0, le=1), # new compared to naive.DDPG
        policy_grad_norm_clip=Field(type='float', required=False, default=None, ge=0),
    )
        
    def initialize(self):
        if self.config['q_learning_starts'] >= self.config['policy_learning_starts']:
            raise ValueError(f'q_learning_starts={self.config["q_learning_starts"]} must be less than policy_learning_starts={self.config["policy_learning_starts"]}')
        if not hasattr(self.action_space, 'low') or not hasattr(self.action_space, 'high'):
            raise ValueError("Action space must have 'low' and 'high' attributes")
        self.q_learning_starts = self.config['q_learning_starts']
        self.policy_learning_starts = self.config['policy_learning_starts']
        self.learning_starts = self.q_learning_starts
        self.policy_frequency = self.config['policy_frequency']
        self.policy_grad_norm_clip = self.config['policy_grad_norm_clip']
        self.actor = get_actor(self.state_space, self.action_space)
        self.critic1 = get_critic(self.state_space, self.action_space)
        self.critic2 = get_critic(self.state_space, self.action_space) # new compared to naive.DDPG
        self.target_actor = get_actor(self.state_space, self.action_space)
        self.target_critic1 = get_critic(self.state_space, self.action_space)
        self.target_critic2 = get_critic(self.state_space, self.action_space) # new compared to naive.DDPG

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config['actor_lr'])
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.config['critic_lr']
        )
        
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
                action = self.actor(torch.FloatTensor(state))
                action += torch.normal(0, self.actor.action_scale * self.config['exploration_noise'])
                action = action.cpu().numpy().clip(self.action_space.low, self.action_space.high)
        return action.astype(self.action_space.dtype)
    
    def step(self, states, actions, next_states, rewards, terminates, truncates, infos, 
             *, global_step_idx=None, episode_step_idx=None, episode_idx=None, **kwargs):
        # 跟naive.DDPG比不变 | same as naive.DDPG
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
        batch_size = batch.states.shape[0] # if buffer not full, batch_size < batch_size
        # batch.states: (batch_size, *state_shape)
        states = batch.states
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_states
        dones = batch.dones

        # compute target q value
        with torch.no_grad():
            vectorized_action_space_low = torch.tensor(np.array([self.action_space.low]*batch_size), device=self.device)
            vectorized_action_space_high = torch.tensor(np.array([self.action_space.high]*batch_size), device=self.device)
            assert vectorized_action_space_high.shape[0] == batch_size
            
            # sarsa-like
            clipped_noise = (torch.randn_like(actions, device=self.device) * self.config['policy_noise']).clamp(
                -self.config['noise_clip'], self.config['noise_clip']
            ) * self.target_actor.action_scale

            actions_of_next_states = (self.target_actor(next_states) + clipped_noise).clamp(
                vectorized_action_space_low, vectorized_action_space_high
            )
            values1_of_next_states = self.target_critic1(next_states, actions_of_next_states)
            values2_of_next_states = self.target_critic2(next_states, actions_of_next_states)
            # 我们认为最小值的critic更可靠 | we believe the critic with the minimum value is more reliable
            min_values_of_next_states = torch.min(values1_of_next_states, values2_of_next_states)
            # compute td target q value
            td_target_q = rewards + (1 - dones) * self.gamma * min_values_of_next_states.view(-1)
            
        # compute critic loss
        current_q1 = self.critic1(states, actions).view(-1)
        current_q2 = self.critic2(states, actions).view(-1)
        
        q_loss1 = nn.MSELoss()(current_q1, td_target_q)
        q_loss2 = nn.MSELoss()(current_q2, td_target_q)
        critic_loss = q_loss1 + q_loss2 # td_target_q.shape: (batch_size,)
        
        self.critic_optimizer.zero_grad()
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
            # XXX: 只使用critic1来更新actor | only use critic1 to update the actor
            # 是否可以使用两个critic的最小值来更新actor，注意：critic1此处不更新 | consider using the minimum of two critics to update the actor
            # XXX 此处和clearnrl不同
            # XXX TODO: 验证猜想
            # 我们认为最小值更可靠，更新actor参数，使得critic评价出的最小值变大
            actions = self.actor(states)
            q1 = self.critic1(states, actions)
            # q2 = self.critic2(states, actions)
            # actor_loss = -torch.min(q1, q2).mean()
            actor_loss = -q1.mean() # stable-baselines3中也只使用q1
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.policy_grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.policy_grad_norm_clip)
            self.actor_optimizer.step()
            self._update_target_networks()
    
    def _update_target_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        polyak_update(self.actor.parameters(), self.target_actor.parameters(), tau)
        polyak_update(self.critic1.parameters(), self.target_critic1.parameters(), tau)
        polyak_update(self.critic2.parameters(), self.target_critic2.parameters(), tau)
        
    def load_model_dict(self, model_dict):
        self.config = self.make_config(model_dict['config'])
        self.initialize()
        self.actor.load_state_dict(model_dict['actor'])
        self.critic1.load_state_dict(model_dict['critic1'])
        self.critic2.load_state_dict(model_dict['critic2'])
        self._update_target_networks(tau=1.0) # copy from actor and critic

    def model_dict(self):
        return {
            'config': self.config.to_dict(),
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
        }
        
    def predict(self, state, deterministic=None):
        if deterministic is not None:
            warn(f'DDPG is not a stochastic policy, deterministic={deterministic} is ignored')
        state = torch.FloatTensor(np.array(state))
        action = self.actor(state)
        return action.detach().numpy()

    def _get_action_noise(self, batch_size):
        """生成动作噪声"""
        noise = torch.randn(batch_size, *self.action_space.shape, device=self.device)
        noise = noise * self.config['policy_noise']
        return torch.clamp(noise, -self.config['noise_clip'], self.config['noise_clip'])