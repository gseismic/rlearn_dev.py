import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from cfgdict import Schema, Field
from ....core.agent.naive.vector import OnlineAgent
from ....utils.misc import polyak_update
from ....utils.replay_buffer.vector import RandomReplayBuffer
from .network.api import get_actor, get_critic

# referece:  clearnrl, stable-baselines3
# based on naive.TD3
class SACAgent(OnlineAgent):
    """
    Soft Actor-Critic
    """
    schema = Schema(
        actor_lr=Field(type='float', default=0.0003, ge=0),
        critic_lr=Field(type='float', default=0.0003, ge=0),
        gamma=Field(type='float', default=0.99, ge=0, le=1),
        buffer_size=Field(type='int', default=1_000, ge=1),
        batch_size=Field(type='int', default=256, ge=0),
        q_learning_starts=Field(type='int', default=1_000, ge=1),
        policy_learning_starts=Field(type='int', default=5_000, ge=1),
        policy_frequency=Field(type='int', default=2, ge=0),
        policy_update_steps=Field(type='int', default=2, ge=1), # new compared to naive.TD3
        tau=Field(type='float', default=0.005, ge=0, le=1),
        # exploration_noise=Field(type='float', default=0.1, ge=0, le=1),
        # policy_noise=Field(type='float', default=0.2, ge=0, le=1), 
        # noise_clip=Field(type='float', default=0.5, ge=0, le=1), 
        policy_grad_norm_clip=Field(type='float', default=1.0, ge=0),
        critic_grad_norm_clip=Field(type='float', default=1.0, ge=0),
        autotune=Field(type='bool', default=True), # new compared to naive.TD3
        alpha=Field(type='float', default=0.2), # new compared to naive.TD3
        log_alpha_lr=Field(type='float', default=0.001, ge=0), # new compared to naive.TD3
        alpha_grad_norm_clip=Field(type='float', default=1.0, ge=0), # new compared to naive.TD3
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
        self.critic_grad_norm_clip = self.config['critic_grad_norm_clip']
        self.alpha_grad_norm_clip = self.config['alpha_grad_norm_clip']

        self.actor = get_actor(self.state_space,
                               self.action_space).to(self.device)
        self.critic1 = get_critic(self.state_space,
                                  self.action_space).to(self.device)
        self.critic2 = get_critic(self.state_space, self.action_space).to(self.device)
        self.target_critic1 = get_critic(self.state_space,
                                         self.action_space).to(self.device)
        self.target_critic2 = get_critic(self.state_space,
                                         self.action_space).to(self.device)
        self._update_target_networks(tau=1.0) # copy critic

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config['actor_lr'])
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.config['critic_lr']
        )
        if self.config['autotune']:
            # TODO XXX: use action_dim
            self.target_entropy = -np.prod(self.action_space.shape)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.config['log_alpha_lr'])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.config['alpha']
        
        # TODO XXX: USE (state_shape, action_dim)
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

    def select_action(self, state, *, episode_step_idx=None, global_step_idx=None, **kwargs):
        if global_step_idx is not None and global_step_idx < self.learning_starts:
            action = np.array([self.action_space.sample() for _ in range(self.num_envs)])
        else:
            with torch.no_grad():
                action, _, _ = self.actor.get_action(
                    torch.FloatTensor(state).to(self.device),
                    compute_log_prob=False,
                    compute_mean=True
                )
                # no need to clip, because action is already in the range of action space
                # 无需添加噪音 | no need to add noise
                action = action.cpu().numpy()
        return action.astype(self.action_space.dtype)
    
    def step(self, states, actions, next_states, rewards, terminates, truncates, infos, 
             *, global_step_idx=None, episode_step_idx=None, episode_idx=None, **kwargs):
        # states: (num_envs, *state_shape)
        # actions: (num_envs, *action_dim)
        # rewards: (num_envs,)
        # next_states: (num_envs, *state_shape)
        # dones: (num_envs,)
        dones = np.logical_or(terminates, truncates)
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
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        next_states = batch.next_states.to(self.device)
        dones = batch.dones.to(self.device)

        # compute target q value
        with torch.no_grad():
            # 与td3不同，此处使用actor而不是target_actor | different from td3, use actor instead of target_actor
            actions_of_next_states, log_pi_of_next_states, _ = self.actor.get_action(
                next_states,
                compute_log_prob=True,
                compute_mean=True
            )
            values1_of_next_states = self.target_critic1(next_states, actions_of_next_states)
            values2_of_next_states = self.target_critic2(next_states, actions_of_next_states)
            # 我们认为最小值的critic更可靠 | we believe the critic with the minimum value is more reliable
            min_values_of_next_states = (
                torch.min(values1_of_next_states, values2_of_next_states) 
                - self.alpha * log_pi_of_next_states
            )
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
        if self.critic_grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.critic_grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.critic_grad_norm_clip)
        self.critic_optimizer.step()
        
        # 策略网络更新比Q网络慢 | policy-network update slower than Q-network
        if (
            (global_step_idx >= self.policy_learning_starts)
            and ((global_step_idx + 1 - self.policy_learning_starts) % self.policy_frequency == 0)
        ):
            if not self._policy_training_started:
                self._policy_training_started = True
                self.logger.info('Policy-network training started')
                
            for _ in range(self.config['policy_update_steps']):
                actions, log_pi, _ = self.actor.get_action(states, compute_log_prob=True, compute_mean=False)
                # 我们认为最小值更可靠，更新actor参数，使得critic评价出的最小值变大
                q1 = self.critic1(states, actions)
                q2 = self.critic2(states, actions)
                # 这里使用min(q1, q2)，而不是td3中的-q1.mean()
                actor_loss = (-torch.min(q1, q2) + self.alpha * log_pi).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.policy_grad_norm_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.policy_grad_norm_clip)
                self.actor_optimizer.step()
                
                if self.config['autotune']:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(states, compute_log_prob=True, compute_mean=False)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    if self.alpha_grad_norm_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.log_alpha, self.alpha_grad_norm_clip)
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()
        
        # naive.TD3中策略更新时才更新target网络 | target network updated only when policy updated in naive.TD3
        # 意味着这里比naive.TD3多更新了几次target网络
        # TODO: 使得二者一致
        self._update_target_networks()
    
    def _update_target_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        # polyak_update(self.actor.parameters(), self.target_actor.parameters(), tau)
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
        
    def predict(self, state, deterministic=True):
        # if deterministic is not None:
        #     warn(f'deterministic={deterministic} parameter is ignored in the current implementation')
        state = torch.FloatTensor(np.array(state)).to(self.device)
        state = state.unsqueeze(0) # (1, *state_shape)
        with torch.no_grad():
            action, _, _ = self.actor.get_action(
                state, compute_log_prob=False, compute_mean=False, deterministic=deterministic
            )
        info = {}
        return action.detach().squeeze(0).cpu().numpy(), info
