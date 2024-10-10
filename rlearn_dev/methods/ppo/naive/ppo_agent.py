import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
# from .network import get_actor_model, get_critic_model
from ....core.agent.main.online_agent_ve import OnlineAgentVE
from .network.actor_critic import ActorCritic

# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
class PPOAgent(OnlineAgentVE):
    def __init__(self, env, config, logger=None, seed=None, **kwargs):
        super().__init__(env, config, logger=logger, seed=seed, **kwargs)
        
    def initialize(self, *args, **kwargs):
        self.num_envs = self.env.num_envs
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        
        # algo
        self.learning_rate = self.config.get('learning_rate', 2.5e-4)
        self.anneal_lr = self.config.get('anneal_lr', True)
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95) 
        self.num_minibatches = self.config.get('num_minibatches', 4)
        self.update_epochs = self.config.get('update_epochs', 4)
        self.norm_adv = self.config.get('norm_adv', True)
        self.clip_coef = self.config.get('clip_coef', 0.2)
        self.clip_vloss = self.config.get('clip_vloss', True)
        self.ent_coef = self.config.get('ent_coef', 0.01)
        self.vf_coef = self.config.get('vf_coef', 0.5)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        self.target_kl = self.config.get('target_kl')
        
        self.optimizer_eps = self.config.get('optimizer_eps', 1e-5)
        
        self.batch_size = None
        self.minibatch_size = None
        # self.minibatch_size = self.config.get('minibatch_size')
        # self.mini_batch_epochs = self.config.get('mini_batch_epochs', 8)
        # self.mini_batch_size = self.config.get('mini_batch_size', 12)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.get('cuda', True) else "cpu")
        assert isinstance(self.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        self.actor_critic = ActorCritic(self.env)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate, eps=self.optimizer_eps)
        # self.actor = get_actor_model(env, model_type='MLPActor').to(self.device)
        # self.critic = get_critic_model(env, model_type='MLPCritic').to(self.device)
        # self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=config.learning_rate)
    
    def before_learn(self, states, infos, **kwargs):
        # self.obs = torch.zeros((self.steps_per_epoch, self.num_envs) + self.env.observation_space.shape).to(self.device)
        # actions = torch.zeros((self.config.num_steps, self.config.num_envs) + self.env.action_space.shape).to(self.device)
        num_envs = self.num_envs
        steps_per_epoch = self.steps_per_epoch
        # batch_size: 单个epoch，所有env的step步数steps_per_epoch
        self.batch_size = int(self.num_envs * steps_per_epoch)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        # self.num_iterations = self.total_timesteps // self.batch_size
    
        # storage
        self.obs = torch.zeros((steps_per_epoch, num_envs) + self.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((steps_per_epoch, num_envs) + self.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((steps_per_epoch, num_envs)).to(self.device)
        self.rewards = torch.zeros((steps_per_epoch, num_envs)).to(self.device)
        self.dones = torch.zeros((steps_per_epoch, num_envs)).to(self.device)
        self.values = torch.zeros((steps_per_epoch, num_envs)).to(self.device)
        
        # var for single step
        self.next_obs = torch.Tensor(states).to(self.device) # (num_envs, *obs_shape)
        self.next_done = torch.zeros(self.num_envs).to(self.device)
    
    def before_episode(self, epoch, **kwargs):
        if self.anneal_lr:
            frac = 1.0 - (epoch - 1.0) / self.max_epochs
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow
    
    def select_action(self, state: torch.Tensor, 
                      epoch_step, *args, **kwargs):
        # state 和 self.next_obs 是相同的, 只是next_obs是torch.Tensor, state是numpy.ndarray
        self.obs[epoch_step] = self.next_obs
        self.dones[epoch_step] = self.next_done
        
        print(f'{self.next_obs.shape, state.shape = }')
        assert np.allclose(self.next_obs.cpu().numpy(), state)

        # ALGO LOGIC: action logic
        # action, action-logprob, action-value
        with torch.no_grad():
            # action: (num_envs, action_dim)
            action, logprob, _, value = self.actor_critic.get_action_and_value(self.next_obs, compute_entropy=False)
            # value: (num_envs, )
            self.values[epoch_step] = value.flatten()
        
        print(f'{action.shape, logprob.shape, value.shape = }')
        assert action.shape == (self.num_envs, )
        assert logprob.shape == (self.num_envs, )
        assert value.shape == (self.num_envs, )
        
        self.actions[epoch_step] = action
        self.logprobs[epoch_step] = logprob
        # required by: env.step
        return action.cpu().numpy()
    
    def step(self, next_obs, rewards, terminates, truncates, 
             infos, epoch, epoch_step):
        next_done = np.logical_or(terminates, truncates)
        print(f'{next_done.shape, next_done.dtype=}')
        print(f'{next_obs.shape, rewards.shape=}')
        # TODO ? view(-1)
        self.rewards[epoch_step] = torch.tensor(rewards).to(self.device).view(-1)
        self.next_obs, self.next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)
    
    def after_episode(self, epoch):
        # self.next_obs, self.next_done := env.step后的结果
        # bootstrap value if not done
        should_exit = False
        device = self.device
        obs, actions, rewards, dones, values, logprobs = self.obs, self.actions, self.rewards, self.dones, self.values, self.logprobs
        next_obs, next_done = self.next_obs, self.next_done
        with torch.no_grad():
            # compute: 
            #  - advantages: GAE_advantage
            #  - returns: GAE_advantage + values
            # value: predicted return
            next_value = self.actor_critic.get_value(next_obs).reshape(1, -1)
            print(f'{next_value.shape=}')
            advantages = torch.zeros_like(rewards).to(device) # shape: (steps_per_epoch, num_envs)
            lastgaelam = 0
            gamma = self.gamma
            gae_lambda = self.gae_lambda
            for t in reversed(range(self.steps_per_epoch)):
                if t == self.steps_per_epoch - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        print(f'**{b_obs.shape, b_actions.shape, b_advantages.shape, b_returns.shape, b_values.shape=}')
        
        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        # 每个数据还是跑一遍
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end] 

                newlogprob, entropy, newvalue = self.actor_critic.get_action_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # 提高稳定性
                    # TODO: move to loc of target_kl
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                # XXX TODO: 这里减去的是minibatch的平均值, 可以考虑减去整个steps_per_epoch*num_envs的平均值
                #           或其他自定义长度的平均值
                # XXX: 这里打乱了顺序，导致非因果，时序非平稳数据，比如金融数据，或应该保存原始序列，减去running_mean更为合适
                #      TODO: 将来校验
                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                # XXX 此处用huber-loss或可考虑，value-clip未必有依据
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                self.logger.info(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl}")
                should_exit = True
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        # XXX ?
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        episode_info = {
            'loss': loss.item(),
            'pg_loss': pg_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'v_loss': v_loss.item(),
            'explained_var': explained_var,
            'clipfracs': np.mean(clipfracs)
        }
        return should_exit, episode_info
    
    def after_learn(self):
        pass
    
    def predict(self, states, deterministic=False):
        state = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            actions, action_probs, entropy, values = self.actor_critic.get_action_and_value(
                state, deterministic=deterministic,
                compute_entropy=True
            )
            info = {
                'action_probs': action_probs.cpu().tolist(),
                'entropy': entropy.cpu().tolist(),
                'values': values.cpu().tolist()
            }
        
        return actions, info
    
    def model_dict(self):
        state = {
            'config': self.config,
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return state
    
    def load_model_dict(self, model_dict):
        self.config = model_dict['config']
        self.initialize()
        self.actor_critic.load_state_dict(model_dict['actor_critic'])
        self.optimizer.load_state_dict(model_dict['optimizer'])
    