import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from ....core.agent.main.online_agent_ve import OnlineAgentVE
from .network.discrete import ActorCritic as ActorCriticDiscrete
from .network.continous import ActorCritic as ActorCriticContinous

class PPOAgent(OnlineAgentVE):

    def __init__(self, env, config, logger=None, seed=None, **kwargs):
        super().__init__(env, config, logger=logger, seed=seed, **kwargs)
        
    def initialize(self, *args, **kwargs):
        self.is_single_env = not hasattr(self.env, 'num_envs')
        if self.is_single_env:
            self.num_envs = 1
            self.single_observation_space = self.env.observation_space
            self.single_action_space = self.env.action_space
        else:
            self.num_envs = self.env.num_envs
            self.single_observation_space = self.env.single_observation_space
            self.single_action_space = self.env.single_action_space
        
        if len(self.single_observation_space.shape) == 0:
            self.state_dim = (1,)
        else:
            self.state_dim = self.single_observation_space.shape
        # algo
        self.learning_rate = self.config.get('learning_rate', 2.5e-4)
        self.anneal_lr = self.config.get('anneal_lr', True) 
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95) 
        self.num_minibatches = self.config.get('num_minibatches', 4)
        self.update_epochs = self.config.get('update_epochs', 4)
        self.norm_adv = self.config.get('norm_adv', True)
        self.clip_coef = self.config.get('clip_coef', 0.2)
        self.clip_coef_v = self.config.get('clip_coef_v', 0.8)
        self.clip_vloss = self.config.get('clip_vloss', False) # False
        self.ent_coef = self.config.get('ent_coef', 0.01)
        self.vf_coef = self.config.get('vf_coef', 0.5)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        self.target_kl = self.config.get('target_kl', None)
        self.kl_coef = self.config.get('kl_coef', None)
        
        self.rpo_alpha = self.config.get('rpo_alpha', 0.0)
        
        self.optimizer_eps = self.config.get('optimizer_eps', 1e-5)
        
        self.batch_size = None
        self.minibatch_size = None
        # self.minibatch_size = self.config.get('minibatch_size')
        # self.mini_batch_epochs = self.config.get('mini_batch_epochs', 8)
        # self.mini_batch_size = self.config.get('mini_batch_size', 12)
        self.clipfrac_stop = self.config.get('clipfrac_stop', 0.12) # 0.12
        self.v_clipfrac_stop = self.config.get('v_clipfrac_stop', None) # 
        self.kl_stop = self.config.get('kl_stop', None) 
        self.norm_adv_eps = self.config.get('norm_adv_eps', 1e-8)        
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.get('cuda', True) else "cpu")
        if isinstance(self.single_action_space, gym.spaces.Box):
            self.logger.info(f'Use continuous action space: {self.single_action_space=}')
            self.actor_critic = ActorCriticContinous(self.state_dim, 
                                                     self.single_action_space, #.shape,
                                                     rpo_alpha=self.rpo_alpha).to(self.device)
        else:
            self.logger.info(f'Use discrete action space: {self.single_action_space=}')
            self.actor_critic = ActorCriticDiscrete(self.state_dim, self.single_action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate, eps=self.optimizer_eps)
        self.logger.info(f'config: {self.config}')
        # self.actor = get_actor_model(env, model_type='MLPActor').to(self.device)
        # self.critic = get_critic_model(env, model_type='MLPCritic').to(self.device)
        # self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=config.learning_rate)
    
    def before_learn(self, states, infos, **kwargs):
        num_envs = self.num_envs
        steps_per_epoch = self.steps_per_epoch
        # batch_size: 单个epoch，所有env的step步数steps_per_epoch
        self.batch_size = int(self.num_envs * steps_per_epoch)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        self.states = torch.zeros((steps_per_epoch, num_envs) + self.state_dim).to(self.device)
        self.actions = torch.zeros((steps_per_epoch, num_envs) + self.single_action_space.shape).to(self.device)
        self.log_probs = torch.zeros((steps_per_epoch, num_envs)).to(self.device)
        self.rewards = torch.zeros((steps_per_epoch, num_envs)).to(self.device)
        self.dones = torch.zeros((steps_per_epoch, num_envs)).to(self.device)
        self.values = torch.zeros((steps_per_epoch, num_envs)).to(self.device)
        
        # variables for single step
        self.next_state = torch.Tensor(states).to(self.device) # (num_envs, *obs_shape)
        self.next_done = torch.zeros(self.num_envs).to(self.device)

        self.logger.debug(f'reset: demo states: {states}')
        self.logger.debug(f'reset: demo infos: {infos}')
    
    def before_episode(self, epoch, **kwargs):
        if self.anneal_lr:
            frac = 1.0 - epoch / self.max_epochs
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow
    
    def save_lr(self):
        self.lr_backup = self.optimizer.param_groups[0]["lr"]
    
    def decay_lr(self, ratio):
        assert 0 < ratio <= 1
        self.optimizer.param_groups[0]["lr"] = self.lr_backup * ratio
        
    def restore_lr(self):
        self.optimizer.param_groups[0]["lr"] = self.lr_backup
    
    def select_action(self, state: torch.Tensor, 
                      epoch_step, *args, **kwargs):
        self.states[epoch_step] = self.next_state
        self.dones[epoch_step] = self.next_done

        with torch.no_grad():
            # action: (num_envs, action_dim)
            action, logprob, _, value = self.actor_critic.get_action_and_value(self.next_state, compute_entropy=False)
            # value: (num_envs, 1)
            assert value.shape == (self.num_envs, 1)
            self.values[epoch_step] = value.flatten()
        
        if isinstance(self.single_action_space, gym.spaces.Box):
            assert len(self.single_action_space.shape) == 1
            assert action.shape == (self.num_envs, self.single_action_space.shape[0])
            assert logprob.shape == (self.num_envs,)
        else:
            assert action.shape == (self.num_envs, )
            assert logprob.shape == (self.num_envs, )
        self.actions[epoch_step] = action
        self.log_probs[epoch_step] = logprob
        
        return action.cpu().numpy()
    
    def step(self, next_state, rewards, terminates, 
             truncates, infos, epoch, epoch_step):
        next_done = np.logical_or(terminates, truncates)
        assert rewards.shape == (self.num_envs, )
        self.rewards[epoch_step] = torch.tensor(rewards).to(self.device) # removed: .view(-1)
        
        # OnlineAgentVE 的 next_obs 已经处理
        # print('next_state', next_state)
        # if next_state is not None:
        #     self.next_state = torch.Tensor(next_state).to(self.device)
        # else:
        #     self.next_state = None
        self.next_state = torch.Tensor(next_state).to(self.device)
        self.next_done = torch.Tensor(next_done).to(self.device)
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         if info and "episode" in info:
        #             self.logger.debug(f"{epoch=}, {epoch_step=}, episodic_return={info['episode']['r']}")
                    
    def _compute_gae_and_returns(self, rewards, dones, values, next_state, next_done, device):
        """Compute GAE and returns
        
        Returns: 
          - advantages: GAE_advantage
          - returns: GAE_advantage + values
        """
        device = self.device
        with torch.no_grad():
            # predicted return
            next_value = self.actor_critic.get_value(next_state).reshape(1, -1) # shape: (1, num_envs)
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
        return advantages, returns
            
    def after_episode(self, epoch, episode_reward, **kwargs):
        should_exit_program = False
        (states, actions, rewards, dones, values, log_probs) = (
            self.states, self.actions, self.rewards, self.dones, self.values, self.log_probs
        )

        advantages, returns = self._compute_gae_and_returns(
            rewards, dones, values, self.next_state, self.next_done, self.device
        )

        batch_states = states.reshape((-1,) + self.state_dim)
        batch_log_probs = log_probs.reshape(-1)
        batch_actions = actions.reshape((-1,) + self.single_action_space.shape)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)
        
        batch_indices = np.arange(self.batch_size)
        clipfracs = []
        v_clipfracs = []
        v_clipfrac = 0.0
        
        # 每个数据还是跑一遍，但分多批
        self.save_lr()
        exit_this_train = False
        for epoch in range(self.update_epochs):
            if exit_this_train:
                break
            
            approx_kls = []
            approx_kl = None
            assert self.minibatch_size > 1
            clipfrac = None
            # 因为数据高度复用了
            # 增加内部循环，因为local_minibatch_size变大了，循环次数为：self.batch_size/self.minibatch_size
            if 0:
                local_minibatch_size = min(int(self.minibatch_size*1.005**epoch), self.batch_size)
                _std_num_loops = int(self.batch_size/self.minibatch_size)
                _current_num_loops = int(self.batch_size/local_minibatch_size)
                recommended_num_loops = int(_std_num_loops/_current_num_loops)
                # print(f'**{recommended_num_loops=}, {_std_num_loops=}, {_current_num_loops=}')   
            else:
                local_minibatch_size = self.minibatch_size
                recommended_num_loops = 1
            for iloop in range(recommended_num_loops):
                if exit_this_train:
                    break
                
                np.random.shuffle(batch_indices) # shuffle for each local_minibatch_size for i.i.d condition
                for start in range(0, self.batch_size, local_minibatch_size):
                    # mbatch_advantages.std() 可能因只有一个元素导致，导致计算失败
                    end = start + local_minibatch_size
                    mini_batch_indices = batch_indices[start:end] 
                    # 因为有update_epochs随机复用数据，丢弃部分数据是可行的
                    # 防止数据太少导致误差
                    if len(mini_batch_indices) != local_minibatch_size:
                        continue

                    # NOTE: 因为是mini-batch, 所有计算很快
                    if isinstance(self.single_action_space, gym.spaces.Box):
                        _, newlogprob, entropy, new_value = self.actor_critic.get_action_and_value(batch_states[mini_batch_indices], batch_actions[mini_batch_indices])
                    else:
                        _, newlogprob, entropy, new_value = self.actor_critic.get_action_and_value(batch_states[mini_batch_indices], batch_actions.long()[mini_batch_indices])
                    logratio = newlogprob - batch_log_probs[mini_batch_indices]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        # `http://joschu.net/blog/kl-approx.html`
                        approx_kl = ((ratio - 1) - logratio).mean()
                        approx_kls += [approx_kl.cpu().item()]
                        clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        clipfracs += [clipfrac]
                    
                    # XXX TODO: 这里减去的是minibatch的平均值, 可以考虑减去整个steps_per_epoch*num_envs的平均值
                    #           或其他自定义长度的平均值
                    # XXX: 这里打乱了顺序，导致非因果，时序非平稳数据，比如金融数据，或应该保存原始序列，减去running_mean更为合适
                    #      TODO: 将来校验
                    
                    # ** normalize mbatch_advantages **
                    mbatch_advantages = batch_advantages[mini_batch_indices]
                    if self.norm_adv:
                        mbatch_advantages = (mbatch_advantages - mbatch_advantages.mean()) / (mbatch_advantages.std() + self.norm_adv_eps)
        
                    # ** compute policy loss **
                    pg_loss1 = - mbatch_advantages * ratio
                    pg_loss2 = - mbatch_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # ** compute value loss **
                    # XXX 此处用huber-loss或可考虑，value-clip未必有依据
                    # 策略相似的地方，价值大部分地方可能也相似，差别过大的, 
                    # 因为v是估计当前策略，非最优策略，不必要求太准 
                    new_value = new_value.view(-1)
                    if self.clip_vloss:
                        # XXX: the code NOT good
                        v_loss_unclipped = (new_value - batch_returns[mini_batch_indices]) ** 2
                        v_clipped = batch_values[mini_batch_indices] + torch.clamp(
                            new_value - batch_values[mini_batch_indices],
                            -self.clip_coef_v,
                            self.clip_coef_v,
                        )
                        # record
                        v_clipfrac = ((v_clipped - batch_returns[mini_batch_indices]).abs() > self.clip_coef_v).float().mean().item()
                        v_clipfracs += [v_clipfrac]
                        
                        v_loss_clipped = (v_clipped - batch_returns[mini_batch_indices]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_value - batch_returns[mini_batch_indices]) ** 2).mean()

                    kl_loss = ((ratio - 1) - logratio).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    if self.kl_coef is not None:
                        loss += kl_loss * self.kl_coef
                        
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if epoch % 20 == 0:
                    self.logger.debug(
                        f'\tepoch:{epoch:2d}:{iloop+1}/{recommended_num_loops} approx_kl:{np.mean(approx_kls):<7.5f} '
                        f"mbs:{local_minibatch_size} frac:{clipfrac:<5.3f} "
                        f"vfrac:{v_clipfrac:<5.2f} loss:{loss.cpu().item():<8.3f} "
                        f"pg:{pg_loss.detach().cpu().item():<8.3f} ent:{entropy_loss.detach().cpu().item():<8.5f} "
                        f" kl:{kl_loss.cpu().item():<7.5f} v:{v_loss.cpu().item():<8.3f}"
                    )
                    # self._debug_test()
                if self.clipfrac_stop is not None and len(clipfracs) > 3 and np.mean(clipfracs[-3:]) >= self.clipfrac_stop:
                    self.logger.debug(f"Early stopping at step {epoch} due to clipping: {clipfrac}")
                    exit_this_train = True
                    break
                
                if self.v_clipfrac_stop is not None and len(v_clipfracs) > 3 and np.mean(v_clipfracs[-3:]) >= self.clipfrac_stop:
                    self.logger.debug(f"Early stopping at step {epoch} due to Value clipping: {v_clipfrac}")
                    exit_this_train = True
                    break
                
                if self.kl_stop is not None and approx_kl > self.kl_stop:
                    self.logger.debug(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl.detach().cpu().item()}")
                    exit_this_train = True
                    break
                
                if self.target_kl is not None and approx_kl > self.target_kl:
                    self.logger.info(f"Early stopping Program at step {epoch} due to reaching max kl: {approx_kl.detach().cpu().item()}")
                    exit_this_train = True
                    should_exit_program = True
                    break
                
        self.restore_lr()
        pred_returns, true_returns = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
        var_true_returns = np.var(true_returns)
        if var_true_returns == 0:
            v_explained_var = np.nan
        else:
            v_explained_var = np.var(true_returns - pred_returns) / var_true_returns

        episode_info = {
            'loss': loss.item(),
            'pg_loss': pg_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'v_loss': v_loss.item(),
            'v_explained_var': v_explained_var,
            'clipfracs': np.mean(clipfracs)
        }
        self.logger.info(f'**{episode_info=}')
        # self._debug_test()

        return should_exit_program, episode_info
        
    def after_learn(self):
        pass
    
    def predict(self, state, deterministic=False):
        # for single env
        if np.isscalar(state):
            state = np.array([state])
        assert state.shape == self.state_dim
        states = torch.FloatTensor(np.array([state])).to(self.device)
        with torch.no_grad():
            actions, action_probs, entropy, values = self.actor_critic.get_action_and_value(
                states, deterministic=deterministic,
                compute_entropy=True
            )
            info = {
                'action_probs': action_probs[0].cpu().tolist(),
                'entropy': entropy[0].cpu().tolist(),
                'values': values[0].cpu().tolist()
            }
        action = actions[0].cpu().numpy()
        return action.item() if np.isscalar(action) else action, info
    
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
    
