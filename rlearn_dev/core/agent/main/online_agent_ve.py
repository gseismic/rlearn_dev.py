from abc import abstractmethod
import time
import uuid
import numpy as np
from pathlib import Path
from collections import deque
from .base_agent import BaseAgent
from ....utils.i18n import Translator
# from ....utils.exit_monitor.exit_monitor_ve import ExitMonitorVE
from ....utils.exit_monitor.exit_monitor import ExitMonitor

class OnlineAgentVE(BaseAgent):
    def __init__(self, env=None, config=None, logger=None, seed=None):
        super().__init__(env, config, logger, seed)
    
    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass
    
    def before_learn(self, *args, **kwargs):
        pass
    
    def before_episode(self, *args, **kwargs):
        pass
    
    def after_episode(self, epoch, episode_reward, *args, **kwargs):
        pass
    
    def after_learn(self, *args, **kwargs):
        pass
    
    def learn(self, 
              max_epochs,
              steps_per_epoch, 
              max_total_steps=None, 
              max_runtime=None,
              reward_threshold=None,
              reward_window_size=100,
              min_reward_threshold=None,
              max_reward_threshold=None,
              reward_check_freq=10,
              verbose_freq=10,
              no_improvement_threshold=50,
              improvement_threshold=None,
              improvement_ratio_threshold=None,
              checkpoint_freq=None,
              checkpoint_dir='checkpoints',
              final_model_name=None,
              final_model_dir='final_models'):
        """
        avg_reward := avg reward of all environments in recent reward_window_size episodes
        exit if any:
            (1) avg_reward >= reward_threshold and min_reward_threshold <= avg_reward
            (2) avg_reward >= max_reward_threshold
            (3) avg_reward >= best_avg_reward + improvement_threshold
            (4) avg_reward > best_avg_reward * (1 + improvement_ratio_threshold)
        """
        if self.env is None:
            raise ValueError("Environment not set. Please call set_env() before learning.")
        exit_monitor = ExitMonitor({
            'max_total_steps': max_total_steps,
            'max_runtime': max_runtime,
            'reward_threshold': reward_threshold,
            'reward_window_size': reward_window_size,
            'min_reward_threshold': min_reward_threshold,
            'max_reward_threshold': max_reward_threshold,
            'reward_check_freq': reward_check_freq,
            'no_improvement_threshold': no_improvement_threshold,
            'improvement_threshold': improvement_threshold,
            'improvement_ratio_threshold': improvement_ratio_threshold
        })
        self.num_envs = self.env.num_envs
        self.single_action_space = self.env.single_action_space
        self.single_observation_space = self.env.single_observation_space
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        
        tr = Translator(to_lang=self.lang)
        
        # here: the only `reset`
        states, infos = self.env.reset()
        if len(states.shape) == 1:
            states = states.reshape(-1, 1)
        # print(states)
        self.before_learn(states, infos, max_epochs=max_epochs, steps_per_epoch=steps_per_epoch)
        total_steps = 0
        start_time = time.time()
        # vec_episode_dones = np.zeros(self.num_envs, dtype=bool)

        vec_episode_rewards = np.zeros(self.num_envs)
        vec_episode_lengths = np.zeros(self.num_envs)
        # 因为是参数共享，所以直接对最近的reward求平均
        rolling_episode_rewards = deque(maxlen=reward_window_size)
        rolling_episode_lengths = deque(maxlen=reward_window_size)
        latest_episode_reward = np.nan
        latest_episode_length = np.nan
        total_episode_rewards = 0
        total_episode_lengths = 0
        for epoch in range(max_epochs):
            self.before_episode(epoch=epoch)
            # 不能在此reset，因为 steps_per_epoch不是真正的结束
            # vec_episode_rewards.fill(0) # reset
            # vec_episode_lengths.fill(0) # reset
            # avg_episode_perstep_reward = np.nan
            # avg_episode_reward = np.nan
            for epoch_step in range(steps_per_epoch):
                actions = self.select_action(states, epoch_step=epoch_step)
                (next_obs, rewards, terminates, truncates, infos) = self.env.step(actions)
                dones = np.logical_or(terminates, truncates)
                
                # 只有episode结束，且next_obs为None时，才用全零状态代替
                next_obs = np.array([
                    np.zeros(self.single_observation_space.shape)
                    if done and obs is None else obs 
                    for obs, done in zip(next_obs, dones)
                ])
                if len(next_obs.shape) == 1:
                    next_obs = next_obs.reshape(-1, 1)
                
                vec_episode_rewards += rewards
                vec_episode_lengths += 1
                
                # vec_episode_dones[~dones] = False
                # vec_episode_dones[dones] = True
                if any(dones):
                    # 因为参数共享，所以直接对最近的reward求平均
                    # rolling_episode_rewards.append(np.mean(vec_episode_rewards[dones]))
                    # rolling_episode_lengths.append(np.mean(vec_episode_lengths[dones]))
                    rolling_episode_rewards.extend(vec_episode_rewards[dones])
                    rolling_episode_lengths.extend(vec_episode_lengths[dones])
                    latest_episode_reward = rolling_episode_rewards[-1]
                    latest_episode_length = rolling_episode_lengths[-1]
                    total_episode_rewards += np.sum(vec_episode_rewards[dones])
                    total_episode_lengths += np.sum(vec_episode_lengths[dones])
                # avg_episode_perstep_reward = np.nanmean(vec_episode_rewards[dones] / vec_episode_lengths[dones])
                # avg_episode_reward = np.nanmean(rolling_episode_rewards)
                vec_episode_rewards[dones] = 0 # reset
                vec_episode_lengths[dones] = 0 # reset
                total_steps += 1
                
                self.step(next_obs, rewards, terminates, truncates, infos,
                          epoch=epoch, epoch_step=epoch_step)
                
                states = next_obs

            ep_should_exit, episode_info = self.after_episode(epoch=epoch, episode_reward=latest_episode_reward, episode_length=latest_episode_length)
            should_exit, exit_reason = exit_monitor.should_exit(latest_episode_reward, latest_episode_length)
            if exit_monitor.episode_count % verbose_freq == 0:
                assert len(rolling_episode_rewards) == len(rolling_episode_lengths)
                if len(rolling_episode_lengths) > 0:
                    avg_episode_reward = np.nanmean(rolling_episode_rewards)
                    avg_episode_length = np.nanmean(rolling_episode_lengths)
                    avg_episode_perstep_reward = np.sum(rolling_episode_rewards) / np.sum(rolling_episode_lengths)
                else:
                    avg_episode_reward = np.nan
                    avg_episode_length = np.nan
                    avg_episode_perstep_reward = np.nan
                self.logger.info(
                    f"Episode {exit_monitor.episode_count}/{max_epochs}: "
                    f"{tr('average_episode_reward')}: {avg_episode_reward}, "
                    f"{tr('average_episode_length')}: {avg_episode_length}, "
                    f"{tr('average_perstep_reward')}: {avg_episode_perstep_reward}"
                )

            if should_exit:
                self.logger.info(f"{tr('exit_reason')}: {tr(exit_reason)}")
                break
            
            if ep_should_exit:
                self.logger.info(f'Early stopping: {str(episode_info)}')
                break
            
            if checkpoint_freq and exit_monitor.episode_count % checkpoint_freq == 0:
                checkpoint_dir = checkpoint_dir or 'checkpoints'
                checkpoint_file = Path(checkpoint_dir) / f'checkpoint_episode_{exit_monitor.episode_count}.pth'
                # in case of user overridding save-method
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(str(checkpoint_file))
                self.logger.info(tr('checkpoint_saved') + f': {checkpoint_file}')
                
        self.after_learn()
        
        end_time = time.time()
        training_duration = end_time - start_time
        if not final_model_dir:
            final_model_dir = 'final_model'
        if not final_model_name:
            final_model_name = f'final_model_{uuid.uuid4().hex[:8]}.pth'
        
        final_model_file = Path(final_model_dir) / final_model_name
        final_model_file.parent.mkdir(parents=True, exist_ok=True)
        self.save(str(final_model_file))
        self.logger.info(tr('final_model_saved') + f': {final_model_file}')
        
        end_time = time.time()
        training_duration = end_time - start_time

        learning_info = {
            'total_episode': exit_monitor.episode_count,
            'total_steps': total_steps,
            'training_duration': training_duration,
            'exit_reason': exit_reason,
            'final_model_file': final_model_file,
            'best_avg_reward': exit_monitor.best_avg_reward,
        }

        return learning_info
    
    @abstractmethod
    def select_action(self, states, epoch_step, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def step(self, next_obs, rewards, terminates, truncates, info,
              epoch, epoch_step):
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, state, deterministic=False):
        raise NotImplementedError()
    
    @abstractmethod 
    def model_dict(self):
        raise NotImplementedError()
    
    @abstractmethod
    def load_model_dict(self, model_dict):
        raise NotImplementedError()  
