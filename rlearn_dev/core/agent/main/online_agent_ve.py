from abc import abstractmethod
import time
import uuid
import numpy as np
from pathlib import Path
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
              checkpoint_path='checkpoints',
              final_model_path=None):
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
        self.before_learn(states, infos, max_epochs=max_epochs, steps_per_epoch=steps_per_epoch)
        total_steps = 0
        # rewards_history = []
        start_time = time.time()
        for epoch in range(max_epochs):
            self.before_episode(epoch=epoch)
            # episode_rewards = [] # (steps_per_epoch, num_envs)
            episode_reward = np.zeros(self.num_envs)
            for epoch_step in range(steps_per_epoch):
                actions = self.select_action(states, epoch_step=epoch_step)
                (next_obs, rewards, terminates, truncates, infos) = self.env.step(actions)
                episode_reward += np.array(rewards)
                # episode_rewards.append(rewards)
                total_steps += 1
                
                self.step(next_obs, rewards, terminates, truncates, infos,
                          epoch=epoch, epoch_step=epoch_step)
                
                states = next_obs

            # rewards_history.append(episode_reward)
            ep_should_exit, episode_info = self.after_episode(epoch=epoch, episode_reward=episode_reward)
            should_exit, exit_reason = exit_monitor.should_exit(np.mean(episode_reward))
            if exit_monitor.episode_count % verbose_freq == 0:  
                self.logger.info(f"Episode {exit_monitor.episode_count}/{max_epochs}: {tr('average_reward')}: {np.mean(episode_reward)}")

            if should_exit:
                self.logger.info(f'{tr('exit_reason')}: {tr(exit_reason)}')
                break
            
            if ep_should_exit:
                self.logger.info(f'Early stopping: {str(episode_info)}')
                break
            
            if checkpoint_freq and exit_monitor.episode_count % checkpoint_freq == 0:
                checkpoint_file = Path(checkpoint_path) / f'checkpoint_episode_{exit_monitor.episode_count}.pth'
                # in case of user-override save method
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(str(checkpoint_file))
                self.logger.info(tr('checkpoint_saved') + f': {checkpoint_file}')
                
        self.after_learn()
        
        end_time = time.time()
        training_duration = end_time - start_time
        # 保存最终模型
        if final_model_path:
            final_model_path = Path(final_model_path)
        else:
            # 如果没有指定路径，则在 checkpoints 目录下生成一个唯一的文件名
            final_model_path = Path(checkpoint_path) / f'final_model_{uuid.uuid4().hex[:8]}.pth'
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.save(str(final_model_path))
        self.logger.info(tr('final_model_saved') + f': {final_model_path}')
        
        end_time = time.time()
        training_duration = end_time - start_time

        # 准备返回的训练信息
        learning_info = {
            'total_episode': exit_monitor.episode_count,
            'total_steps': total_steps,
            'training_duration': training_duration,
            'exit_reason': exit_reason,
            'final_model_path': final_model_path,
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