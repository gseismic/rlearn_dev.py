from abc import abstractmethod
import time
import uuid
import numpy as np
from pathlib import Path
from ..base_agent import BaseAgent
from .....utils.i18n import Translator
# from ....utils.exit_monitor.exit_monitor_ve import ExitMonitorVE
from .....utils.exit_monitor.exit_monitor import ExitMonitor

class OnlineAgent(BaseAgent):
    def __init__(self, env=None, config=None, seed=None, lang=None, logger=None):
        super().__init__(env, config, seed, lang, logger)
    
    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass
    
    def before_learn(self, *args, **kwargs):
        pass
    
    def before_episode(self, *args, **kwargs):
        pass
    
    def after_episode(self, episode_idx, episode_reward, *args, **kwargs):
        pass
    
    def after_learn(self, *args, **kwargs):
        pass
    
    def learn(self, 
              max_episodes,
              max_episode_steps,
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
              final_model_dir='final_model'):
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
        if not self.is_vec_env:
            raise ValueError("This agent only supports vectorized environments.")
        
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
        self.single_action_space = self.env.single_action_space
        self.single_observation_space = self.env.single_observation_space
        self.max_episodes = max_episodes
        self.max_episode_steps = max_episode_steps
        
        tr = Translator(to_lang=self.lang)
        
        # here: the only `reset`
        states, infos = self.env.reset()
        if len(states.shape) == 1:
            states = states.reshape(-1, 1)
            
        self.before_learn(states, infos, max_episodes=max_episodes, max_episode_steps=max_episode_steps)
        global_step_idx = 0
        start_time = time.time()
        episode_reward = np.ones(self.num_envs)*float('-inf')
        episode_reward_rolling = np.ones(self.num_envs)*float('-inf')
        for episode_idx in range(max_episodes):
            self.before_episode(episode_idx=episode_idx)
            for episode_step_idx in range(max_episode_steps):
                actions = self.select_action(states, episode_step_idx=episode_step_idx, global_step_idx=global_step_idx)
                (next_states, rewards, terminates, truncates, infos) = self.env.step(actions)
    
                if len(next_states.shape) == 1:
                    next_states = next_states.reshape(-1, 1)
                dones = np.logical_or(terminates, truncates)
                reward_initialized = episode_reward_rolling != float('-inf')
                episode_reward_rolling[reward_initialized] += np.array(rewards[reward_initialized])
                episode_reward_rolling[~reward_initialized] = np.array(rewards[~reward_initialized])
                
                self.step(states, actions, next_states, 
                          rewards, terminates, truncates, infos,
                          global_step_idx=global_step_idx,
                          episode_idx=episode_idx, episode_step_idx=episode_step_idx)
                
                global_step_idx += 1
                # episode_reward[~dones] = np.max(np.stack([episode_reward[~dones], episode_reward_rolling[~dones]]), axis=0)
                episode_reward[dones] = episode_reward_rolling[dones]
                episode_reward_rolling[dones] = float('-inf')
                states = next_states

            ep_rv_info = self.after_episode(episode_idx=episode_idx, episode_reward=episode_reward)
            if ep_rv_info is not None:
                ep_should_exit, episode_info = ep_rv_info
            else:
                ep_should_exit = False
                episode_info = None
            should_exit, exit_reason = exit_monitor.should_exit(np.mean(episode_reward))
            if exit_monitor.episode_count % verbose_freq == 0:  
                self.logger.info(f"Episode {exit_monitor.episode_count}/{max_episodes} [{global_step_idx}]: {tr('average_reward')}: {np.mean(episode_reward):.5f}, detail: {episode_reward}")

            if should_exit:
                self.logger.info(f'{tr('exit_reason')}: {tr(exit_reason)}')
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
            'total_steps': global_step_idx,
            'training_duration': training_duration,
            'exit_reason': exit_reason,
            'final_model_file': final_model_file,
            'best_avg_reward': exit_monitor.best_avg_reward,
        }

        return learning_info
    
    @abstractmethod
    def select_action(self, states, *, episode_step_idx=None, global_step_idx=None, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def step(self, states, actions, next_states, rewards, terminates, truncates, infos,
              *, global_step_idx=None, episode_idx=None, episode_step_idx=None):
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