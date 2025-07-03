from abc import abstractmethod
import time
import uuid
from pathlib import Path
from ....utils.recorder import TrajectoryRecorder
from ....utils.exit_monitor import ExitMonitor
from ....utils.i18n import Translator
from .base_agent import BaseAgent

class OnlineAgent(BaseAgent):
    
    def __init__(self, 
                 env=None, 
                 config=None, 
                 seed=None,
                 lang=None,
                 logger=None):
        super().__init__(env, config, seed, lang, logger)

    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass
    
    def before_learn(self, *args, **kwargs):
        pass
    
    def before_episode(self, state, info, **kwargs):
        pass
            
    @abstractmethod
    def select_action(self, state, 
                      *, episode_idx=None, global_step_idx=None, episode_step_idx=None):
        raise NotImplementedError()

    @abstractmethod
    def step(self, state, action, 
             next_state, reward, terminated, truncated, info,
             *, episode_idx=None, global_step_idx=None, episode_step_idx=None):
        raise NotImplementedError()
    
    def after_episode(self, 
                      episode_rewards=None, 
                      episode_total_reward=None,
                      episode_idx=None,
                      **kwargs):
        pass
    
    def after_learn(self, *args, **kwargs):
        pass
    
    @abstractmethod 
    def model_dict(self):
        raise NotImplementedError()
    
    @abstractmethod
    def load_model_dict(self, model_dict):
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, state, deterministic=False):
        raise NotImplementedError()

    def learn(self, 
              max_episodes, 
              max_episode_steps=None, 
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
        Train the agent.
        
        Args:
            max_episodes: Maximum number of episodes to train.
            max_episode_steps: Maximum number of steps per episode.
            max_total_steps: Maximum number of steps to train.
            max_runtime: Maximum training time in seconds.
        """
        if self.env is None:
            raise ValueError("Environment not set. Please call set_env() before learning.")
        exit_monitor = ExitMonitor({
            'max_episodes': max_episodes,
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
        tr = Translator(to_lang=self.lang)
        trajectory_recorder = TrajectoryRecorder()

        self.before_learn()
        global_step_idx = 0
        rewards_history = []
        episode_lengths = []
        start_time = time.time()
        episode_idx = 0
        while True:
            episode_kwargs = {'episode_idx': episode_idx}
            state, info = self.env.reset()
            trajectory_recorder.start_episode(state)
            episode_step_idx = 0
            episode_rewards = []
            episode_total_reward = 0

            self.before_episode(state, info, **episode_kwargs)
            while True:
                step_kwargs = {'episode_step_idx': episode_step_idx, 'global_step_idx': global_step_idx}
                action = self.select_action(state, **episode_kwargs, **step_kwargs)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                trajectory_recorder.record_step(state, action, next_state, reward, terminated, truncated, info)
                episode_total_reward += reward
                episode_rewards.append(reward)
                
                self.step(state, action, 
                          next_state, reward, terminated, truncated, info,
                          **episode_kwargs, **step_kwargs)
                
                episode_step_idx += 1
                global_step_idx += 1
                
                state = next_state
                if (
                    (terminated or truncated)
                    or (max_episode_steps is not None and episode_step_idx >= max_episode_steps)
                ):
                    break
                
            trajectory_recorder.end_episode()
            rewards_history.append(episode_total_reward)
            episode_lengths.append(episode_step_idx)
            reward_kwargs = {
                'episode_rewards': episode_rewards, 
                'episode_total_reward': episode_total_reward
            }
            self.after_episode(**reward_kwargs, **episode_kwargs)
            should_exit, exit_reason = exit_monitor.should_exit(episode_total_reward)

            if exit_monitor.episode_count % verbose_freq == 0:
                self.logger.info(f"Episode {exit_monitor.episode_count}/{max_episodes}: {tr('total_reward')}: {episode_total_reward}")
            
            if should_exit:
                self.logger.info(f"{tr('exit_reason')}: {tr('exit_reason')}")
                break

            if checkpoint_freq and exit_monitor.episode_count % checkpoint_freq == 0:
                checkpoint_file = Path(checkpoint_path) / f'checkpoint_episode_{exit_monitor.episode_count}.pth'
                # in case of user-override save method
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(str(checkpoint_file))
                self.logger.info(tr('checkpoint_saved') + f': {checkpoint_file}')
            
            episode_idx += 1
        self.after_learn()
        if final_model_path:
            final_model_path = Path(final_model_path)
        else:
            final_model_path = Path(checkpoint_path) / f'final_model_{uuid.uuid4().hex[:8]}.pth'
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.save(str(final_model_path))
        self.logger.info(tr('final_model_saved') + f': {final_model_path}')
        
        end_time = time.time()
        training_duration = end_time - start_time

        learning_info = {
            'rewards_history': rewards_history,
            'episode_lengths': episode_lengths,
            'total_episodes': exit_monitor.episode_count,
            'total_steps': global_step_idx,
            'training_duration': training_duration,
            'exit_reason': exit_reason,
            'final_model_path': final_model_path,
            'best_avg_reward': exit_monitor.best_avg_reward,
            'last_avg_reward': sum(rewards_history[-reward_window_size:]) / min(reward_window_size, len(rewards_history))
        }

        learning_info['full_trajectory'] = trajectory_recorder.get_full_trajectory()

        return learning_info
    
