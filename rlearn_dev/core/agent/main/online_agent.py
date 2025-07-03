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
                 logger=None, 
                 seed=None):
        super().__init__(env, config, logger, seed)

    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass
    
    def before_learn(self, *args, **kwargs):
        pass
    
    def before_episode(self, *args, **kwargs):
        pass
    
    def after_episode(self, i_episode, episode_rewards, *args, **kwargs):
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
    def select_action(self, state, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def step(self, state, action, reward, next_state, done, 
             episode_steps, total_steps, *args, **kwargs):
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
        total_steps = 0
        rewards_history = []
        episode_lengths = []
        start_time = time.time()
        while True:
            state, _ = self.env.reset()
            trajectory_recorder.start_episode(state)
            episode_rewards = []
            episode_reward = 0
            episode_steps = 0

            self.before_episode()
            while True:
                action = self.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                trajectory_recorder.record_step(state, action, next_state, reward, done, truncated, info)
                episode_reward += reward
                episode_rewards.append(reward)
                episode_steps += 1
                total_steps += 1
                
                self.step(state, action, reward, next_state, done,
                          episode_steps, total_steps)

                state = next_state

                if done or truncated or (max_episode_steps and episode_steps >= max_episode_steps):
                    break
                
            trajectory_recorder.end_episode()
            rewards_history.append(episode_reward)
            episode_lengths.append(episode_steps)
            self.after_episode(exit_monitor.episode_count, episode_rewards)
            should_exit, exit_reason = exit_monitor.should_exit(episode_reward)

            if exit_monitor.episode_count % verbose_freq == 0:
                self.logger.info(f"Episode {exit_monitor.episode_count}/{max_episodes}: {tr('total_reward')}: {episode_reward}")
            
            if should_exit:
                self.logger.info(f"{tr('exit_reason')}: {tr(exit_reason)}")
                break

            if checkpoint_freq and exit_monitor.episode_count % checkpoint_freq == 0:
                checkpoint_file = Path(checkpoint_path) / f'checkpoint_episode_{exit_monitor.episode_count}.pth'
                # in case of user-override save method
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(str(checkpoint_file))
                self.logger.info(tr('checkpoint_saved') + f': {checkpoint_file}')

        self.after_learn()
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
            'rewards_history': rewards_history,
            'episode_lengths': episode_lengths,
            'total_episodes': exit_monitor.episode_count,
            'total_steps': total_steps,
            'training_duration': training_duration,
            'exit_reason': exit_reason,
            'final_model_path': final_model_path,
            'best_avg_reward': exit_monitor.best_avg_reward,
            'last_avg_reward': sum(rewards_history[-reward_window_size:]) / min(reward_window_size, len(rewards_history))
        }

        learning_info['full_trajectory'] = trajectory_recorder.get_full_trajectory()

        return learning_info
    
