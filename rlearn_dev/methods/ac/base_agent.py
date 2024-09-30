from abc import ABC, abstractmethod
import time
import uuid
import torch
from pathlib import Path
from loguru import logger as default_logger
from ...utils.recorder import TrajectoryRecorder
from ...utils.exit_monitor import ExitMonitor
from ...utils.i18n import Translator

class BaseAgent(ABC):
    def __init__(self, 
                 env, 
                 config, 
                 logger=None, 
                 seed=None):
        self.env = env
        self.config = config or {}
        self.logger = logger or default_logger
        self.lang = self.config.get('lang', 'en')
        self.seed_all(seed)
        self.initialize()

    def seed_all(self, seed):
        self._seed = seed
        self.env.reset(seed=seed)
        if seed is None:
            torch.seed()
        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if torch.cuda.is_available() and seed is not None:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError()

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path):
        raise NotImplementedError()

    @abstractmethod
    def load(self, path):
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
              checkpoint_freq=100,
              checkpoint_path='checkpoints',
              final_model_path=None):
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
        rewards_history = []
        episode_lengths = []
        total_steps = 0
        start_time = time.time()

        trajectory_recorder = TrajectoryRecorder()
        
        tr = Translator(to_lang=self.lang)

        while True:
            state, _ = self.env.reset()
            trajectory_recorder.start_episode(state)
            episode_reward = 0
            episode_steps = 0

            while True:
                action = self.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                trajectory_recorder.record_step(state, action, next_state, reward, done, truncated, info)
                episode_reward += reward

                self.step(state, action, reward, next_state, done)

                state = next_state
                episode_steps += 1
                total_steps += 1

                if done or truncated or (max_episode_steps and episode_steps >= max_episode_steps):
                    break

            trajectory_recorder.end_episode()
            rewards_history.append(episode_reward)
            episode_lengths.append(episode_steps)
            should_exit, exit_reason = exit_monitor.should_exit(episode_reward)
            
            if should_exit:
                self.logger.info(tr(exit_reason))
                break

            if exit_monitor.episode_count % verbose_freq == 0:
                self.logger.info(f"Episode {exit_monitor.episode_count}: {tr('total_reward')}: {episode_reward}")

            if checkpoint_freq and exit_monitor.episode_count % checkpoint_freq == 0:
                checkpoint_file = Path(checkpoint_path) / f'checkpoint_episode_{exit_monitor.episode_count}.pth'
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                self.save(str(checkpoint_file))
                self.logger.info(tr('checkpoint_saved') + f': {checkpoint_file}')

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
        training_info = {
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

        training_info['full_trajectory'] = trajectory_recorder.get_full_trajectory()

        return training_info