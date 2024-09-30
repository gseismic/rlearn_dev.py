from typing import Any, Tuple, Dict

class EnvPlayer:
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.current_episode = 0
        self.current_step = 0
        self.observation_space = None  # 这需要从原始环境中获取
        self.action_space = None  # 这需要从原始环境中获取

    def reset(self) -> Tuple[Any, Dict]:
        if self.current_episode >= len(self.trajectory):
            raise StopIteration("No more episodes")
        self.current_step = 0
        initial_state = self.trajectory[self.current_episode].initial_state
        return initial_state, {}

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        if self.current_step >= len(self.trajectory[self.current_episode].steps):
            raise StopIteration("Episode finished")
        step = self.trajectory[self.current_episode].steps[self.current_step]
        self.current_step += 1
        return step.next_state, step.reward, step.done, step.truncated, step.info

    def close(self):
        pass