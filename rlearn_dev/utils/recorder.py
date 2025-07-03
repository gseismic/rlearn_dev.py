from dataclasses import dataclass, field
from typing import List, Any, Dict

@dataclass
class Step:
    state: Any
    action: Any
    next_state: Any
    reward: float
    done: bool
    truncated: bool
    info: Dict

@dataclass
class Episode:
    initial_state: Any
    steps: List[Step] = field(default_factory=list)

class TrajectoryRecorder:
    def __init__(self):
        self.episodes: List[Episode] = []
        self.current_episode: Episode = None

    def start_episode(self, initial_state):
        self.current_episode = Episode(initial_state)

    def record_step(self, state, action, next_state, reward, done, truncated, info):
        step = Step(state, action, next_state, reward, done, truncated, info)
        self.current_episode.steps.append(step)

    def end_episode(self):
        self.episodes.append(self.current_episode)
        self.current_episode = None

    def get_full_trajectory(self):
        return self.episodes