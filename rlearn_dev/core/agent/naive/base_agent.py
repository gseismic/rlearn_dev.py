from abc import (
    ABC,
    abstractmethod
)
import torch
import random
import numpy as np
from pathlib import Path
from cfgdict import make_config #, Schema
from gymnasium.vector import VectorEnv
from ....logger import user_logger
from ....utils.cuda import get_device
from ...env_player import BaseVecEnvPlayer

def is_vec_env(env):
    return isinstance(env, (VectorEnv, BaseVecEnvPlayer))

class BaseAgent(ABC):
    schema = None
    default_lang = 'en'
    
    def __init__(self,
                 env=None,
                 config=None,
                 seed=None,
                 lang=None,
                 logger=None):
        self.logger = logger or user_logger
        self.config = self.make_config(config)
        self.seed = seed
        self.lang = lang or self.default_lang
        self.is_vec_env = False
        self.state_space = None
        self.action_space = None
        self.device = get_device(self.config.get('device'))
        self.set_env(env)
    
    def make_config(self, config):
        return make_config(config, schema=self.schema, logger=self.logger)
    
    def set_env(self, env):
        self.env = env
        self.seed_all(self.seed)
        if self.env is not None:
            self.is_vec_env = is_vec_env(self.env)
            if self.is_vec_env:
                self.state_space = self.env.single_observation_space
                self.action_space = self.env.single_action_space
            else:
                self.state_space = self.env.observation_space
                self.action_space = self.env.action_space
            self._num_envs = self.env.num_envs if self.is_vec_env else 1
            self.initialize()
    
    def seed_all(self, seed):
        if self.env is not None:
            self.env.reset(seed=seed)
        if seed is None:
            torch.seed()
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    
    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def learn(self, *args, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def select_action(self, state, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def step(self, *args, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, state, deterministic=False):
        raise NotImplementedError()
    
    def before_learn(self):
        pass
    
    def before_episode(self):
        pass
    
    def after_episode(self, *args, **kwargs):
        pass
    
    def after_learn(self):
        pass
    
    @abstractmethod 
    def model_dict(self):
        raise NotImplementedError()
    
    @abstractmethod
    def load_model_dict(self, model_dict):
        raise NotImplementedError()  
    
    def checkpoint_dict(self):
        return self.model_dict()
    
    def load_checkpoint_dict(self, checkpoint_dict):
        return self.load_model_dict(checkpoint_dict)
    
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_dict(), str(path))
    
    @classmethod
    def load(cls, path, env):
        model_dict = torch.load(path)
        agent = cls(env, config=model_dict['config'])
        agent.load_model_dict(model_dict)
        return agent
    
    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_dict(), str(path))
    
    @classmethod
    def load_checkpoint(cls, path, env):
        checkpoint_dict = torch.load(path)
        agent = cls(env, config=checkpoint_dict['config'])
        agent.load_checkpoint_dict(checkpoint_dict)
        return agent
    
    @property
    def num_envs(self):
        return self._num_envs