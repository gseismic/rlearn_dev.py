# rlearn.dev
A Reinforcement Learning Library [dev]

## Installation
```bash
pip install -e .
```

## Versions
- draft: draft version, used for design algorithm, api-not-stable but working
- naive: naive version, raw implementation with raw or minor optimization, could be used for benchmark
- main: stable version, ready for production, with more docs and tests

## Methods
| state | Agent | version | description | env | demo |   
|:---:|:---:|:---:|:---|:---|:---|
| ✅ | QAC  | naive | Env | Q Actor-Critic | [demo](tests/methods/qac/test_qac_naive.py)
| ✅ | DDPG | naive | VecEnv | DDPG | [demo](tests/methods/ddpg/test_ddpg_naive.py)
| ✅ | PPO  | draft | VecEnv | Proximal Policy Optimization | [demo](tests/methods/ppo/test_ppo_draft.py)
| ✅ | TD3  | naive | VecEnv | Twin Delayed DDPG | [demo](tests/methods/td3/test_td3_naive.py)
| ✅ | SAC  | naive | VecEnv | Soft Actor-Critic | [demo](tests/methods/sac/test_sac_naive.py)
| ✅ | C51  | naive | VecEnv | Categorical DQN | [demo](tests/methods/c51/test_c51_naive.py)

## Usages
```python
import gymnasium as gym
from rlearn_dev.methods.sac.naive.agent import SACAgent as Agent

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                name_prefix=f"sac_{env_id}",
                episode_trigger=lambda episode_idx: episode_idx % 100 == 0,
                video_length=0, # 0 means infinite
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def test_sac_naive_autotune():
    capture_video = True
    # capture_video = False
    run_name = 'test'
    num_envs = 5
    env_id = 'Hopper-v4'
    env = gym.vector.SyncVectorEnv([make_env(env_id, 36, i, capture_video, run_name)
                                     for i in range(num_envs)])
    g_seed = 36
    config = {
        'gamma': 0.99,
        'batch_size': 64,
        'tau': 0.005,
        'actor_lr': 0.0003,
        'critic_lr': 0.0003,
        'buffer_size': 100_000,
        'exploration_noise': 0.1,
        'autotune': True, # True
    }
    agent = Agent(env, config=config, seed=g_seed)
    learn_config = {
        'max_episodes': 30_000//256,
        'max_episode_steps': 256,
        'max_total_steps': 1000_000,
        'verbose_freq': 1,
    }
    agent.learn(**learn_config)
    env.close() 
    
if __name__ == '__main__':
    test_sac_naive_autotune()
```

## TODO
- [ ] add more docs and tests
- [ ] add more papers and references
- [ ] add more envs
- [ ] abstract nnblock
- [ ] make `EnvPlayer` more flexible
- [ ] more unified interfaces
- [ ] SACT, SACT-v2
- [ ] PPG (phasic policy gradient)
- [ ] tabular methods
- [ ] more envs

## Reference
### Common
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [Spinning Up](https://spinningup.openai.com/)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- more: [awesome-rl](https://github.com/aikorea/awesome-rl)
### PPO
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
### DDPG
- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
### TD3
### SAC
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
### C51
- [Categorical DQN](https://arxiv.org/abs/1707.06887)

## ChangeLog
- [@2024-10-02] v0.0.2 c51-naive
- [@2024-11-03] v0.0.7 PPO-draft, DDPG-naive, TD3-naive, SAC-naive, QAC-naive
