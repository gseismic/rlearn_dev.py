# rlearn.dev
A Reinforcement Learning Library [dev]

In the future, this repo will move to [rlearn](http://github.com/gseismic/rlearn.py)
创建`rlearn_dev`的原因是: 最初开发rlearn.py时使用了自定义的非gym环境，包袱过重

## Installation
```bash
pip install -e .
```

## Versions
- draft: Draft version, used for designing algorithms, API-not-stable but working
- naive: Naive version, raw implementation with raw or minor optimizations, could be used for benchmarking
- main: Stable version, ready for production, with more docs and tests

## Features

## showcase
### MuJoco-ant-v5
使用`rlearn_dev`的naive版本SAC训练mujoco Ant-v5，含主动添加的噪音下，训练得分6000+
trained by naive.SAC, with deterministic=False, score=6000+

![ant-sac](./docs/ant_sac.gif)

对比第三方知名库[ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)
benchmark: [用GPU并行环境Isaac Gym 训练机器人Ant，3小时6000分，最高12000分（代码开源](https://zhuanlan.zhihu.com/p/508378146)

训练log:
```
2024-11-23 07:45:24 | INFO   | Episode 2019/5000 [4134912]: Average Reward: 5611.59677, detail: [6339.41974365 6330.48039567 6412.41894184 6317.87476665 6403.85789968
   99.67731813 6430.00267087 6559.04241766]
2024-11-23 07:45:41 | INFO   | Episode 2020/5000 [4136960]: Average Reward: 6320.81176, detail: [6131.89531841 6141.40146106 6371.71031779 6146.98573506 6462.2939436
 6572.73806615 6467.93920164 6271.53004489]
2024-11-23 07:45:59 | INFO   | Episode 2021/5000 [4139008]: Average Reward: 6108.65877, detail: [6366.46026193 6499.5197772  3587.1468893  6650.29738074 6514.41845971
 6469.3829467  6313.48620102 6468.5582827 ]
2024-11-23 07:46:17 | INFO   | Episode 2022/5000 [4141056]: Average Reward: 6082.90059, detail: [3445.91923514 6416.88338573 6495.99878767 6572.26091719 6577.24471379
 6474.37533201 6336.68020873 6343.8421299 ]
2024-11-23 07:46:35 | INFO   | Episode 2023/5000 [4143104]: Average Reward: 6054.10073, detail: [6386.18024652 6042.66135112 6577.68305068 6432.44589925 6414.15561133
 6458.69392698 3451.87516831 6669.11062416]
2024-11-23 07:46:53 | INFO   | Episode 2024/5000 [4145152]: Average Reward: 6399.13154, detail: [6017.71464908 6581.0859381  6549.04273248 6514.8605336  6270.06884443
 6471.11665318 6546.70169277 6242.46127681]
```

## Methods
| state | Agent | version | env | description | demo |   
|:---:|:---:|:---:|:---|:---|:---|
| ✅ | C51  | naive | VecEnv | Categorical DQN | [demo](tests/methods/c51/test_c51_naive.py)
| ✅ | QAC  | naive | Env | Q Actor-Critic | [demo](tests/methods/qac/test_qac_naive.py)
| ✅ | DDPG | naive | VecEnv | DDPG | [demo](tests/methods/ddpg/test_ddpg_naive.py)
| ✅ | TD3  | naive | VecEnv | Twin Delayed DDPG | [demo](tests/methods/td3/test_td3_naive.py)
| ✅ | SAC  | naive | VecEnv | Soft Actor-Critic | [demo](tests/methods/sac/test_sac_naive.py)
| ✅ | MCPG | basic |   Env  | Monte-Carlo REINFORCE | [demo](tests/methods/mcpg/test_mcpg_basic.py)
| ✅ | PPO  | naive | VecEnv | Proximal Policy Optimization | [demo](tests/methods/ppo/test_ppo_draft.py)

## Usages
```python
import gymnasium as gym
from rlearn_dev.methods.sac.naive.agent import SACAgent as Agent

def make_env(env_id, seed, idx, capture_video, run_name):
    def _make_env():
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

    return _make_env

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
### Methods
- [ ] SACT, SACT-v2
- [ ] PPG (phasic policy gradient)
- [ ] PPO-RND
- [ ] tabular methods

### Common
- [ ] add more docs and tests
- [ ] add more papers and references
- [ ] abstract nnblock
- [ ] make `EnvPlayer` more flexible
- [ ] more unified interfaces
- [ ] self-contained gymnasium-like(v1) envs
- [ ] read source code [sb3-off-policy](https://github.com/DLR-RM/stable-baselines3/blob/06498e8be71b9c8aee38226176dbd28443afbb4f/stable_baselines3/common/off_policy_algorithm.py#L439)
- [ ] process TimeLimit & modify ReplayBuffer [gym-time-limit](https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py#L19)
- [ ] support multi-agent
- [ ] support gymnasium-v1
- [ ] support distributed training
- [ ] parallel DataCollector

## 可能遇到的问题 Frequently Asked Questions
### 无法录制视频 Unable to record videos
最新版本的gymnasium依赖的moviepy有一些问题  Latest version of gymnasium depends on moviepy with some issues
```bash
1. pip install --upgrade decorator==4.0.2
2. pip uninstall moviepy decorator
3. pip install moviepy
```

## Reference
### Common
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [Spinning Up](https://spinningup.openai.com/)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [@zh 强化学习导论](https://hrl.boyuai.com/chapter/3/%E6%A8%A1%E4%BB%BF%E5%AD%A6%E4%B9%A0)
- more: [awesome-rl](https://github.com/aikorea/awesome-rl)
### PPO
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
### DDPG
- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
### TD3
### SAC
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
### C51
- [Categorical DQN](https://arxiv.org/abs/1707.06887)

## Tutorials for Reinforcement Learning
- [@en policy-gradient](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- [@zh LiuJianping](https://www.cnblogs.com/pinard)
- [@en coach](https://intellabs.github.io/coach/components/agents/index.html)
- [@en Deep Reinforcement Learning: An Overview](https://arxiv.org/pdf/1810.06339.pdf)
- [@en google-scholar on RL](https://scholar.google.com/scholar?q=reinforcement+learning)
- [@en stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- [@en Practical_RL](https://github.com/yandexdataschool/Practical_RL)