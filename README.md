# rlearn.dev
A Reinforcement Learning Library [dev]

## Methods
| state | Agent | version | description | env | demo |   
|:---:|:---:|:---:|:---|:---|:---|
| ✅ | QAC  | naive | Env | Q Actor-Critic | [demo](tests/methods/qac/test_qac_naive.py)
| ✅ | DDPG | naive | VecEnv | DDPG | [demo](tests/methods/ddpg/test_ddpg_naive.py)
| ✅ | PPO  | draft | VecEnv | Proximal Policy Optimization | [demo](tests/methods/ppo/test_ppo_draft.py)
| ✅ | TD3  | naive | VecEnv | Twin Delayed DDPG | [demo](tests/methods/td3/test_td3_naive.py)
| ✅ | SAC  | naive | VecEnv | Soft Actor-Critic | [demo](tests/methods/sac/test_sac_naive.py)
| ✅ | C51  | naive | VecEnv | Categorical DQN | [demo](tests/methods/c51/test_c51_naive.py)

## TODO
- [ ] add more docs
- [ ] SACT, SACT-v2
- [ ] PPG (phasic policy gradient)
- [ ] tabular methods

## ChangeLog
- [@2024-10-02] v0.0.2 c51-naive
- [@2024-11-03] v0.0.7 PPO-draft, DDPG-naive, TD3-naive, SAC-naive, QAC-naive
