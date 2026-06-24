# PLAN-003 PPO 教学代码自包含与注释补充

## 背景

PLAN-002 中为了避免重复，将 PPO 的 explained variance 与 early stop 判断抽到了 `rlearn_dev/methods/ppo/_utils.py`。用户明确要求本库偏教学用途，`naive`、`draft` 等版本应独立自包含，不应通过共享工具隐藏关键算法细节。

## 目标

1. 移除 PPO 共享 `_utils.py`。
2. 将上一轮修复的 explained variance 与 recent clip fraction 判断分别内联到 `naive` 与 `draft` 版本。
3. 在代码中补充中文教学注释，说明修改点和算法含义。
4. 更新测试，使测试直接覆盖两个版本各自的 helper，而不是覆盖共享工具。
5. 生成结果文档并提交推送。

## 实施范围

1. `rlearn_dev/methods/ppo/naive/ppo_agent.py`
   - 移除 `.._utils` 依赖。
   - 添加本文件内的 explained variance 与 recent clip fraction helper。
   - 在 KL 数值化、value clip early stop 和 explained variance 处补注释。
2. `rlearn_dev/methods/ppo/draft/agent.py`
   - 同步做自包含改造和注释补充。
3. `tests/methods/ppo/`
   - 删除依赖共享工具的测试。
   - 新增覆盖两个 PPO 版本本地 helper 的测试。

## 验证

1. `git diff --check`
2. `python -m pytest tests/methods/ppo/test_ppo_agent_helpers.py tests/methods/ppo/test_ppo_continous_network.py tests/methods/ppo/test_vec_env_gym.py`
