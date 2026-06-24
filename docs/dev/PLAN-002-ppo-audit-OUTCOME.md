# PLAN-002 PPO 算法集中审核结果

## 已处理问题

1. `naive` PPO 的 KL early stop 混用 tensor 和 Python 数值
   - 问题：`approx_kls` 中记录的是数值，但 `kl_stop` 和 `target_kl` 判断仍使用 tensor。
   - 处理：统一使用 `approx_kl_value` 做日志、均值和 early stop 判断。

2. `draft` 与 `naive` 的 value clip early stop 阈值引用错误
   - 问题：`v_clipfrac_stop` 分支实际比较的是 `clipfrac_stop`，会导致 value clip 阈值配置无效。
   - 处理：新增 `recent_mean_reaches_threshold`，两个 agent 同时使用正确阈值。

3. value explained variance 指标公式错误
   - 问题：原实现返回 `Var(y - y_hat) / Var(y)`，标准 explained variance 应为 `1 - Var(y - y_hat) / Var(y)`。
   - 处理：新增 `explained_variance` 工具函数，并在 `draft`、`naive` 中复用。

4. `AGENTS.md` 纳入跟踪
   - 按用户要求将仓库协作规范文件纳入 Git 提交范围。

## 新增测试

1. `tests/methods/ppo/test_ppo_utils.py`
   - 覆盖 explained variance 的完美预测和常量 target 场景。
   - 覆盖 recent clip fraction 阈值判断。

## 仍需后续设计的问题

1. time-limit truncation 的 GAE bootstrap 语义
   - 当前 `step()` 中使用 `terminates OR truncates` 作为 `done`。
   - 对自然终止是合理的；对时间限制截断，PPO 通常应基于 final observation 或下一状态继续 bootstrap。
   - 这需要统一处理 Gymnasium vector env 的 `final_observation`/自动 reset 语义，建议单独制定设计。

2. 连续动作空间的 action clipping 与 log_prob 一致性
   - `naive` 连续网络会对动作进行裁剪后再返回 log_prob。
   - 若采样动作越界，裁剪后的动作并不是原 Normal 分布采样值，PPO ratio 会近似但不严格。
   - 后续应在以下方案中选择一种：保留未裁剪动作并由环境 wrapper 裁剪、使用 tanh-squashed policy 并加入 log_prob correction、或实现截断分布。

3. `draft` 与 `naive` 两套 PPO 实现长期分叉
   - 两套 agent 共享大量训练逻辑，但连续动作网络、单环境支持和 debug 行为不同。
   - 建议后续确认一个主实现，另一个转为兼容层或删除。

4. PPO 测试分层不足
   - 当前多个 `tests/methods/ppo/test_ppo_*.py` 是耗时训练示例，不适合作为默认快速单元测试。
   - 建议后续拆分 `unit`、`smoke`、`slow/integration`，并给 MuJoCo 依赖测试加 skip 条件。

## 验证结果

已执行：

```bash
git diff --check
python -m pytest tests/methods/ppo/test_ppo_utils.py tests/methods/ppo/test_ppo_continous_network.py tests/methods/ppo/test_vec_env_gym.py
```

结果：

- `git diff --check` 通过。
- PPO 快速测试 `7 passed`。

未执行：

- 长耗时 PPO 训练示例。
- 依赖 MuJoCo 的连续控制训练示例。
