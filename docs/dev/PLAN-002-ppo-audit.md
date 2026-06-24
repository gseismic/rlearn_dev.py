# PLAN-002 PPO 算法集中审核

## 背景

当前仓库中 PPO 存在 `draft` 与 `naive` 两套实现，二者大量逻辑相似但维护状态不同。旧库迁移后，需要集中审核 PPO 训练流程，确认可直接修复的问题，并记录后续需要设计处理的算法风险。

## 目标

1. 将 `AGENTS.md` 纳入 Git 跟踪。
2. 集中审阅 PPO 相关实现：
   - rollout 数据采样与 done/truncated 处理。
   - GAE 与 return 计算。
   - policy loss、value loss、entropy、KL 与 early stop。
   - 连续动作空间分布、裁剪和 log_prob 一致性。
   - 测试覆盖。
3. 对低风险、确定性的实现错误直接修复并补测试。
4. 对需要更大设计变更的算法问题记录到结果文档，不在本次提交中做侵入式改造。

## 实施范围

本次计划直接修复：

1. `naive` PPO 中 `approx_kl` 与 early stop 判断混用 tensor 和 Python 数值的问题。
2. `draft` 和 `naive` PPO 中 value clip early stop 使用了 `clipfrac_stop` 而非 `v_clipfrac_stop` 的问题。
3. `draft` 和 `naive` PPO 中 explained variance 指标缺少 `1 -` 的问题。
4. 为 shared PPO 指标和 early stop 判断补单元测试。

本次只记录、不直接修复：

1. time-limit truncation 是否应在 GAE 中继续 bootstrap。
2. 连续动作空间裁剪动作与 `log_prob` 计算的一致性。
3. `draft` 与 `naive` 两套 PPO 实现的长期合并策略。

## 验证

1. 运行 `git diff --check`。
2. 运行 PPO 相关快速单元测试。
3. 生成 `PLAN-002-ppo-audit-OUTCOME.md`。
4. 提交并 push，提交信息使用中文并记录计划和结果文档。
