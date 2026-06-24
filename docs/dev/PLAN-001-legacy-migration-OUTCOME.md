# PLAN-001 旧库迁移与审阅隔离结果

## 结果概览

已以当前主仓库 `/Users/mac/pai-studio/rlearn_dev.py` 为主线完成旧库迁移整理。旧库 `/Users/mac/turing/repository/rlearn_dev.py` 未被删除。

当前主仓库已移除指向旧库的未跟踪符号链接 `rlearn_dev.py`，并将旧库待审阅材料复制到本地审阅目录：

`legacy_review/rlearn_dev_legacy_20260624/`

该目录已加入 `.gitignore`，避免大型参考包和训练产物误提交。

## 已迁移到当前代码的内容

1. `rlearn_dev/methods/ppo/draft/agent.py`
   - 将 `approx_kl` 写入 `approx_kls` 前转为 Python 数值。
   - 早停判断和日志输出使用同一个数值，避免 tensor 混入 numpy 均值或日志判断。

2. `rlearn_dev/methods/ppo/naive/network/continous.py`
   - 增加 action space 上下界有限性检查。
   - 当 `high` 或 `low` 包含 `inf` 时禁用 action scaling，避免无穷边界参与缩放。

3. `tests/methods/ppo/test_ppo_continous_network.py`
   - 新增快速单元测试，覆盖有限边界启用 scaling、无穷边界禁用 scaling。

## 已归档到审阅区的旧库材料

1. `patches/`
   - `old-local-tracked-changes.patch`：旧库本地已跟踪文件 diff。
   - `old-head-to-origin-main.patch`：旧库落后远端 2 个提交的差异。

2. `tracked-working-tree/`
   - 旧库本地修改过的 tracked 文件快照。

3. `tracked-head-files/`
   - 旧库本地删除、但旧 HEAD 中仍存在的文件快照。

4. `untracked/`
   - 旧库未跟踪 debug 脚本。

5. `external/packages/cleanrl/`
   - 旧库中的 CleanRL 参考副本，已排除其嵌套 `.git`。

6. `ignored-artifacts/`
   - 旧库中被 ignore 的模型、日志、图片、视频和 checkpoint 等训练产物。

7. `manifests/`
   - 旧库状态、提交信息、审阅目录大小和文件清单。

审阅目录当前大小约 `141M`。

## 验证

已执行：

```bash
git diff --check
python -m pytest tests/methods/ppo/test_ppo_continous_network.py
find . -maxdepth 1 -type l -ls
```

结果：

- `git diff --check` 通过。
- 新增测试 `2 passed`。
- 当前仓库顶层不存在符号链接。

## 删除旧库前确认

删除 `/Users/mac/turing/repository/rlearn_dev.py` 前，建议确认：

1. `legacy_review/rlearn_dev_legacy_20260624/README.md` 中的目录说明满足审阅需要。
2. `legacy_review/rlearn_dev_legacy_20260624/manifests/review-files.txt` 中列出的旧库材料已覆盖需要保留的内容。
3. 当前仓库状态中不再出现 `rlearn_dev.py -> /Users/mac/turing/repository/rlearn_dev.py` 这样的符号链接。
