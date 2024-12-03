# TODOs
- [ ] 支持done=True时, next_state = None 的情况
- [ ] 检查单变量 OnlineAgentVE 的 next_obs 处理
```python
                next_obs = np.array([
                    np.zeros(self.single_observation_space.shape)
                    if done and obs is None else obs 
                    for obs, done in zip(next_obs, dones)
                ])
                if len(next_obs.shape) == 1:
                    next_obs = next_obs.reshape(-1, 1)
```
