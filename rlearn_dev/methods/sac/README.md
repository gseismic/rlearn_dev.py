# SAC

## Reference
- [SAC](https://intellabs.github.io/coach/components/agents/policy_optimization/sac.html)

## Note
- 对于回归问题，目标熵不应过小，gamma过小, 策略震荡原因？
- 可以考虑critic先更新e.g.100步，actor才更新，之后critic 和 actor交替更新，actor更新慢于critic
- 
## Blog
- [SAC](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- [SAC(Soft Actor-Critic)阅读笔记](https://zhuanlan.zhihu.com/p/85003758)
- [SAC@SpinningUp](https://spinningup.openai.com/en/latest/algorithms/sac.html)
