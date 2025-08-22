import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# 1. 创建环境
env = gym.make("Pendulum-v1")

# 2. 初始化PPO模型
model = PPO(
    "MlpPolicy",            # 使用多层感知机策略
    env,
    verbose=1,              # 打印训练日志
    learning_rate=3e-4,     # 学习率
    batch_size=64,          # 批大小
    n_steps=2048,           # 每次更新收集的步数
    gamma=0.99,             # 折扣因子
    gae_lambda=0.95,        # GAE参数
    ent_coef=0.01,          # 熵系数（鼓励探索）
    device="auto"           # 自动选择CPU/GPU
)

# 3. 添加评估回调（每5000步评估一次）
eval_callback = EvalCallback(
    env,
    eval_freq=5000,
    best_model_save_path="./logs/",
    log_path="./logs/",
    deterministic=True
)

# 4. 训练模型（10万步）
model.learn(
    total_timesteps=100_000,
    callback=eval_callback
)

# 5. 保存模型
model.save("ppo_pendulum")

# 6. 评估模型（可选）
mean_reward, _ = evaluate_policy(
    model, env, n_eval_episodes=10, deterministic=True
)
print(f"平均奖励: {mean_reward:.2f}")

# 7. 可视化演示
obs, _ = env.reset()
for _ in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _, _ = env.step(action)
    env.render()

env.close()