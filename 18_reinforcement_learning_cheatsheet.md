# Ch18 速查表：Reinforcement Learning

> **核心主旨**：RL 的核心循環：Agent 觀察狀態 → 選擇動作 → 環境返回獎勵 → 更新策略。DQN 是深度 RL 的基礎。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Environment (`gym`) | 提供狀態、接收動作、返回獎勵的模擬環境 | 所有 RL 任務 |
| Policy (π) | 從狀態到動作的映射（函數或查找表） | RL 的核心 |
| Reward (r) | 每步的立即回報，RL 的唯一訓練訊號 | 獎勵設計至關重要 |
| Discounted Return (G) | $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$，折扣累積回報 | 考量長期回報 |
| Policy Gradient (REINFORCE) | 直接優化 E[G]，梯度 ∝ log_prob × G | 連續動作空間 |
| Q-Value Q(s,a) | 在狀態 s 採取動作 a 後的折扣期望回報 | Q-Learning 基礎 |
| DQN | 用神經網路近似 Q 函數 + experience replay + target network | 離散動作空間的深度 RL |
| Epsilon-Greedy | 以 ε 機率探索隨機動作，否則選最佳動作 | 探索 vs 利用平衡 |
| Experience Replay | 用 replay buffer 打亂相關性，提升樣本效率 | DQN 的關鍵技巧 |
| Target Network | 定期複製的凍結 Q-network，穩定訓練目標 | DQN 訓練穩定性 |

---

## 2. 關鍵 API 速查

| API | 重點參數 | 用途 |
|-----|---------|------|
| `gym.make("CartPole-v1")` | 環境名稱 | 建立 Gym 環境 |
| `env.reset(seed=42)` | – | 重置環境，回傳初始觀測 |
| `env.step(action)` | action | 執行動作，回傳 (obs, reward, terminated, truncated, info) |
| `env.action_space` | – | 動作空間描述 |
| `env.observation_space` | – | 觀測空間描述 |
| `env.render()` | – | 渲染畫面（human/rgb_array mode） |
| `collections.deque(maxlen=N)` | – | 固定大小的 replay buffer |

---

## 3. 必備代碼片段

```python
import gymnasium as gym
import tensorflow as tf
import numpy as np
from collections import deque

# 建立環境與探索
env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, info = env.reset(seed=42)
print(f"觀測空間: {env.observation_space}")  # Box(4,), 連續狀態
print(f"動作空間: {env.action_space}")       # Discrete(2), 左/右

# 基本環境互動迴圈
total_reward = 0
obs, _ = env.reset()
for step in range(200):
    action = env.action_space.sample()  # 隨機策略
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
print(f"Total reward: {total_reward}")

# Policy Gradient (REINFORCE) 網路
n_inputs = env.observation_space.shape[0]  # 4 for CartPole
n_outputs = env.action_space.n              # 2 for CartPole

policy_net = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu", input_shape=[n_inputs]),
    tf.keras.layers.Dense(n_outputs, activation="softmax")
])

def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        probas = model(obs[tf.newaxis])  # 前向傳播
        action = tf.random.categorical(tf.math.log(probas), 1)[0, 0]
        loss = tf.reduce_mean(loss_fn(tf.expand_dims(action, 0),
                                      probas))  # log_prob
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, terminated, truncated, _ = env.step(int(action))
    return obs, reward, terminated or truncated, grads

def discount_rewards(rewards, discount_factor=0.95):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

# DQN（完整範例）
class DQN:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=2000)
        self.gamma = 0.95       # 折扣率
        self.epsilon = 1.0      # 探索率（初始）
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model(state_size, action_size)
        self.target_model = self._build_model(state_size, action_size)
        self.update_target_network()

    def _build_model(self, state_size, action_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation="relu", input_shape=[state_size]),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(action_size, activation="linear")  # Q-values 不用 softmax
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss="mse")
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()  # Epsilon-greedy 探索
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        batch = np.array(self.replay_buffer)[
            np.random.choice(len(self.replay_buffer), batch_size, replace=False)]
        states, actions, rewards, next_states, dones = (
            np.stack(batch[:, i]) for i in range(5))
        next_q_values = self.target_model.predict(next_states, verbose=0)
        target_q = rewards + (1 - dones.astype(float)) * self.gamma * next_q_values.max(axis=1)
        q_values = self.model.predict(states, verbose=0)
        q_values[np.arange(batch_size), actions.astype(int)] = target_q
        self.model.train_on_batch(states, q_values)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 訓練迴圈
dqn = DQN(state_size=4, action_size=2)
for episode in range(1000):
    obs, _ = env.reset()
    for step in range(200):
        action = dqn.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        dqn.remember(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            break
    dqn.replay(batch_size=32)
    if episode % 10 == 0:
        dqn.update_target_network()  # 定期更新目標網路
```

---

## 4. 常見陷阱

- **Gymnasium vs Gym**：新版本 `gymnasium` 的 `env.step()` 回傳 5 個值（多了 `truncated`），舊版 `gym` 只回傳 4 個，注意版本。
- **Target Network 更新頻率**：太頻繁（每步）→ 不穩定；太少（萬步一次）→ 學習緩慢。通常每 100-1000 步更新一次。
- **Epsilon 衰減**：`epsilon` 過快衰減 → 過早停止探索；太慢 → 效率低下。通常訓練的前 50-80% 期間維持探索。
- **獎勵設計（Reward Shaping）**：原始環境的 reward 可能稀疏（只有最後才有），可以手動加中間獎勵，但要小心引入偏差。

---

## 5. 決策指南

```
RL 算法選擇：
├── 離散動作（遊戲、選擇題）      → DQN / Double DQN / Dueling DQN
├── 連續動作（機器人控制）        → PPO / SAC / TD3
├── 環境模型已知                  → 動態規劃（Value Iteration）
└── 需要可解釋的策略              → Policy Gradient (REINFORCE)

DQN 進階改進：
├── Double DQN     → 用 online 網路選動作，target 網路評估，減少高估 Q 值
├── Dueling DQN    → 分開估計 V(s) 和 A(s,a)，學習更穩定
└── Prioritized ER → 優先回放 TD-error 大的樣本

常用 Gym 環境：
├── CartPole-v1        → 倒立擺（入門首選）
├── MountainCar-v0     → 稀疏獎勵的挑戰性任務
├── LunarLander-v2     → 連續動作空間入門
└── Atari Breakout     → 圖像輸入的進階任務（需 CNN）
```
