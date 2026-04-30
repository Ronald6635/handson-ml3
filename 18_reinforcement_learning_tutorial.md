<!-- meta-title: 強化學習完整指南：MDP、Q-Learning、DQN 與 Double DQN 實作 -->
<!-- meta-description: 深入強化學習：馬可夫決策過程（MDP）、OpenAI Gymnasium 環境、REINFORCE 策略梯度、Q-Learning 與 Bellman 方程、DQN（經驗回放、目標網路）、Double DQN，以及如何訓練 Atari 遊戲 AI。 -->
<!-- meta-keywords: Python, 強化學習, MDP, Q-Learning, DQN, 深度強化學習, Gymnasium, OpenAI, TensorFlow, 策略梯度 -->
<!-- meta-hashtags: #Python #強化學習 #MDP #QLearning #DQN #DeepRL #Gymnasium #TensorFlow #AI #教學 -->

# 🐍 強化學習：從馬可夫決策過程到 DQN 打 Atari 遊戲

AlphaGo 下棋、Tesla 自動駕駛、資料中心冷卻系統優化——**強化學習（Reinforcement Learning, RL）** 是讓 AI 自主學習最優策略的技術。本教學從最基礎的 MDP 概念出發，帶你一步步實作 Q-Learning、DQN，直到能訓練出可以玩 Atari 遊戲的 AI。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [🌍 強化學習基本概念](#rl-basics)
- [🏋️ Gymnasium 環境](#gymnasium)
- [📊 策略梯度（REINFORCE）](#reinforce)
- [🧮 Q-Learning 與 Bellman 方程](#qlearning)
- [🤖 深度 Q 網路（DQN）](#dqn)
- [🎯 Double DQN 改進](#double-dqn)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **MDP** 的五元組：(狀態 S, 動作 A, 獎勵 R, 轉移 T, 折扣因子 γ)
- **Q-Learning** 是離線策略方法：學習最優 Q 值，不需要遵循當前策略
- **DQN** 用神經網路替代 Q-Table，兩個關鍵技巧：**經驗回放（Experience Replay）** 和 **目標網路（Target Network）**
- **ε-greedy** 策略平衡探索（Exploration）和利用（Exploitation）
- **Double DQN** 解決 Q-Learning 的過度估計問題，通常比標準 DQN 表現更好

---

## <a id="rl-basics"></a>🌍 強化學習基本概念

💡 **實際應用情境：** 想像訓練一隻機器狗學習行走——沒有人告訴它如何走路（無監督），但每次成功往前走一步就得到正獎勵，摔倒就得到負獎勵。通過無數次嘗試，它學會了最優的行走策略。

### 強化學習的要素

| 要素 | 描述 | 例子（CartPole） |
|------|------|----------------|
| 環境（Environment） | 智能體（Agent）互動的世界 | CartPole 物理模擬器 |
| 狀態（State, s） | 環境的當前描述 | 小車位置、桿子角度 |
| 動作（Action, a） | 智能體可執行的操作 | 向左推、向右推 |
| 獎勵（Reward, r） | 執行動作後的即時反饋 | 桿子直立 +1；倒下 0 |
| 策略（Policy, π） | 從狀態到動作的映射 | π(s) → a |
| 折扣因子（γ） | 未來獎勵的重要性（0~1） | γ=0.99 重視長期 |

**累積折扣獎勵（Return）：**

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

---

## <a id="gymnasium"></a>🏋️ Gymnasium 環境

💡 **實際應用情境：** Gymnasium（原 OpenAI Gym）提供標準化的 RL 測試環境，從簡單的 CartPole 到複雜的 Atari 遊戲，讓我們可以快速測試算法。

### 範例 1: 基本環境操作

```python
import gymnasium as gym
import numpy as np

# 建立 CartPole 環境（最常用的 RL 入門環境）
env = gym.make("CartPole-v1", render_mode=None)

print(f"狀態空間: {env.observation_space}")      # Box(4,) - 4 個連續數值
print(f"動作空間: {env.action_space}")            # Discrete(2) - 0=左, 1=右
print(f"動作數量: {env.action_space.n}")          # 2

# 與環境互動的基本流程
state, info = env.reset(seed=42)  # 重置環境，返回初始狀態
total_reward = 0

for step in range(200):
    # 隨機策略（基準）
    action = env.action_space.sample()  # 隨機選擇動作

    # 執行動作，獲取下一狀態、獎勵、終止信號
    next_state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode 結束，共 {step+1} 步，總獎勵: {total_reward:.1f}")
        break

    state = next_state

env.close()

# 狀態含義：[小車位置, 小車速度, 桿子角度, 桿子角速度]
print(f"\nCartPole 狀態示例: {state}")
print("目標：讓桿子保持直立盡量久（最多 500 步）")
```

**✅ 程式碼逐行解析：**

1. `env.reset()`: 重置環境到初始狀態，每個 Episode 開始時呼叫
2. `env.step(action)`: 執行動作，返回 (next_state, reward, terminated, truncated, info)
3. `terminated`: 達到終止條件（桿子倒了）；`truncated`: 達到最大步數限制

---

## <a id="reinforce"></a>📊 策略梯度（REINFORCE）

### 範例 2: REINFORCE 算法

```python
import tensorflow as tf
from tensorflow import keras

# 策略網路（Policy Network）：輸入狀態 → 輸出每個動作的機率
def build_policy_network(n_inputs: int, n_outputs: int) -> keras.Model:
    return keras.Sequential([
        keras.layers.Dense(32, activation="relu", input_shape=(n_inputs,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(n_outputs, activation="softmax")  # 動作機率分佈
    ])

n_inputs  = env.observation_space.shape[0]  # 4（CartPole 狀態維度）
n_outputs = env.action_space.n              # 2（動作數）
policy_net = build_policy_network(n_inputs, n_outputs)

def discount_rewards(rewards: list, gamma: float = 0.99) -> np.ndarray:
    """計算折扣累積獎勵（從後往前計算）"""
    discounted = np.zeros(len(rewards), dtype=np.float32)
    discounted[-1] = rewards[-1]
    for t in range(len(rewards) - 2, -1, -1):
        discounted[t] = rewards[t] + gamma * discounted[t + 1]
    # 標準化（減均值除標準差）→ 降低方差，穩定訓練
    mean, std = discounted.mean(), discounted.std()
    return (discounted - mean) / (std + 1e-8)

def reinforce_train_step(policy_net: keras.Model,
                          optimizer: keras.optimizers.Optimizer,
                          episode_states: list,
                          episode_actions: list,
                          episode_rewards: list) -> float:
    """REINFORCE 算法的一步更新"""
    discounted = discount_rewards(episode_rewards)

    with tf.GradientTape() as tape:
        logits = policy_net(np.array(episode_states), training=True)
        action_probs = tf.nn.softmax(logits)

        # 選擇實際執行動作的 log 機率
        action_masks = tf.one_hot(episode_actions, n_outputs)
        log_probs = tf.reduce_sum(
            tf.math.log(action_probs + 1e-8) * action_masks, axis=1
        )

        # REINFORCE 損失：-E[log π(a|s) * G_t]（負號因為要最大化）
        loss = -tf.reduce_mean(log_probs * discounted)

    gradients = tape.gradient(loss, policy_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_net.trainable_variables))
    return loss.numpy()


# 訓練迴圈
optimizer_reinforce = keras.optimizers.Adam(1e-3)
env_train = gym.make("CartPole-v1")

best_reward = 0
for episode in range(200):
    states, actions, rewards = [], [], []
    state, _ = env_train.reset()

    while True:
        probs = policy_net(state[np.newaxis]).numpy()[0]
        action = np.random.choice(n_outputs, p=probs)  # 按機率採樣

        states.append(state)
        actions.append(action)
        next_state, reward, terminated, truncated, _ = env_train.step(action)
        rewards.append(reward)
        state = next_state

        if terminated or truncated:
            break

    loss = reinforce_train_step(policy_net, optimizer_reinforce,
                                  states, actions, rewards)
    total = sum(rewards)
    best_reward = max(best_reward, total)
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}: total_reward={total:.0f}, "
              f"best={best_reward:.0f}, loss={loss:.4f}")

env_train.close()
```

**🎯 重點摘要:**

- REINFORCE 是**蒙地卡羅策略梯度**：整個 episode 結束後才更新
- 獎勵標準化（減均值除標準差）大幅降低梯度估計的方差，加速收斂

---

## <a id="qlearning"></a>🧮 Q-Learning 與 Bellman 方程

💡 **實際應用情境：** Q-Learning 學習「在狀態 s 執行動作 a 的長期價值 Q(s,a)」，而不是直接學習策略。

$$Q^*(s, a) = r + \gamma \cdot \max_{a'} Q^*(s', a') \quad \text{（Bellman 方程）}$$

### 範例 3: 表格 Q-Learning（離散環境）

```python
# FrozenLake：4×4 網格，目標是從 S 走到 G，避開冰洞 H
env_lake = gym.make("FrozenLake-v1", is_slippery=False)

n_states  = env_lake.observation_space.n   # 16 個格子
n_actions = env_lake.action_space.n        # 4 個方向

# Q-Table：行=狀態，列=動作，值=Q(s,a)
Q_table = np.zeros((n_states, n_actions))

# Q-Learning 超參數
alpha   = 0.1    # 學習率
gamma   = 0.99   # 折扣因子
epsilon = 1.0    # 初始探索率
epsilon_min = 0.01
epsilon_decay = 0.995

# 訓練
for episode in range(5000):
    state, _ = env_lake.reset()
    done = False

    while not done:
        # ε-greedy 策略（平衡探索與利用）
        if np.random.random() < epsilon:
            action = env_lake.action_space.sample()  # 探索：隨機動作
        else:
            action = np.argmax(Q_table[state])        # 利用：取最大 Q 值

        next_state, reward, terminated, truncated, _ = env_lake.step(action)
        done = terminated or truncated

        # Q-Learning 更新（Bellman 方程的增量形式）
        td_target = reward + gamma * np.max(Q_table[next_state]) * (not done)
        td_error  = td_target - Q_table[state, action]
        Q_table[state, action] += alpha * td_error

        state = next_state

    # 衰減探索率
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

env_lake.close()
print(f"Q-Table 訓練完成，探索率衰減至: {epsilon:.4f}")
print(f"學到的最優策略（每個格子的最佳動作）:")
action_names = ["←", "↓", "→", "↑"]
policy = [action_names[np.argmax(Q_table[s])] for s in range(n_states)]
print(np.array(policy).reshape(4, 4))
```

**✅ 程式碼逐行解析：**

1. `td_target = r + γ * max Q(s')`: Bellman 方程——當前獎勵 + 折扣後的未來最大價值
2. `Q[s,a] += α * (target - Q[s,a])`: 增量更新，alpha 控制更新速度（學習率）
3. ε-greedy 的 epsilon 隨訓練遞減：早期多探索，後期多利用

---

## <a id="dqn"></a>🤖 深度 Q 網路（DQN）

💡 **實際應用情境：** CartPole 只有 4 個狀態變數，Q-Table 可行。但 Atari 遊戲的狀態是 84×84 像素圖像——Q-Table 的大小會是天文數字。DQN 用神經網路替代 Q-Table，解決高維連續狀態空間的問題。

### 範例 4: DQN 的核心組件

```python
import collections
import random

# ── 1. DQN 網路結構 ──
def build_dqn(n_inputs: int, n_outputs: int) -> keras.Model:
    """Q 網路：輸入狀態 → 輸出每個動作的 Q 值"""
    return keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(n_inputs,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(n_outputs)  # 線性輸出（Q 值可為負）
    ])

# ── 2. 經驗回放緩衝區（Experience Replay Buffer）──
class ReplayBuffer:
    """環形緩衝區，儲存過去的 (s, a, r, s', done) 轉換"""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """隨機採樣一個 mini-batch"""
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (np.array(states, dtype=np.float32),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ── 3. DQN 訓練邏輯 ──
class DQNAgent:
    """DQN 智能體（含目標網路和經驗回放）"""

    def __init__(self, n_states: int, n_actions: int,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 target_update_freq: int = 100):
        self.n_actions = n_actions
        self.gamma  = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.step_count = 0

        # 主 Q 網路（頻繁更新）
        self.q_network = build_dqn(n_states, n_actions)
        # 目標網路（緩慢更新，提供穩定的訓練目標）
        self.target_network = build_dqn(n_states, n_actions)
        self.target_network.set_weights(self.q_network.get_weights())

        self.optimizer = keras.optimizers.Adam(lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy 動作選擇"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.q_network(state[np.newaxis], training=False)[0]
        return int(np.argmax(q_values))

    def train_step(self) -> float | None:
        """從 Replay Buffer 採樣並更新 Q 網路"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        with tf.GradientTape() as tape:
            # 當前 Q 值
            q_values = self.q_network(states, training=True)
            action_masks = tf.one_hot(actions, self.n_actions)
            q_selected = tf.reduce_sum(q_values * action_masks, axis=1)

            # 目標 Q 值（使用目標網路，提供穩定目標）
            next_q = tf.reduce_max(
                self.target_network(next_states, training=False), axis=1
            )
            td_target = rewards + self.gamma * next_q * (1 - dones)

            # Huber 損失（比 MSE 對離群值更健壯）
            loss = keras.losses.huber(tf.stop_gradient(td_target), q_selected)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )

        # 定期更新目標網路
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.set_weights(self.q_network.get_weights())

        # 衰減探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss)


# ── 4. 訓練迴圈 ──
env_dqn = gym.make("CartPole-v1")
agent = DQNAgent(
    n_states=env_dqn.observation_space.shape[0],
    n_actions=env_dqn.action_space.n
)

episode_rewards = []
for episode in range(300):
    state, _ = env_dqn.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env_dqn.step(action)
        done = terminated or truncated

        agent.replay_buffer.add(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"Episode {episode+1}: avg_reward={avg_reward:.1f}, "
              f"epsilon={agent.epsilon:.3f}")

env_dqn.close()
```

**✅ 程式碼逐行解析：**

1. **經驗回放** (`ReplayBuffer`): 打亂時間相關性——若直接用連續 batch，相鄰樣本高度相關，導致訓練不穩定
2. **目標網路** (`target_network`): 每隔 N 步從主網路複製權重，在此期間提供固定目標——避免「追逐移動目標」的發散問題
3. `tf.stop_gradient(td_target)`: 計算損失時，target 不應產生梯度（只更新 q_network）

**🎯 重點摘要:**

- DQN 的兩個關鍵技巧缺一不可：沒有 Replay Buffer → 訓練不穩定；沒有目標網路 → 訓練發散

---

## <a id="double-dqn"></a>🎯 Double DQN 改進

### 範例 5: Double DQN 的差異（只需修改一行）

```python
# 標準 DQN：目標 Q 值 = r + γ * max Q_target(s')
# 問題：用 max 選擇和評估同一個動作 → 高估 Q 值

# Double DQN：
#   選擇：用主網路選擇最佳動作 a* = argmax Q_main(s')
#   評估：用目標網路評估該動作 Q_target(s', a*)
# 分離選擇和評估 → 減少過度估計

class DoubleDQNAgent(DQNAgent):
    """Double DQN：分離動作選擇和 Q 值評估"""

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            action_masks = tf.one_hot(actions, self.n_actions)
            q_selected = tf.reduce_sum(q_values * action_masks, axis=1)

            # ── Double DQN 的關鍵修改 ──
            # 1. 主網路選擇最佳動作
            next_q_main = self.q_network(next_states, training=False)
            best_actions = tf.argmax(next_q_main, axis=1)  # 主網路選動作

            # 2. 目標網路評估該動作的 Q 值
            next_q_target = self.target_network(next_states, training=False)
            best_action_mask = tf.one_hot(best_actions, self.n_actions)
            next_q = tf.reduce_sum(next_q_target * best_action_mask, axis=1)

            td_target = rewards + self.gamma * next_q * (1 - dones)
            loss = keras.losses.huber(tf.stop_gradient(td_target), q_selected)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.set_weights(self.q_network.get_weights())
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss)

print("Double DQN 定義完成！與標準 DQN 的唯一差異：目標 Q 值的計算方式。")
```

**🎯 重點摘要:**

- Double DQN 通常比標準 DQN 穩定，且最終效能更好
- 其他 DQN 改進：Dueling DQN（分離狀態值和優勢函數）、Prioritized Experience Replay（高 TD-Error 樣本更頻繁採樣）

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: ε-greedy 的 epsilon 如何設定？**

A: 通常從 1.0（完全隨機探索）開始，指數衰減到 0.01~0.1（以利用為主）。衰減速度取決於任務複雜度——太快則沒探索完就開始利用；太慢則浪費時間在隨機探索上。

**Q2: 為什麼需要 Replay Buffer？**

A: 連續互動的樣本高度相關（每一步的狀態和下一步非常相似），直接訓練相當於只用一個樣本——梯度高方差。Replay Buffer 儲存過去的經驗並隨機採樣，打破時間相關性，提供獨立同分佈的 mini-batch。

**Q3: 目標網路更新頻率如何選擇？**

A: 通常每 100~1000 步更新一次。更新太頻繁（接近 1）→ 等於沒有目標網路；更新太慢 → 目標過於陳舊，學習緩慢。另一種方式：使用軟更新（`θ_target = τ*θ_main + (1-τ)*θ_target`，τ≈0.01）。

**Q4: DQN 適用的場景？離散 vs 連續動作空間？**

A: DQN 只適用於**離散動作空間**（因為需要對所有動作計算 max Q）。連續動作空間需用 DDPG（Deep Deterministic Policy Gradient）或 SAC（Soft Actor-Critic）。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #強化學習 #MDP #QLearning #DQN #DoubleDQN #DeepRL #Gymnasium #TensorFlow #策略梯度 #AI #程式設計 #教學 #MachineLearning
