# 課程講義：強化學習 (Chapter 18)

強化學習（Reinforcement Learning，RL）是機器學習的第三大典範：Agent 不依賴標籤資料，而是透過**在環境中行動、接收獎勵**來學習最優策略。從 OpenAI Gym 的遊戲環境，到 DeepMind AlphaGo、OpenAI ChatGPT 的 RLHF——強化學習正在重塑人工智慧的邊界。本章從 MDP 的數學基礎出發，帶你實作策略梯度、Q 學習和深度 Q 網路（DQN）。

---

## 1. 強化學習的基本框架

### 理論背景

**馬可夫決策過程 (MDP, Markov Decision Process)**：

- **狀態 (State) $s$**：Agent 對環境的觀測
- **動作 (Action) $a$**：Agent 可以執行的操作
- **獎勵 (Reward) $r$**：執行動作後環境給的即時回饋
- **策略 (Policy) $\pi(a|s)$**：在狀態 $s$ 下選擇動作 $a$ 的機率

**折扣累積獎勵 (Return)**：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

其中折扣因子 $\gamma \in [0, 1)$：

- $\gamma = 0$：只在乎即時獎勵（短視）
- $\gamma \to 1$：長遠考慮（有遠見）

**`gymnasium` 環境介面**（OpenAI Gym 的繼承者）：

```
env.reset()  → state
env.step(action) → (next_state, reward, terminated, truncated, info)
env.close()
```

### 核心代碼

```python
import gymnasium as gym
import numpy as np

# 建立 CartPole 環境（平衡桿問題）
env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, info = env.reset(seed=42)
print(f"初始狀態: {obs}")   # [位置, 速度, 角度, 角速度]
print(f"動作空間: {env.action_space}")    # Discrete(2): 向左(0) / 向右(1)
print(f"狀態空間: {env.observation_space}")  # Box([-4.8, -inf, -0.41, -inf], ...)

# 隨機策略基準線
total_rewards = 0
obs, info = env.reset(seed=42)
for step in range(1000):
    action = env.action_space.sample()  # 隨機選動作
    obs, reward, terminated, truncated, info = env.step(action)
    total_rewards += reward
    if terminated or truncated:
        obs, info = env.reset()
        break

print(f"隨機策略總獎勵: {total_rewards}")
env.close()
```

### ⚡ 補充練習 1

**理論題：** 在 CartPole 中，折扣因子 $\gamma = 0.99$ 和 $\gamma = 0.9$ 對 Agent 的行為有何影響？哪個設定會讓 Agent 更傾向於避免長期崩潰而非最大化即時獎勵？

**實作題：** 用 `gymnasium` 的 `MountainCar-v0` 環境跑 10 個隨機策略 episode，記錄每個 episode 的總獎勵，計算平均和標準差。觀察隨機策略是否能偶爾解決這個問題（達到山頂）。

---

## 2. 策略梯度：REINFORCE 演算法

### 理論背景

**策略梯度定理**：直接優化策略 $\pi_\theta$ 以最大化期望回報：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [G_0]$$

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[\sum_t G_t \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

**REINFORCE 演算法**（Monte Carlo 策略梯度）：

1. 使用當前策略 $\pi_\theta$ 跑完整個 episode，收集軌跡 $(s_0, a_0, r_0, s_1, \ldots)$
2. 計算每個時間步的折扣回報 $G_t$
3. 梯度更新：$\theta \leftarrow \theta + \eta G_t \nabla_\theta \log \pi_\theta(a_t | s_t)$

**Keras 的實現方式**：自定義損失函數 $\mathcal{L} = -\log \pi_\theta(a_t | s_t) \cdot G_t$（梯度上升等效於最小化負值）。

**方差問題**：REINFORCE 的梯度方差很高（每次完整 episode 的 $G_t$ 差異大）。解法：減去基線（Baseline）如 $G_t - b$。

### 核心代碼

```python
import tensorflow as tf

tf.random.set_seed(42)

# 策略網路（給定狀態，輸出各動作的機率）
policy_net = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu", input_shape=[4]),  # CartPole: 4 個狀態
    tf.keras.layers.Dense(2, activation="softmax")  # 2 個動作
])

def play_one_step(env, obs, model, loss_fn):
    """執行一步，記錄梯度"""
    with tf.GradientTape() as tape:
        proba = model(obs[np.newaxis], training=True)
        # 從機率分布採樣動作
        action = tf.random.categorical(tf.math.log(proba), num_samples=1)[0, 0]
        loss = loss_fn(tf.constant([[1, 0]] if action == 0 else [[0, 1]]), proba)
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, terminated, truncated, info = env.step(int(action))
    return obs, reward, terminated or truncated, grads

def discount_rewards(rewards, discount_rate=0.95):
    """計算折扣回報"""
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted

def normalize_rewards(all_discounted_rewards):
    """正規化回報（降低方差）"""
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std  = flat_rewards.std() + 1e-8
    return [(dr - reward_mean) / reward_std for dr in all_discounted_rewards]
```

### ⚡ 補充練習 2

**理論題：** REINFORCE 演算法在稀疏獎勵環境（如 MountainCar，只有到達山頂才有正獎勵）中表現極差，為什麼？Actor-Critic 方法如何緩解這個問題？

**實作題：** 在 CartPole 上訓練 REINFORCE（50 個 episodes/批次，discount_rate=0.95），繪製每 10 個批次的平均總獎勵曲線，觀察學習進程。策略何時開始穩定地讓桿子平衡超過 100 步？

---

## 3. Q 學習與 Bellman 方程

### 理論背景

**Q 函數 (Action-Value Function)**：在狀態 $s$ 下執行動作 $a$，然後遵循策略 $\pi$ 的期望回報：

$$Q^\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$

**Bellman 最優方程**（Q 學習的核心）：

$$Q^*(s, a) = \mathbb{E} \left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

**Q 學習（表格型）**：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

括號內是 **TD 誤差 (Temporal Difference Error)**：目標值與估計值的差。

**$\varepsilon$-greedy 策略**：以 $\varepsilon$ 的機率隨機探索，以 $1-\varepsilon$ 的機率利用（選最大 Q 值的動作）。$\varepsilon$ 隨訓練逐漸衰減。

### 核心代碼

```python
# 表格型 Q 學習（MountainCar 離散化示例）
class QTable:
    def __init__(self, n_states, n_actions):
        self.Q = np.zeros((n_states, n_actions))

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.Q.shape[1])  # 探索
        return self.Q[state].argmax()  # 利用

    def update(self, state, action, reward, next_state, gamma=0.99, alpha=0.1):
        td_error = (reward + gamma * self.Q[next_state].max()
                    - self.Q[state, action])
        self.Q[state, action] += alpha * td_error

# 訓練循環
epsilon_start, epsilon_end = 1.0, 0.05
n_episodes = 1000

for episode in range(n_episodes):
    epsilon = max(epsilon_end, epsilon_start - episode / (n_episodes * 0.8))
    obs, _ = env.reset()
    done = False
    while not done:
        state = discretize(obs)  # 連續狀態→離散
        action = q_table.get_action(state, epsilon)
        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(obs)
        q_table.update(state, action, reward, next_state)
        done = terminated or truncated
```

### ⚡ 補充練習 3

**理論題：** Q 學習是「off-policy」演算法（可以從其他策略收集的資料中學習），而 REINFORCE 是「on-policy」演算法（只能從當前策略收集的資料中學習）。解釋這個區別，以及為何 off-policy 演算法可以使用「經驗回放緩衝區」？

**實作題：** 手動實作 Bellman 方程的值迭代（Value Iteration）在一個小型網格世界（如 5×5，有陷阱和獎勵格）上，計算每個格子的最優 Q 值，並推導最優策略（用箭頭圖表示）。

---

## 4. DQN：深度 Q 網路

### 理論背景

**DQN (Deep Q-Network)**：用神經網路逼近 Q 函數：

$$Q_\theta(s, a) \approx Q^*(s, a)$$

**DQN 的兩個關鍵創新**（解決訓練不穩定問題）：

**1. 經驗回放 (Experience Replay)**：

將過去的轉移 $(s, a, r, s')$ 存入緩衝區（Replay Buffer），訓練時隨機取樣 mini-batch。

- 打破序列相關性（相鄰步驟的資料高度相關，違反 SGD 假設）
- 同一個轉移可以多次用於訓練（提高資料效率）

**2. 目標網路 (Target Network)**：

使用**凍結**的舊網路參數 $\theta^-$ 計算 TD 目標：

$$y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a')$$

- 穩定訓練目標（避免「移動靶」問題）
- 每 $N$ 步才更新一次目標網路的參數

**Double DQN**：分離「選動作」（用主網路）和「評估 Q 值」（用目標網路），減少 Q 值高估。

### 核心代碼

```python
from collections import deque
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

# 建立 DQN 網路（CartPole: 4 個輸入, 2 個動作）
def build_dqn(n_inputs, n_outputs):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=[n_inputs]),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_outputs)  # 輸出每個動作的 Q 值（無活化函數）
    ])

model        = build_dqn(4, 2)
target_model = build_dqn(4, 2)
target_model.set_weights(model.get_weights())  # 初始化相同

optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)
loss_fn   = tf.keras.losses.MeanSquaredError()

# 經驗回放緩衝區
replay_buffer = deque(maxlen=2000)

def training_step(batch_size=64, gamma=0.99):
    # 從緩衝區隨機取樣
    batch = [replay_buffer[np.random.randint(len(replay_buffer))]
             for _ in range(batch_size)]
    states, actions, rewards, next_states, dones = zip(*batch)

    states = np.array(states, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    dones   = np.array(dones,   dtype=np.float32)

    # 計算 TD 目標（用目標網路）
    next_Q_values = target_model.predict(next_states, verbose=0)
    max_next_Q = np.max(next_Q_values, axis=1)
    target_Q = rewards + (1 - dones) * gamma * max_next_Q

    with tf.GradientTape() as tape:
        Q_values = model(states, training=True)
        action_mask = tf.one_hot(actions, depth=2)
        Q_action = tf.reduce_sum(Q_values * action_mask, axis=1)
        loss = loss_fn(target_Q, Q_action)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# 每 50 步同步目標網路
TARGET_UPDATE_FREQ = 50
```

### ⚡ 補充練習 4

**理論題：** 「移動靶（Moving Target）」問題是什麼？若用同一個網路計算 TD 目標和 Q 值估計，為何訓練會不穩定（類比：你一邊移動，一邊試圖追著自己的影子跑）？

**實作題：** 在 CartPole-v1 上訓練 DQN（回放緩衝區 2000、batch_size=64、gamma=0.99、目標網路每 50 步更新），繪製每 10 個 episode 的平均總獎勵，觀察 Agent 何時學會穩定平衡（總獎勵 > 400）。

---

## 結論

強化學習的核心工具箱：

- **MDP + Gymnasium**：環境交互的標準介面
- **REINFORCE**：策略梯度方法的起點；高方差，適合連續動作空間
- **Q 學習 + Bellman 方程**：值函數方法；表格型適合小型離散空間
- **DQN**：結合深度學習和 Q 學習；經驗回放 + 目標網路是訓練穩定的關鍵
- **Double DQN**：緩解 Q 值高估問題

現代 RL 的最新進展（本書未覆蓋但值得了解）：PPO、SAC、AlphaZero、RLHF（ChatGPT 的訓練方法）。

下一章（Ch19）聚焦於大規模生產部署：如何將模型從 Jupyter Notebook 推進到真正服務用戶的生產系統。

---

## 課後作業

**作業：DQN 訓練與分析**

在 `CartPole-v1` 環境上：

1. 訓練完整 DQN（含經驗回放和目標網路），直到平均總獎勵超過 450（CartPole 的最大獎勵為 500）。記錄訓練所需的 episode 數。

2. **消融實驗**：分別移除「經驗回放」和「目標網路」之一，觀察訓練穩定性的變化。製成比較圖（x 軸：episode，y 軸：總獎勵的移動平均）。

3. **思考題**：CartPole 的最優策略是什麼（用自然語言描述：在哪種情況下應該向左/向右推）？觀察訓練好的 DQN Agent 的行為，它的策略是否符合你的直覺？
