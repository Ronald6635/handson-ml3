---
marp: true
theme: gaia
size: 16:9
paginate: true
style: |
  h1, h2, h3 {
    color: #265D94;
  }
  section.lead h1 {
    color: white;
    text-align: center;
  }
  section.lead {
    background-color: #265D94;
    color: white;
    text-align: center;
  }
  section {
    font-size: 28px;
  }
---

<!-- _class: lead -->

# 🎯 深入理解優化理論：原始問題與對偶問題完全指南

---

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [🧐 什麼是原始問題](#primal-problem)
- [🪞 什麼是對偶問題](#dual-problem)
- [🤝 原始與對偶的關鍵關係](#duality-relationships)
- [💡 對偶問題的重要性](#importance)
- [📐 標準原始問題的轉換](#standard-transformation)
- [🔬 拉格朗日對偶性數學推導](#lagrangian-derivation)
- [🧮 凸優化中的對偶理論](#convex-optimization)
- [💼 實際應用案例](#applications)
- [❓ 常見問答](#faq)
- [🎓 總結與最佳實踐](#summary)

---

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **原始問題 (Primal Problem)**：我們最初想要解決的優化問題。
- **對偶問題 (Dual Problem)**：從原始問題衍生的鏡像問題，揭示資源的內在價值。
- **弱對偶性 (Weak Duality)**：對偶解為原始解提供一個界限。
- **強對偶性 (Strong Duality)**：在特定條件下，兩個問題的最優值相等。
- **影子價格 (Shadow Prices)**：對偶變數的經濟意義，代表資源的邊際價值。
- **拉格朗日對偶性**：推導對偶問題的嚴謹數學框架。
- **實務應用**：廣泛應用於機器學習（如 SVM）、資源分配、生產規劃等。

---

## <a id="primal-problem"></a>🧐 什麼是原始問題 (Primal Problem)？

**原始問題**（Primal Problem）通常是我們**最初想要解決的問題**。它直接反映了我們的優化目標和面臨的限制條件。

**範例：最大化利潤**
一家工廠生產兩種產品 A 和 B，目標是在有限的**工時**和**原料**下，最大化總利潤。

---

### <a id="primal-example"></a>範例：原始問題數學表述

**目標函數：**
$$
\text{Maximize } Z = 3x_A + 5x_B \quad \text{(總利潤)}
$$

**限制條件 (Constraints)：**
$$
\begin{align}
1x_A + 2x_B &\le 100 \quad \text{(工時限制)} \\
4x_A + 3x_B &\le 300 \quad \text{(原料限制)} \\
x_A, x_B &\ge 0 \quad \text{(產量非負)}
\end{align}
$$

- **核心功能**：尋找最佳的產品組合 $(x_A, x_B)$ 以最大化總利潤。

---

## <a id="dual-problem"></a>🪞 什麼是對偶問題 (Dual Problem)？

**對偶問題**（Dual Problem）是從原始問題的數學結構中**衍生**出來的「鏡像」問題。它提供了看待原始問題的**不同視角**，特別是從**資源價值**的角度來思考。

**範例：最小化資源價值**
假設有競爭者想購買你的所有資源，他希望用**最小的總成本**來購買，但他開出的「資源價格」必須讓你覺得「賣資源」比「自己生產」更划算。

---

### <a id="dual-example"></a>範例：對偶問題數學表述

**定義影子價格變數：**
- $y_1$：每小時工時的價格
- $y_2$：每單位原料的價格

**目標函數：**
$$
\text{Minimize } Y = 100y_1 + 300y_2 \quad \text{(購買所有資源的總成本)}
$$

**限制條件 (Constraints)：**
$$
\begin{align}
1y_1 + 4y_2 &\ge 3 \quad \text{(對 A 產品的利潤約束)} \\
2y_1 + 3y_2 &\ge 5 \quad \text{(對 B 產品的利潤約束)} \\
y_1, y_2 &\ge 0 \quad \text{(資源價格非負)}
\end{align}
$$

---

## <a id="duality-relationships"></a>🤝 原始與對偶的關鍵關係

原始問題和對偶問題之間的關係，由幾個強大的數學定理來定義。

1.  **弱對偶性 (Weak Duality)**
2.  **強對偶性 (Strong Duality)**
3.  **互補鬆弛 (Complementary Slackness)**

---

### <a id="weak-duality"></a>1. 弱對偶性 (Weak Duality)

**定理陳述：**
對於任何**可行的**原始解（利潤 $Z$）和任何**可行的**對偶解（成本 $Y$），一定會滿足：
$$
Z \le Y \quad \text{(在最大化問題中)}
$$

**直觀理解：**
你（原始問題）能賺到的最大利潤，永遠**不會**超過競爭者（對偶問題）為了買你資源所需要付出的最小成本。

**實際意義：**
對偶問題提供了原始問題最優值的**上界**。

---

### <a id="strong-duality"></a>2. 強對偶性 (Strong Duality)

**定理陳述：**
如果原始問題存在一個**最佳解**（$Z^*$），那麼對偶問題也**必然**存在一個最佳解（$Y^*$），並且：
$$
Z^* = Y^* \quad \text{(最佳利潤 = 最小資源總成本)}
$$

**實際意義：**
- 當兩者相等時，我們就知道已經找到了最佳方案。
- 可以選擇求解較簡單的那個問題（原始或對偶）。

**⚠️ 重要條件：** 強對偶性並非總是成立，但線性規劃問題通常滿足。

---

### <a id="complementary-slackness"></a>3. 互補鬆弛 (Complementary Slackness)

這個關係連結了原始問題的「限制」和對偶問題的「變數」。

**直觀解釋：**

- **資源有剩餘（鬆弛）**
  - 如果原始問題的某個資源「沒有用完」。
  - **結論**：對偶問題中該資源的「價格」（影子價格）必定為 0。
  - **經濟解釋**：既然資源沒用完，表示它不稀缺，邊際價值就是 0。

- **資源被完全使用（緊湊）**
  - 如果對偶問題中某個資源的「價格」為正。
  - **結論**：原始問題中該資源必定「剛好用完」。
  - **經濟解釋**：只有當資源被完全用盡時，它才具有正的邊際價值。

---

## <a id="importance"></a>💡 為什麼對偶問題很重要？

1.  **📈 提供經濟解釋 (影子價格)**
    - 對偶變數代表資源的**邊際價值**，是決策的關鍵依據。
    - 幫助管理者決定「是否該花錢購買更多資源」。

2.  **💻 帶來計算優勢**
    - 當原始問題限制多、變數少時，對偶問題可能更容易求解。
    - 在 SVM 中，求解對偶問題能引入「核技巧 (Kernel Trick)」。

3.  **📊 進行敏感度分析**
    - 分析當原始問題的參數（如利潤、資源總量）發生變化時，最佳解會如何改變。

---

## <a id="standard-transformation"></a>📐 標準原始問題的轉換

將原始問題轉換為對偶問題有一套清晰的鏡像規則。

| 原始問題 (Primal - Max) | 鏡像轉換 | 對偶問題 (Dual - Min) |
| :--- | :--- | :--- |
| **目標**：最大化 (Max) | $\longleftrightarrow$ | **目標**：最小化 (Min) |
| **目標函數係數**：$c$ | $\longleftrightarrow$ | **限制式右側**：$c$ |
| **限制式右側**：$b$ | $\longleftrightarrow$ | **目標函數係數**：$b$ |
| **限制矩陣**：$A$ | $\longleftrightarrow$ | **限制矩陣**：$A^T$ (轉置) |
| **限制類型**：$\le$ | $\longleftrightarrow$ | **變數符號**：$\ge 0$ |
| **變數符號**：$\ge 0$ | $\longleftrightarrow$ | **限制類型**：$\ge$ |

---

## <a id="lagrangian-derivation"></a>🔬 拉格朗日對偶性數學推導

最嚴謹的推導方法是使用**拉格朗日對偶性 (Lagrangian Duality)**。

**核心思想：**
1.  將原始問題的「限制條件」透過「拉格朗日乘數」融入到「目標函數」中，形成拉格朗日函數 $L(x, y)$。
2.  從一個新的「對偶」視角來優化這個函數，從而自然地推導出對偶問題。

---

### 拉格朗日推導步驟

1.  **定義原始問題 (P)**
    $$
    \text{Maximize } Z = c^T x \quad \text{s.t. } Ax \le b, x \ge 0
    $$
2.  **建構拉格朗日函數 $L(x, y)$**
    $$
    L(x, y) = c^T x + y^T (b - Ax) \quad \text{其中 } y \ge 0
    $$
3.  **定義對偶函數 $g(y)$**
    $$
    g(y) = \sup_{x \ge 0} L(x, y)
    $$
4.  **分析 $g(y)$ 的值**
    $$
    g(y) = \begin{cases} b^T y & \text{如果 } A^T y \ge c \\ \infty & \text{其他} \end{cases}
    $$
5.  **定義對偶問題 (D)**：尋找最好的上界
    $$
    \text{Minimize } g(y) \quad \text{s.t. } y \ge 0 \implies \text{Minimize } b^T y \quad \text{s.t. } A^T y \ge c, y \ge 0
    $$

---

## <a id="convex-optimization"></a>🧮 凸優化中的對偶理論

對偶理論在**凸優化 (Convex Optimization)** 中扮演著更廣泛且關鍵的角色。

### KKT 條件 (Karush–Kuhn–Tucker Conditions)

在滿足強對偶性時，最優解會同時滿足 KKT 條件，這是非線性規劃最佳解的必要條件：

1.  **原始可行性 (Primal feasibility)**
2.  **對偶可行性 (Dual feasibility)**
3.  **互補鬆弛 (Complementary slackness)**
4.  **梯度條件 (Stationarity)**

在 SVM 的推導中，KKT 條件被用來找出支持向量 (Support Vectors)。

---

## <a id="applications"></a>💼 實際應用案例

| 領域 | 應用案例 | 如何利用對偶理論 |
| :--- | :--- | :--- |
| **機器學習** | **支援向量機 (SVM)** | 透過求解對偶問題，引入核技巧 (Kernel Trick)，解決非線性分類問題。 |
| **經濟學** | **資源定價與分配** | 對偶變數（影子價格）直接反映了資源的邊際價值，幫助制定最優策略。 |
| **生產管理** | **產品組合優化** | 分析哪些產品利潤貢獻最大，以及哪些資源是生產瓶頸。 |
| **網路流量** | **最大流最小割問題** | 網路中的最大流問題與其對偶問題——最小割問題的最優值相等。 |

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: 為什麼影子價格可以是 0？**
> A1: 當一個資源沒有被完全用完時，代表它不稀缺。因此，多給你一單位這種資源，並不會增加總利潤，其邊際價值就是 0。

**Q2: 求解對偶問題一定比原始問題容易嗎？**
> A2: 不一定。這取決於問題的結構。通常我們會選擇變數較少或結構更適合特定演算法的問題來求解。

**Q3: 強對偶性在什麼情況下不成立？**
> A3: 在非凸優化問題中，強對偶性通常不成立。即使在凸優化中，也需要滿足某些條件（如 Slater 條件）才能保證。

---

## <a id="summary"></a>🎓 總結與最佳實踐

- **從兩個角度思考**：遇到優化問題時，不僅要思考如何直接求解（原始視角），也要思考其資源的價值評估（對偶視角）。
- **利用影子價格做決策**：在資源配置問題中，務必計算影子價格。它是衡量資源價值的黃金標準。
- **檢查 KKT 條件**：對於非線性優化問題，KKT 條件是驗證解的最優性的重要工具。
- **選擇合適的問題求解**：根據問題的變數與限制數量，靈活選擇求解原始問題或對偶問題。

---

<!-- _class: lead -->

# Q & A

## <a id="hashtags"></a>🏷️ #優化理論 #線性規劃 #機器學習