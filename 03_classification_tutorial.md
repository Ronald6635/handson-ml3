<!-- meta-title: Python 分類實戰：從 MNIST 到 ROC 曲線的完整機器學習教學 -->
<!-- meta-description: 使用 MNIST 資料集完整實作分類任務：SGDClassifier、混淆矩陣、精確率/召回率、ROC 曲線、多類別分類、多標籤分類。包含逐行程式碼解析，適合 ML 學習者。 -->
<!-- meta-keywords: Python, 機器學習, 分類, MNIST, Scikit-Learn, 混淆矩陣, ROC曲線, AUC, SGD, 精確率, 召回率 -->
<!-- meta-hashtags: #Python #機器學習 #分類 #MNIST #ScikitLearn #混淆矩陣 #ROC曲線 #程式設計 #教學 #DataScience -->

# 🐍 Python 分類實戰：MNIST 到 ROC 曲線完整指南

分類（Classification）是監督學習中最常見的任務類型。本教學以 **MNIST 手寫數字辨識**為主線，帶你從零開始建構二元分類器，深入了解如何評估模型效能，並延伸至多類別、多標籤、多輸出分類。

## 📝 本文目錄

- [🎯 關鍵重點](#key-takeaways)
- [📦 MNIST 資料集](#mnist)
- [🔢 訓練二元分類器](#binary-classifier)
- [📊 效能評估指標](#performance-measures)
- [🎯 精確率/召回率權衡](#precision-recall)
- [📈 ROC 曲線與 AUC](#roc)
- [🔢 多類別分類](#multiclass)
- [🔬 錯誤分析](#error-analysis)
- [🏷️ 多標籤與多輸出分類](#multilabel)
- [❓ 常見問答](#faq)
- [🏷️ 推薦標籤](#hashtags)

## <a id="key-takeaways"></a>🎯 關鍵重點 (Key Takeaways)

- **準確率（Accuracy）** 不是萬能指標——在不平衡資料集上可能嚴重誤導
- **混淆矩陣（Confusion Matrix）** 提供比準確率更細緻的效能視圖
- **精確率/召回率（Precision/Recall）** 是評估不平衡分類問題的核心指標
- **ROC-AUC** 評估分類器在所有閾值下的整體表現
- `cross_val_predict` + `confusion_matrix` 是標準的模型診斷流程

---

## <a id="mnist"></a>📦 MNIST 資料集

💡 **實際應用情境：** MNIST 被稱為 ML 的「Hello World」——70,000 張 28×28 灰階手寫數字圖片，是評估分類演算法的標準基準資料集。

### 範例 1: 載入並探索 MNIST

```python
from sklearn.datasets import fetch_openml
import numpy as np

# 載入 MNIST（首次需要聯網下載，會快取到本地）
mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
X, y = mnist["data"], mnist["target"]

print(f"特徵矩陣形狀: {X.shape}")  # (70000, 784) — 70k 張 28×28 圖片
print(f"標籤形狀: {y.shape}")       # (70000,) — 字串類別 '0'~'9'

# MNIST 已按慣例分割：前 60k 為訓練，後 10k 為測試
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# 標準化：將像素值從 [0, 255] 縮放到 [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0
```

**✅ 程式碼逐行解析：**

1. `fetch_openml("mnist_784", as_frame=False)`: 從 OpenML 下載資料集，`as_frame=False` 回傳 numpy 陣列
2. `X.shape = (70000, 784)`: 784 = 28 × 28，每張圖片被展平為一維向量
3. `y` 是字串類型（`'0'`, `'1'`, ...），需要在訓練前轉換

**🎯 重點摘要:**

- 像素值標準化（÷255）能加速梯度下降收斂
- MNIST 的固定分割使不同論文的結果可直接比較

---

## <a id="binary-classifier"></a>🔢 訓練二元分類器 (Binary Classifier)

💡 **實際應用情境：** 二元分類是最基礎的分類形式。以 MNIST 為例，「是否為數字 5」就是一個典型的二元問題，等同於工業檢測中的「良品/不良品」判斷。

### 範例 2: SGDClassifier 辨識數字 5

```python
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

# 建立二元標籤：是 5 → True，否則 → False
y_train_5 = (y_train == "5")
y_test_5  = (y_test  == "5")

# SGDClassifier（隨機梯度下降，適合大型資料集）
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train_5)

# 預測單張圖片
some_digit = X_test[0]
print(f"預測: {'是5' if sgd_clf.predict([some_digit])[0] else '不是5'}")

# 交叉驗證準確率
scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(f"3-Fold 準確率: {scores.mean():.4f}")  # 約 97%
# ⚠️ 危險！「全預測為非5」的笨模型也有 ~90% 準確率
```

**✅ 程式碼逐行解析：**

1. `y_train == "5"`: 向量化比較，建立布林陣列（10% 為 True）
2. `SGDClassifier(random_state=42)`: SGD 每次看一個樣本更新梯度，速度快，適合大型資料集
3. 97% 準確率看起來很好，但「全說不是5」也有90%——說明準確率在不平衡資料時會誤導

**🎯 重點摘要:**

- 準確率在類別不平衡（Imbalanced Classes）時毫無意義
- 需要更好的評估指標：混淆矩陣、精確率、召回率

---

## <a id="performance-measures"></a>📊 效能評估指標

### 範例 3: 混淆矩陣 (Confusion Matrix)

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 用交叉驗證取得每個樣本的「乾淨」預測（每個樣本只在驗證集時被預測一次）
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# 混淆矩陣
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)
# [[TN, FP],
#  [FN, TP]]
# 例如：[[53892, 687],   ← 非5：53892 正確，687 誤報為5
#         [1891, 3530]]  ← 是5：1891 漏報，3530 正確

# 視覺化
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["非5", "是5"])
disp.plot(cmap="Blues")
plt.title("SGDClassifier 混淆矩陣")
plt.show()
```

**✅ 程式碼逐行解析：**

1. `cross_val_predict(cv=3)`: 用 3-Fold CV 取得訓練集每個樣本的「跨折預測」，比直接在訓練集預測更能反映真實效能
2. 混淆矩陣的四個格：**TP**（真陽性）、**TN**（真陰性）、**FP**（假陽性/誤報）、**FN**（假陰性/漏報）

**🎯 重點摘要:**

- **FP（假警報）** 和 **FN（漏報）** 的業務代價往往不同（如醫療診斷漏報比假警報更嚴重）
- 混淆矩陣讓你看清模型的「錯誤模式」

---

## <a id="precision-recall"></a>🎯 精確率/召回率權衡

### 範例 4: 精確率、召回率、F1

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# 精確率（Precision）= TP / (TP + FP)：預測為正時，有多少真的是正的
precision = precision_score(y_train_5, y_train_pred)
print(f"精確率: {precision:.4f}")  # 約 0.84 — 預測為5的中，84%真的是5

# 召回率（Recall）= TP / (TP + FN)：真正是正的，被找出多少
recall = recall_score(y_train_5, y_train_pred)
print(f"召回率: {recall:.4f}")  # 約 0.65 — 真正是5的，只找出65%

# F1 分數：精確率和召回率的調和平均數
f1 = f1_score(y_train_5, y_train_pred)
print(f"F1 分數: {f1:.4f}")  # 約 0.73

# 精確率/召回率曲線
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# 找到精確率 ≥ 90% 的閾值
idx_90_precision = (precisions >= 0.90).argmax()
print(f"90% 精確率對應閾值: {thresholds[idx_90_precision]:.2f}")
print(f"對應召回率: {recalls[idx_90_precision]:.4f}")
```

**✅ 程式碼逐行解析：**

1. `precision_score`: 衡量「說是5有多可靠」——假警報少則精確率高
2. `recall_score`: 衡量「把5找完整有多好」——漏報少則召回率高
3. F1 = 2 × (P × R) / (P + R): 兩者的調和平均，當 P 和 R 都高時 F1 才高
4. `decision_function`: 回傳原始決策分數（而非二元預測），用於 PR 曲線計算

**🎯 重點摘要:**

- **精確率 vs 召回率的取捨**：提高閾值 → 精確率↑召回率↓；降低閾值 → 精確率↓召回率↑
- 業務決策決定如何設定閾值：垃圾郵件過濾（希望精確率高）vs 癌症篩查（希望召回率高）

---

## <a id="roc"></a>📈 ROC 曲線與 AUC

### 範例 5: ROC 曲線比較兩個分類器

```python
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# SGD 的 ROC
fpr_sgd, tpr_sgd, _ = roc_curve(y_train_5, y_scores)
auc_sgd = roc_auc_score(y_train_5, y_scores)

# 隨機森林（用 predict_proba 取機率分數）
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
)
fpr_forest, tpr_forest, _ = roc_curve(y_train_5, y_probas_forest[:, 1])
auc_forest = roc_auc_score(y_train_5, y_probas_forest[:, 1])

# 繪製比較圖
plt.figure(figsize=(8, 6))
plt.plot(fpr_sgd,    tpr_sgd,    "b-", linewidth=2, label=f"SGD (AUC={auc_sgd:.3f})")
plt.plot(fpr_forest, tpr_forest, "r-", linewidth=2, label=f"Random Forest (AUC={auc_forest:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="隨機分類器 (AUC=0.5)")
plt.xlabel("False Positive Rate (偽陽率)")
plt.ylabel("True Positive Rate (真陽率 / 召回率)")
plt.title("ROC 曲線比較")
plt.legend()
plt.grid(True)
plt.show()
print(f"SGD AUC: {auc_sgd:.3f},  Random Forest AUC: {auc_forest:.3f}")
```

**✅ 程式碼逐行解析：**

1. `roc_curve(y_true, y_scores)`: 在所有可能閾值下計算 FPR 和 TPR 組成 ROC 曲線
2. `roc_auc_score`: AUC 為 ROC 曲線下面積，範圍 [0, 1]，越高越好，0.5 = 隨機猜測
3. `predict_proba[:, 1]`: 取正類（是5）的機率作為分數

**🎯 重點摘要:**

- **何時用 PR 曲線**：正負樣本極不平衡時（ROC 曲線可能過於樂觀）
- **何時用 ROC**：正負樣本相對平衡時，AUC 是最常用的單一評估指標

---

## <a id="multiclass"></a>🔢 多類別分類 (Multiclass Classification)

### 範例 6: 辨識 0~9 所有數字

```python
from sklearn.svm import SVC
from sklearn.multiclass import OvRClassifier

# SVC 預設使用 OvO（One-vs-One）策略
# 10 類 → C(10,2) = 45 個二元分類器
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000])  # 先用小資料測試

print(f"預測結果: {svm_clf.predict([X_test[0]])}")

# 強制使用 OvR（One-vs-Rest）策略：10 個二元分類器
ovr_clf = OvRClassifier(SVC())
ovr_clf.fit(X_train[:1000], y_train[:1000])
print(f"OvR 分類器數量: {len(ovr_clf.estimators_)}")  # 10

# SGDClassifier 天生支援多類別（直接輸出 10 類）
sgd_multi = SGDClassifier(random_state=42, max_iter=1000)
sgd_multi.fit(X_train, y_train)
scores_multi = cross_val_score(sgd_multi, X_train, y_train, cv=3, scoring="accuracy")
print(f"SGD 多類別準確率: {scores_multi.mean():.4f}")
```

**✅ 程式碼逐行解析：**

1. **OvO（One-vs-One）**：每對類別訓練一個分類器，適合訓練較慢但對大資料效能好的分類器（如 SVM）
2. **OvR（One-vs-Rest）**：每個類別 vs 其他，10 個分類器，更常用
3. `SGDClassifier` 天生多類別：內部使用 OvR 策略

**🎯 重點摘要:**

- Sklearn 的大多數分類器直接支援多類別，無需手動設定
- SVM 預設 OvO，SGD/LogReg 預設 OvR

---

## <a id="error-analysis"></a>🔬 錯誤分析

### 範例 7: 混淆矩陣熱力圖診斷錯誤

```python
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# 取得 10 類分類的交叉驗證預測
y_train_pred_multi = cross_val_predict(sgd_multi, X_train, y_train, cv=3)

# 混淆矩陣正規化（每行除以該行總數）
cm = confusion_matrix(y_train, y_train_pred_multi)
cm_normalized = cm / cm.sum(axis=1, keepdims=True)  # 每行除以真實數量
np.fill_diagonal(cm_normalized, 0)  # 對角線設為0，聚焦於錯誤

# 視覺化
plt.figure(figsize=(8, 6))
plt.matshow(cm_normalized, cmap="hot", fignum=1)
plt.colorbar()
plt.title("正規化混淆矩陣（對角線設為0）")
plt.xlabel("預測類別")
plt.ylabel("真實類別")
plt.show()
# 亮格 = 常見錯誤，例如 3 和 5 互相混淆
```

**🎯 重點摘要:**

- 正規化混淆矩陣揭示「哪些類別最難分辨」（3↔5、4↔9、7↔1 最容易混淆）
- 根據錯誤模式決定改進方向：擴充容易混淆的訓練樣本、或加入更具區分力的特徵

---

## <a id="multilabel"></a>🏷️ 多標籤與多輸出分類

### 範例 8: 多標籤分類

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import numpy as np

# 建立多標籤目標：[是否 ≥ 7, 是否為奇數]
y_train_large = (y_train.astype(int) >= 7)  # 7, 8, 9
y_train_odd   = (y_train.astype(int) % 2 == 1)  # 1, 3, 5, 7, 9
y_multilabel  = np.c_[y_train_large, y_train_odd]

# KNN 支援多標籤
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train[:1000], y_multilabel[:1000])  # 用小資料示範

# 預測：回傳兩個布林值
pred = knn_clf.predict([X_test[0]])
print(f"預測 [≥7, 奇數]: {pred}")  # 例如 [[False, False]] 表示 <7 且 偶數

# 評估多標籤的 F1（每個標籤的 F1 加權平均）
# y_pred_multi = knn_clf.predict(X_train[:1000])
# print(f"多標籤 F1: {f1_score(y_multilabel[:1000], y_pred_multi, average='macro'):.3f}")
```

**✅ 程式碼逐行解析：**

1. `y_train_large = (y_train.astype(int) >= 7)`: 建立布林陣列（True/False）
2. `np.c_[...]`: 水平串接，建立 (n, 2) 形狀的多標籤矩陣
3. KNN 的多標籤預測：每個樣本輸出一個布林向量

**🎯 重點摘要:**

- 多標籤分類：每個樣本可同時屬於多個類別（如圖片同時含有貓和狗）
- 多輸出分類：輸出的每個標籤可以有多個可能值（更一般化的多標籤）

---

## <a id="faq"></a>❓ 常見問答 (FAQ)

**Q1: 準確率高是否代表模型好？**

A: 不一定。以 MNIST 的 5-detector 為例：若資料中只有 10% 是 5，「全部預測非5」能達到 90% 準確率，卻毫無用處。不平衡資料集應使用 F1 分數、AUC 等指標。

**Q2: 精確率和召回率哪個更重要？**

A: 取決於業務場景。癌症篩查優先召回率（漏掉一個病患代價高）；垃圾郵件過濾優先精確率（把正常郵件誤判為垃圾代價高）。可用 `fbeta_score(beta)` 調整側重比例。

**Q3: `cross_val_predict` 和 `cross_val_score` 的區別？**

A: `cross_val_score` 回傳每折的分數；`cross_val_predict` 回傳每個樣本的「乾淨」預測值（每個樣本只在作為驗證集時被預測），適合用來計算混淆矩陣。

**Q4: OvO 和 OvR 哪個更好？**

A: 一般 OvR 更常用（訓練和預測更快）。OvO 的優點是每個分類器只需要看兩個類別的資料，對某些二元分類器（如 SVM）的核計算更有效率。

---

## <a id="hashtags"></a>🏷️ 推薦標籤 (Suggested Hashtags)

\#Python #機器學習 #分類 #MNIST #ScikitLearn #混淆矩陣 #ROC曲線 #AUC #精確率 #召回率 #程式設計 #教學 #DataScience #MachineLearning #AI
