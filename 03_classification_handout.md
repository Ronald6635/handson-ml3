# 課程講義：分類 (Chapter 03)

本章聚焦於機器學習中的**分類問題**，以 MNIST 手寫數字資料集為實驗場域。你將發現：看似直觀的「準確率」在面對不平衡資料時完全失效，而精確率、召回率、F1 分數、ROC 曲線才是評估分類器的真正工具。掌握這些指標，是工業界機器學習工程師的必備技能。

---

## 1. 訓練二元分類器與混淆矩陣

### 理論背景

二元分類（Binary Classification）：將樣本判斷為正類（Positive）或負類（Negative）。

**混淆矩陣 (Confusion Matrix)**：

$$\begin{pmatrix} TN & FP \\ FN & TP \end{pmatrix}$$

- **TP (True Positive)**：實際為正，預測為正（正確）
- **TN (True Negative)**：實際為負，預測為負（正確）
- **FP (False Positive)**：實際為負，預測為正（型一誤差，False Alarm）
- **FN (False Negative)**：實際為正，預測為負（型二誤差，Miss）

**準確率的陷阱**：若正類只佔 1%（不平衡資料），一個永遠預測負類的分類器準確率高達 99%，但完全無用！

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### 核心代碼

```python
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 載入 MNIST
mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target

# 分割（MNIST 前 60000 為訓練集）
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# 建立二元分類器（判斷是否為數字 5）
y_train_5 = (y_train == "5")
y_test_5  = (y_test  == "5")

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# 用交叉驗證取得預測（不偷看測試集）
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)
ConfusionMatrixDisplay(cm).plot()
```

### ⚡ 補充練習 1

**理論題：** 醫療診斷中（陽性=有病），FP 和 FN 哪個代價更高？對應到垃圾郵件過濾（陽性=垃圾郵件），答案有何不同？

**實作題：** 建立一個「笨分類器」：永遠預測 `False`（非 5）。計算它的準確率，並與 `SGDClassifier` 比較，解釋為何準確率在此不是好指標。

---

## 2. 精確率、召回率與 F1 分數

### 理論背景

從混淆矩陣衍生的更有意義的指標：

$$\text{精確率 (Precision)} = \frac{TP}{TP + FP}$$

$$\text{召回率 (Recall) / 靈敏度 (Sensitivity)} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

**精確率-召回率的權衡**：

- 降低決策閾值 → 召回率↑、精確率↓（找到更多正例，但誤報增加）
- 提高決策閾值 → 精確率↑、召回率↓（更保守，但可能漏掉正例）

選擇哪個指標取決於業務目標：

- 影片內容審查（避免誤殺好內容）→ 偏重精確率
- 癌症篩查（避免漏掉病患）→ 偏重召回率

### 核心代碼

```python
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              precision_recall_curve, roc_curve, roc_auc_score)
import matplotlib.pyplot as plt

# 基本指標
print(f"Precision: {precision_score(y_train_5, y_train_pred):.3f}")
print(f"Recall:    {recall_score(y_train_5, y_train_pred):.3f}")
print(f"F1 Score:  {f1_score(y_train_5, y_train_pred):.3f}")

# 取得決策分數（非二元預測）
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5,
                             cv=3, method="decision_function")

# 精確率-召回率曲線
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1],    "g-",  label="Recall")
plt.xlabel("Threshold")
plt.legend()
plt.title("Precision-Recall vs Threshold")
plt.grid()

# 設定目標精確率，找出對應閾值
idx_90_precision = (precisions >= 0.90).argmax()
threshold_90_precision = thresholds[idx_90_precision]
y_pred_90 = (y_scores >= threshold_90_precision)
print(f"Precision @90: {precision_score(y_train_5, y_pred_90):.3f}")
print(f"Recall  @90:   {recall_score(y_train_5, y_pred_90):.3f}")
```

### ⚡ 補充練習 2

**理論題：** 若模型的精確率為 0.8、召回率為 0.6，計算 $F_1$ 分數。若業務目標是「召回率至少 95%，在此前提下精確率越高越好」，應如何設定閾值？

**實作題：** 繪製 Precision-Recall 曲線（以召回率為 x 軸，精確率為 y 軸），並計算曲線下面積（Average Precision）。找出使 $F_1$ 最大化的閾值。

---

## 3. ROC 曲線與 AUC

### 理論背景

**ROC 曲線 (Receiver Operating Characteristic)**：

- x 軸：偽陽性率（FPR）$= \frac{FP}{TN + FP}$，即 1 - 特異度
- y 軸：真陽性率（TPR）$= \frac{TP}{TP + FN}$，即召回率

**AUC (Area Under Curve)**：

- AUC = 1.0 → 完美分類器
- AUC = 0.5 → 隨機猜測（對角線）
- AUC 越大越好

**ROC vs Precision-Recall 曲線選擇**：

- 正負類別**嚴重不平衡**時，PR 曲線更具鑑別力（ROC 曲線可能過於樂觀）
- 兩類別比例均衡時，ROC 曲線更常用

### 核心代碼

```python
from sklearn.metrics import roc_curve, roc_auc_score

# ROC 曲線
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, "b-",  linewidth=2, label=f"SGD AUC={roc_auc_score(y_train_5, y_scores):.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR = Recall)")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# 與 RandomForestClassifier 比較
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(rf_clf, X_train, y_train_5,
                                   cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]  # 正類的機率
print(f"RF ROC AUC: {roc_auc_score(y_train_5, y_scores_forest):.4f}")
```

### ⚡ 補充練習 3

**理論題：** 在同一張圖上繪製 SGDClassifier 與 RandomForestClassifier 的 ROC 曲線，從圖形判斷哪個分類器更好。AUC 的物理意義是什麼（從機率角度解釋）？

**實作題：** 使用 `classification_report` 輸出完整的分類報告（Precision、Recall、F1 for each class），並解讀 `macro avg` 與 `weighted avg` 的差異。

---

## 4. 多類別與多標籤分類

### 理論背景

**多類別分類 (Multi-class)**：每個樣本屬於一個類別（如 MNIST 的 0-9）。

- **OvR (One-vs-Rest)**：訓練 10 個二元分類器，取分數最高者
- **OvO (One-vs-One)**：訓練 $C(n,2)$ 個分類器（每對類別一個），適合 SVM 等對樣本數敏感的算法

**多標籤分類 (Multi-label)**：每個樣本可同時屬於多個類別（如圖片標籤：{人臉, 微笑, 帽子}）。

**多輸出分類 (Multi-output)**：每個輸出可以是多類別（如影像去噪：每個像素的灰階值）。

### 核心代碼

```python
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# SVC 預設使用 OvO 策略
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000])  # 用子集加速
print(svm_clf.predict([X_test[0]]))          # 預測數字類別

# 多標籤分類（判斷是否為大數字 & 是否為奇數）
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train.astype(int) >= 7)
y_train_odd   = (y_train.astype(int) % 2 == 1)
y_multilabel  = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
print(knn_clf.predict([X_test[0]]))  # [[True, False]] → 7 是大數字但不是奇數（非7的情況）
```

### ⚡ 補充練習 4

**理論題：** 對 10 個類別的多類別問題，OvR 需要訓練幾個分類器？OvO 需要幾個？當訓練速度很慢時，哪個策略比較好？

**實作題：** 對完整 MNIST（10 類）訓練 `SGDClassifier`（預設使用 OvR），計算測試集的混淆矩陣，繪製熱力圖（`plt.matshow`），找出最常被混淆的數字對。

---

## 結論

本章揭示了分類問題評估的完整工具箱：

- **混淆矩陣**是一切指標的起點
- **精確率、召回率、F1** 比準確率更能反映不平衡資料的模型效能
- **ROC / AUC** 是模型比較的通用基準
- **閾值調整**允許根據業務目標在精確率與召回率之間取捨

下一章（Ch04）深入線性模型的數學原理，理解梯度下降如何最小化損失函數。

---

## 課後作業

**作業：MNIST 錯誤分析與改進**

1. 訓練 `SGDClassifier` 在完整 MNIST（10 類），計算測試集 `f1_score(average="macro")`。
2. 繪製**正規化後的混淆矩陣**（每列除以該類別的真實樣本數），用顏色深淺顯示，找出模型最常搞混的兩個數字。
3. 針對最容易混淆的那對數字（例如 4 和 9），查看被分類錯誤的圖片，思考：能否透過特徵工程（如計算影像的某種統計量）幫助區分？
