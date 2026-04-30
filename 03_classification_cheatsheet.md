# Ch03 速查表：Classification

> **核心主旨**：分類任務的評估指標 —— 準確率只是起點，Precision/Recall/ROC-AUC 才是關鍵。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Binary Classification | 二元分類，輸出 0 或 1 | 垃圾郵件偵測、詐欺偵測 |
| Confusion Matrix | TP/FP/TN/FN 的完整統計 | 模型錯誤分析 |
| Precision | $\frac{TP}{TP+FP}$：預測為正中有多少真的是正 | 假陽性代價高時（如垃圾郵件） |
| Recall (Sensitivity) | $\frac{TP}{TP+FN}$：真正的正有多少被找出來 | 假陰性代價高時（如癌症篩檢） |
| F1 Score | Precision 與 Recall 的調和平均 | 兩者都重要時 |
| ROC-AUC | 不同閾值下的 TPR vs FPR 曲線面積 | 比較分類器整體能力 |
| PR Curve | Precision-Recall 曲線 | 類別極度不平衡時優先看 |
| OvR / OvO | 多類別策略：One-vs-Rest / One-vs-One | 將二元分類器用於多類別 |

---

## 2. 關鍵 API 速查

| sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `SGDClassifier` | `loss="hinge"/"log_loss"`, `random_state=42` | 線性分類器（支援 SGD） |
| `cross_val_predict` | `cv=3`, `method="decision_function"` | 取每個樣本的 OOF 預測 |
| `confusion_matrix` | – | 輸出 TP/FP/TN/FN 矩陣 |
| `precision_score` | `average="binary"/"macro"/"weighted"` | 精確率 |
| `recall_score` | `average="binary"` | 召回率 |
| `f1_score` | – | F1 Score |
| `roc_auc_score` | – | ROC AUC |
| `precision_recall_curve` | – | PR 曲線資料點 |
| `roc_curve` | – | ROC 曲線資料點 |
| `classification_report` | – | 一次印出所有指標 |
| `RandomForestClassifier` | `n_estimators=100` | 隨機森林分類 |

---

## 3. 必備代碼片段

```python
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              precision_recall_curve, roc_curve)
import matplotlib.pyplot as plt

# 訓練分類器
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)  # y_train_5: True/False

# 交叉驗證取 OOF 預測
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5,
                              cv=3, method="decision_function")

# 混淆矩陣與各項指標
y_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_pred)
print(f"Precision: {precision_score(y_train_5, y_pred):.3f}")
print(f"Recall:    {recall_score(y_train_5, y_pred):.3f}")
print(f"F1:        {f1_score(y_train_5, y_pred):.3f}")
print(f"ROC AUC:   {roc_auc_score(y_train_5, y_scores):.3f}")
print(classification_report(y_train_5, y_pred))

# PR 曲線
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plt.plot(recalls, precisions)
plt.xlabel("Recall"); plt.ylabel("Precision")

# 調整決策閾值
threshold_90_precision = thresholds[precisions >= 0.90][0]
y_pred_90 = (y_scores >= threshold_90_precision)
```

---

## 4. 常見陷阱

- **準確率的陷阱**：極度不平衡資料（如 99% 負例），全預測負例也有 99% 準確率，但完全沒用。
- **`cross_val_score` vs `cross_val_predict`**：前者給摘要分數，後者給每個樣本的 OOF 預測（才能畫 PR/ROC 曲線）。
- **多類別指標 `average` 參數**：`"macro"` 對每個類別平均處理，`"weighted"` 依樣本數加權，二元分類用 `"binary"`。
- **`roc_auc_score` 需要機率/分數**：不能直接傳 0/1 預測，要傳 `decision_function()` 或 `predict_proba()` 的輸出。

---

## 5. 決策指南

```
評估指標的選擇：
├── 假陽性代價高（如垃圾郵件誤判） → 最大化 Precision
├── 假陰性代價高（如癌症漏診）     → 最大化 Recall
├── 兩者平衡                        → F1 Score
└── 整體比較多個模型                → ROC-AUC

類別不平衡時：
├── 優先看 PR Curve（而非 ROC Curve）
├── 使用 class_weight="balanced"
└── 考慮 SMOTE 過採樣（需 imblearn 套件）
```

**精確率 vs 召回率的取捨**：
- 一個分類器不能同時最大化 precision 和 recall
- 提高閾值 → precision ↑, recall ↓
- 降低閾值 → precision ↓, recall ↑
