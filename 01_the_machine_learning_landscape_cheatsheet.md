# Ch01 速查表：Machine Learning Landscape

> **核心主旨**：ML 的分類體系、基本術語與核心挑戰 —— 選對演算法前先搞清楚問題的類型。

---

## 1. 核心概念一覽

| 概念 | 一句話說明 | 適用情境 |
|------|-----------|---------|
| Supervised Learning | 用有標籤資料訓練，學習 input → output 映射 | 分類、迴歸（房價、垃圾郵件） |
| Unsupervised Learning | 無標籤，自動發現資料結構 | 聚類、降維、異常偵測 |
| Semi-supervised Learning | 少量標籤 + 大量無標籤 | 圖片標注資料稀缺時 |
| Reinforcement Learning | Agent 透過 reward/penalty 學習策略 | 遊戲 AI、機器人控制 |
| Batch Learning | 離線使用全部資料訓練，定期重新訓練 | 資料量固定、不需即時更新 |
| Online Learning | 資料逐步到達，增量更新模型 | 串流資料、記憶體有限時 |
| Instance-based Learning | 直接記憶訓練資料，用相似度預測 | KNN |
| Model-based Learning | 從資料中學習模型參數 | 線性迴歸、決策樹 |
| Overfitting | 模型在訓練集表現好但泛化差 | 模型太複雜 / 資料太少 |
| Underfitting | 模型太簡單，連訓練集都學不好 | 需要更複雜模型或更多特徵 |

---

## 2. 關鍵 API 速查

| sklearn Class / Function | 重點參數 | 用途 |
|--------------------------|---------|------|
| `train_test_split()` | `test_size`, `random_state`, `stratify` | 切分資料集 |
| `cross_val_score()` | `cv=5`, `scoring="accuracy"` | K-Fold 交叉驗證 |
| `StandardScaler` | – | 特徵標準化（μ=0, σ=1） |
| `SimpleImputer` | `strategy="median"` | 填補缺失值 |
| `Pipeline` | `steps=[("scaler", ...), ("clf", ...)]` | 串接多個轉換器 |

---

## 3. 必備代碼片段

```python
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# 切分資料（分層抽樣）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 建立 Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# 交叉驗證
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
print(f"RMSE: {-scores.mean():.2f} ± {scores.std():.2f}")
```

---

## 4. 常見陷阱

- **資料洩漏（Data Leakage）**：StandardScaler 必須在 Pipeline 內，不能在 split 前 fit 整個資料集。
- **Snooping Bias**：多次在測試集上評估會讓測試集變成「驗證集」，最終指標過度樂觀。
- **不平衡類別**：`train_test_split` 加 `stratify=y` 確保類別比例一致。
- **測試集僅用一次**：調超參數用驗證集（或 CV），測試集只在最終評估時碰一次。

---

## 5. 決策指南

```
問題有標籤嗎？
├── 是 → Supervised
│   ├── 預測連續值 → 迴歸 (LinearRegression, RandomForest)
│   └── 預測類別 → 分類 (LogisticRegression, SVM, GBM)
└── 否 → Unsupervised
    ├── 找群體結構 → 聚類 (KMeans, DBSCAN)
    ├── 降低維度 → PCA, t-SNE
    └── 找異常點 → IsolationForest, EllipticEnvelope
```

**資料量考量**：
- 小資料（< 10k）：SVM、GBM 表現佳
- 大資料（> 100k）：深度學習、線性模型（SGD）更有效率
