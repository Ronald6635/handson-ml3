# 機器學習與深度學習實戰速查手冊

基於《Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition》

## 第1章：機器學習景觀

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **機器學習類型：監督學習、無監督學習、強化學習** | `無` | 機器學習的三種主要類型：監督學習基於標籤資料學習預測，無監督學習從未標籤資料中發現模式，強化學習通過試錯學習最佳策略。應用於分類、聚類、推薦系統、遊戲AI等。 | 無 | `無` |
| **機器學習專案的典型步驟** | `無` | 機器學習專案的標準流程：定義問題、獲取資料、探索資料、準備資料、選擇模型、訓練模型、微調模型、展示解決方案。應用於確保專案的系統性和有效性。 | 無 | `無` |
| **資料管道 (Data Pipeline)** | `sklearn.pipeline.Pipeline` | 將資料預處理和模型訓練結合的管道。應用於自動化機器學習工作流程，確保一致性和可重現性。 | steps, memory | `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())])
pipeline.fit(X_train, y_train)` |
| **交叉驗證** | `sklearn.model_selection.cross_val_score` | 評估模型在不同資料子集上的表現。應用於檢測過擬合和選擇最佳模型。 | cv, scoring | `from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)` |
| **常見陷阱：過擬合 vs. 欠擬合** | `無` | 過擬合：模型在訓練資料上表現良好，但在新資料上表現差；欠擬合：模型無法捕捉資料模式。最佳實踐：使用交叉驗證、正規化、增加資料量。 | 無 | `無` |

## 第2章：端到端機器學習專案

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **數據探索與分析函數** | `head(), info(), describe(), corr()` | 數據探索：檢查數據結構、統計摘要和相關性。應用於了解數據集特性和品質。 | N/A。 | `df.head()  # 查看前5行
df.info()  # 數據類型和非空值
df.describe()  # 統計摘要
df.corr()  # 相關性矩陣` |
| **超參數調優** | `GridSearchCV，RandomizedSearchCV` | 超參數調優：使用交叉驗證自動搜索最佳超參數組合。應用於尋找最佳模型參數。 | param_grid (參數網格)，n_iter (隨機搜索的迭代次數)，cv (交叉驗證折疊數)。 | `from sklearn.model_selection import GridSearchCV
GridSearchCV(estimator, param_grid, cv=5).fit(X, y)` |
| **數據預處理** | `Pipeline，ColumnTransformer，make_pipeline` | 將多個數據轉換步驟和模型串聯成一個序列，實現自動化流程。應用於處理混合數據類型 (數值和類別)。 | steps (包含的轉換器和最終模型列表)；memory (緩存目錄，用於加速重複調用)。 | `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())]).fit(X, y)` |
|  | `StandardScaler，MinMaxScaler，SimpleImputer，OneHotEncoder` | 數據預處理：標準化/縮放特徵，處理缺失值（SimpleImputer 使用 median 或 most_frequent 策略），將類別特徵轉換為數值（獨熱編碼）。 | strategy (Imputer，如 "median")；handle_unknown (OneHotEncoder，如 "ignore")；feature_range (MinMaxScaler)。 | `from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)` |
| **數據清理選項** | `dropna()，fillna()，drop_duplicates()` | 數據清理：處理缺失值、重複數據和異常值。應用於提高數據品質和模型性能。 | axis (刪除方向)，how (刪除條件)，value (填充值)，method (填充方法)。 | `# 刪除缺失值
df.dropna(axis=0, how='any')
# 填充缺失值
df.fillna(value=0)
# 刪除重複行
df.drop_duplicates()` |
| **特徵工程** | `pd.cut()，pd.qcut()，np.c_[]` | 特徵工程：創建新特徵，如分箱特徵、組合特徵等。應用於提升模型性能和捕捉複雜關係。 | bins (分箱數量)，labels (分箱標籤)，q (分位數)。 | `import pandas as pd
import numpy as np
# 分箱特徵
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
# 組合特徵
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']` |
| **數據分割** | `train_test_split` | 將數據集分割為訓練集和測試集。 | test_size, train_size, random_state, shuffle, stratify。 | `from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)` |
| **自定義數據分割函數** | `shuffle_and_split_data` | 自定義數據分割：實現自定義的數據分割邏輯，確保訓練和測試集的獨立性。 | test_ratio (測試集比例)，random_state (隨機種子)。 | `def shuffle_and_split_data(data, test_ratio, random_state=42):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]` |
| **分層分割** | `StratifiedShuffleSplit` | 分層隨機分割，確保每個折疊中的類別分佈與原始數據相同。 | n_splits, test_size, train_size, random_state。 | `from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)` |
| **交叉驗證** | `StratifiedKFold` | 分層 K 折交叉驗證。 | n_splits, shuffle, random_state。 | `from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
|  | `cross_val_predict, cross_val_score` | 交叉驗證預測和評分。 | estimator, X, y, cv, scoring。 | `from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator, X, y, cv=5, scoring='accuracy')` |
| **學習曲線** | `learning_curve` | 繪製學習曲線，顯示訓練和驗證分數隨樣本數量的變化。 | estimator, X, y, train_sizes, cv。 | `from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=5)` |
| **類別編碼** | `OrdinalEncoder` | 將類別特徵編碼為整數。 | categories, dtype。 | `from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder().fit(X)
X_encoded = encoder.transform(X)` |
| **Pandas數據處理** | `pd.concat()，pd.merge()，pd.pivot_table()` | Pandas數據處理：數據合併、連接和重塑。應用於數據整合和特徵工程。 | axis (合併軸向)，how (合併方式)，on (合併鍵)，values (聚合值)，index (行索引)，columns (列索引)。 | `import pandas as pd
# 合併數據框
pd.concat([df1, df2], axis=0)
# 連接數據
pd.merge(df1, df2, on='key', how='left')
# 數據透視表
pd.pivot_table(df, values='value', index='row', columns='col')` |
| **自定義轉換器** | `FunctionTransformer` | 使用任意可調用函數構造轉換器。 | func, inverse_func, validate。 | `from sklearn.preprocessing import FunctionTransformer
import numpy as np
transformer = FunctionTransformer(func=np.log1p).fit(X)` |
| **聚類相似度轉換器** | `ClusterSimilarity` | 聚類相似度轉換器：基於聚類質心的相似度計算新特徵。應用於將地理位置等特徵轉換為相似度分數。 | n_clusters (聚類數量)，gamma (相似度參數)，random_state (隨機種子)。 | `from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)` |
| **自定義估計器** | `BaseEstimator, TransformerMixin` | 基類，用於繼承自定義估計器。 | N/A。 | `from sklearn.base import BaseEstimator, TransformerMixin
class MyEstimator(BaseEstimator, TransformerMixin): pass` |
| **輸入驗證** | `check_array, check_is_fitted` | 輸入驗證工具。 | 各種參數。 | `from sklearn.utils.validation import check_array
X = check_array(X, ensure_2d=True)` |
| **目標變數轉換** | `TransformedTargetRegressor` | 目標變數轉換：對目標變數進行轉換的元估計器。應用於目標變數需要正規化或轉換的迴歸任務。 | regressor (基礎迴歸器)，transformer (目標變數轉換器)。 | `from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
ttr = TransformedTargetRegressor(regressor=LinearRegression(), transformer=StandardScaler())` |
| **數據下載** | `fetch_openml` | 從 OpenML 下載數據集。 | name, version, as_frame。 | `from sklearn.datasets import fetch_openml
data = fetch_openml('mnist_784', version=1, as_frame=False)` |

## 第3章：分類

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **混淆矩陣** | `confusion_matrix` | 計算真陽性、假陽性、真陰性、假陰性的矩陣：用於評估分類模型的基本性能，識別哪些類別容易混淆。 | y_true (真實標籤), y_pred (預測標籤), labels (類別標籤), sample_weight (樣本權重)。 | `from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)` |
| **精確率** | `precision_score` | 計算預測為正類的樣本中真正為正類的比例：適用於關注假陽性的場景，如垃圾郵件檢測。 | y_true (真實標籤), y_pred (預測標籤), average (多分類平均方式), pos_label (正類標籤)。 | `from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred)` |
| **召回率** | `recall_score` | 計算真正為正類的樣本中被正確預測的比例：適用於關注假陰性的場景，如疾病檢測。 | y_true (真實標籤), y_pred (預測標籤), average (多分類平均方式), pos_label (正類標籤)。 | `from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred)` |
| **F1 分數** | `f1_score` | 精確率和召回率的調和平均：用於平衡精確率和召回率的權衡，適用於不平衡數據集。 | y_true (真實標籤), y_pred (預測標籤), average (多分類平均方式), pos_label (正類標籤)。 | `from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)` |
| **ROC AUC 分數** | `roc_auc_score` | 計算ROC曲線下的面積：用於評估二分類器的整體性能，越接近1表示性能越好。 | y_true (真實標籤), y_score (預測分數), average (多分類平均方式), max_fpr (最大假陽率)。 | `from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_scores)` |
|  | `SGDClassifier，f1_score()，ROC curve` | 分類評估：隨機梯度下降分類器適用於大規模數據。性能評估使用 F1 分數 (Precision/Recall 權衡) 和 ROC 曲線。 | 調整決策閾值 (Threshold)；loss (e.g., "hinge")。 | `from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
clf = SGDClassifier(loss='hinge').fit(X, y)
f1 = f1_score(y_true, y_pred)` |
| **多分類** | `OneVsRestClassifier，OneVsOneClassifier` | 多分類任務：處理多個類別的分類，使用一對多或一對一策略將二分類器擴展到多分類。 | estimator (基礎二分類器)。 | `from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
ovr_clf = OneVsRestClassifier(SVC()).fit(X, y)` |
| **混淆矩陣分析** | `ConfusionMatrixDisplay` | 錯誤分析：詳細分析分類錯誤的分佈，識別哪些類別容易混淆。 | confusion_matrix (混淆矩陣)，display_labels (類別標籤)。 | `from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot()` |
| **精確率/召回率權衡** | `precision_recall_curve，PrecisionRecallDisplay` | 閾值調整：調整決策閾值以平衡精確率和召回率，適用於不平衡數據集。 | pos_label (正類標籤)，response_method (預測方法)。 | `from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)` |
| **ROC曲線** | `roc_curve，auc，RocCurveDisplay` | 分類器評估：繪製真陽率vs假陽率曲線，計算AUC面積評估分類器性能。 | pos_label (正類標籤)。 | `from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)` |
| **多標籤分類** | `MultiOutputClassifier，ClassifierChain` | 多標籤任務：每個樣本可以屬於多個類別的分類，使用二分類器組合處理。 | estimator (基礎分類器)。 | `from sklearn.multioutput import MultiOutputClassifier
multi_clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)` |
| **多輸出分類** | `MultiOutputClassifier` | 多輸出任務：同時預測多個相關輸出的分類任務。 | estimator (基礎分類器)。 | `from sklearn.multioutput import MultiOutputClassifier
multi_clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)` |
| **錯誤分析** | `cross_val_predict` | 模型診斷：使用交叉驗證預測分析模型錯誤模式，識別改進方向。 | cv (交叉驗證折數)，method (預測方法)。 | `from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)` |
| **羅吉斯迴歸** | `LogisticRegression` | 邏輯斯蒂回歸：應用於二分類和多分類任務，估計類別機率。 | C (正規化參數，控制正規化強度，值越大正規化越弱)；penalty ('l1', 'l2', 'elasticnet', 'l2'適用大多數情況)。 | `from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=1.0).fit(X, y)
probs = log_reg.predict_proba(X_test)` |

## 第4章：訓練線性模型

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **線性與廣義線性模型** | `LinearRegression` | 線性迴歸：使用 Normal Equation 或 SVD 直接計算最佳模型參數（最小化 MSE）。 | 無（封閉式解法）。 | `from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X, y)
y_pred = lr.predict(X_test)` |
| **Normal Equation** | `np.linalg.inv` | 手動實現線性回歸：使用矩陣運算直接計算參數，適用於小規模數據集。 | 無。 | `import numpy as np
X_b = np.c_[np.ones((100, 1)), X]  # 添加截距項
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)` |
| **Batch Gradient Descent** | `np.random.randn` | 批量梯度下降：使用整個訓練集計算梯度，適用於小規模數據集。 | eta (學習率)；n_iterations (迭代次數)。 | `eta = 0.1
n_iterations = 1000
m = 100
theta = np.random.randn(2, 1)
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients` |
| **Stochastic Gradient Descent** | `np.random.randint` | 隨機梯度下降：每次使用一個樣本計算梯度，適用於大規模數據集。 | eta (學習率)；n_epochs (訓練輪數)。 | `n_epochs = 50
t0, t1 = 5, 50
def learning_schedule(t):
    return t0 / (t + t1)
theta = np.random.randn(2, 1)
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients` |
| **Mini-batch Gradient Descent** | `np.random.permutation` | 小批量梯度下降：使用小批量樣本計算梯度，平衡效率和穩定性。 | batch_size (批量大小)；eta (學習率)。 | `batch_size = 20
n_iterations = 1000
theta = np.random.randn(2, 1)
for iteration in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, batch_size):
        xi = X_b_shuffled[i:i+batch_size]
        yi = y_shuffled[i:i+batch_size]
        gradients = 2/batch_size * xi.T.dot(xi.dot(theta) - yi)
        theta = theta - eta * gradients` |
| **學習曲線** | `learning_curve` | 學習曲線分析：繪製訓練和驗證分數隨樣本數量的變化，檢測過擬合和欠擬合。 | estimator (估計器)；X, y (數據)；train_sizes (訓練樣本比例)。 | `from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')` |
| **Early Stopping** | `SGDRegressor` | 早期停止：監控驗證誤差，當誤差不再改善時停止訓練，防止過擬合。 | validation_fraction (驗證集比例)；n_iter_no_change (無改善的最大迭代次數)。 | `from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X_train, y_train)
# 手動實現早期停止
best_score = float('inf')
for epoch in range(1000):
    sgd_reg.partial_fit(X_train, y_train)
    y_val_predict = sgd_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < best_score:
        best_score = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)` |
| **(Linear & Generalized Linear Models)** | `Ridge，Lasso，ElasticNet` | 正規化線性模型：分別使用 ℓ_2、ℓ_1 或兩者組合來約束權重，以減少過度擬合。 | 正規化強度 α (alpha)：值越大，約束越強。 | `from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1).fit(X, y)` |
| **隨機梯度下降** | `SGDRegressor` | 線性模型，使用 SGD 訓練：適用於大規模數據集的線性迴歸，通過隨機梯度下降優化參數。 | loss ('squared_error', 'huber'，'squared_error'適用標準迴歸)；penalty ('l2', 'l1', 'elasticnet')；alpha (正規化強度)；learning_rate ('constant', 'optimal', 'invscaling')；eta0 (初始學習率)。 | `from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(loss='squared_error').fit(X, y)` |
| **目標變數轉換** | `TransformedTargetRegressor` | 對目標變數進行轉換的元估計器：適用於目標變數需要正規化或轉換的迴歸任務。 | regressor (基礎迴歸器，如 LinearRegression)；transformer (目標變數轉換器，如 StandardScaler)。 | `from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
ttr = TransformedTargetRegressor(regressor=LinearRegression(), transformer=StandardScaler())` |
| **特徵工程** | `add_dummy_feature` | 向數據集添加截距列：適用於線性模型需要顯式截距項的情況。 | X (輸入數據矩陣)；value (添加的截距值，通常為1.0)。 | `from sklearn.preprocessing import add_dummy_feature
X_with_dummy = add_dummy_feature(X, value=1.0)` |
|  | `PolynomialFeatures` | 生成多項式和交互特徵：適用於捕捉非線性關係和特徵交互的迴歸任務。 | degree (多項式維度，值越大複雜度越高)；interaction_only (是否只生成交互項)；include_bias (是否包含偏置項)。 | `from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2).fit_transform(X)` |
| **多項式回歸** | `PolynomialFeatures + LinearRegression` | 多項式回歸：使用多項式特徵進行線性回歸，捕捉非線性關係。 | degree (多項式維度)。 | `from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)` |
| **迴歸指標** | `mean_squared_error, root_mean_squared_error` | 均方誤差和均方根誤差：評估迴歸模型性能的標準指標。 | y_true (真實值)；y_pred (預測值)。 | `from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)` |

## 第5章：支援向量機

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **支持向量機** | `SVC，LinearSVC，SVR` | SVM 分類與迴歸：大邊界分類 (Large Margin Classification) 尋找最優決策邊界。使用 核心技巧 (Kernel Trick) 處理非線性數據，適用於文本分類、圖像識別、生物資訊學等任務。 | C (正規化參數，控制邊界寬度，值越大越嚴格)；kernel ('rbf', 'poly', 'linear'，'rbf'適用大多數非線性問題)；gamma (RBF核心參數，'scale'自動調整，值越大越複雜)。 | `from sklearn.svm import SVC
svc = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X, y)` |
| **(Support Vector Machines, SVMs)** | `hinge loss function` | SVM 理論基礎：硬邊界與軟邊界分類的數學原理。通過對偶問題優化求解，理解SVM的核心概念和損失函數設計。 | N/A；相關參數為 C 和 gamma。 | N/A |
|  | `LinearSVR` | 線性支持向量迴歸：適用於線性可分的迴歸任務，比標準SVR更高效。適合大規模數據集的迴歸預測。 | epsilon (不敏感區間寬度，控制容忍度)；C (正規化參數)；loss ('epsilon_insensitive' 或 'squared_epsilon_insensitive')。 | `from sklearn.svm import LinearSVR
svr = LinearSVR(epsilon=0.1, C=1.0).fit(X, y)` |
| **核心函數** | `rbf_kernel` | RBF 核心函數：高斯徑向基函數，將數據映射到無限維空間。適用於大多數非線性分類任務，特別適合複雜的決策邊界。 | gamma (核心係數，控制影響範圍，值越大局部性越強)；X, Y (輸入數據矩陣)。 | `from sklearn.metrics.pairwise import rbf_kernel
K = rbf_kernel(X, Y, gamma=1.0)` |
| **多項式核心** | `SVC(kernel="poly")` | 多項式核心函數：將數據映射到高階多項式特徵空間。適用於非線性分類任務，特別適合具有多項式決策邊界的數據。 | degree (多項式階數，預設3)；coef0 (獨立項係數，控制高階項影響)；C (正規化參數)；gamma (核心係數)。 | `from sklearn.svm import SVC
poly_svm = SVC(kernel="poly", degree=3, coef0=1, C=5).fit(X, y)` |
| **相似性特徵** | `gaussian_rbf` | 相似性特徵：基於高斯RBF函數的相似性度量，將數據映射到高維空間。適用於非線性SVM的核技巧實現。 | gamma (控制鐘形曲線寬度，值越大影響範圍越小)；landmark (參考點坐標)；x (輸入數據點)。 | `def gaussian_rbf(x, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1)**2)

similarity_features = gaussian_rbf(X, landmark, gamma=0.3)` |
| **高斯RBF核心** | `SVC(kernel="rbf")` | 高斯RBF核心：使用高斯徑向基函數作為核函數的SVM。適用於大多數非線性分類任務，特別適合複雜且未知的決策邊界。 | gamma (核心係數，'scale'自動調整，值越大越複雜)；C (正規化參數，控制邊界寬度)。 | `from sklearn.svm import SVC
rbf_svm = SVC(kernel="rbf", gamma=5, C=0.001).fit(X, y)` |
| **SVM迴歸** | `SVR` | 支持向量迴歸：SVM的迴歸版本，尋找最大邊界的迴歸線。適用於非線性迴歸任務，對異常值具有魯棒性。 | kernel ('rbf', 'poly', 'linear')；C (正規化參數)；epsilon (不敏感區間寬度)；gamma (RBF核心參數)。 | `from sklearn.svm import SVR
svr = SVR(kernel="rbf", C=1.0, epsilon=0.1).fit(X, y)` |
| **Hinge損失函數** | `hinge loss` | Hinge損失：SVM分類的損失函數，懲罰分類錯誤的樣本。數學公式：max(0, 1 - t·s)，其中t是真實標籤，s是決策函數值。 | N/A；相關參數為 C (控制懲罰強度)。 | `# Hinge loss calculation
hinge_loss = np.maximum(0, 1 - t * s)` |
| **平方Hinge損失函數** | `squared hinge loss` | 平方Hinge損失：Hinge損失的平方版本，對異常值更敏感。數學公式：max(0, 1 - t·s)²，常用於LinearSVC的預設損失函數。 | N/A；相關參數為 C (控制懲罰強度)。 | `# Squared hinge loss calculation
squared_hinge_loss = np.maximum(0, 1 - t * s) ** 2` |
| **自定義SVM實現** | `MyLinearSVC` | 從頭實現線性SVM：使用批量梯度下降實現線性SVM分類器。適用於學習SVM內部機制和自定義優化。 | C (正規化參數)；eta0 (初始學習率)；n_epochs (訓練輪數)。 | `class MyLinearSVC(BaseEstimator):
    def __init__(self, C=1, eta0=1, n_epochs=1000):
        self.C = C
        self.eta0 = eta0
        self.n_epochs = n_epochs

    def fit(self, X, y):
        # Batch gradient descent implementation
        # ... (implementation details)` |

## 第6章：決策樹

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **決策樹與集成方法** | `DecisionTreeClassifier/Regressor` | 決策樹：白箱模型。利用 CART 演算法 和 Gini 不純度/熵 進行分割。 | max_depth (最大深度，控制樹的深度以防止過擬合)；min_samples_leaf (葉節點最小樣本數，用於正則化)。 | `from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5).fit(X, y)` |
|  | `load_iris` | 載入鳶尾花數據集，用於決策樹訓練和可視化：適用於分類任務的標準數據集。 | as_frame (是否返回DataFrame格式)；return_X_y (是否分離特徵和標籤)。 | `from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target` |
| **預測方法** | `predict_proba` | 估計類別概率，用於決策樹的概率預測：適用於需要概率輸出的分類任務。 | X (輸入特徵矩陣)。 | `proba = dt.predict_proba(X_test)
print(proba)  # 輸出每個類別的概率` |
|  | `predict` | 進行類別預測，用於決策樹的分類輸出：適用於標準分類任務。 | X (輸入特徵矩陣)。 | `y_pred = dt.predict(X_test)
print(y_pred)  # 輸出預測類別` |
| **正則化參數** | `min_samples_leaf` | 葉節點最小樣本數，用於控制樹的複雜度：適用於防止過擬合，提高泛化能力。 | min_samples_leaf (葉節點最小樣本數，整數或比例)。 | `dt_reg = DecisionTreeClassifier(min_samples_leaf=5)
dt_reg.fit(X, y)` |
| **決策樹回歸** | `DecisionTreeRegressor` | 決策樹回歸器，用於連續值預測：適用於回歸任務，處理非線性關係。 | max_depth (最大深度)；min_samples_split (最小分割樣本數)。 | `from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth=3).fit(X, y)` |
| **PCA預處理** | `make_pipeline(StandardScaler, PCA)` | 使用PCA對數據進行預處理，以處理軸方向敏感性：適用於改善決策樹對旋轉數據的魯棒性。 | n_components (主成分數量)；whiten (是否白化)。 | `from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pca_pipe = make_pipeline(StandardScaler(), PCA())
X_pca = pca_pipe.fit_transform(X)` |
| **Gini不純度** | `Gini impurity` | Gini不純度公式，用於評估節點的純度：$G = 1 - \sum_{k=1}^{K} p_k^2$，適用於CART演算法的分割標準。 | criterion='gini' (分割標準)。 | `# Gini不純度計算
def gini_impurity(counts):
    total = sum(counts)
    return 1 - sum((count/total)**2 for count in counts)` |
| **樹結構訪問** | `tree_` | 訪問決策樹的內部結構，用於分析和調試：適用於檢查樹的屬性，如深度、節點數等。 | tree_.node_count (節點數)；tree_.max_depth (最大深度)。 | `tree = dt.tree_
print(f"節點數: {tree.node_count}")
print(f"最大深度: {tree.max_depth}")` |
| **超參數調優** | `GridSearchCV` | 網格搜索交叉驗證，用於決策樹超參數調優：適用於系統性地搜索最佳參數組合。 | param_grid (參數網格)；cv (交叉驗證折數)。 | `from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [3, 5, 7]}
grid = GridSearchCV(dt, param_grid, cv=5).fit(X, y)` |
| **多數投票** | `mode (from scipy.stats)` | 多數投票聚合，用於集成決策樹的預測：適用於隨機森林等集成方法的實現。 | axis (投票軸)。 | `from scipy.stats import mode
y_pred_ensemble = mode(y_pred_matrix, axis=0)[0]` |
|  | `export_graphviz` | 將決策樹導出為 GraphViz DOT 格式文件，用於可視化：適用於理解和展示決策樹的結構。 | out_file (輸出文件路徑)；feature_names (特徵名稱列表)；class_names (類別名稱列表)；rounded (是否圓角)；filled (是否填充顏色)。 | `from sklearn.tree import export_graphviz
export_graphviz(dt, out_file='tree.dot', feature_names=feature_names)` |
| **決策樹可視化** | `Source.from_file` | 從 DOT 文件讀取並顯示決策樹圖形，用於可視化 export_graphviz 導出的樹結構：適用於在 Jupyter Notebook 中展示決策樹的可視化圖形。 | filepath (DOT 文件路徑)。 | `from graphviz import Source
graph = Source.from_file('tree.dot')
graph.view()` |
| **數據生成** | `make_moons` | 生成月亮形數據集，用於測試非線性分類：適用於測試決策樹等非線性模型的性能。 | n_samples (樣本數)；noise (噪聲標準差)；random_state (隨機種子)。 | `from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)` |
| **交叉驗證** | `ShuffleSplit` | 隨機分割數據集，用於交叉驗證：適用於隨機分割訓練和測試集，確保模型評估的可靠性。 | n_splits (分割次數)；train_size (訓練集比例)；test_size (測試集比例)；random_state (隨機種子)。 | `from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)` |

## 第7章：集成學習和隨機森林

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **決策樹與集成學習** | `RandomForestClassifier，BaggingClassifier，VotingClassifier` | 隨機森林：通過 Bagging 結合多棵決策樹，有效降低變異性。VotingClassifier 結合不同模型的預測結果。 | n_estimators (樹的數量)；max_features (每次分割時隨機採樣的特徵數)；voting 類型 ('hard' 或 'soft')。 | `from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100).fit(X, y)` |
| **梯度提升** | `GradientBoostingRegressor/Classifier` | 梯度提升 (Boosting)：序列訓練，每一步都糾正前一個模型的殘差誤差，以降低偏差：適用於提高模型準確率，特別適合處理複雜的非線性關係。 | learning_rate (學習率，控制每棵樹的貢獻)；n_estimators (樹的數量)；max_depth (每棵樹的最大深度)。 | `from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1).fit(X, y)` |
| **AdaBoost** | `AdaBoostClassifier` | AdaBoost 分類器：通過迭代訓練弱學習器並調整樣本權重來提高整體性能。 | n_estimators (弱學習器的數量)；learning_rate (學習率)；algorithm (演算法類型，如 'SAMME')。 | `from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50).fit(X, y)` |
| **極端隨機樹** | `ExtraTreesClassifier` | 極端隨機樹分類器：通過隨機選擇分割特徵來增加多樣性，提高泛化能力。 | n_estimators (樹的數量)；criterion (分割標準，如 'gini')；max_depth (最大深度)。 | `from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators=100).fit(X, y)` |
| **直方圖梯度提升** | `HistGradientBoostingRegressor` | 基於直方圖的梯度提升迴歸樹：使用直方圖近似來加速訓練，適合大規模數據。 | learning_rate (學習率)；max_iter (最大迭代次數)；max_leaf_nodes (葉節點的最大數量)。 | `from sklearn.ensemble import HistGradientBoostingRegressor
hgb = HistGradientBoostingRegressor(max_iter=100).fit(X, y)` |
| **堆疊** | `StackingClassifier` | 堆疊估計器：通過訓練元模型來組合多個基礎模型的預測結果。 | estimators (基礎估計器列表)；final_estimator (最終估計器)；cv (交叉驗證折疊數)。 | `from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
stack = StackingClassifier(estimators=[('rf', RandomForestClassifier()), ('gb', GradientBoostingClassifier())], final_estimator=LogisticRegression())` |
| **包外評估** | `oob_score，oob_decision_function_` | 包外評估：使用未參與訓練的樣本評估Bagging模型性能，無需額外驗證集。 | oob_score (是否啟用包外評估)。 | `bag_clf = BaggingClassifier(DecisionTreeClassifier(), oob_score=True, random_state=42)
bag_clf.fit(X_train, y_train)
print(bag_clf.oob_score_)
print(bag_clf.oob_decision_function_[:3])` |
| **特徵重要性** | `feature_importances_` | 特徵重要性：衡量每個特徵在隨機森林中的貢獻程度，用於特徵選擇。 | 無特定參數，屬性返回重要性分數數組。 | `from sklearn.datasets import load_iris
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)
for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
    print(f'importance: {round(score, 2)}, feature: {name}')` |
| **提前停止** | `n_iter_no_change` | 提前停止：當驗證分數在指定迭代次數內無改善時停止訓練，防止過擬合。 | n_iter_no_change (無改善迭代次數閾值)。 | `gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=500,
    n_iter_no_change=10, random_state=42)
gbrt.fit(X, y)
print(gbrt.n_estimators_)  # 實際使用的估計器數量` |
| **分類特徵管道處理** | `make_pipeline，OrdinalEncoder` | 使用管道處理分類特徵：結合編碼器和HistGradientBoostingRegressor處理混合數據類型。 | categorical_features (分類特徵位置)。 | `from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
hgb_reg = make_pipeline(
    make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
                            remainder="passthrough"),
    HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
)` |

## 第8章：降維

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **降維與流形學習** | `PCA，RandomizedPCA，KernelPCA` | 主成分分析 (PCA)：最流行降維演算法。通過 SVD 找出保留最大變異數的超平面：適用於數據可視化、噪聲去除和特徵提取。 | n_components (目標維度或變異數比例)；svd_solver (SVD求解器，如 'randomized' 用於加速)。 | `from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)` |
| **(Dimensionality Reduction)** | `LocallyLinearEmbedding (LLE)` | 流形學習：應用於處理非線性數據集（如 Swiss Roll）：適用於揭示數據的低維流形結構。 | n_neighbors (近鄰數)；max_iter (最大迭代次數)。 | `from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2).fit_transform(X)` |
| **增量PCA** | `IncrementalPCA` | 增量主成分分析：適用於無法一次性載入全部數據的大規模數據集。 | n_components (目標維度)；whiten (是否白化數據)。 | `from sklearn.decomposition import IncrementalPCA
ipca = IncrementalPCA(n_components=2).fit(X)` |
| **線性判別分析** | `LinearDiscriminantAnalysis` | 線性判別分析：同時進行降維和分類的監督學習方法。 | solver (求解器)；shrinkage (收縮參數)；n_components (目標維度)。 | `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)` |
| **Isomap** | `Isomap` | Isomap 嵌入：基於測地線距離的非線性降維方法。 | n_neighbors (近鄰數)；n_components (目標維度)。 | `from sklearn.manifold import Isomap
iso = Isomap(n_neighbors=10, n_components=2).fit_transform(X)` |
| **多維縮放** | `MDS` | 多維縮放：通過保留成對距離來進行降維。 | n_components (目標維度)；metric (是否使用度量距離)。 | `from sklearn.manifold import MDS
mds = MDS(n_components=2).fit_transform(X)` |
| **t-SNE** | `TSNE` | t-分佈隨機鄰域嵌入：適用於高維數據的可視化，特別適合聚類分析。 | n_components (目標維度)；perplexity (困惑度)；learning_rate (學習率)。 | `from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30).fit_transform(X)` |
| **隨機投影** | `GaussianRandomProjection, SparseRandomProjection` | 隨機投影：使用隨機矩陣進行快速降維，基於Johnson-Lindenstrauss引理。 | n_components (目標維度)；eps (誤差容忍度)。 | `from sklearn.random_projection import GaussianRandomProjection
grp = GaussianRandomProjection(n_components=2).fit_transform(X)` |
| **Johnson-Lindenstrauss引理** | `johnson_lindenstrauss_min_dim` | 找到安全的隨機投影維度：根據Johnson-Lindenstrauss引理計算最小維度。 | n_samples (樣本數)；eps (誤差容忍度)。 | `from sklearn.random_projection import johnson_lindenstrauss_min_dim
min_dim = johnson_lindenstrauss_min_dim(n_samples=1000, eps=0.1)` |

## 第9章：無監督學習

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **異常檢測** | `IsolationForest` | 異常檢測演算法：通過隔離森林來識別異常數據點。 | n_estimators (樹的數量)；contamination (污染比例)；random_state (隨機種子)。 | `from sklearn.ensemble import IsolationForest
clf = IsolationForest(random_state=42).fit(X)` |
| **無監督學習核心演算法** | `KMeans，DBSCAN，GaussianMixture (GMM)` | 聚類分析 (Clustering)：用於客戶分群、異常偵測。DBSCAN 基於密度，GMM 基於概率分佈：適用於發現數據中的自然群組結構。 | n_clusters (聚類數，適用於KMeans和GMM)；eps (密度半徑，適用於DBSCAN)；min_samples (最小樣本數，適用於DBSCAN)。 | `from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(X)` |
| **層次聚類** | `AgglomerativeClustering` | 凝聚聚類：通過自下而上合併聚類來構建層次結構。 | n_clusters (聚類數)；affinity (距離度量)；linkage (鏈接方法)。 | `from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3).fit(X)` |
| **增量聚類** | `MiniBatchKMeans` | 小批量 K-Means 聚類：適用於大規模數據集的增量聚類。 | n_clusters (聚類數)；batch_size (批次大小)。 | `from sklearn.cluster import MiniBatchKMeans
mbk = MiniBatchKMeans(n_clusters=3, batch_size=100).fit(X)` |
| **圖論聚類** | `SpectralClustering` | 譜聚類：使用圖論中的譜圖分割來進行聚類。 | n_clusters (聚類數)；affinity (親和度矩陣構造方式)。 | `from sklearn.cluster import SpectralClustering
sc = SpectralClustering(n_clusters=3).fit(X)` |
| **聚類評估** | `silhouette_samples, silhouette_score` | 輪廓係數計算：評估聚類質量的指標，值越接近1表示聚類越好。 | X (數據矩陣)；labels (聚類標籤)；metric (距離度量)。 | `from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)` |
| **概率聚類** | `BayesianGaussianMixture` | 變分貝葉斯高斯混合：自動確定最佳組件數的概率聚類方法。 | n_components (組件數)；covariance_type (協方差類型)。 | `from sklearn.mixture import BayesianGaussianMixture
bgm = BayesianGaussianMixture(n_components=3).fit(X)` |

## 第10章：使用Keras的神經網路

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **多層感知器** | `MLPClassifier` | 多層感知器分類器：使用反向傳播訓練的神經網路分類器。 | hidden_layer_sizes (隱藏層大小)；activation (激活函數)；solver (優化器)。 | `from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,)).fit(X, y)` |
| **深度學習核心組件** | `tf.keras.models.Sequential，tf.keras.layers.Dense` | 序列模型 (Sequential API)：構建 MLP 網路。全連接層：適用於構建簡單的前饋神經網路。 | units (神經元數量)；activation (激活函數)；kernel_initializer (權重初始化器)。 | `import tensorflow as tf
model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu')])` |
| **感知器和多層感知器** | `Perceptron, MLPClassifier, MLPRegressor` | 感知器和多層感知器：基本的線性分類器和多層神經網路。 | hidden_layer_sizes (隱藏層大小)；activation (激活函數)；solver (優化器)。 | `from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,)).fit(X, y)` |

## 第11章：訓練深度神經網路

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **(Deep Learning Core Components)** | `tf.keras.layers.BatchNormalization` | 批次正規化 (BN)：加速訓練並緩解梯度不穩定性：適用於穩定訓練過程，提高收斂速度。 | momentum (移動平均動量)；axis (正規化軸)。 | `model.add(tf.keras.layers.BatchNormalization())` |
| **正則化技術** | `tf.keras.layers.Dropout，tf.keras.regularizers.l1/l2` | 正則化技術：Dropout 減少神經元共適應，提升泛化能力。L1/L2 懲罰項約束權重：適用於防止過擬合，提高模型泛化性能。 | rate (丟棄率)；factor (正規化因子)。 | `model.add(tf.keras.layers.Dropout(0.5))` |

## 第12章：使用TensorFlow的自訂模型和訓練

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **TensorFlow 底層與客製化** | `tf.GradientTape，@tf.function` | 自動微分 (Autodiff)：記錄計算過程以計算梯度。使用 @tf.function 將 Python 函數轉換為 TensorFlow Graph 進行優化：適用於自定義訓練循環和複雜的梯度計算。 | 無特定參數，主要用於梯度計算上下文。 | `import tensorflow as tf
with tf.GradientTape() as tape:
    loss = compute_loss()
    grads = tape.gradient(loss, variables)` |
| **張量與操作** | `tf.Variable，tf.constant，tf.placeholder` | 張量與操作：Variable (可變參數)；constant (常量)；placeholder (外部輸入)：TensorFlow的基本數據結構和操作。 | 無特定參數，這些是基本張量類型。 | `var = tf.Variable([1.0, 2.0])
const = tf.constant([3.0, 4.0])` |

## 第13章：載入和預處理資料

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **資料API** | `tf.data.Dataset` | 高效能的資料載入和預處理管道。應用於大規模資料集的批次處理和預取：適用於優化數據輸入管道，提高訓練效率。 | buffer_size (緩衝區大小)；batch_size (批次大小)；prefetch (預取數量)。 | `dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(1)` |
| **數據管道與特徵預處理** | `tf.data.Dataset，tf.data.TextLineDataset，interleave()` | 高效數據加載：處理大規模數據 (如 CSV, TFRecords)。使用 interleave() 進行並行讀取以提高效率 ：適用於處理多個文件和大規模數據集。 | cycle_length (並行讀取的文件數)；batch_size (批次大小)。 | `dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)` |
| **(Data Pipelines & Feature Prep)** | `tf.keras.layers.Embedding，TextVectorization，CategoryEncoding` | 特徵預處理層：將文本/類別特徵轉換為數值向量。詞嵌入 (Word Embeddings) 處理高維文本：適用於NLP和類別數據的自動預處理。 | output_dim (嵌入維度)；output_mode (輸出模式，如"tf_idf")。 | `embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)` |
|  | `tensorflow.train` | 用於創建 TFRecord 文件的工具：適用於高效存儲和讀取大規模數據。 | 無特定參數，用於TFRecord文件操作。 | `writer = tf.io.TFRecordWriter('data.tfrecord')` |
|  | `tensorflow_datasets` | 數據集集合，可與 TensorFlow 一起使用：提供標準化的數據集加載接口。 | 無特定參數，提供標準數據集。 | `import tensorflow_datasets as tfds
ds = tfds.load('mnist')` |
|  | `tensorflow_hub` | 訓練過的機器學習模型倉庫：適用於遷移學習和模型重用。 | 無特定參數，提供預訓練模型。 | `import tensorflow_hub as hub
model = hub.load('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4')` |
| **文字預處理** | `tf.strings` | 將原始文字轉換為數值特徵的預處理層。應用於NLP任務的文字向量化：適用於文本數據的自動處理和轉換。 | output_mode (輸出模式)；max_tokens (最大詞彙數)。 | `text_vec_layer = keras.layers.TextVectorization(
    output_mode="tf_idf", max_tokens=1000)
text_vec_layer.adapt(train_texts)` |
| **影像預處理** | `tf.image` | 常見的影像轉換和增強操作。應用於電腦視覺任務的資料預處理：適用於圖像數據的標準化處理。 | size (目標大小)；method (插值方法)。 | `def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    return resized_image, label
dataset = dataset.map(preprocess)` |

## 第14章：使用CNN的深度電腦視覺

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **建立CNN** | `keras.layers.Conv2D` | 建立卷積神經網路進行影像特徵提取。應用於影像分類、物體檢測等電腦視覺任務：適用於處理圖像數據的特徵學習。 | filters (濾波器數量)；kernel_size (卷積核大小)；strides (步長)；padding (填充方式)；activation (激活函數)。 | `model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="same",
                        input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])` |
| **使用預訓練模型進行遷移學習** | `keras.applications.ResNet50` | 利用預訓練的深度網路進行遷移學習。應用於小資料集上的高效能模型訓練：適用於利用現有知識快速適應新任務。 | weights (權重類型)；include_top (是否包含分類頭)；input_shape (輸入形狀)。 | `base_model = keras.applications.ResNet50(weights="imagenet",
                                         include_top=False)
base_model.trainable = False
model = keras.models.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax")
])` |

## 第15章：使用RNN和CNN處理序列

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **建立RNN** | `keras.layers.SimpleRNN` | 建立簡單的遞歸神經網路處理序列資料。應用於時間序列預測和序列分類：適用於基本的序列建模任務。 | units (神經元數量)；return_sequences (是否返回所有時間步)；input_shape (輸入形狀)。 | `model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1)
])` |
| **使用LSTM** | `keras.layers.LSTM` | 使用長短期記憶網路處理長序列。應用於需要記憶長期依賴的序列任務：適用於處理長距離依賴關係。 | units (神經元數量)；return_sequences (是否返回所有時間步)；input_shape (輸入形狀)。 | `model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20),
    keras.layers.Dense(1)
])` |
| **使用GRU** | `keras.layers.GRU` | 使用門控遞歸單元處理序列資料。應用於計算效率和效能平衡的序列任務：適用於需要平衡性能和效率的序列處理。 | units (神經元數量)；return_sequences (是否返回所有時間步)；input_shape (輸入形狀)。 | `model = keras.models.Sequential([
    keras.layers.GRU(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.GRU(20),
    keras.layers.Dense(1)
])` |

## 第16章：使用RNN和注意力進行NLP

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **字元級RNN模型** | `keras.layers.GRU` | 建立字元級的文字生成模型。應用於文字生成和語言模型任務：適用於生成連續文本和語言建模。 | units (神經元數量)；return_sequences (是否返回所有時間步)；input_shape (輸入形狀)。 | `model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id]),
    keras.layers.GRU(128),
    keras.layers.Dense(max_id, activation="softmax")
])` |
| **使用遮罩** | `keras.layers.Masking` | 處理可變長度序列的遮罩層。應用於處理填充序列時忽略填充值：適用於處理不等長序列數據。 | mask_value (遮罩值)；input_shape (輸入形狀)。 | `model = keras.models.Sequential([
    keras.layers.Masking(mask_value=0., input_shape=[None, 1]),
    keras.layers.GRU(20),
    keras.layers.Dense(1)
])` |
| **使用預訓練的詞嵌入** | `keras.layers.Embedding` | 將詞彙轉換為密集向量表示。應用於捕捉詞語語義關係的NLP任務：適用於學習詞語的語義嵌入。 | input_dim (詞彙大小)；output_dim (嵌入維度)；input_length (輸入序列長度)。 | `model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embed_size, input_length=max_length),
    keras.layers.GRU(128),
    keras.layers.Dense(1, activation="sigmoid")
])` |

## 第17章：自動編碼器、GAN和擴散模型

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **線性自動編碼器** | `keras.models.Sequential` | 學習資料壓縮和重建的無監督模型。應用於降維和特徵學習：適用於學習數據的緊湑表示。 | input_shape (輸入形狀)；units (編碼器神經元數量)。 | `encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[784])])
decoder = keras.models.Sequential([keras.layers.Dense(784, input_shape=[2])])
autoencoder = keras.models.Sequential([encoder, decoder])` |
| **卷積自動編碼器** | `keras.layers.Conv2D` | 使用卷積層的自動編碼器。應用於影像壓縮和生成任務：適用於圖像數據的壓縮和重建。 | filters (濾波器數量)；kernel_size (卷積核大小)；strides (步長)；padding (填充方式)。 | `conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten()
])
conv_decoder = keras.models.Sequential([
    keras.layers.Reshape([7, 7, 16], input_shape=[49*16]),
    keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="same", activation="relu"),
    keras.layers.Reshape([28, 28, 1])
])` |
| **GAN** | `keras.models.Sequential` | 生成對抗網路學習資料分佈。應用於影像生成和風格轉換：適用於生成逼真的新數據樣本。 | units (神經元數量)；activation (激活函數)；input_shape (輸入形狀)。 | `codings_size = 30
generator = keras.models.Sequential([
    keras.layers.Dense(100, activation="relu", input_shape=[codings_size]),
    keras.layers.Dense(150, activation="relu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(150, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])` |

## 第18章：強化學習

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **OpenAI Gym** | `gym.make()` | 強化學習的標準環境介面。應用於測試和開發RL演算法：適用於標準化RL環境的交互。 | env_name (環境名稱)。 | `import gym
env = gym.make("CartPole-v1")
obs = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        break` |
| **神經網路策略** | `keras.models.Sequential` | 使用神經網路學習策略函數。應用於複雜環境的策略梯度方法：適用於學習最優行動策略。 | units (神經元數量)；activation (激活函數)；input_shape (輸入形狀)。 | `n_inputs = 4  # 環境觀測
model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid")
])` |

## 第19章：大規模訓練和部署

| 關鍵概念 | 重要函數 | 功能簡介與應用場景 | Fine-tune 用參數 | 範例代碼片段 |
|:---|:---|:---|:---|:---|
| **規模化與部署** | `tf.distribute.MirroredStrategy，MultiWorkerMirroredStrategy` | 分散式訓練：在多 GPU 或多台機器上進行數據并行化 (Data Parallelism)，以加速大規模模型的訓練：適用於大規模深度學習模型的訓練加速。 | batch_size (批次大小，需被副本數整除)。 | `strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_model()` |
| **(Scaling & Deployment)** | `TFLiteConverter，tf.saved_model.save，Keras Tuner` | 部署與優化：將模型轉換為 TFLite 格式以部署到行動/嵌入式設備；使用 Keras Tuner 進行超參數自動優化：適用於模型壓縮和自動超參數調優。 | max_trial_count (最大嘗試次數)；搜索範圍 (超參數搜索空間)。 | `converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_path')
tflite_model = converter.convert()` |
| **使用TensorFlow Serving進行服務** | `tf.saved_model.save()` | 將訓練好的模型儲存為SavedModel格式。應用於生產環境的模型部署：適用於TensorFlow Serving等服務框架。 | model (要保存的模型)；export_dir (導出目錄)。 | `tf.saved_model.save(model, "/tmp/my_mnist_model/1/")` |
| **使用Google Cloud AI Platform進行預測** | `googleapiclient.discovery.build` | 在雲端平台部署和服務機器學習模型。應用於大規模的模型推論服務：適用於雲端AI服務的模型部署。 | service_name (服務名稱)；version (版本)；credentials (憑證)。 | `from googleapiclient import discovery
ai_platform = discovery.build('ml', 'v1')
response = ai_platform.projects().predict(
    name=project_name,
    body={'instances': instances}
).execute()` |
|  | `google.cloud.aiplatform` | Vertex AI 用於機器學習：適用於Google Cloud的端到端ML平台。 | 無特定參數，Vertex AI 服務接口。 | `from google.cloud import aiplatform
aiplatform.init(project='my-project')` |
|  | `hypertune` | 超參數調優庫。 | N/A。 | `import hypertune
hpt = hypertune.HyperTune()` |
|  | `keras_tuner` | Keras 模型的超參數調優。 | N/A。 | `import keras_tuner as kt
tuner = kt.Hyperband(build_model, objective='val_accuracy')` |
|  | `tensorboard` | TensorFlow 可視化工具包。 | N/A。 | `import tensorboard
%tensorboard --logdir logs` |
|  | `tqdm` | 進度條庫。 | N/A。 | `from tqdm import tqdm
for i in tqdm(range(100)): pass` |
|  | `transformers` | 最新的自然語言處理庫。 | N/A。 | `from transformers import pipeline
classifier = pipeline('sentiment-analysis')` |
|  |  | Python 圖像處理庫。 | N/A。 | `from PIL import Image
img = Image.open('image.jpg')` |
|  | `nltk` | 自然語言處理工具包。 | N/A。 | `import nltk
nltk.download('punkt')` |
|  | `graphviz` | 圖形可視化軟體。 | N/A。 | `import graphviz
dot = graphviz.Digraph()` |

