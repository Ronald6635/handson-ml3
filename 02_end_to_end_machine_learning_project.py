"""
端到端機器學習專案 (End-to-End Machine Learning Project) 模組

本模組演示了從數據獲取到模型部署的完整機器學習專案流程。

主要特點:
- 數據獲取與載入
- 探索性數據分析 (EDA) 與可視化
- 使用 Scikit-Learn Pipeline 進行數據前處理
- 模型訓練、評估與微調

範例基於 Markdown 文件 02_end_to_end_machine_learning_project.md
"""

import sys
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tarfile
import urllib.request
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from scipy import stats

# =============================================================================
# EXAMPLE 1: 環境設定
# =============================================================================
print("=== 範例 1: 環境設定 ===")

# 檢查 Python 版本
assert sys.version_info >= (3, 7)
print(f"Python 版本: {sys.version}")

# 檢查 Scikit-Learn 版本
assert sklearn.__version__ >= "1.0.1"
print(f"Scikit-Learn 版本: {sklearn.__version__}")

# 設定 Matplotlib 繪圖樣式
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
print("Matplotlib 繪圖樣式設定完成。")

# =============================================================================
# EXAMPLE 2: 獲取數據
# =============================================================================
print("\n=== 範例 2: 獲取數據 ===")

def load_housing_data() -> pd.DataFrame:
    """
    下載並載入加州房價數據集。
    
    如果數據集不存在，會自動從網路上下載並解壓縮。
    
    Returns:
        pd.DataFrame: 包含房價數據的 DataFrame
        
    Raises:
        urllib.error.URLError: 當無法下載數據時
        FileNotFoundError: 當解壓縮後找不到 CSV 檔案時
        
    Example:
        >>> housing_data = load_housing_data()
        >>> print(housing_data.shape)
    """
    tarball_path: Path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        print("下載數據中...")
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url: str = "https://github.com/ageron/handson-ml3/raw/main/datasets/housing.tgz"
        try:
            urllib.request.urlretrieve(url, tarball_path)
            with tarfile.open(tarball_path) as housing_tarball:
                housing_tarball.extractall(path="datasets")
            print("下載並解壓縮完成。")
        except urllib.error.URLError as e:
            raise urllib.error.URLError(f"無法下載數據: {e}")
    
    csv_path: Path = Path("datasets/housing/housing.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到數據檔案: {csv_path}")
    
    return pd.read_csv(csv_path)

housing: pd.DataFrame = load_housing_data()
print("房價數據載入成功。")

# =============================================================================
# EXAMPLE 3: 數據初步探索
# =============================================================================
print("\n=== 範例 3: 數據初步探索 ===")

print("--- 數據前五行 (head) ---")
print(housing.head())

print("\n--- 數據資訊 (info) ---")
housing.info()

print("\n--- 'ocean_proximity' 欄位類別計數 ---")
print(housing["ocean_proximity"].value_counts())

print("\n--- 數值欄位描述性統計 (describe) ---")
print(housing.describe())

# =============================================================================
# EXAMPLE 4: 繪製數據分佈直方圖
# =============================================================================
print("\n=== 範例 4: 繪製數據分佈直方圖 ===")
# 由於這是在腳本中運行，我們將保存圖像而不是顯示它
Path("images").mkdir(parents=True, exist_ok=True)
housing.hist(bins=50, figsize=(12, 8))
plt.suptitle("所有數值特徵的直方圖")
plt.savefig("images/housing_histograms.png")
print("直方圖已保存至 images/housing_histograms.png")
plt.close() # 關閉圖形以釋放記憶體

# 為了後續步驟，我們需要創建分層抽樣所需的 'income_cat' 欄位
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# 創建訓練集與測試集 (使用分層抽樣確保收入分佈的代表性)
strat_train_set: pd.DataFrame
strat_test_set: pd.DataFrame
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

# 移除 'income_cat' 欄位 (這是臨時創建的輔助欄位)
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# 創建一個探索用的數據副本 (避免意外修改原始訓練數據)
housing = strat_train_set.copy()

# =============================================================================
# EXAMPLE 5: 地理位置散點圖
# =============================================================================
print("\n=== 範例 5: 地理位置散點圖 ===")
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.title("加州房價地理分佈圖 (依人口與房價中位數)")
plt.xlabel("經度 (Longitude)")
plt.ylabel("緯度 (Latitude)")
plt.savefig("images/housing_geographical_plot.png")
print("地理位置散點圖已保存至 images/housing_geographical_plot.png")
plt.close()

# =============================================================================
# EXAMPLE 6, 7, 8: 數據前處理 Pipeline
# =============================================================================
print("\n=== 範例 6, 7, 8: 數據前處理 Pipeline ===")

# 從訓練集中分離特徵和標籤
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels: pd.Series = strat_train_set["median_house_value"].copy()

# 選擇數值和類別欄位
housing_num: pd.DataFrame = housing.select_dtypes(include=[np.number])
num_attribs: List[str] = list(housing_num)
cat_attribs: List[str] = ["ocean_proximity"]

# 建立數值特徵的處理 pipeline
# 步驟1: 填補缺失值 (使用中位數)
# 步驟2: 特徵縮放 (標準化)
num_pipeline: Pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# 建立完整的欄位轉換器
# 對數值欄位應用 num_pipeline
# 對類別欄位應用 OneHotEncoder
full_pipeline: ColumnTransformer = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(sparse_output=False), cat_attribs),
])

# 應用 pipeline 到數據上
housing_prepared: np.ndarray = full_pipeline.fit_transform(housing)
print("數據前處理完成。")
print(f"處理後數據的維度: {housing_prepared.shape}")

# =============================================================================
# EXAMPLE 9: 訓練線性回歸模型
# =============================================================================
print("\n=== 範例 9: 訓練線性回歸模型 ===")

lin_reg: LinearRegression = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
print("線性回歸模型訓練完成。")

# 測試一下模型 (選取前5個樣本進行預測測試)
some_data: pd.DataFrame = housing.iloc[:5]
some_labels: pd.Series = housing_labels.iloc[:5]
some_data_prepared: np.ndarray = full_pipeline.transform(some_data)
predictions: np.ndarray = lin_reg.predict(some_data_prepared)
print("預測值:", predictions.round(2))
print("實際值:", list(some_labels))

# =============================================================================
# EXAMPLE 10: 使用交叉驗證評估模型
# =============================================================================
print("\n=== 範例 10: 使用交叉驗證評估模型 ===")

# 評估線性回歸 (使用10折交叉驗證獲得更穩健的性能評估)
lin_scores: np.ndarray = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores: np.ndarray = np.sqrt(-lin_scores)
print("線性回歸 RMSE 分數:", lin_rmse_scores.round(2))
print(f"平均 RMSE: {lin_rmse_scores.mean():.2f}")
print(f"標準差: {lin_rmse_scores.std():.2f}")

# 訓練並評估決策樹 (可能會過擬合，但提供對比基準)
tree_reg: DecisionTreeRegressor = DecisionTreeRegressor(random_state=42)
tree_scores: np.ndarray = cross_val_score(tree_reg, housing_prepared, housing_labels,
                                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores: np.ndarray = np.sqrt(-tree_scores)
print("\n決策樹 RMSE 分數:", tree_rmse_scores.round(2))
print(f"平均 RMSE: {tree_rmse_scores.mean():.2f}")
print(f"標準差: {tree_rmse_scores.std():.2f}")

# 訓練並評估隨機森林 (通常性能較好，能減少過擬合)
forest_reg_pipeline = make_pipeline(full_pipeline, RandomForestRegressor(random_state=42))
forest_scores: np.ndarray = -cross_val_score(forest_reg_pipeline, housing, housing_labels,
                                           scoring="neg_root_mean_squared_error", cv=10)
print("\n隨機森林 RMSE 分數:", forest_scores.round(2))
print(f"平均 RMSE: {forest_scores.mean():.2f}")
print(f"標準差: {forest_scores.std():.2f}")

# =============================================================================
# EXAMPLE 11: 使用網格搜索進行超參數調優
# =============================================================================
print("\n=== 範例 11: 使用網格搜索進行超參數調優 ===")

# 建立包含前處理和模型的完整 Pipeline
full_model_pipeline = Pipeline([
    ("preprocessing", full_pipeline),
    ("random_forest", RandomForestRegressor(random_state=42)),
])

# 定義超參數搜索空間 (組合會產生大量的模型訓練)
param_grid: List[Dict[str, Any]] = [
    # 嘗試 12 (3×4) 種 n_estimators 和 max_features 的組合
    {'random_forest__n_estimators': [3, 10, 30], 'random_forest__max_features': [2, 4, 6, 8]},
    # 然後嘗試 6 (2×3) 種 bootstrap 為 False 的組合
    {'random_forest__bootstrap': [False], 'random_forest__n_estimators': [3, 10], 'random_forest__max_features': [2, 3, 4]},
]

# 使用 5 折交叉驗證，總共會訓練 (12 + 6) * 5 = 90 次
grid_search: GridSearchCV = GridSearchCV(full_model_pipeline, param_grid, cv=5,
                                        scoring='neg_mean_squared_error',
                                        return_train_score=True)

grid_search.fit(housing, housing_labels)

print("網格搜索完成。")
print("最佳超參數組合:", grid_search.best_params_)
print("最佳估計器:", grid_search.best_estimator_)

# 顯示每個組合的評估分數 (幫助理解不同參數的影響)
cvres: Dict[str, np.ndarray] = grid_search.cv_results_
print("\n--- 網格搜索結果 ---")
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    rmse_score: float = np.sqrt(-mean_score)
    print(f"RMSE: {rmse_score:.2f}, 參數組合: {params}")

# =============================================================================
# EXAMPLE 12: 在測試集上評估最終模型
# =============================================================================
print("\n=== 範例 12: 在測試集上評估最終模型 ===")

# 從網格搜索中獲取最佳模型 (已經過超參數調優)
final_model = grid_search.best_estimator_

# 準備測試集數據 (分離特徵和目標變數)
X_test: pd.DataFrame = strat_test_set.drop("median_house_value", axis=1)
y_test: pd.Series = strat_test_set["median_house_value"].copy()

# **重要**: 這裡直接使用 fit 好的 final_model (包含 pipeline)
# 它會自動對 X_test 進行 transform
final_predictions: np.ndarray = final_model.predict(X_test)

# 計算最終的 RMSE (模型在未見過數據上的泛化性能)
final_mse: float = mean_squared_error(y_test, final_predictions)
final_rmse: float = np.sqrt(final_mse)
print(f"最終模型在測試集上的 RMSE: {final_rmse:.2f}")

# 計算 95% 信賴區間 (提供性能估計的不確定性範圍)
confidence: float = 0.95
squared_errors: np.ndarray = (final_predictions - y_test) ** 2

# 使用 stats.bootstrap (需要 SciPy 1.7.0+)
try:
    def rmse_func(data, axis):
        return np.sqrt(np.mean(data, axis=axis))
    
    res = stats.bootstrap((squared_errors,), rmse_func, confidence_level=0.95, random_state=42)
    confidence_interval = res.confidence_interval
    print(f"RMSE 的 {confidence*100}% 信賴區間 (bootstrap): [{confidence_interval.low:.2f}, {confidence_interval.high:.2f}]")

except (AttributeError, TypeError):
    # Fallback for older SciPy versions
    t_score = stats.t.ppf((1 + confidence) / 2, len(squared_errors) - 1)
    sem = stats.sem(squared_errors)
    margin_of_error = t_score * sem
    mean_se = np.mean(squared_errors)
    confidence_interval_manual = np.sqrt([mean_se - margin_of_error, mean_se + margin_of_error])
    print(f"RMSE 的 {confidence*100}% 信賴區間 (manual t-dist): [{confidence_interval_manual[0]:.2f}, {confidence_interval_manual[1]:.2f}]")


print("\n=== 所有範例執行完畢 ===")
