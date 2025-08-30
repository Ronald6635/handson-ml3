<!-- meta-title: 🐍 端到端機器學習專案實戰：從數據到模型的完整指南 -->
<!-- meta-description: 本教學將引導您完成一個完整的端到端機器學習專案。我們將從獲取真實世界的房價數據開始，一路進行數據探索、預處理、模型訓練、微調，最終部署一個能夠預測房價的模型。這是一個專為初學者到中級開發者設計的實戰演練。 -->
<!-- meta-keywords: Python, 機器學習, Scikit-Learn, Pandas, 數據科學, 端到端專案, 房價預測, 數據分析, 模型訓練 -->
<!-- meta-hashtags: #Python #機器學習 #數據科學 #ScikitLearn #Pandas #專案實戰 #房價預測 #技術教學 #程式設計 -->

# 🐍 端到端機器學習專案實戰

歡迎來到您的第二個機器學習專案！本章節將引導您完成一個完整的端到端（End-to-End）專案，模擬真實世界中數據科學家的工作流程。我們將使用加州房價數據集，從數據獲取、探索、準備，到模型選擇、訓練、微調，最終評估我們的模型。

## 📝 本文目錄
- [專案設定與環境準備](#setup)
- [獲取數據](#get-data)
- [探索與可視化數據](#explore-data)
- [數據前處理](#prepare-data)
- [選擇與訓練模型](#select-train-model)
- [模型微調](#fine-tune-model)
- [評估最終模型](#evaluate-model)

## 🎯 關鍵重點 (Key Takeaways)
- **完整流程**: 學習一個典型機器學習專案從頭到尾的完整步驟。
- **數據處理**: 掌握使用 Pandas 進行數據清洗、處理缺失值、特徵縮放和轉換的關鍵技術。
- **模型訓練**: 了解如何使用 Scikit-Learn 訓練多種回歸模型，如線性回歸、決策樹和隨機森林。
- **模型評估**: 學習使用交叉驗證（Cross-Validation）來更穩健地評估模型性能。
- **超參數調優**: 探索網格搜索（Grid Search）和隨機搜索（Randomized Search）等方法來找到模型的最佳超參數。

## <a id="setup"></a>🚀 專案設定與環境準備
💡 **實際應用情境：** 在任何專案開始之前，確保您的開發環境已經準備就緒是至關重要的第一步。這包括安裝必要的 Python 版本、函式庫，並設定好專案的工作目錄。一個穩定且一致的環境可以避免許多不必要的問題。

### 範例 1: 環境設定
```python
# Python ≥3.7 is required
import sys
assert sys.version_info >= (3, 7)

# Scikit-Learn ≥1.0.1 is required
from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

# To plot pretty figures
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

**✅ 程式碼逐行解析：**
1.  `第 2-3 行`: 檢查並斷言（assert）當前的 Python 版本是否大於等於 3.7。如果版本不符，程式將會中斷並報錯，這確保了程式碼在兼容的環境中運行。
2.  `第 6-7 行`: 同樣地，檢查 Scikit-Learn 函式庫的版本是否大於等於 1.0.1。這是為了確保我們能使用到特定版本之後的新功能或修復。
3.  `第 10-16 行`: 設定 Matplotlib 繪圖的預設樣式。`%matplotlib inline` 是一個 Jupyter Notebook 的魔法指令，它讓繪圖結果直接顯示在儲存格下方。後續的 `plt.rc` 指令則是為了讓圖表有更好的一致性與可讀性，統一設定了字體大小、座標軸標籤大小等。

**🎯 重點摘要:**
- **核心功能**: 建立一個穩定且版本一致的開發環境，並設定好繪圖的基礎樣式。
- **潛在問題**: 如果環境中的函式庫版本不符，可能會導致程式碼無法執行或出現預期外的錯誤。
- **最佳使用情境**: 在任何 Python 專案或 Notebook 的開頭進行此類設定，以確保可重現性（reproducibility）。

## <a id="get-data"></a>📥 獲取數據
💡 **實際應用情境：** 機器學習專案的第一步通常是獲取數據。數據可以來自資料庫、API、或是像本範例一樣的壓縮檔。我們需要編寫函式來自動化下載和解壓縮的過程，這樣不僅方便自己，也方便他人重現我們的專案。

### 範例 2: 下載並載入數據
```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    # 設定數據檔案的路徑
    tarball_path = Path("datasets/housing.tgz")
    
    # 檢查檔案是否已存在，避免重複下載
    if not tarball_path.is_file():
        # 建立資料夾結構，parents=True 允許建立多層目錄
        Path("datasets").mkdir(parents=True, exist_ok=True)
        
        # 指定數據來源 URL
        url = "https://github.com/ageron/handson-ml3/raw/main/datasets/housing.tgz"
        
        # 下載檔案到本地
        urllib.request.urlretrieve(url, tarball_path)
        
        # 解壓縮 tar.gz 檔案
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    
    # 讀取解壓縮後的 CSV 檔案並回傳 DataFrame
    return pd.read_csv(Path("datasets/housing/housing.csv"))

# 執行函式獲取數據
housing = load_housing_data()
```

**✅ 程式碼逐行解析：**
1.  `第 6 行`: 定義一個名為 `load_housing_data` 的函式，將獲取數據的邏輯封裝起來。
2.  `第 7-13 行`: 檢查 `datasets/housing.tgz` 這個壓縮檔是否存在。如果不存在，就建立 `datasets` 資料夾，然後從指定的 URL 下載檔案，並使用 `tarfile` 函式庫將其解壓縮。
3.  `第 14 行`: 使用 Pandas 的 `read_csv` 函式讀取解壓縮後的 `housing.csv` 檔案，並將其作為一個 DataFrame 回傳。
4.  `第 16 行`: 呼叫 `load_housing_data()` 函式，將回傳的 DataFrame 存儲在 `housing` 變數中。

**🎯 重點摘要:**
- **核心功能**: 自動化數據獲取流程，包括下載、解壓縮和載入。
- **潛在問題**: 如果網路連線失敗或 URL 失效，下載過程會失敗。函式庫 `pathlib` 提供了現代化的檔案路徑操作方式，比傳統的 `os.path` 更直觀。
- **最佳使用情境**: 當您需要分發您的專案，並希望其他人能夠輕鬆地獲取所需數據時，這種自動化腳本非常有用。
- **故障排除**: 如果下載失敗，請檢查網路連線或嘗試手動下載檔案。相關的數據探索技巧請參考[範例 3](#explore-data)。

### 範例 3: 數據初步探索
```python
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
```

**✅ 程式碼逐行解析：**
1.  `第 1 行`: `housing.head()` 顯示 DataFrame 的前五行，讓我們可以快速一覽數據的樣貌和欄位。
2.  `第 2 行`: `housing.info()` 提供 DataFrame 的簡潔摘要，包括每個欄位的名稱、非空值的數量以及數據類型（Dtype）。這對於快速發現缺失值和了解數據結構至關重要。
3.  `第 3 行`: `housing["ocean_proximity"].value_counts()` 針對 `ocean_proximity` 這個類別欄位，計算每個類別的出現次數。
4.  `第 4 行`: `housing.describe()` 針對數值型欄位，產生描述性統計數據，如計數、平均值、標準差、最小值、最大值以及百分位數。

**🎯 重點摘要:**
- **核心功能**: 快速了解數據集的整體情況，包括大小、欄位、數據類型、缺失值和基本統計特性。
- **潛在問題**: `info()` 顯示的非空值數量可以幫助我們快速識別哪些欄位有缺失數據。`describe()` 只對數值欄位有效。
- **最佳使用情境**: 在數據載入後，立即進行這些初步探索，是數據分析的標準起手式。

### 範例 4: 繪製數據分佈直方圖
```python
housing.hist(bins=50, figsize=(12, 8))
plt.show()
```

**✅ 程式碼逐行解析：**
1.  `第 1 行`: `housing.hist()` 是 Pandas DataFrame 內建的繪圖方法，可以為所有數值型欄位繪製直方圖（histogram）。
2.  `bins=50`: 指定直方圖的條數為 50。越多的條數可以讓我們更細緻地觀察數據分佈，但太多也可能引入雜訊。
3.  `figsize=(12, 8)`: 設定整個圖表的大小為 12x8 英吋。
4.  `plt.show()`: 顯示圖表。

**🎯 重點摘要:**
- **核心功能**: 可視化各個數值特徵的數據分佈。
- **潛在問題**: 從直方圖中，我們可以觀察到特徵的尺度（scale）差異很大，例如 `median_house_value` 的值遠大於 `housing_median_age`。我們也看到某些特徵有著「長尾」分佈，或是被限制在某個最大/最小值（如 `median_house_value`）。
- **最佳使用情境**: 在數據探索階段，使用直方圖來了解特徵的分佈、範圍、偏態（skewness）以及是否存在異常值。

## <a id="explore-data"></a>🗺️ 探索與可視化數據
💡 **實際應用情境：** 在深入模型訓練之前，花時間探索數據是非常有價值的。透過地理位置繪圖，我們可以將經緯度數據與房價等其他屬性結合，從而發現數據中的地理模式或群聚現象。

### 範例 5: 地理位置散點圖
```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()
```

**✅ 程式碼逐行解析：**
1.  `第 1 行`: 使用 DataFrame 的 `plot` 方法，設定 `kind="scatter"` 來繪製散點圖。`x` 和 `y` 軸分別對應經度和緯度。
2.  `s=housing["population"] / 100`: 設定每個點的大小（size）與該地區的人口數成正比。除以 100 是為了縮放點的大小，使其在圖上更合適。
3.  `c="median_house_value"`: 設定每個點的顏色（color）對應於房價中位數。
4.  `cmap="jet"`: 使用名為 "jet" 的顏色映射（colormap），從藍色（低房價）到紅色（高房價）進行漸變。
5.  `colorbar=True`: 顯示一個顏色條，標示顏色與數值的對應關係。

**🎯 重點摘要:**
- **核心功能**: 將地理數據與其他特徵（人口、房價）結合，進行多維度可視化。
- **潛在問題**: 這張圖清楚地顯示了房價與地理位置（特別是沿海地區）以及人口密度有很強的關聯性。
- **最佳使用情境**: 當數據集包含地理資訊（如經緯度）時，繪製此類圖表有助於發現空間上的關聯性。

## <a id="prepare-data"></a>🛠️ 數據前處理
💡 **實際應用情境：** 真實世界的數據很少是完美的。它們通常包含缺失值、異常值，或者其格式不適合直接用於機器學習模型。數據前處理是整個專案中至關重要且通常最耗時的階段。Scikit-Learn 的 `Pipeline` 和 `ColumnTransformer` 是處理這個階段的強大工具。

### 範例 6: 處理缺失值
```python
from sklearn.impute import SimpleImputer

# 建立填補器，策略設為中位數（對異常值較不敏感）
imputer = SimpleImputer(strategy="median")

# 只選擇數值型欄位進行處理
housing_num = housing.select_dtypes(include=[np.number])

# 學習每個欄位的中位數（fit 階段）
imputer.fit(housing_num)

# 使用學習到的中位數填補缺失值（transform 階段）
X = imputer.transform(housing_num)
```

**✅ 程式碼逐行解析：**
1.  `第 2 行`: 從 Scikit-Learn 導入 `SimpleImputer`，這是一個用於處理缺失值的轉換器。
2.  `第 4 行`: 建立一個 `SimpleImputer` 的實例，並設定 `strategy="median"`，表示我們將用每個欄位的中位數來填補該欄位的缺失值。
3.  `第 5 行`: `select_dtypes` 方法只選擇數值型的欄位。
4.  `第 6 行`: `imputer.fit(housing_num)` 計算 `housing_num` 中每個欄位的中位數，並將結果存儲在 `imputer` 的 `statistics_` 屬性中。
5.  `第 7 行`: `imputer.transform(housing_num)` 使用計算好的中位數來填補缺失值，並回傳一個包含轉換後數據的 NumPy 陣列。

**🎯 重點摘要:**
- **核心功能**: 使用特徵的中位數來系統性地填補數據集中的缺失值。
- **潛在問題**: 選擇 "median" 策略對處理有異常值的數據比較穩健。其他策略還有 "mean"（平均值）和 "most_frequent"（眾數）。
- **最佳使用情境**: 當數值特徵存在缺失值時，`SimpleImputer` 提供了一個簡單而有效的解決方案。

### 範例 7: 處理文本與類別屬性
```python
from sklearn.preprocessing import OneHotEncoder

# 建立獨熱編碼器，sparse_output=False 確保輸出為密集陣列
cat_encoder = OneHotEncoder(sparse_output=False)

# 對類別欄位進行獨熱編碼轉換
# 注意: 需要用雙層中括號 [[]] 來保持 DataFrame 格式
housing_cat_1hot = cat_encoder.fit_transform(housing[["ocean_proximity"]])
```

**✅ 程式碼逐行解析：**
1.  `第 2 行`: 導入 `OneHotEncoder`，它用於將類別特徵轉換為數值格式。
2.  `第 4 行`: 建立 `OneHotEncoder` 的實例。`sparse_output=False` 確保輸出是一個密集的 NumPy 陣列，而不是稀疏矩陣。
3.  `第 5 行`: `fit_transform` 方法會先學習類別（`fit`），然後將類別轉換為獨熱編碼（`transform`）。例如，如果 `ocean_proximity` 有 5 個類別，那麼每個地區都會被轉換成一個長度為 5 的向量，其中只有一個元素是 1，其餘都是 0。

**🎯 重點摘要:**
- **核心功能**: 將無法直接計算的文本類別轉換為機器學習模型可以處理的數值格式。
- **潛在問題**: 獨熱編碼會產生很多新的欄位，如果一個類別特徵有非常多的類別，可能會導致維度災難。
- **最佳使用情境**: 當類別特徵的類別數量不多時，獨熱編碼是處理標稱型（Nominal）類別數據的標準方法。

### 範例 8: 建立數據轉換 Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# 建立數值特徵的處理流水線
num_pipeline = Pipeline([
    # 步驟1: 用中位數填補缺失值
    ('imputer', SimpleImputer(strategy="median")),
    # 步驟2: 標準化特徵（平均值=0，標準差=1）
    ('std_scaler', StandardScaler()),
])

# 定義欄位類型列表
num_attribs = list(housing_num)  # 數值欄位
cat_attribs = ["ocean_proximity"]  # 類別欄位

# 建立完整的欄位轉換器，可對不同類型欄位應用不同處理
full_pipeline = ColumnTransformer([
    # 對數值欄位應用數值處理流水線
    ("num", num_pipeline, num_attribs),
    # 對類別欄位應用獨熱編碼
    ("cat", OneHotEncoder(sparse_output=False), cat_attribs),
])

# 一次性應用所有轉換步驟，得到處理完成的數據
housing_prepared = full_pipeline.fit_transform(housing)
```

**✅ 程式碼逐行解析：**
1.  `第 5-8 行`: 建立一個名為 `num_pipeline` 的 `Pipeline`。它將一系列轉換步驟串聯起來：首先用中位數填補缺失值（`imputer`），然後進行特徵縮放（`std_scaler`）。
2.  `第 13-16 行`: 建立一個 `ColumnTransformer`。這個強大的工具可以讓我們對不同的欄位應用不同的轉換流程。這裡，我們對所有數值屬性（`num_attribs`）應用 `num_pipeline`，並對類別屬性（`cat_attribs`）應用 `OneHotEncoder`。
3.  `第 18 行`: `full_pipeline.fit_transform(housing)` 對整個 `housing` DataFrame 應用我們定義好的所有前處理步驟，回傳一個準備好用於模型訓練的 NumPy 陣列。

**🎯 重點摘要:**
- **核心功能**: 將所有數據前處理步驟封裝成一個單一的轉換器，極大地簡化了工作流程。
- **潛在問題**: 確保 `num_attribs` 和 `cat_attribs` 的欄位列表是正確且沒有重疊的。
- **最佳使用情境**: 在任何需要對不同欄位進行不同處理的複雜數據準備場景中，`Pipeline` 和 `ColumnTransformer` 都是最佳實踐。
- **故障排除**: 如果遇到欄位類型錯誤，請檢查數據類型。相關的模型訓練步驟請參考[選擇與訓練模型](#select-train-model)。

## <a id="select-train-model"></a>🧠 選擇與訓練模型
💡 **實際應用情境：** 在數據準備好之後，下一步就是選擇、訓練和評估模型。通常我們會從一些簡單的模型開始，建立一個基準性能，然後再嘗試更複雜的模型。

### 範例 9: 訓練線性回歸模型
```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing["median_house_value"])
```

**✅ 程式碼逐行解析：**
1.  `第 2 行`: 導入 `LinearRegression` 模型。
2.  `第 4 行`: 建立 `LinearRegression` 模型的實例。
3.  `第 5 行`: `fit` 方法使用準備好的數據 `housing_prepared`（特徵）和 `housing["median_house_value"]`（目標值）來訓練模型。

**🎯 重點摘要:**
- **核心功能**: 訓練一個基本的線性回歸模型。
- **最佳使用情境**: 作為建立性能基準的第一個模型。線性回歸速度快、易於解釋，但可能因為對數據的假設過於簡單而導致欠擬合（underfitting）。

### 範例 10: 使用交叉驗證評估模型
```python
from sklearn.model_selection import cross_val_score

# 使用 K 折交叉驗證評估線性回歸模型
scores = cross_val_score(lin_reg, housing_prepared, housing["median_house_value"],
                         scoring="neg_mean_squared_error",  # 負的均方誤差
                         cv=10)  # 10 折交叉驗證

# 轉換為 RMSE（均方根誤差）- 注意要先取負號再開根號
lin_rmse_scores = np.sqrt(-scores)
```

**✅ 程式碼逐行解析：**
1.  `第 2 行`: 導入 `cross_val_score`，這是 Scikit-Learn 中用於交叉驗證的工具。
2.  `第 4-5 行`: `cross_val_score` 會自動將數據集分成 `cv=10` 折。在每一折中，它會用 9 折的數據訓練模型，並在剩下的 1 折上進行評估。`scoring="neg_mean_squared_error"` 指定了評估指標。
3.  `第 6 行`: Scikit-Learn 的評分函式通常回傳的是效用（utility，越大越好），而不是成本（cost，越小越好），所以它計算的是負的 MSE。我們需要取負號再開根號，才能得到 RMSE。

**🎯 重點摘要:**
- **核心功能**: 使用 K-折交叉驗證來獲得比單純的訓練/測試集劃分更穩健的模型性能評估。
- **潛在問題**: 交叉驗證的計算成本是單次訓練的 K 倍。
- **最佳使用情境**: 在模型評估和比較階段，交叉驗證是標準做法，它可以有效避免因為某次特定的訓練/測試集劃分而導致的偶然性結果。
- **故障排除**: 如果計算時間過長，可以減少 `cv` 的值或使用較小的數據樣本。接下來的超參數調優請參考[模型微調](#fine-tune-model)。

## <a id="fine-tune-model"></a>⚙️ 模型微調
💡 **實際應用情境：** 大多數機器學習模型都帶有超參數（Hyperparameters），這些是在訓練前設定的參數，例如決策樹的最大深度。找到一組好的超參數組合可以顯著提升模型性能。網格搜索和隨機搜索是兩種常用的自動化超參數調優方法。

### 範例 11: 使用網格搜索進行超參數調優
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, housing["median_house_value"])
```

**✅ 程式碼逐行解析：**
1.  `第 5-8 行`: 定義一個 `param_grid`，這是一個字典列表，其中包含了我們想要測試的超參數名稱和值的組合。`GridSearchCV` 將會遍歷所有可能的組合。
2.  `第 12-15 行`: 建立 `GridSearchCV` 的實例。它接收一個模型（`forest_reg`）、超參數網格（`param_grid`），並使用 `cv=5` 的交叉驗證來評估每種組合的性能。
3.  `第 17 行`: `fit` 方法會執行搜索。這是一個計算密集型的過程，因為它需要訓練 `(3*4 + 2*3) * 5 = 90` 個模型。

**🎯 重點摘要:**
- **核心功能**: 自動化地搜索最佳的超參數組合。
- **潛在問題**: 當超參數空間很大時，網格搜索的計算成本會非常高。在這種情況下，`RandomizedSearchCV`（隨機搜索）通常是更好的選擇。
- **最佳使用情境**: 當您想要徹底地探索一個較小的超參數空間時，網格搜索非常有用。

## <a id="evaluate-model"></a>📊 評估最終模型
💡 **實際應用情境：** 在經過多輪的模型選擇和微調後，我們選出了最終的模型。現在，我們需要在「從未見過」的測試集上對其進行最後一次評估，以估計它在真實世界中的泛化性能。

### 範例 12: 在測試集上評估最終模型
```python
# 獲取網格搜索找到的最佳模型
final_model = grid_search.best_estimator_

# 從測試集分離特徵和目標變數
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# 【關鍵】只使用 transform，不用 fit_transform
# 必須使用訓練時學習的轉換參數，避免數據洩漏
X_test_prepared = full_pipeline.transform(X_test)

# 使用最終模型進行預測
final_predictions = final_model.predict(X_test_prepared)

# 計算測試集上的最終性能指標
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```

**✅ 程式碼逐行解析：**
1.  `第 1 行`: 從 `grid_search` 物件中獲取性能最好的模型。
2.  `第 3-4 行`: 從測試集中分離出特徵（`X_test`）和標籤（`y_test`）。
3.  `第 6 行`: **非常重要**：這裡我們只使用 `transform` 方法，而不是 `fit_transform`。我們必須使用從訓練集中學習到的轉換規則（如中位數、縮放參數）來處理測試集，以避免數據洩漏（data leakage）。
4.  `第 8 行`: 使用最終模型對處理過的測試集進行預測。
5.  `第 10-11 行`: 計算並打印出最終的均方根誤差（RMSE）。

**🎯 重點摘要:**
- **核心功能**: 在獨立的測試集上評估最終模型的性能，得到其泛化誤差的估計。
- **潛在問題**: 絕對不能在測試集上進行任何的 `fit` 操作（包括前處理和模型訓練），否則評估結果將會過於樂觀且不可信。
- **最佳使用情境**: 這是機器學習專案的最後一步，用於報告模型的最終性能。

## 💡 總結與最佳實踐
這個端到端的專案向我們展示了一個完整的機器學習工作流程。我們從定義問題開始，獲取數據，進行探索性分析，然後投入大量精力進行數據前處理。我們學習了如何使用 Scikit-Learn 的 `Pipeline` 和 `ColumnTransformer` 來建立可重用的數據準備流程。接著，我們訓練了多個模型，使用交叉驗證進行評估，並透過網格搜索來微調超參數。最後，我們在測試集上評估了最終模型的性能。

**最佳實踐：**
- **自動化**: 盡可能將數據獲取和轉換的過程編寫成函式或 `Pipeline`，以方便重現。
- **數據探索**: 不要急著訓練模型，花足夠的時間去理解你的數據。
- **交叉驗證**: 使用交叉驗證來獲得更可靠的模型性能評估。
- **避免數據洩漏**: 嚴格區分訓練集和測試集，絕不在測試集上進行任何形式的擬合（fitting）。

## ❓ 常見問答 (FAQ)
**Q: 為什麼需要將數據分成訓練集和測試集？**
A: 這是為了評估模型的泛化能力。模型在訓練集上表現好並不意味著它在從未見過的新數據上也會表現好。測試集模擬了這種新數據，讓我們可以估計模型在真實世界中的性能。如果我們在所有數據上訓練模型，我們就無法知道它是否只是「記住」了訓練數據（過擬合），而不是學到了通用的模式。

**Q: `fit`, `transform`, `fit_transform` 有什麼區別？**
A: - `fit()`: 轉換器學習數據的參數（例如，`SimpleImputer` 學習中位數，`StandardScaler` 學習平均值和標準差）。它只計算，不應用。
   - `transform()`: 使用**已經學習到**的參數來轉換數據。
   - `fit_transform()`: 先執行 `fit()`，然後執行 `transform()`。這是一個方便的捷徑，但只能對訓練數據使用。對於測試數據，必須只使用 `transform()`，以確保使用從訓練數據中學到的相同參數。

**Q: 網格搜索（Grid Search）和隨機搜索（Randomized Search）哪個更好？**
A: 這取決於情況。如果超參數的可能組合不多，網格搜索可以窮盡所有可能，找到理論上的最優組合。但如果超參數空間很大，網格搜索的計算成本會急劇增加。在這種情況下，隨機搜索通常更高效，它在指定的範圍內隨機抽樣組合，通常能在更短的時間內找到一個非常好（即使不一定是絕對最優）的解。

**Q: 如果我的電腦記憶體不足，無法處理大型數據集怎麼辦？**
A: 您可以考慮以下幾種策略：
   - 使用數據抽樣，先在較小的數據子集上驗證您的方法
   - 採用增量學習算法（如 `SGDRegressor`）
   - 使用更高效的數據格式（如 Parquet）
   - 考慮雲端運算平台的資源

**Q: 為什麼我的模型在訓練集上表現很好，但在測試集上表現很差？**
A: 這是典型的過擬合現象。可能的解決方案包括：
   - 增加更多的訓練數據
   - 使用正則化技術
   - 減少模型複雜度
   - 使用交叉驗證進行更穩健的模型選擇
   - 檢查數據前處理是否存在數據洩漏

**Q: 如何選擇最適合的評估指標？**
A: 選擇評估指標應該基於您的業務目標：
   - RMSE：當大誤差比小誤差更不可接受時使用
   - MAE：當所有誤差同等重要時使用
   - R²：當您想了解模型解釋了多少變異時使用
   - 對於房價預測，RMSE 通常是合適的選擇，因為大的預測誤差（如預測錯誤 10 萬美元）比小誤差更嚴重

## 🏷️ 推薦標籤 (Suggested Hashtags)
#Python #機器學習 #數據科學 #ScikitLearn #Pandas #專案實戰 #房價預測 #技術教學 #程式設計 #數據分析 #模型訓練 #特徵工程
