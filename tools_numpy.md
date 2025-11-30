<!-- meta-title: 🐍 NumPy 工具教學：從陣列建立到線性代數 -->
<!-- meta-description: 學習 NumPy 的完整指南，從基本陣列操作到進階線性代數功能。包含實用範例、逐行解析和最佳實踐。 -->
<!-- meta-keywords: NumPy, Python, 陣列, 線性代數, 科學計算, 教學 -->
<!-- meta-hashtags: #Python #程式設計 #教學 #NumPy #陣列 #線性代數 #科學計算 #編程 #學習筆記 #程式開發者 #軟體工程 -->

# 🐍 NumPy 完整教學指南：陣列建立、操作與線性代數實戰技巧

NumPy 是 Python 科學計算領域的核心函式庫，專為高效能 N 維陣列運算設計。

其底層以 C 語言實作，提供極快的數值處理速度，並支援線性代數、傅立葉變換、隨機數生成等進階功能。

NumPy 不僅是資料科學、機器學習、人工智慧等領域的基礎工具，也是許多 Python 標準函式庫和第三方套件的效能關鍵。透過 Python 封裝，使用者能輕鬆進行大規模數據分析與科學運算，提升開發效率與程式效能。

以下就讓我們深入探索 NumPy 的強大功能，從基礎的陣列建立到複雜的線性代數運算，並透過實際範例與逐行解析，幫助你掌握這個不可或缺的工具。

## 📝 本文目錄

- [建立陣列](#建立陣列)
- [陣列資料](#陣列資料)
- [重塑陣列](#重塑陣列)
- [算術運算](#算術運算)
- [廣播](#廣播)
- [條件運算子](#條件運算子)
- [數學和統計函數](#數學和統計函數)
- [陣列索引](#陣列索引)
- [疊加陣列](#疊加陣列)
- [分割陣列](#分割陣列)
- [轉置陣列](#轉置陣列)
- [線性代數](#線性代數)
- [向量化](#向量化)
- [儲存和載入](#儲存和載入)
- [常見問答](#常見問答)
- [最佳實踐](#最佳實踐)
- [推薦標籤](#推薦標籤)

---

## 🎯 關鍵重點

- NumPy以高效能的 N 維陣列 (N-dimensional arrays) 為核心，提供廣泛的數學和科學計算功能
- 掌握陣列建立 (array creation)、索引 (indexing)、運算 (operations) 和重塑 (reshaping) 是使用 NumPy 的基礎
- 廣播機制 (broadcasting mechanism) 允許不同形狀陣列間的靈活運算
- 向量化操作 (vectorized operations) 比迴圈 (loops) 更有效率，能充分利用 NumPy 的優化
- 線性代數 (linear algebra) 模組提供矩陣運算 (matrix operations)、特徵值 (eigenvalues) 等進階功能

---

## <a id="建立陣列"></a>建立陣列

💡 **實際應用情境：** 在資料科學 (data science) 和機器學習 (machine learning) 中，陣列(array) 是處理數值資料的基本單位，從簡單的向量(vector) 到複雜的張量(tensor)，都依賴 NumPy 的陣列建立功能。

### 基本概念

NumPy 中的每個維度稱為軸（axis）。軸的數量稱為秩（rank），軸長度的清單稱為形狀（shape），總元素數稱為大小（size）。

如果一個陣列只有一個軸，則稱為一維陣列（向量）；如果有兩個軸，則稱為二維陣列（矩陣）。

假設有一個形狀為 (3, 5) 的二維陣列，它有 2 個軸（秩為 2），第一個軸長度為 3，第二個軸長度為 5，總共有 15 個元素。

```python
import numpy as np

# 建立一個包含 6 個 0 的陣列
a = np.zeros(6)
print("a =", a)

# 建立 3x5 矩陣
b = np.zeros((3, 5))
print("b =", b)
print("b.shape =", b.shape)
print("b.ndim =", b.ndim)
print("b.size =", b.size)
```

**✅ 程式碼逐行解析：**

1. `import numpy as np`：匯入 NumPy 模組並簡稱為 np，這是標準慣例。
2. `a = np.zeros(6)`：使用 zeros 函數建立一個包含 6 個 0 的一維陣列。
3. `print("a =", a)`：輸出陣列 a 的內容。
4. `b = np.zeros((3, 5))`：建立一個形狀為 (3, 5) 的二維陣列（3 行 5 列），所有元素為 0。
5. `print("b =", b)`：輸出陣列 b 的內容。
6. `print("b.shape =", b.shape)`：輸出陣列 b 的形狀，即 (3, 5)。
7. `print("b.ndim =", b.ndim)`：輸出陣列 b 的維度數，即 2。
8. `print("b.size =", b.size)`：輸出陣列 b 的總元素數，即 15。

**🎯 重點摘要：**

- **核心功能**：zeros 函數用於建立指定形狀的全零陣列，是初始化陣列的常用方法。
- **潛在問題**：忘記指定形狀的括號會導致錯誤，如誤用 np.zeros(3,4) 而非正確的 np.zeros((3,4))。
- **最佳使用情境**：需要初始化陣列但尚未有具體數值時，或在演算法中需要重置陣列為零。

### 其他建立函數

💡 實際應用情境： 在陣列初始化中，除了全零陣列，還需全一陣列、指定值陣列或未初始化陣列，以適應計數、常數或效能關鍵場景。

此節介紹 NumPy 的 ones、full 和 empty 函數，分別用於建立全 1、指定值和未初始化陣列，實現高效率的初始化。

```python
# 全 1 陣列
ones = np.ones((3, 8))
print("ones =", ones)

# 指定值的陣列
full = np.full((3, 3), np.pi)
print("full =", full)

# 未初始化的陣列
empty = np.empty((2, 3))
print("empty =", empty)
```

**✅ 程式碼逐行解析：**

1. `ones = np.ones((3, 8))`：建立一個形狀為 (3, 8) 的全 1 陣列。
2. `print("ones =", ones)`：輸出全 1 陣列。
3. `full = np.full((3, 3), np.pi)`：建立一個形狀為 (3, 3) 的陣列，所有元素為 π（圓周率）。
4. `print("full =", full)`：輸出指定值陣列。
5. `empty = np.empty((2, 3))`：建立一個形狀為 (2, 3) 的未初始化陣列，其內容不可預測。
6. `print("empty =", empty)`：輸出未初始化陣列。

**🎯 重點摘要：**

- **核心功能**：ones、full 和 empty 分別用於建立全 1、指定值和未初始化陣列。
- **潛在問題**：empty 陣列的內容不可預測，可能包含垃圾值，只在需要立即覆寫時使用。
- **最佳使用情境**：ones 用於計數初始化，full 用於常數陣列，empty 用於效能關鍵的初始化。

### 從序列建立陣列

💡 **實際應用情境**：
在資料處理和數值計算中，經常需要從現有序列（如特定範圍數值、等間距點或 Python 列表）建立陣列，這是初始化資料結構的基礎步驟。

此節介紹 NumPy 提供的高效函數：

  * `arange`：用於生成整數或浮點數序列（類似 Python 的 range）。
  * `linspace`：用於建立固定數量的等間距點（Linear Space）。
  * `array`：將 Python 列表（List）轉換為多維陣列。

這些方法支援自訂步長和資料類型，適合各種數值範圍的應用。

```python
# 1. 使用 arange (指定範圍與步長)
# 整數：[1, 2, 3, 4, 5]
arange_int = np.arange(1, 6)
print("arange_int =", arange_int)

# 浮點數：[1., 2., 3., 4., 5.]
arange_float = np.arange(1.0, 6.0)
print("arange_float =", arange_float)

# 指定步長 0.5：[1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5]
arange_step = np.arange(1, 6, 0.5)
print("arange_step =", arange_step)

# 2. 比較 arange 和 linspace 的差異（使用相同範圍）
# 試圖用 arange 切分出 6 個點 (步長 = 總長度 / 5)
# 注意：由於浮點數誤差，arange 有時可能無法準確包含或排除終點
arange_like_linspace = np.arange(0, 10/3, (10/3)/5)
print("arange_like_linspace =", arange_like_linspace)

# 3. 使用 linspace (指定範圍與點數)
# 建立 6 個點，自動計算步長，且精確包含終點
linspace = np.linspace(0, 10/3, 6)
print("linspace =", linspace)

# 4. 從 Python 列表 (List) 轉換
array_from_list = np.array([[1, 2, 3, 4], [6, 7, 8, 9]])
print("array_from_list =\n", array_from_list)
print("array_from_list.shape =", array_from_list.shape)
```

**✅ 程式碼逐行解析：**

1.  `np.arange(1, 6)`：建立從 1 到 5 的整數陣列（**不包含 6**，即左閉右開區間）。
2.  `np.arange(1.0, 6.0)`：建立從 1.0 到 5.0 的浮點數陣列，預設步長為 1。
3.  `np.arange(1, 6, 0.5)`：建立從 1 開始，步長為 0.5 的陣列，直到小於 6 為止。
4.  `np.arange(0, 10/3, (10/3)/5)`：嘗試模擬等間距，但在浮點數運算下，`arange` 對於「終點是否包含」的判定較不穩定，不建議這樣做。
5.  `np.linspace(0, 10/3, 6)`：在 0 到 10/3 之間建立 **6 個**等間距點（預設**包含終點**）。這是在浮點數範圍取樣的最佳方式。
6.  `np.array(...)`：將 Python 的巢狀列表 `[[...], [...]]` 轉換為 NumPy 的二維陣列。
7.  `.shape`：屬性顯示陣列為 `(2, 4)`，代表 2 列 4 行。

**🎯 重點摘要：**

  * **核心區別**：
      * `arange` 關注的是 **「步長 (Step)」**（例如：每隔 0.5 走一步）。
      * `linspace` 關注的是 **「點數 (Count)」**（例如：我要這段路中間有 10 個點）。
  * **潛在陷阱**：使用 `arange` 處理浮點數時，因為二進位浮點數精度的關係，可能會導致結果陣列的長度不如預期（多一個或少一個點）。
  * **最佳實踐**：
      * **整數序列** ➝ 使用 `arange`。
      * **浮點數範圍 / 固定取樣點數** ➝ 優先使用 `linspace`。

### 隨機陣列

💡實際應用情境： 在模擬資料、初始化神經網路權重或進行隨機取樣時，需要產生隨機數陣列，以引入變異性和不確定性。

此節介紹 NumPy 的 `random.rand` 和 `random.randn` 函數，分別用於建立**均勻分佈**（範圍 [0, 1)）和**常態分佈**（均值 0，標準差 1）的隨機浮點數陣列。

```python
# 均勻分佈隨機數
rand = np.random.rand(3, 5)
print("rand =", rand)

# 常態分佈隨機數
randn = np.random.randn(3, 5)
print("randn =", randn)
```

**✅ 程式碼逐行解析：**

1. `rand = np.random.rand(3, 5)`：建立形狀為 (3, 5) 的隨機浮點數陣列，範圍在 [0, 1) 間的均勻分佈 (Uniform Distribution)。
2. `print("rand =", rand)`：輸出均勻分佈隨機陣列。
3. `randn = np.random.randn(3, 5)`：建立形狀為 (3, 5) 的隨機浮點數陣列，均值 0 標準差 1 的常態分佈 (Normal Distribution)。
4. `print("randn =", randn)`：輸出常態分佈隨機陣列。

**🎯 重點摘要：**

- **核心功能**：rand 產生均勻分佈，randn 產生常態分佈的隨機數。
- **潛在問題**：隨機數每次執行結果不同，測試時需設定種子確保重現性。
- **最佳使用情境**：模擬資料、初始化權重、隨機取樣等需要隨機性的應用。

### 使用函數建立

💡實際應用情境： 當陣列中的元素值取決於其「位置索引（Index）」時（例如建立網格座標、棋盤格紋或距離矩陣），使用 fromfunction 可以避免撰寫低效的巢狀迴圈。

我們使用 NumPy 的 fromfunction 函數，它會將每個座標點的索引值傳入指定的函數中進行計算。

```python
# 定義計算規則
def rule_function(i, j):
    # i 代表第 0 軸 (row) 的索引
    # j 代表第 1 軸 (col) 的索引
    return j + 10 * i

# 建立 (3, 2) 的陣列
# dtype=int 確保傳入函數的座標是整數 (預設為 float)
result = np.fromfunction(rule_function, (3, 2), dtype=int)

print("fromfunction result shape:", result.shape)
print("result =\n", result)
```

**✅ 程式碼逐行解析：**

1. `def rule_function(i, j):`  
    定義一個函數，參數數量需與陣列維度相符（此例為 2 維）。

2. `return j + 10 * i`  
    計算邏輯：十位數代表列索引，個位數代表行索引。

3. `result = np.fromfunction(rule_function, (3, 2), dtype=int)`  
    建立形狀為 (3, 2) 的陣列。NumPy 會自動生成座標網格：  
    - `i` 為 `[[0, 0], [1, 1], [2, 2]]`  
    - `j` 為 `[[0, 1], [0, 1], [0, 1]]`  
    並將座標傳入函數計算。

4. `print(...)`  
    輸出結果，例如 (1, 0) 的位置值為 10，(2, 1) 的位置值為 21。

**🎯 重點摘要：**

- **核心功能**：利用座標（索引）生成陣列內容，屬於向量化運算，效能遠高於 Python 迴圈。
- **參數對應**：函數的第 1 個參數對應 Axis 0（列），第 2 個參數對應 Axis 1（行），依序類推，順序一致。
- **注意事項**：fromfunction 預設傳入座標為浮點數，若需整數運算請加上 `dtype=int`。
- **最佳使用情境**：建立幾何變換矩陣、特殊數列圖案（如 Pascal 三角形）、初始化物理場等。

---

## <a id="陣列資料"></a>陣列資料 (Array Data)

💡 **實際應用情境：** 在處理大規模資料集時，精確掌握 NumPy 陣列的資料類型（Data Types）和記憶體佈局（Memory Layout）是提升效能和避免錯誤的關鍵。資料類型決定了數值範圍、精確度和運算效率，例如使用 `int32` 而非 `int64` 可節省記憶體並加速計算；記憶體佈局則影響資料存取速度，如連續佈局（Contiguous Layout）能充分利用 CPU 快取，減少快取遺漏（Cache Miss）。在機器學習中，這有助於優化模型訓練速度和資源使用；在影像處理中，選擇合適類型（如 `uint8`）可避免溢位並提升處理效率。若忽略這些細節，可能導致記憶體浪費、運算緩慢或資料損壞，因此建議在陣列建立時明確指定 `dtype`，並使用工具如 `nbytes` 屬性監控記憶體使用量。

### 資料類型

```python
c = np.arange(1, 6)
print("c.dtype =", c.dtype, "c =", c)

c_float = np.arange(1.0, 6.0)
print("c_float.dtype =", c_float.dtype, "c_float =", c_float)

c_complex = np.arange(1, 6, dtype=np.complex64)
print("c_complex.dtype =", c_complex.dtype, "c_complex =", c_complex)
```

**✅ 程式碼逐行解析：**

1. `c = np.arange(1, 6)`：建立從 1 到 5 的整數陣列，預設為 int32 或 int64。
2. `print("c.dtype =", c.dtype, "c =", c)`：輸出陣列的資料類型和內容。
3. `c_float = np.arange(1.0, 6.0)`：建立浮點數陣列，自動推斷為 float64。
4. `print("c_float.dtype =", c_float.dtype, "c_float =", c_float)`：輸出浮點數陣列的類型和內容。
5. `c_complex = np.arange(1, 6, dtype=np.complex64)`：明確指定複數類型建立陣列。
6. `print("c_complex.dtype =", c_complex.dtype, "c_complex =", c_complex)`：輸出複數陣列的類型和內容。

**🎯 重點摘要：**

- **核心功能**：NumPy 陣列有統一資料類型，可顯式指定 dtype。
- **潛在問題**：不同類型混合運算會自動提升，可能導致記憶體使用增加。
- **最佳使用情境**：根據資料範圍選擇適當類型，如 uint8 用於影像，float32 用於一般計算。

### 記憶體資訊

💡 實際應用情境： 在處理大規模資料時，了解陣列的記憶體佈局和大小對於效能調優和資源管理至關重要。

在這裡，我們將介紹 NumPy 陣列的 itemsize 和 data 屬性，用於檢查元素大小和存取原始位元組資料。

```python
d_cpx64 = np.arange(1, 6, dtype=np.complex64)
print(f"The itemsize of d_cpx64 elements = {d_cpx64.itemsize}")

d_int32 = np.array([[1, 2], [-3000, 4000]], dtype=np.int32)
print(f"Raw byte representation of d_int32 array (first 20 bytes): {d_int32.data[:20].tobytes()}")
```

**✅ 程式碼逐行解析：**

1. `d_cpx64 = np.arange(1, 6, dtype=np.complex64)`：建立複數陣列。
2. `print(f"The itemsize of d_cpx64 elements = {d_cpx64.itemsize}")`：輸出每個元素的位元組大小（複數 64 為 8 位元組）。
3. `d_int32 = np.array([[1, 2], [-3000, 4000]], dtype=np.int32)`：建立 int32 二維陣列。
4. `print(f"Raw byte representation of d_int32 array (first 20 bytes): {d_int32.data[:20].tobytes()}")`：輸出陣列資料緩衝區的前 20 位元組，以位元組字串格式顯示。
    - `data` 屬性提供對底層位元組的直接存取，`tobytes()` 方法將其轉換為位元組字串。
    - 注意直接操作 `data` 可能導致資料損壞，應謹慎使用。

**🎯 重點摘要：**

- **核心功能**：itemsize 顯示元素大小，data 提供原始位元組存取。
- **潛在問題**：直接操作 data 緩衝區可能導致資料損壞，應謹慎使用。
- **最佳使用情境**：檢查記憶體使用量或與低階程式設計介面互動時。

---

## <a id="重塑陣列"></a>重塑陣列 (Reshaping Arrays)

💡 實際應用情境： 陣列重塑（reshape）是資料預處理和張量操作的常用技巧，能將一維資料轉換為多維結構。例如，將原始數據重塑為影像矩陣，方便後續分析或模型訓練。這在資料整理、特徵工程及深度學習中非常實用。

### 就地重塑 (使用 shape 屬性)

當需要直接改變陣列的形狀而不複製資料時，可利用就地重塑，快速將一維資料轉換為多維結構，方便後續分析或模型訓練。

```python
a24 = np.arange(24)
print("a24 =", a24)
print("a24.shape =", a24.shape)

a24.shape = (6, 4)
print("after reshape a24 =", a24)
print("a24.shape =", a24.shape)

a24.shape = (2, 3, 4)
print("3D a24 =", a24)
print("a24.shape =", a24.shape)
```

**✅ 程式碼逐行解析：**

1. `a24 = np.arange(24)`：建立包含 0 到 23 的陣列。
2. `print("a24 =", a24)`：輸出原始一維陣列。
3. `print("a24.shape =", a24.shape)`：輸出原始形狀 (24,)。
4. `a24.shape = (6, 4)`：將形狀修改為 (6, 4)，元素數必須相同。
5. `print("after reshape a24 =", a24)`：輸出重塑後的陣列。
6. `print("a24.shape =", a24.shape)`：輸出新形狀。
7. `a24.shape = (2, 3, 4)`：進一步重塑為三維。
8. `print("3D a24 =", a24)`：輸出三維陣列。
9. `print("a24.shape =", a24.shape)`：輸出三維形狀。

**🎯 重點摘要：**

- **核心功能**：直接修改 shape 屬性進行就地重塑，元素數必須保持不變。
- **潛在問題**：形狀乘積不等於元素數會引發錯誤。
- **最佳使用情境**：需要改變陣列維度結構但保持資料不變的情況。

### 使用 reshape

💡實際應用情境： 在資料分析和機器學習中，經常需要將一維資料重塑為多維結構以符合模型輸入需求。使用 reshape 函數可以方便地創建新的視圖，適應不同的形狀要求。

```python
a12 = np.arange(12)
a12_rs = a12.reshape(3, 4)
print("a12_rs =", a12_rs)

a12_rs[1, 2] = 999
print(f'a12_rs after modification =\n{a12_rs}')
print(f'a12 after a12_rs modification =\n{a12}')
```

**✅ 程式碼逐行解析：**

1. `a12_rs = a12.reshape(3, 4)`：建立 a12 的重塑視圖，形狀為 (3, 4)。
2. `print("a12_rs =", a12_rs)`：輸出重塑後的陣列。
3. `a12_rs[1, 2] = 999`：修改 a12_rs 的元素。
4. `print(f'a12_rs after modification =\n{a12_rs}')`：顯示修改後的 a12_rs。
5. `print(f'a12 after a12_rs modification =\n{a12}')`：顯示原始陣列 a12 也被修改，因為共享資料。

**🎯 重點摘要：**

- **核心功能**：reshape 返回新視圖，*與原陣列共享資料*。
- **潛在問題**：修改任一陣列都會影響另一個，造成意外副作用。
- **最佳使用情境**：需要不同形狀視圖但不想複製資料時。

### 陣列展平：使用 ravel

💡實際應用情境： 在將多維陣列輸入到只能接受一維資料的演算法（如 Scikit-Learn 的某些模型）或進行繪圖時，我們需要將陣列「拉平」。ravel 是最有效率的方法，因為它通常不會複製資料。

```python
# 1. 建立一個多維陣列 md_array
md_array = np.array([[1, 2, 3], [4, 5, 6]])
print("Original md_array:\n", md_array)

# 2. 使用 ravel 展平
flat = md_array.ravel()
print("flat =", flat)

# 3. 驗證：修改展平後的視圖，是否會影響原陣列？
flat[0] = 888 
print("\nAfter modifying flat:")
print("md_array (Original) has changed:\n", md_array)
```

**✅ 程式碼逐行解析：**

1. `md_array = np.array(...)`：初始化一個 (2, 3) 的二維陣列。
2. `flat = md_array.ravel()`：將多維陣列展平為一維。注意：這裡返回的是「視圖 (View)」，記憶體位置仍指向原本的資料。
3. `flat[0] = 888`：修改 flat 的第一個元素。
4. `print("md_array (Original) has changed:\n", md_array)`：你會發現原陣列 md_array 的 (0, 0) 位置也變成了 888。這證明了兩者共享記憶體。

**🎯 重點摘要：**

- **核心功能**：`ravel` 可將多維陣列展平成一維序列。
- **關鍵特性**：通常返回「視圖 (View)」而非複製，速度快且節省記憶體。
- **潛在風險**：因為與原陣列共享資料，修改展平後的結果會同步影響原始陣列。若需獨立副本，請使用 `flatten()`。
- **最佳使用情境**：適合唯讀存取所有元素，或需同步修改原資料時。

---

## <a id="算術運算"></a>算術運算 (Arithmetic Operations)

💡**實際應用情境：** 在資料科學中，我們很少使用迴圈來逐個計算數據。NumPy 的算術運算皆為 **元素級 (Element-wise)**，這意味著運算是並行作用於陣列中對應位置的元素。這在影像處理（如調整亮度 img + 10）或神經網路權重更新中無處不在。

```python
a = np.array([26, 22, 30, 11])
b = np.array([4, 3, 2, 1])

# 基礎四則運算
print("a + b =", a + b)  # [30 25 32 12]
print("a - b =", a - b)  # [22 19 28 10]
print("a * b =", a * b)  # [104 66 60 11] (注意：這是元素對應相乘)
print("a / b =", a / b)  # [6.5 7.333 15. 11.] (結果為浮點數)

# 進階運算
print("a // b =", a // b) # [6 7 15 11] (整數除法/地板除法)
print("a % b =", a % b)   # [2 1 0 0] (取餘數)
print("a ** b =", a ** b) # [456976 10648 900 11] (指數運算)
```

**✅ 程式碼逐行解析：**

1. a 與 b：建立兩個形狀相同的陣列。NumPy 要求參與運算的陣列形狀必須相同，或符合「廣播 (Broadcasting)」規則。
2. a * b：特別注意，這是將對應位置的數字相乘（例如 26*4），並非線性代數中的矩陣乘法。
3. a / b：即使輸入是整數，普通除法 / 的結果預設也會是浮點數（float）。
4. a // b：若只需要整數部分的商，使用 //。
5. a ** b：對應元素的指數運算（例如 $26^4$），這在 Python 中非常方便。

**🎯 重點摘要：**

- **核心機制**：所有標準運算子（`+`, `-`, `*`, `/`）在 NumPy 中預設為元素級（Element-wise）運算，即對應位置的元素逐一計算。

- **常見陷阱**：
    - `*` 是元素乘法，不是矩陣乘法。
    - 若需矩陣乘法（Matrix Multiplication），請使用 `@` 運算子（如 `A @ B`）或 `np.dot(A, B)`。
- **最佳使用情境**：適用於批次數據的統一數學變換，如單位換算、影像濾鏡、訊號增益等。

---

## <a id="廣播"></a>廣播 (Broadcasting)

💡 **實際應用情境：** 廣播機制允許 NumPy 在不同形狀的陣列之間進行算術運算，而無需手動複製資料。這在機器學習中極為常見，例如：將一個偏置向量 (Bias) 加到整個資料批次 (Batch) 的每一行，或進行影像的通道正規化。

廣播的優勢在於能自動擴展較小的陣列，使其形狀與較大的陣列相容，從而實現批次運算、特徵標準化、資料中心化等操作。例如，對每個樣本加上同一個偏置向量，或將一維陣列加到多維矩陣的每一行或每一列。這不僅簡化程式碼，也提升運算效率，避免不必要的記憶體複製。

常見應用包括：

- 對所有樣本批次加上偏置或標準化參數
- 對影像資料進行通道級正規化（如每個 RGB 通道減去平均值）
- 批次資料的特徵縮放與中心化
- 將純量或一維陣列自動擴展到多維陣列進行運算

廣播機制是 NumPy 向量化運算的基礎，建議在資料處理和科學計算中充分利用，能顯著提升程式效能與可讀性。

### 廣播規則

廣播規則（Broadcasting Rules）是 NumPy 允許不同形狀的陣列進行算術運算的核心機制。其運作方式如下：

1. **維度對齊**：從右往左比較兩個陣列的每個軸（dimension）。
2. **相容條件**：每個軸的長度必須「相等」或其中之一為 1。
3. **自動擴展**：若某一軸長度為 1，NumPy 會自動將其複製（虛擬擴展）至另一陣列的長度。
4. **最終形狀**：運算結果的形狀為兩陣列在每個軸上的最大值。

**範例：**  

- (2, 3) + (3,) → (2, 3)（第二個陣列自動擴展為 (2, 3)）  
- (4, 1, 6) + (3, 6) → (4, 3, 6)（第二個陣列自動擴展為 (1, 3, 6)，再複製到 (4, 3, 6)）  
- (5, 4) + (1,) → (5, 4)（純量或一維長度為 1時可廣播到任意形狀）

**注意：**  

- 若任一軸長度不相等且都不為 1，則無法廣播，會拋出錯誤。
- 廣播不會真正複製資料，只是調整資料讀取方式，效能高且節省記憶體。

**實務建議：**  

- 若遇到形狀不符，可用 `reshape` 或 `np.newaxis` 調整陣列形狀以符合廣播規則。

```python
# 模擬資料：2 個樣本，每個樣本有 3 個特徵 (Shape: 2, 3)
data = np.array([[1, 2, 3],
                 [10, 20, 30]])

# 模擬偏置：要加到每個樣本上的數值 (Shape: 3,)
bias = np.array([1, 0, 1])

print("Data shape:", data.shape)
print("Bias shape:", bias.shape)

# 觸發廣播：Bias 會自動「擴展」應用到 Data 的每一列
result = data + bias
print("Broadcast result =\n", result)
```

**✅ 程式碼逐行解析：**

1. `data` 是形狀 `(2, 3)` 的矩陣。
2. `bias` 是形狀 `(3,)` 的向量。
3. NumPy 廣播規則：
    - 從右往左比對維度。
    - 最後一個維度：`data` 和 `bias` 都是 3，直接匹配。
    - 倒數第二個維度：`data` 是 2，`bias` 沒有（視為 1），1 可廣播擴展為 2。
4. 廣播結果：
    - `bias` 被邏輯上複製為兩行，分別加到 `data` 的每一列。
    - 計算如下：
        - 第一列：[1, 2, 3] + [1, 0, 1] = [2, 2, 4]
        - 第二列：[10, 20, 30] + [1, 0, 1] = [11, 20, 31]

**🎯 重點摘要：**

- **核心規則**：形狀從**右向左**對齊。只要對應軸的長度**相等**，或其中一方為 **1**，即可進行廣播運算。
- **優勢**：廣播不會真正複製資料，只是調整資料讀取方式，因此能節省記憶體並提升運算速度。
- **常見錯誤**：若維度不匹配且都不為 1（例如 (2, 3) 加 (2,)），會拋出 ValueError。此時可用 `reshape` 或 `np.newaxis` 調整維度以符合廣播規則。

### 更多廣播範例

💡 **實際應用情境：** 這些不同的廣播方式在資料處理中超級實用：

- **列向量（形狀為 (N, 1)）**：適合「列級」運算，例如對每一**列 (Axis 0)**（樣本）乘上不同的正規化因子或權重。這種方式常見於批次資料的標準化或特徵縮放。

- **一維向量（形狀為 (M,)）**：適合「行級」運算，例如將偏置（Bias）或特徵平均值加到所有樣本的每一**行 (Axis 1)**。這在特徵工程、資料中心化等場景非常常見。

- **純量（單一數值）**：可自動廣播到整個陣列，適用於所有元素的統一加減或乘除。

這些廣播機制讓 NumPy 能夠用簡潔的語法完成複雜的批次運算，大幅提升程式效率與可讀性。

```python
b = np.arange(6).reshape(2, 3)
print("b =", b)

# Column vector broadcasting (shape: (2, 1))
result2 = b + [[10], [20]]
print("b + [[10], [20]] =", result2)

# Row vector broadcasting (shape: (3,))
result3 = b + [100, 200, 300]
print("b + [100, 200, 300] =", result3)

# Scalar broadcasting
result4 = b + 10000
print("b + 10000 =", result4)
```

**✅ 程式碼逐行解析：**

1. `b = np.arange(6).reshape(2, 3)`：建立 2x3 陣列，形狀為 (Axis 0: 2, Axis 1: 3)。
2. `result2 = b + [[10], [20]]`：`b` 為 (2, 3)，`[[10], [20]]` 為 (2, 1)，廣播沿著Axis 1 擴展 (因為其長度為 1，可以擴展)。
3. `result3 = b + [100, 200, 300]`：`b` 為 (2, 3)，`[100, 200, 300]` 為 (3,)，廣播沿著Axis 0 擴展。
4. `result4 = b + 10000`：所有元素加同一值。

**🎯 重點摘要：**

- **核心功能**：廣播的核心是**從右向左對齊**且**長度必須相等或為1**。
- **潛在問題**：複雜廣播可能難以理解，建議用簡單規則。
- **關鍵區分 (軸)**：在二維陣列中， `(N, 1)` 和 `(N,)` 的效果截然不同：
    - **`(N, 1)` 向量**：強迫擴展發生在 Axis 1，常用於對 Axis 0 進行獨立操作。
    - **`(N,)` 向量**：強迫擴展發生在 Axis 0，常用於對 Axis 1 進行獨立操作（如加 Bias）。
- **最佳使用情境**：特徵標準化、批次偏移等需要一致運算的情況。

### 類型提升

💡實際應用情境： 類型提升是 NumPy 確保混合資料類型運算正確性的機制。它會自動將所有運算元提升到一個「能安全容納所有可能結果」的通用類型。這在處理來自不同來源（如資料庫或檔案）的數據時尤其重要，但需注意它對記憶體和性能的影響。

```python
c1 = np.arange(0, 5, dtype=np.uint8)
# c1: [0 1 2 3 4], dtype=uint8 (無符號整數)

c2 = c1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)
# c2: [ 5 7 9 11 13]

c3 = c1 + 1.5
# c3: [1.5 2.5 3.5 4.5 5.5]

print("Type of c1 is", c1.dtype)
print("Type of c2 is", c2.dtype) # 預期: int16 (或更高)
print("Type of c3 is", c3.dtype) # 預期: float64
```

**✅ 程式碼逐行解析：**

1. `c1 = np.arange(0, 5, dtype=np.uint8)`：建立 uint8 (8 位元無符號整數) 陣列，範圍 0-255。
2. `c2 = c1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)`：與 int8 (8 位元有符號整數) 陣列相加，觸發類型提升以避免溢位。
3. `c3 = c1 + 1.5`：與浮點數相加，進一步提升為 float64 以保持精確度。
4. `print(...)`：輸出各陣列的資料類型，觀察類型提升結果。

**🎯 重點摘要：**

- **核心功能**：NumPy 的類型提升機制自動選擇能安全容納運算結果的通用類型，確保數值正確性，但可能導致記憶體使用量增加（如從 uint8 提升至 float64）。

- **潛在問題**：
    1. **效能與記憶體**：意外提升至高位元類型（如 float64）會大幅增加記憶體佔用和運算成本。
    2. **精確度**：提升雖常為精確度考量，但浮點數運算可能引入近似誤差。

- **最佳使用情境**：適用於混合類型運算的自動處理；為效能優化，建議預先統一資料類型，如統一使用 np.float32。

---

## <a id="條件運算子"></a>條件運算子 (Conditional Operators)
💡 **實際應用情境：** 條件運算是資料過濾和布林索引的基礎。在資料分析中，它用於快速識別和選擇滿足特定標準（例如，找出所有收入超過 $50000 且年齡小於 30 歲的用戶）的值，是取代傳統 Python 迴圈進行資料清理和篩選的關鍵。

```python
m = np.array([23, -15, 32, 57])
print(f"m = {m}")
print("-" * 25)

# 1. 元素級比較 (Element-wise Comparison)
mask = m < [16, 0, 35, 20]
print("m < [16, 0, 35, 20] =", mask)
# Output: [False  True  True False]

# 2. 純量比較 (Scalar Comparison) - 觸發廣播
mask2 = m < 30
print("Mask of m < 30 =", mask2)
# Output: [ True  True False False]

# 3. 布林索引 (Boolean Indexing)
filtered = m[mask2]
print("m[m < 30] =", filtered)
# Output: [23 -15]

# 4. 進階：結合多個條件 (使用 & 進行 AND 運算)
# 注意：必須使用 & (位元運算子) 而非 Python 的 and
mask_combined = (m > 0) & (m < 30)
print("\nMask of (m > 0) & (m < 30) =", mask_combined)
# Output: [ True False False False]
filtered_combined = m[mask_combined]
print("Combined filter =", filtered_combined)
# Output: [23]
```

**✅ 程式碼逐行解析：**

1. `m = np.array([23, -15, 32, 57])`：建立測試陣列，包含四個整數元素。
2. `mask = m < [16, 0, 35, 20]`：進行元素級比較，將 m 的每個元素與對應位置的比較值進行小於比較，返回布林陣列。
    - `print("m < [16, 0, 35, 20] =", mask)`：輸出比較結果的布林陣列。
3. `mask2 = m < 30`：與純量 30 比較，觸發廣播機制，將 30 與 m 的所有元素比較。
    - `print("Mask of m < 30 =", mask2)`：輸出純量比較的布林陣列。
4. `filtered = m[mask2]`：使用布林索引，根據 mask2 選擇 m 中對應 True 的元素。
    - `print("m[m < 30] =", filtered)`：輸出過濾後的陣列。
5. `mask_combined = (m > 0) & (m < 30)`：結合多個條件，使用位元運算子 & 進行 AND 運算，必須用括號包裹每個條件。
    - `print("\nMask of (m > 0) & (m < 30) =", mask_combined)`：輸出結合條件的布林陣列。
    - `filtered_combined = m[mask_combined]`：使用結合的布林遮罩進行索引。
    - `print("Combined filter =", filtered_combined)`：輸出結合過濾後的陣列。

**🎯 重點摘要：**

- **核心功能**：條件運算子（如 `<`, `>`, `==`, `!=` 等）返回布林陣列（遮罩），是進行資料過濾和條件選擇的關鍵工具。
- **布林索引特性**：使用布林遮罩進行索引時，返回的結果總是一維陣列，會丟失原始陣列的形狀（例如無法保留 `(N, M)` 的結構）。
- **多條件組合**：必須使用 NumPy 的位元邏輯運算子（`&` 而非 `and`，`|` 而非 `or`），並以括號包裹每個條件以確保正確的運算優先級。
- **最佳使用情境**：資料清理、條件式元素選擇、統計過濾和條件聚合。

---

## <a id="數學和統計函數"></a>數學和統計函數 (Math and Statistical Functions)

💡 **實際應用情境：** 這些函數在資料分析和統計計算中不可或缺，如計算平均值、標準差等。

### 陣列方法

```python
a_stat = np.array([[-2.2, 3.0, 86], [3, 300, -3]])
print("a_stat =", a_stat)
print("mean =", a_stat.mean()) 
print("max =", a_stat.max())
print("sum =", a_stat.sum())
print("std =", a_stat.std())
print("var =", a_stat.var())
print("mean along axis 1 =", a_stat.mean(axis=1).reshape(-1, 1))

```

**✅ 程式碼逐行解析：**

1. `a_stat = np.array([[-2.2, 3.0, 86], [3, 300, -3]])`：建立二維陣列，維度為 (2, 3)。
2. `print("a_stat =", a_stat)`：輸出陣列。
3. `print("mean =", a_stat.mean())`：計算所有元素的平均值。
4. `print("max =", a_stat.max())`：找到最大值。
5. `print("sum =", a_stat.sum())`：計算總和。
6. `print("std =", a_stat.std())`：計算標準差。
7. `print("var =", a_stat.var())`：計算變異數。
8. `print("mean along axis 1 =", a_stat.mean(axis=1).reshape(-1, 1))`：沿著第 1 軸計算平均值，並重塑為行向量 (column vector)（即形狀為 (n, 1) 的二維陣列）。

**🎯 重點摘要：**

- **核心功能**：NumPy 陣列方法（如 `mean()`、`sum()`、`max()` 等）提供統計運算，*預設對所有元素進行計算*，支援向量化處理以提升效能。
- **潛在問題**：若不指定 `axis` 參數，運算會將多維陣列展平為一維後進行，可能導致意外結果或效能損失。
- **最佳使用情境**：適用於整體資料統計分析，如計算整個資料集的平均值、總和或標準差。

### 指定軸運算

在機器學習中，資料通常是多維的（例如：`[樣本, 時間步, 特徵]` 或 `[批次, 頻道, 高度, 寬度]`）。`axis` 參數允許我們計算特定維度的統計量，例如：計算所有樣本的平均值 (Mean)、或某個時間序列的總和 (Sum)。

```python
mat_axis = np.arange(30).reshape(2, 3, 5)
print("mat_axis =", mat_axis)
print("="*30)
print("mat_axis.shape =", mat_axis.shape)
print("sum axis 0 =", mat_axis.sum(axis=0))
print("sum axis 1 =", mat_axis.sum(axis=1))
print("sum axis (0,2) =", mat_axis.sum(axis=(0, 2)))
```

**✅ 程式碼逐行解析：**

1. `mat_axis = np.arange(30).reshape(2, 3, 5)`：建立 2x3x5 三維陣列。
2. `print("mat_axis.shape =", mat_axis.shape)`：輸出形狀。
3. `print("sum axis 0 =", mat_axis.sum(axis=0))`：沿第 0 軸求和，代表著將兩個「3x5 平面」對應相加。結果陣列的形狀會是 (3, 5)。
4. `print("sum axis 1 =", mat_axis.sum(axis=1))`：沿第 1 軸求和，代表著將三個「2x5 平面」對應相加。結果陣列的形狀會是 (2, 5)。
5. `print("sum axis (0,2) =", mat_axis.sum(axis=(0, 2)))`：沿 (0,2) 軸求和，將第 0 軸和第 2 軸的元素相加，結果陣列的形狀會是 (3, )。

**🎯 重點摘要：**

- **核心功能**：聚合函數（如 sum, mean, max, std）搭配 axis 參數，能計算特定維度的統計量。
- **軸消失原則**：運算時，被指定的 axis 會在結果的形狀中消失。例如，對 (2, 3, 4) 沿 axis=0 求和，結果為 (3, 4)。
- **軸索引**：軸索引從 0 開始，依序往上數。使用負數（如 axis=-1）則表示從最後一個維度開始算。
- **最佳使用情境**：多維資料分析，例如在影像處理中，對 (高, 寬, 頻道) 的陣列沿 axis=2 求平均值，可以得到一張灰階影像。

### 通用函數(Universal Functions / ufunc)

💡 **實際應用情境：** 在 Python 原生語法中，若要計算百萬筆數據的平方或對數，需要寫 `for` 迴圈，速度極慢。NumPy 的 `ufunc` 是在底層 C 語言實作的「向量化」函數，能對陣列中的每個元素進行快速的並行運算。

```python
a = np.array([[-12.5, 3.14, 99], [10, 100, 1000]])
print("Original array:")
print(a)
print("-" * 30)
print("np.sin(a) =", np.sin(a))
print("np.square(a) =", np.square(a))
print("np.abs(a) =", np.abs(a))
print("np.exp(a) =", np.exp(a))
print("np.log(a) =", np.log(a))
```

**✅ 程式碼逐行解析：**

- `np.sin(a)`：計算正弦值，輸入值被視為弧度 (Radians) 而非角度。
- `np.square(a)`：計算元素級平方 ($x^2$)。
- `np.abs(a)`：計算絕對值 ($|x|$)，負數轉正。
- `np.exp(a)`：計算指數函數 ($e^x$)。
- `np.log(a)`：計算自然對數 ($\ln x$)。  
  重要細節：由於陣列包含負數，該位置的結果會變成 `nan`，並且 Python 介面通常會顯示 `RuntimeWarning: invalid value encountered in log`。這是處理真實數據時常見的情況

**🎯 重點摘要：**

- **核心功能**：ufunc 提供高效的「向量化」數學運算，直接作用於整個陣列，效能遠優於 Python 迴圈。
- **常見數學定義**：
    - `np.log` 是**自然對數 ($\ln$)**；若需底數為 10，請用 `np.log10`。
    - 三角函數（如 `sin`, `cos`）皆使用**弧度**。
- **異常處理**：當數學運算定義域不符（如對負數取對數、除以零）時，NumPy 通常不會報錯中斷程式，而是返回 `nan` (Not a Number) 或 `inf` (Infinity) 並給出警告。
- **最佳使用情境**：科學運算、訊號處理、特徵轉換（如將偏斜分佈的數據取 log 轉常態分佈）。

### 二元通用函數 (Binary ufuncs)

💡**實際應用情境：** 二元通用函數接受兩個陣列作為輸入，並進行元素級的運算。這在比較兩組數據（例如：比較預測值與真實值的大小，或取兩張圖像中較亮的像素）時非常有用。

```python
a = np.array([-1, -2, 3, 4])
b = np.array([7, 8, -9, 10])
print("a =", a)
print("b =", b)
print("-" * 30)

# 1. 加法 (等同於 a + b)
print("np.add(a, b) =", np.add(a, b))
# Output: [ 6  6 -6 14]

# 2. 比較運算 (等同於 a > b)
print("np.greater(a, b) =", np.greater(a, b))
# Output: [False False  True False]

# 3. 元素級最大值 (注意：這不是找出整個陣列的最大值)
print("np.maximum(a, b) =", np.maximum(a, b))
# Output: [ 7  8  3 10] (取兩者中較大者)
```

**✅ 程式碼逐行解析：**

1. `np.add(a, b)`：元素級相加。這與運算子 `a + b` 功能完全相同。
2. `np.greater(a, b)`：元素級比較。這與運算子 `a > b` 功能完全相同，返回布林陣列。
3. `np.maximum(a, b)`：逐一比較 `a` 和 `b` 對應位置的元素，並保留較大的那個值。例如第一個位置：-1 vs 7，取 7。

**🎯 重點摘要：**

- **核心功能**：二元 ufunc 進行「一對一」的元素級運算，若陣列形狀不同，會自動觸發廣播機制。
- **常見陷阱 (maximum vs max)**：
    - `np.maximum(a, b)`：是二元運算，比較兩個陣列，返回一個新陣列。
    - `np.max(a)` (或 `a.max()`)：是聚合運算，找出單一陣列中的最大值，返回一個純量。
    - 請勿混用。
- **最佳使用情境**：需要對兩組數據進行邏輯比較或數值篩選（例如 ReLU 函數的實作就是 `np.maximum(0, x)`）。

---

## <a id="陣列索引"></a>陣列索引 (Array Indexing)

💡 **實際應用情境：** 索引是存取和修改陣列元素的關鍵，在資料選擇和特徵工程中廣泛應用。

例如：

- 影像處理，你可能需要提取特定像素區域（如裁剪臉部特徵）；
- 在時間序列分析中，索引能快速篩選特定時間段的資料（如過去一週的銷售記錄）；
- 在機器學習，索引用於交叉驗證 (Cross Validation) 的資料分割或特徵子集選擇。

此外，索引支援就地修改，能快速更新大型資料集而不需重建陣列，這在即時資料處理和記憶體受限的環境中特別重要。

NumPy 支援多種索引方式，包括基本切片（slicing）、花式索引（fancy indexing）和布林索引（boolean indexing），可靈活存取和操作一維向量到多維張量。這些語法不僅提升程式碼可讀性，也優化資料處理效能，適用於各種科學計算與資料分析場景。

### 一維陣列

```python
a = np.linspace(100, 109, 10, dtype=int)
print("Example 1-D array a =", a)
print("-" * 30)

print("a[3] =", a[3])
print("a[3:5] =", a[3:5])
print("a[3:-1] =", a[3:-1])
print("a[3:] =", a[3:])
print("a[:3] =", a[:3])
print("a[3::2] =", a[3::2])
print("a[::-1] =", a[::-1])

a[3] = 99
print(f"a after modification = {a}")

a[1:4] = [-87, 88, 89]
print(f"a after slice assignment = {a}")
```

**✅ 程式碼逐行解析：**

1. `a = np.linspace(100, 109, 10, dtype=int)`：使用 linspace 建立從 100 到 109 的 10 個整數陣列。
2. `print("a[3] =", a[3])`：存取第 4 個元素（索引從 0 開始）。
3. `print("a[3:5] =", a[3:5])`：切片索引 3 到 4（不含 5）。
4. `print("a[3:-1] =", a[3:-1])`：從索引 3 到倒數第 2 個。
5. `print("a[3:] =", a[3:])`： 從索引 3 到結尾。
6. `print("a[:3] =", a[:3])`：前 3 個元素。
7. `print("a[3::2] =", a[3::2])`：從索引 3 開始，取步長(Step)為 2。
8. `print("a[::-1] =", a[::-1])`：反轉陣列。
9. `a[3] = 99`：修改第 4 個元素為 99。
10. `print(f"a after modification = {a}")`：輸出修改後陣列。
11. `a[1:4] = [-87, 88, 89]`：將索引 1 到 3 的元素替換為新值。
12. `print(f"a after slice assignment = {a}")`：輸出切片賦值後陣列。

**🎯 重點摘要：**

- **核心功能**：NumPy 一維陣列支援完整的 Python 索引語法，包括正索引、負索引、切片（含起始、結束和步長）、以及就地修改和切片賦值。這些操作都是 O(1) 或 O(k) 時間複雜度，非常有效率。
- **索引特性**：索引從 0 開始，支援負數從末尾計數；切片返回視圖（view）而非複製，提升記憶體效率；步長允許靈活的資料抽樣。
- **修改機制**：單元素修改和切片賦值都支援，切片賦值時若新值數量不匹配會觸發廣播，但在此例中數量相等。
- **效能優勢**：相較於 Python 列表，NumPy 陣列的索引和切片操作更快，尤其在大資料集上；視圖機制避免不必要的記憶體複製。
- **常見應用**：資料預處理中的子集選取、特徵工程的資料轉換、演算法實作中的陣列操作等。
- **潛在問題**：切片返回視圖，若修改視圖會影響原陣列；大型陣列的切片可能消耗額外記憶體；索引超出範圍會引發 IndexError。
- **最佳使用情境**：需要高效能資料存取和修改的科學計算、資料分析和機器學習任務；適合處理結構化資料如時間序列、訊號或影像的一維表示。

### 多維陣列

```python
b = np.arange(15).reshape(3, 5)
print("Original Shape of this example =", b.shape)
print("b[1, 3] =", b[1, 3])      # 取第 2 列第 4 行元素（定義：列=axis 0，行=axis 1）
print("b[1, :] =", b[1, :])      # 取第 2 列所有行
print("b[:, -1] =", b[:, -1])    # 取所有列的最後一行
```

**✅ 程式碼逐行解析（台灣定義：列=axis 0，行=axis 1）：**

1. `b = np.arange(15).reshape(3, 5)`：建立 3 列 5 行的二維陣列（shape = (3, 5)）。
2. `print("b.shape =", b.shape)`：輸出陣列形狀，顯示為 (3, 5)。
3. `print("b[1, 3] =", b[1, 3])`：存取第 2 列第 4 行的元素（索引從 0 開始）。
4. `print("b[1, :] =", b[1, :])`：取第 2 列的所有行（即第 2 橫列）。
5. `print("b[:, -1] =", b[:, -1])`：取所有列的最後一行（即每一列的最右側元素）。

**🎯 重點摘要：**

- **核心功能**：多維索引使用逗號分隔的索引或切片。
- **潛在問題**：切片返回不同維度的視圖。
- **最佳使用情境**：矩陣和張量操作，如影像處理。

### 花式索引 (Fancy Indexing)

💡**實際應用情境：** 一般的切片（如 3:5）只能選取連續的區域。當我們需要選取不連續的特定列或行（例如：選取第 1、3、5 個樣本，或打亂資料順序）時，就需要使用整數陣列來進行索引，這被稱為「花式索引」。

```python
# 建立一個 (4, 6) 的陣列以便示範
# Axis 0 (列): 0~3
# Axis 1 (行): 0~5
b = np.arange(24).reshape(4, 6)
print("b (Shape: 4, 6) =\n", b)
print("-" * 30)

# 1. 混合索引：花式索引 (Axis 0) + 切片 (Axis 1)
# 選取 Axis 0 的索引 0 和 2 (第1, 3列)
# 選取 Axis 1 的索引 3 到 5 (不含5)
# 結果形狀預期: (2, 2)
print("b[[0,2], 3:5] =\n", b[[0, 2], 3:5])

# 2. 單軸花式索引：任意重排 Axis 1
# 選取所有列，並依序選取 Axis 1 的索引 -1(最後), 2, -3(倒數第3)
# 結果形狀預期: (4, 3)
print("b[:, [-1, 2, -3]] =\n", b[:, [-1, 2, -3]])

# 3. 雙軸花式索引：點對點選取 (Point-wise Selection)
# 這是最容易混淆的部分！
# 它不是選取「列(-1, 2) 與 行(3, 4) 的交叉區域」，而是選取座標點：
# 第一點: (-1, 3) -> (最後一列, 索引3)
# 第二點: ( 2, 4) -> (索引2列, 索引4)
# 結果形狀預期: (2,) -> 一維陣列
print("b[[-1, 2], [3, 4]] =", b[[-1, 2], [3, 4]])
```

**✅ 程式碼逐行解析：**

1. `b[[0, 2], 3:5]`：

- 選取 Axis 0（列）索引 0 和 2，並對 Axis 1（行）切片 3:5。
- 結果是 shape (2, 2) 的子矩陣，保留原有的矩陣結構。
- 適合同時選取多列的連續行區段。

2. `b[:, [-1, 2, -3]]`：

- 選取所有列（:），但行索引順序重組為最後一行（-1）、索引 2、倒數第三行（-3）。
- 結果 shape 為 (4, 3)，可用於重排或抽取特定行。

3. `b[[-1, 2], [3, 4]]`：

- 在兩個軸都提供整數列表時，NumPy 會將它們配對為座標點 (axis0_indices, axis1_indices)。
- 只選取這些座標上的元素，結果是一維陣列（此例 shape 為 (2,)）。
- 適合點對點選取，不會返回子矩陣。
- 例如：選取座標 (-1,3) 和 (2,4) 的值（即最後一列的第4行元素和第3列的第5行元素）。


**🎯 重點摘要：**

**核心差異與最佳實踐：**

- **切片 (Slicing)**：使用 `start:end` 語法，選取連續區域，返回的是原陣列的「視圖 (View)」，修改視圖會影響原陣列。
- **花式索引 (Fancy Indexing)**：使用整數列表 `[list]`，選取不連續元素，返回的是「副本 (Copy)」，修改不會影響原陣列。
- **座標選取陷阱**：同時在兩個軸使用列表索引（如 `b[[r1, r2], [c1, c2]]`）時，行為是「點對點 (Point-wise)」選取，只取 `(r1, c1)` 和 `(r2, c2)`，而非交叉區域。
- **最佳使用情境**：花式索引適合資料打亂、重新排序、或根據複雜條件選取特定樣本。

### 布林索引 (Boolean Indexing)

💡 **實際應用情境：**
除了針對元素值過濾（如 `b > 5`），布林索引也常被用來「選擇特定的樣本或特徵」。例如：若我們有一個長度為 N 的布林陣列 `valid_samples`，可以用它來篩選掉無效的數據行。

```python
b = np.arange(24).reshape(4, 6)
print("b (Shape: 4, 6) =\n", b)
print("-" * 30)

# 1. 針對 Axis 0 (列) 進行篩選
# 陣列有 4 列，遮罩長度必須為 4
mask_axis0 = np.array([True, False, True, False])
# 選取第 0 和第 2 列 (Axis 0)
print("b[mask_axis0, :] =\n", b[mask_axis0, :])

# 2. 針對 Axis 1 (行) 進行篩選
# 陣列有 6 行，遮罩長度必須為 6
# 產生 [False, True, False, True, False, True]
mask_axis1 = np.array([False, True] * 3) 
# 選取第 1, 3, 5 行 (Axis 1)
print("b[:, mask_axis1] =\n", b[:, mask_axis1])

# 3. 針對全陣列條件篩選
# 選取所有奇數元素
# 注意：這種過濾方式會破壞形狀，返回一維陣列
print("b[b % 2 == 1] =", b[b % 2 == 1])
```

**✅ 程式碼逐行解析：**

1.  `mask_axis0 = ...`：建立對應 **Axis 0** 長度的布林遮罩。因為 `b` 有 4 列，所以遮罩長度必須是 **4**。
2.  `b[mask_axis0, :]`：將遮罩應用於第一個維度。`True` 的位置保留，`False` 的位置剔除。結果保留了原本的二維結構。
3.  `mask_axis1 = np.array([False, True] * 3)`：建立對應 **Axis 1** 長度的布林遮罩（長度為 6）。
4.  `b[:, mask_axis1]`：將遮罩應用於第二個維度。
5.  `b[b % 2 == 1]`：這是不指定軸的過濾。NumPy 會先計算 `b % 2 == 1` 得到一個與 `b` 形狀相同的布林矩陣，然後選取所有 `True` 的元素，並**展平為一維陣列**返回。

**🎯 重點摘要：**

- **核心規則 (Shape Matching)**：若要對特定軸進行布林索引，**遮罩的長度必須嚴格等於該軸的長度**，否則會報錯 `IndexError`。
- **維度保留 vs 展平**：
    - `b[mask, :]` (切片式布林索引)：通常會**保留**維度結構（例如從 4x6 變成 2x6）。
    - `b[condition]` (直接條件索引)：總是返回**一維陣列**（因為符合條件的元素在記憶體中可能不連續，無法維持矩陣形狀）。
- **最佳使用情境**：資料清理（剔除無效樣本）、特徵選擇（只保留特定欄位）。

---

## <a id="疊加陣列"></a>疊加與串接陣列 (Stacking & Concatenation)

💡 **實際應用情境：**
在機器學習中，我們常需要將多個資料集（例如訓練集 A 和訓練集 B）合併，或者將多張 2D 圖片堆疊成一個 3D 的批次 (Batch)。NumPy 提供了多種方法來處理這些需求。

```python
# 準備測試資料
# m1: Shape (3, 5)
m1 = np.full((3, 5), 1.0)
# m2: Shape (5, 5) -> 注意：Axis 0 (高度) 與 m1 不同
m2 = np.full((5, 5), 2.0)
# m3: Shape (3, 5) -> 與 m1 形狀完全相同
m3 = np.full((3, 5), 3.0)

print("Original shapes:", m1.shape, m2.shape, m3.shape)
print("-" * 30)

# 1. 垂直串接 (Vertical Stack) - 沿 Axis 0
# 要求：Axis 1 (寬度) 必須一致
q4 = np.vstack((m1, m2, m3))
print(f"vstack (m1, m2, m3): {q4.shape}") 
# 結果: (3+5+3, 5) = (11, 5)

# 2. 水平串接 (Horizontal Stack) - 沿 Axis 1
# 要求：Axis 0 (高度) 必須一致
# 注意：這裡不能放 m2，因為 m2 的 Axis 0 是 5，與其他人(3)不同
q5 = np.hstack((m1, m3))
print(f"hstack (m1, m3):     {q5.shape}") 
# 結果: (3, 5+5) = (3, 10)

# 3. 通用串接 (Concatenate) - 指定 Axis
# axis=0 等同於 vstack
q7 = np.concatenate((m1, m2, m3), axis=0)
print(f"concatenate axis=0:  {q7.shape}")

# 4. 堆疊 (Stack) - 增加新維度
# 要求：所有輸入陣列的形狀必須「完全一致」
q8 = np.stack((m1, m3)) 
print(f"stack (m1, m3):      {q8.shape}")
# 結果: 產生新的 Axis 0 -> (2, 3, 5)
```

**✅ 程式碼逐行解析：**

1.  `m1, m2, m3`：建立三個陣列。注意 `m2` 的高度 (Axis 0) 是 5，與其他兩個不同，這會限制它只能參與垂直串接。
2.  `np.vstack(...)`：沿 **Axis 0 (垂直方向)** 串接。它將陣列「上下」接在一起。
    * 條件：**Axis 1 (寬度)** 必須相同。
    * 結果：列數增加 (3+5+3 = 11)，行數不變。
3.  `np.hstack(...)`：沿 **Axis 1 (水平方向)** 串接。它將陣列「左右」接在一起。
    * 條件：**Axis 0 (高度)** 必須相同。
    * 結果：列數不變，行數增加 (5+5 = 10)。
4.  `np.concatenate(..., axis=0)`：這是最底層的函數，`vstack` 其實就是 `axis=0` 的特例，`hstack` 是 `axis=1` 的特例。
5.  `np.stack(...)`：**這是與前三者最大的不同**。它不是延伸現有維度，而是**插入一個新的 Axis 0**。
    * 條件：輸入陣列的形狀必須**完全相同**。
    * 結果：從 2D 變成 3D。原來的 `(3, 5)` 變成了 `(2, 3, 5)`，代表「2 個 (3, 5) 的矩陣」。

**🎯 重點摘要：**

  * **功能區分**：
    * `concatenate` / `vstack` / `hstack`：**延伸 (Extend)** 現有的維度（2D 拼完還是 2D）。
    * `stack`：**增加 (Increase)** 維度（2D 拼完變 3D）。
  * **形狀匹配規則**：
    * 串接 (Concatenate)：除了「串接軸」之外，其他所有軸的長度必須相等。
    * 堆疊 (Stack)：所有軸的長度都必須嚴格相等。
  * **最佳使用情境**：
    * 合併資料集（增加樣本數）➝ `vstack` / `concatenate`。
    * 合併特徵（增加欄位）➝ `hstack`。
    * 將圖片打包成 Batch（增加批次維度）➝ `stack`。

---

## <a id="分割陣列"></a>分割陣列 (Splitting Arrays)

💡 **實際應用情境：** 分割用於資料分批處理，如將大資料集分成小批次進行訓練。

```python
m = np.arange(48).reshape(6,8)
print("Original example is m =", m)

v1, v2, v3 = np.vsplit(m, 3)
print("v1 =", v1)
print("v2 =", v2)
print("v3 =", v3)

h1, h2 = np.hsplit(m, 2)
print("h1 =", h1)
print("h2 =", h2)
```

**✅ 程式碼逐行解析：**

1. `m = np.arange(48).reshape(6,8)`：建立 6x8 陣列。
2. `print("m =", m)`：輸出原始陣列。
3. `v1, v2, v3 = np.vsplit(m, 3)`：垂直分割成 3 等份。
4. `print("v1 =", v1)`：輸出第一份2x8陣列。
5. `print("v2 =", v2)`：輸出第二份2x8陣列。
6. `print("v3 =", v3)`：輸出第三份2x8陣列。
7. `h1, h2 = np.hsplit(m, 2)`：水平分割成 2 等份。
8. `print("h1 =", h1)`：輸出左半6x4陣列。
9. `print("h2 =", h2)`：輸出右半6x4陣列。

**🎯 重點摘要：**

- **核心功能**：vsplit 垂直分割，hsplit 水平分割。
- **潛在問題**：分割數必須能整除對應維度。
- **最佳使用情境**：資料分批，如交叉驗證或記憶體管理。

---

## <a id="轉置陣列"></a>轉置陣列 (Transposing Arrays)

💡 **實際應用情境：** 轉置在線性代數中至關重要，如矩陣運算和座標變換。

```python
t = np.arange(24).reshape(4,2,3)
print("t.shape =", t.shape)

t1 = t.transpose((1,2,0))
print("t1.shape =", t1.shape)

t2 = t.transpose()
print("t2.shape =", t2.shape)

t3 = t.swapaxes(0,1)
print("t3.shape =", t3.shape)
```

**✅ 程式碼逐行解析：**

1. `t = np.arange(24).reshape(4,2,3)`：建立 4x2x3 三維陣列。
2. `print("t.shape =", t.shape)`：輸出原始形狀。
3. `t1 = t.transpose((1,2,0))`：重新排列軸順序。
    - 原本軸順序為 (0,1,2)，轉置後變為 (1,2,0)。
4. `print("t1.shape =", t1.shape)`：輸出轉置後形狀。
5. `t2 = t.transpose()`：預設反轉軸順序。
    - 原本 (0,1,2) 變為 (2,1,0)。
6. `print("t2.shape =", t2.shape)`：輸出預設轉置形狀。
7. `t3 = t.swapaxes(0,1)`：交換指定兩軸。
    - 將軸 0 和軸 1 互換，變為 (1,0,2)。
8. `print("t3.shape =", t3.shape)`：輸出軸交換後形狀。

**🎯 重點摘要：**

- **核心功能**：transpose 重新排列軸，swapaxes 交換兩軸。
- **潛在問題**：軸索引超出範圍會錯誤。
- **最佳使用情境**：矩陣運算、資料重塑。

---

## <a id="線性代數"></a>線性代數 (Linear Algebra)

💡 **實際應用情境：** 線性代數運算是機器學習和科學計算的核心，如解線性方程組和特徵分解。

### 矩陣運算

```python
m1 = np.arange(15).reshape(3, 5)
print("m1 =", m1)
print("Transpose of m1 (m1.T) =\n", m1.T)

n1 = np.arange(10).reshape(2, 5)
n2 = np.arange(15).reshape(5, 3)
print("n1.dot(n2) =", n1.dot(n2))
```

**✅ 程式碼逐行解析：**

1. `m1 = np.arange(15).reshape(3, 5)`：建立 3x5 矩陣。
2. `print("m1 =", m1)`：輸出矩陣。
3. `print("Transpose of m1 (m1.T) =\n", m1.T)`：輸出轉置矩陣。
4. `n1 = np.arange(10).reshape(2, 5)`：建立第一個2x5矩陣。
5. `n2 = np.arange(15).reshape(5, 3)`：建立第二個5x3矩陣。
6. `print("n1.dot(n2) =", n1.dot(n2))`：計算矩陣乘法，得到2x3矩陣。
    - 注意：矩陣乘法 `n1.dot(n2)` 需要 `n1` 的**行數** (5) 與 `n2` 的**列數** (5) 相同，本例中是正確的，因此可以進行矩陣乘法運算，結果為形狀 (2,3) 的矩陣。

**🎯 重點摘要：**

- **核心功能**：
    - `T` 屬性可快速取得矩陣的轉置（即將行與列互換），常用於線性代數運算。
    - `dot` 方法或 `@` 運算子用於執行矩陣乘法（Matrix Multiplication），遵循線性代數規則：若 A 形狀為 (m, n)，B 形狀為 (n, k)，則 A.dot(B) 結果為 (m, k)。
    - 注意：`*` 運算子僅執行元素級相乘（Element-wise Multiplication），而非矩陣乘法。
- **潛在問題**：
    - 若參與矩陣乘法的陣列維度不相容（如 A 的列數不等於 B 的行數），會引發 ValueError。
    - 初學者常誤用 `*` 進行矩陣乘法，導致結果錯誤。
- **最佳使用情境**：
    - 線性變換（如旋轉、縮放）、特徵工程（如主成分分析）、神經網路層的權重計算、解線性方程組等科學計算場景。

### 求反矩陣和分解 (Matrix Inversion & Decomposition)

💡 **實際應用情境：**  
在機器學習、統計分析和工程計算中，計算反矩陣和分解是解線性方程組、特徵分解、主成份分析（PCA）等核心步驟。例如，解 $Ax = b$ 時需計算 $A$ 的逆矩陣 ($A^{-1}$)，特徵分解則用於資料降維和模式識別。NumPy 的 `linalg` 模組提供的矩陣運算工具，能處理大規模數據並支援多種分解方法。

- **逆矩陣 (`inv`)**：計算方陣的反矩陣，僅適用於非奇異（可逆; non-singular、invertible）矩陣。
- **行列式 (`det`)**：判斷矩陣是否可逆，行列式為 0 表示矩陣奇異。
- **特徵分解 (`eig`)**：取得矩陣的特徵值 (eigenvalues) 與特徵向量 (eigenvectors)，常用於資料降維與系統分析。

這些運算在數值分析、物理模擬、金融建模等領域都非常重要，能幫助你深入理解資料結構與系統行為。

```python
import numpy.linalg as linalg

m3 = np.array([[1,2,3],[0,1,4],[5,6,0]])
print("m3 =", m3)
det_m3 = linalg.det(m3) # Calculate determinant
if det_m3 == 0:
    print("Matrix m3 is singular and cannot be inverted.")
else:
    inv_m3 = linalg.inv(m3)
    print("The determinant is", det_m3)
    print("Inverse is", inv_m3)
    print("Verification (m3.dot(inv_m3)) =\n", m3.dot(inv_m3))

eigenvalues, eigenvectors = linalg.eig(m3)
print("eigenvalues =", eigenvalues)
print("eigenvectors =", eigenvectors)
```

**✅ 程式碼逐行解析：**

1. `import numpy.linalg as linalg`：匯入 NumPy 的線性代數模組，提供矩陣運算函數。
2. `m3 = np.array([[1,2,3],[0,1,4],[5,6,0]])`：建立一個 3x3 矩陣（注意：此矩陣非奇異，可求逆）。
3. `print("m3 =", m3)`：輸出原始矩陣以供檢查。
4. `det_m3 = linalg.det(m3)`：計算矩陣的行列式，用於判斷是否可逆。
5. `if det_m3 == 0:`：條件檢查：若行列式為零，矩陣奇異，無法求逆。
6. `print("Matrix m3 is singular and cannot be inverted.")`：輸出警告訊息。
7. `else:`：若行列式非零，進入求逆分支。
8. `inv_m3 = linalg.inv(m3)`：計算逆矩陣。
9. `print("The determinant is", det_m3)`：輸出行列式值。
10. `print("Inverse is", inv_m3)`：輸出逆矩陣。
11. `print("Verification (m3.dot(inv_m3)) =\n", m3.dot(inv_m3))`：驗證逆矩陣正確性，結果應接近單位矩陣。
12. `eigenvalues, eigenvectors = linalg.eig(m3)`：計算矩陣的特徵值和特徵向量。
13. `print("eigenvalues =", eigenvalues)`：輸出特徵值。
14. `print("eigenvectors =", eigenvectors)`：輸出特徵向量。

**🎯 重點摘要：**

- **核心功能**：矩陣求逆和分解是線性代數的核心，`inv` 計算逆矩陣，`det` 檢查可逆性，`eig` 分解為特徵值和向量。
- **潛在問題**：奇異矩陣（det=0）無逆矩陣；浮點數精度可能導致近似奇異矩陣誤判。
- **最佳使用情境**：解線性方程組、主成分分析、變換矩陣等。

- **核心功能**：inv 求逆，det 行列式，eig 特徵分解。
- **潛在問題**：奇異矩陣無反矩陣。
- **最佳使用情境**：解方程組、主成份分析。

### 奇異值分解 (Singular Value Decomposition, SVD)

奇異值分解（SVD）是線性代數中最重要的矩陣分解技術之一。它能將任意形狀的矩陣 $A$ 分解為三個部分：$A = U \Sigma V^T$，其中 $U$ 和 $V$ 是正交矩陣，$\Sigma$ 是只含奇異值的對角矩陣。SVD 廣泛應用於資料降維（如主成份分析 PCA）、壓縮、去噪、推薦系統等領域。NumPy 的 `linalg.svd` 函數可計算 SVD，適合處理大型資料集和稀疏矩陣。

```python
# 建立 4x5 的非方陣
m4 = np.array([[1, 0, 0, 0, 2], 
               [0, 3, 0, 0, 0], 
               [0, 0, 0, 2, 0], 
               [0, 0, 0, 0, 0]])

# 1. 執行 SVD
# U: (4, 4), S: (4,), Vt: (5, 5)
# 注意：NumPy 回傳的第三個值已經是 V 的轉置 (Vt)
U, S, Vt = np.linalg.svd(m4)

print("U Matrix =\n", U)
print("S (Singular Values) =", S)
print("Vt Matrix =\n", Vt)
print("-" * 30)

# 2. 重建矩陣的關鍵技巧
# 原始矩陣是 (4, 5)，所以中間的 Sigma 矩陣必須建立為 (4, 5) 才能與 Vt 相乘
Sigma = np.zeros((4, 5))

# 將 S 填入對角線 (S 的長度為 4，填入前 4x4 的區域)
Sigma[:4, :4] = np.diag(S)

# 3. 重建
# 公式：A = U @ Sigma @ Vt
# 注意：不需要對 Vt 再次轉置
reconstructed = U @ Sigma @ Vt

print("SVD 重建矩陣 =\n", reconstructed)
print("重建成功？", np.allclose(m4, reconstructed))
```

**✅ 程式碼逐行解析：**

1. `m4 = ...`：建立原始矩陣。觀察可知第 2 行（索引 1）元素 3 最大，這通常會反映在最大的奇異值上。
2. `U, S, Vt = np.linalg.svd(m4)`：
    - `S`：奇異值陣列。計算結果約為 `[3., 2.236, 2., 0.]`，分別對應原矩陣各行的特徵（如 3 來自第二列，2.236 來自第一列 $\sqrt{1^2+2^2}$，2 來自第三列）。
    - `Vt`：已是 $V^T$，不需再轉置。
3. `Sigma = np.zeros((4, 5))`：重建步驟關鍵。直接用 `np.diag(S)` 只會得到 (4, 4) 方陣，無法與 (5, 5) 的 Vt 相乘。需手動建立形狀為 (4, 5) 的 Sigma，並將 S 填入前 4x4 的對角線。
4. `reconstructed = U @ Sigma @ Vt`：矩陣乘法順序為 (4, 4) @ (4, 5) @ (5, 5)，最終得到 (4, 5) 的重建矩陣，與原始 m4 形狀一致。

**🎯 重點摘要：**

- **核心功能**：SVD 分解矩陣為 $U \Sigma V^T$。
- **維度對齊**：當處理非方陣時，必須小心構建 $\Sigma$ 矩陣，通常需要補零行或補零列來匹配維度。
- **物理意義**：觀察輸出的 S，最後一個值為 0（或極小的浮點數），這精確地反映了 m4 的最後一行是全零行（線性相依/無效資訊）。

---

## <a id="向量化"></a>向量化 (Vectorization)

💡 **實際應用情境：**  
向量化是 NumPy 提升效能的核心技巧，能將原本需要巢狀 Python 迴圈的運算（如逐元素計算、資料轉換、數學函數套用）轉換為一次性批次運算。這種方式充分利用底層 C 語言優化，顯著加快資料處理速度並減少記憶體存取延遲。  

在資料科學、影像處理、機器學習等領域，向量化能讓你用一行程式碼完成複雜的數值運算，例如：  

- 對整張影像進行濾鏡處理（如 `img = np.clip(img * 1.2 + 10, 0, 255)`），無需逐像素迴圈  
- 批次計算所有樣本的特徵轉換（如 `X_norm = (X - X.mean(axis=0)) / X.std(axis=0)`）  
- 建立座標網格並進行函數運算（如 `Z = np.sin(X * Y / 40.5)`）  

這不僅提升程式碼可讀性，也讓大規模資料分析和模型訓練變得可行。實務上，建議盡量將所有資料處理步驟向量化，僅在必要時才使用 Python 迴圈。

```python
import math
import numpy as np

# 低效率方式
data_loop = np.empty((768, 1024))
for y in range(768):
    for x in range(1024):
        data_loop[y, x] = math.tan(x + y)

# 高效率方式
x_coords = np.arange(0, 1024)
y_coords = np.arange(0, 768)
X, Y = np.meshgrid(x_coords, y_coords)
data_vec = np.tan(X + Y)

print("Shapes match:", data_loop.shape == data_vec.shape)
print("Results match:", np.allclose(data_loop, data_vec))
```

**✅ 程式碼逐行解析：**

1. `import math`：匯入數學模組。
2. `data_loop = np.empty((768, 1024))`：建立空陣列。
3. 迴圈賦值：使用巢狀迴圈計算每個元素（效率低）。
4. `x_coords = np.arange(0, 1024)`：建立 x 座標陣列。
5. `y_coords = np.arange(0, 768)`：建立 y 座標陣列。
6. `X, Y = np.meshgrid(x_coords, y_coords)`：建立座標網格。
7. `data_vet = np.tan(X + Y)`：向量化計算。
8. 比較結果：確認兩種方法結果相同。

**🎯 重點摘要：**

- **核心功能**：meshgrid 建立座標網格，向量化運算避免迴圈。
- **潛在問題**：記憶體使用量大於迴圈方式。
- **最佳使用情境**：數值計算、影像處理。

**補充說明：**
使用 Matplotlib 視覺化向量化結果：

```python
import matplotlib.pyplot as plt
# 視覺化結果
fig = plt.figure(1, figsize=(8, 6))
plt.imshow(data_vec, cmap="autumn")
plt.show()
# 其他常用色彩映射 (cmap)：
# "gray"      灰階
# "viridis"   綠-藍-黃，現代預設
# "plasma"    紫-黃，對比強烈
# "inferno"   黑-紅-黃，暗背景
# "magma"     黑-紫-橘，柔和
# "jet"       彩虹色（不建議科學用途）
# "cool"      青-粉紅
# "hot"       黑-紅-黃-白
# "spring"    粉紅-黃
# "autumn"    紅-黃
# "winter"    藍-綠
```

---

## <a id="儲存和載入"></a>儲存和載入 (Saving and Loading)

💡 **實際應用情境：**  
在機器學習和資料科學工作流程中，資料持久化 (persistence) 是不可或缺的步驟。常見用途包括：  

- 儲存訓練好的模型權重，以便日後載入並進行預測或微調。
- 保存中間計算結果（如特徵工程後的陣列、驗證集預測值），避免重複運算，提升效率。
- 交換資料給團隊成員或跨平台工具（如將 NumPy 陣列匯出為 CSV 供 Excel 或 R 使用）。
- 長期保存原始資料集或處理後的資料，方便版本管理與重現實驗。

選擇合適的儲存格式（如二進位、文字或壓縮）能兼顧效能、精確度與可攜性，是專業資料處理流程的重要一環。

### 二進位格式 (Binary Format)

NumPy 的二進位格式（.npy）能完整保存陣列的形狀、資料類型和內容，適合高效率地儲存與載入大型科學資料。與文字格式相比，二進位格式讀寫速度更快且不會有精度損失，特別適合模型權重、特徵矩陣等需精確重現的場景。若需儲存多個陣列，可使用壓縮格式（.npz），將多個命名陣列打包於同一檔案，方便資料管理與交換。

```python
a = np.random.rand(3,3)*2-1
print("a =", a)
np.save("my_array", a)

a_loaded = np.load("my_array.npy")
print("a_loaded =", a_loaded)
print("Arrays equal:", np.array_equal(a, a_loaded))
```

**✅ 程式碼逐行解析：**

1. `a = np.random.rand(3,3)*2-1`：建立隨機陣列，範圍在 -1 到 1 之間。
2. `print("a =", a)`：輸出原始陣列。
3. `np.save("my_array", a)`：將陣列 `a` 儲存為二進位格式的 `.npy` 檔案。
    - 此方法會自動在檔名後加上 `.npy` 副檔名，能完整保存陣列的形狀、資料類型和內容
    - 注意：若檔案已存在，將直接覆蓋且不會警告。
4. `a_loaded = np.load("my_array.npy")`：載入檔案。
5. `print("a_loaded =", a_loaded)`：輸出載入陣列。
6. `print("Arrays equal:", np.array_equal(a, a_loaded))`：檢查是否相同。

**🎯 重點摘要：**

- **核心功能**：save/load 以二進位格式儲存單一陣列。
- **潛在問題**：檔案覆蓋無警告。
- **最佳使用情境**：單一陣列持久化。

### 文字格式 (Text Format)

NumPy 支援將陣列儲存為純文字格式（如 CSV、TXT），方便與 Excel、R、Pandas 等其他工具交換資料。這種格式可直接用文字編輯器檢視，但儲存速度較慢且可能有精度損失（尤其是浮點數）。適合小型資料集或跨平台資料交換。

- `np.savetxt`：將陣列儲存為文字檔（可指定分隔符、格式）。
- `np.loadtxt`：從文字檔載入陣列（可指定分隔符、資料類型）。

常見應用：
- 匯出分析結果給非 Python 用戶
- 與資料庫、試算表或其他程式語言互通
- 檢查資料內容或進行手動編輯

注意事項：
- 儲存時可用 `fmt` 參數控制數值格式（如 `fmt="%.6f"` 保留 6 位小數）。
- 載入時若資料有標題列，可用 `skiprows` 跳過。
- 文字格式不會保存陣列形狀資訊，僅儲存元素本身，載入時需自行重塑形狀（如 `.reshape()`）。
- 若資料包含非數值型態（如字串），需額外指定 `dtype=str`。

範例：
- 儲存為 CSV：`np.savetxt("data.csv", arr, delimiter=",")`
- 載入 CSV：`arr = np.loadtxt("data.csv", delimiter=",")`
- 儲存為 TXT：`np.savetxt("data.txt", arr, fmt="%.6f")`
- 載入 TXT：`arr = np.loadtxt("data.txt")`
- 若需儲存多維陣列，建議先展平或重塑為 2D，再儲存。

```python
np.savetxt("my_array.csv", a, delimiter=",")
with open("my_array.csv", "rt") as f:
    print("CSV content:")
    print(f.read())

a_loaded_txt = np.loadtxt("my_array.csv", delimiter=",")
print("Loaded from CSV:", a_loaded_txt)
print("CSV arrays equal:", np.allclose(a, a_loaded_txt))
```

**✅ 程式碼逐行解析：**

1. `np.savetxt("my_array.csv", a, delimiter=",")`：以逗號分隔符儲存陣列為 CSV 格式。
2. `with open("my_array.csv", "rt") as f:`：以文字模式開啟 CSV 檔案。
3. `print("CSV content:")`：印出標題。
4. `print(f.read())`：讀取並印出檔案內容。
5. `a_loaded_txt = np.loadtxt("my_array.csv", delimiter=",")`：從 CSV 載入陣列，使用逗號分隔符。
6. `print("Loaded from CSV:", a_loaded_txt)`：輸出載入的陣列。
7. `print("CSV arrays equal:", np.allclose(a, a_loaded_txt))`：檢查載入陣列是否與原始陣列相等。

**🎯 重點摘要：**

- **核心功能**：savetxt/loadtxt 以文字格式儲存，可讀但較慢。
- **潛在問題**：精確度可能損失。
- **最佳使用情境**：與其他工具交換資料。

### 壓縮格式 (Compressed Format)

NumPy 的壓縮格式（`.npz`）允許一次儲存多個陣列於單一檔案，並以壓縮方式減少磁碟空間使用。這對於保存模型參數、特徵集合或多組資料特別有用。每個陣列都可指定名稱（key），載入時可依名稱存取，方便管理和交換資料。

**主要特點：**

- 支援多個陣列同時儲存，並以鍵值（key）管理。
- 檔案自動壓縮，適合大型資料集。
- 載入後為類字典物件（dict-like），可用 `keys()` 查詢所有陣列名稱。
- 適合保存模型多層權重、資料集分批結果等。

```python
b = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
np.savez("my_arrays", my_a=a, my_b=b)

my_arrays = np.load("my_arrays.npz")
print("Keys:", list(my_arrays.keys()))
print("my_a =", my_arrays["my_a"])
```

**✅ 程式碼逐行解析：**

1. `b = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)`：建立另一個陣列。
2. `np.savez("my_arrays", my_a=a, my_b=b)`：儲存多個陣列到壓縮檔案。
3. `my_arrays = np.load("my_arrays.npz")`：載入壓縮檔案。
4. `print("Keys:", list(my_arrays.keys()))`：輸出鍵名。
5. `print("my_a =", my_arrays["my_a"])`：存取特定陣列。

**🎯 重點摘要：**

- **核心功能**：savez 儲存多個陣列到單一壓縮檔案。
- **潛在問題**：鍵名管理複雜。
- **最佳使用情境**：
    - 儲存多個模型權重或中間結果。
    - 打包多組特徵資料，方便跨平台交換。
    - 長期保存多個相關陣列，便於版本管理。
- **注意事項**：
    - 儲存時建議使用具意義的鍵名（如 `train_X`, `train_y`）。
    - 載入後需以鍵名存取陣列（如 `arrays["train_X"]`）。
    - 若未指定鍵名，則以 `arr_0`, `arr_1`...自動命名。

---

## ❓ 常見問答 (FAQ)

**Q: NumPy 陣列和 Python 列表有什麼區別？**  
A: NumPy 陣列更有效率，支援向量化運算和廣播，所有元素必須同類型。

**Q: 如何選擇適當的資料類型？**  
A: 根據資料範圍選擇，如影像用 uint8，科學計算用 float64。

**Q: 廣播規則是什麼？**  
A: 較小陣列自動擴展以匹配較大陣列形狀，從後往前比較維度。

**Q: 為什麼要避免 Python 迴圈？**  
A: 迴圈在 Python 中效率低，NumPy 向量化運算利用 C 語言優化。

**Q: 如何處理記憶體不足？**  
A: 使用適當資料類型，考慮記憶體映射或分批處理。

---

## 🏷️ 最佳實踐

- **效能優化**：優先使用向量化運算，避免 Python 迴圈
- **記憶體管理**：選擇最小適當資料類型，使用 in-place 操作
- **程式碼清晰**：使用有意義的變數名，添加註釋
- **錯誤處理**：檢查陣列形狀相容性，處理數值異常
- **測試驗證**：使用 np.allclose 比較浮點數結果
- **文件記錄**：記錄陣列形狀和資料類型資訊

---

## 🏷️ 推薦標籤 (Suggested Hashtags)

#Python #程式設計 #教學 #NumPy #陣列 #線性代數 #科學計算 #編程 #學習筆記 #程式開發者 #軟體工程
