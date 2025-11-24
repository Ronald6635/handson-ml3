<!-- meta-title: 🐍 NumPy 工具教學：從陣列建立到線性代數 -->
<!-- meta-description: 學習 NumPy 的完整指南，從基本陣列操作到進階線性代數功能。包含實用範例、逐行解析和最佳實踐。 -->
<!-- meta-keywords: NumPy, Python, 陣列, 線性代數, 科學計算, 教學 -->
<!-- meta-hashtags: #Python #程式設計 #教學 #NumPy #陣列 #線性代數 #科學計算 #編程 #學習筆記 #程式開發者 #軟體工程 -->

# 🐍 NumPy 完整教學指南：陣列建立、操作與線性代數實戰技巧

NumPy 是 Python 科學計算領域的核心函式庫，專為高效能 N 維陣列運算設計。

其底層以 C 語言實作，提供極快的數值處理速度，並支援線性代數、傅立葉變換、隨機數生成等進階功能。

NumPy 不僅是資料科學、機器學習、人工智慧等領域的基礎工具，也是許多 Python 標準函式庫和第三方套件的效能關鍵。透過 Python 封裝，使用者能輕鬆進行大規模數據分析與科學運算，提升開發效率與程式效能。

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

## 🎯 關鍵重點

- NumPy 以高效能的 N 維陣列為核心，提供廣泛的數學和科學計算功能
- 掌握陣列建立、索引、運算和重塑是使用 NumPy 的基礎
- 廣播機制允許不同形狀陣列間的靈活運算
- 向量化操作比迴圈更有效率，能充分利用 NumPy 的優化
- 線性代數模組提供矩陣運算、特徵值等進階功能

## <a id="建立陣列"></a>建立陣列

💡 **實際應用情境：** 在資料科學和機器學習中，陣列是處理數值資料的基本單位，從簡單的向量到複雜的張量，都依賴 NumPy 的陣列建立功能。

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

### 從序列建立

```python
# 使用 arange
arange_int = np.arange(1, 5)
print("arange_int =", arange_int)

arange_float = np.arange(1.0, 5.0)
print("arange_float =", arange_float)

arange_step = np.arange(1, 5, 0.5)
print("arange_step =", arange_step)

# 使用 linspace
linspace = np.linspace(0, 5/3, 6)
print("linspace =", linspace)

# 從 Python 陣列
array_from_list = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
print("array_from_list =", array_from_list)
```

**✅ 程式碼逐行解析：**

1. `arange_int = np.arange(1, 5)`：建立從 1 到 4 的整數陣列（不包含 5）。
2. `print("arange_int =", arange_int)`：輸出整數序列陣列。
3. `arange_float = np.arange(1.0, 5.0)`：建立從 1.0 到 5.0 的浮點數陣列。
4. `print("arange_float =", arange_float)`：輸出浮點數序列陣列。
5. `arange_step = np.arange(1, 5, 0.5)`：建立從 1 到 5 步長為 0.5 的陣列。
6. `print("arange_step =", arange_step)`：輸出帶步長的序列陣列。
7. `linspace = np.linspace(0, 5/3, 6)`：建立從 0 到 5/3 的 6 個等間距點陣列（包含終點）。
8. `print("linspace =", linspace)`：輸出等間距陣列。
9. `array_from_list = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])`：從巢狀列表建立二維陣列。
10. `print("array_from_list =", array_from_list)`：輸出從列表建立的陣列。

**🎯 重點摘要：**

- **核心功能**：arange 類似 range 但返回陣列，linspace 建立等間距點，array 從序列轉換。
- **潛在問題**：浮點數 arange 可能因精確度問題不包含終點，linspace 更適合浮點數範圍。
- **最佳使用情境**：arange 用於整數序列，linspace 用於需要精確點數的浮點數範圍。

### 隨機陣列

```python
# 均勻分佈隨機數
rand = np.random.rand(3, 4)
print("rand =", rand)

# 常態分佈隨機數
randn = np.random.randn(3, 4)
print("randn =", randn)
```

**✅ 程式碼逐行解析：**

1. `rand = np.random.rand(3, 4)`：建立形狀為 (3, 4) 的隨機浮點數陣列，範圍在 [0, 1) 間的均勻分佈。
2. `print("rand =", rand)`：輸出均勻分佈隨機陣列。
3. `randn = np.random.randn(3, 4)`：建立形狀為 (3, 4) 的隨機浮點數陣列，均值 0 標準差 1 的常態分佈。
4. `print("randn =", randn)`：輸出常態分佈隨機陣列。

**🎯 重點摘要：**

- **核心功能**：rand 產生均勻分佈，randn 產生常態分佈的隨機數。
- **潛在問題**：隨機數每次執行結果不同，測試時需設定種子確保重現性。
- **最佳使用情境**：模擬資料、初始化權重、隨機取樣等需要隨機性的應用。

### 使用函數建立

```python
def my_function(z, y, x):
    return x + 10 * y + 100 * z

result = np.fromfunction(my_function, (3, 2, 10))
print("fromfunction result shape:", result.shape)
print("result =", result)
```

**✅ 程式碼逐行解析：**

1. `def my_function(z, y, x):`：定義一個函數，接受 z, y, x 三個參數。
2. `return x + 10 * y + 100 * z`：返回計算結果，z 權重最大。
3. `result = np.fromfunction(my_function, (3, 2, 10))`：使用 fromfunction 建立形狀為 (3, 2, 10) 的陣列，每個元素由函數計算。
4. `print("fromfunction result shape:", result.shape)`：輸出結果陣列的形狀。
5. `print("result =", result)`：輸出完整的結果陣列。

**🎯 重點摘要：**

- **核心功能**：fromfunction 根據座標函數建立陣列，避免顯式迴圈。
- **潛在問題**：函數參數順序與形狀軸順序相反，需小心對應。
- **最佳使用情境**：需要根據位置計算值的陣列，如網格座標、距離矩陣。

## <a id="陣列資料"></a>陣列資料
💡 **實際應用情境：** 理解陣列的資料類型和記憶體佈局對於效能優化和資料處理至關重要。

### 資料類型

```python
c = np.arange(1, 5)
print("c.dtype =", c.dtype, "c =", c)

c_float = np.arange(1.0, 5.0)
print("c_float.dtype =", c_float.dtype, "c_float =", c_float)

c_complex = np.arange(1, 5, dtype=np.complex64)
print("c_complex.dtype =", c_complex.dtype, "c_complex =", c_complex)
```

**✅ 程式碼逐行解析：**

1. `c = np.arange(1, 5)`：建立從 1 到 4 的整數陣列，預設為 int32 或 int64。
2. `print("c.dtype =", c.dtype, "c =", c)`：輸出陣列的資料類型和內容。
3. `c_float = np.arange(1.0, 5.0)`：建立浮點數陣列，自動推斷為 float64。
4. `print("c_float.dtype =", c_float.dtype, "c_float =", c_float)`：輸出浮點數陣列的類型和內容。
5. `c_complex = np.arange(1, 5, dtype=np.complex64)`：明確指定複數類型建立陣列。
6. `print("c_complex.dtype =", c_complex.dtype, "c_complex =", c_complex)`：輸出複數陣列的類型和內容。

**🎯 重點摘要：**

- **核心功能**：NumPy 陣列有統一資料類型，可顯式指定 dtype。
- **潛在問題**：不同類型混合運算會自動提升，可能導致記憶體使用增加。
- **最佳使用情境**：根據資料範圍選擇適當類型，如 uint8 用於影像，float32 用於一般計算。

### 記憶體資訊

```python
e = np.arange(1, 5, dtype=np.complex64)
print("e.itemsize =", e.itemsize)

f = np.array([[1, 2], [1000, 2000]], dtype=np.int32)
print("f.data (first 20 bytes) =", f.data[:20])
```

**✅ 程式碼逐行解析：**

1. `e = np.arange(1, 5, dtype=np.complex64)`：建立複數陣列。
2. `print("e.itemsize =", e.itemsize)`：輸出每個元素的位元組大小（複數 64 為 8 位元組）。
3. `f = np.array([[1, 2], [1000, 2000]], dtype=np.int32)`：建立 int32 二維陣列。
4. `print("f.data (first 20 bytes) =", f.data[:20])`：輸出陣列資料緩衝區的前 20 位元組。

**🎯 重點摘要：**

- **核心功能**：itemsize 顯示元素大小，data 提供原始位元組存取。
- **潛在問題**：直接操作 data 緩衝區可能導致資料損壞，應謹慎使用。
- **最佳使用情境**：檢查記憶體使用量或與低階程式設計介面互動時。

## <a id="重塑陣列"></a>重塑陣列
💡 **實際應用情境：** 陣列重塑在資料預處理和張量操作中非常常見，如將一維資料重塑為影像矩陣。

### 就地重塑

```python
g = np.arange(24)
print("g =", g)
print("g.shape =", g.shape)

g.shape = (6, 4)
print("after reshape g =", g)
print("g.shape =", g.shape)

g.shape = (2, 3, 4)
print("3D g =", g)
```

**✅ 程式碼逐行解析：**

1. `g = np.arange(24)`：建立包含 0 到 23 的陣列。
2. `print("g =", g)`：輸出原始一維陣列。
3. `print("g.shape =", g.shape)`：輸出原始形狀 (24,)。
4. `g.shape = (6, 4)`：將形狀修改為 (6, 4)，元素數必須相同。
5. `print("after reshape g =", g)`：輸出重塑後的陣列。
6. `print("g.shape =", g.shape)`：輸出新形狀。
7. `g.shape = (2, 3, 4)`：進一步重塑為三維。
8. `print("3D g =", g)`：輸出三維陣列。

**🎯 重點摘要：**

- **核心功能**：直接修改 shape 屬性進行就地重塑，元素數必須保持不變。
- **潛在問題**：形狀乘積不等於元素數會引發錯誤。
- **最佳使用情境**：需要改變陣列維度結構但保持資料不變的情況。

### 使用 reshape

```python
g2 = g.reshape(4, 6)
print("g2 =", g2)

g2[1, 2] = 999
print("g after g2 modification =", g)
```

**✅ 程式碼逐行解析：**

1. `g2 = g.reshape(4, 6)`：建立 g 的重塑視圖，形狀為 (4, 6)。
2. `print("g2 =", g2)`：輸出重塑後的陣列。
3. `g2[1, 2] = 999`：修改 g2 的元素。
4. `print("g after g2 modification =", g)`：顯示原始陣列 g 也被修改，因為共享資料。

**🎯 重點摘要：**

- **核心功能**：reshape 返回新視圖，與原陣列共享資料。
- **潛在問題**：修改任一陣列都會影響另一個，造成意外副作用。
- **最佳使用情境**：需要不同形狀視圖但不想複製資料時。

### ravel

```python
flat = g.ravel()
print("flat =", flat)
```

**✅ 程式碼逐行解析：**

1. `flat = g.ravel()`：將多維陣列展平為一維，共享資料。
2. `print("flat =", flat)`：輸出展平後的陣列。

**🎯 重點摘要：**

- **核心功能**：ravel 返回一維視圖，適用於需要線性存取的情況。
- **潛在問題**：同樣共享資料，修改會影響原陣列。
- **最佳使用情境**：需要將多維資料視為序列處理時。

## <a id="算術運算"></a>算術運算
💡 **實際應用情境：** 元素級運算在向量化和矩陣計算中無處不在，如影像處理和神經網路前向傳播。

```python
a = np.array([14, 23, 32, 41])
b = np.array([5, 4, 3, 2])
print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)
print("a // b =", a // b)
print("a % b =", a % b)
print("a ** b =", a ** b)
```

**✅ 程式碼逐行解析：**

1. `a = np.array([14, 23, 32, 41])`：建立第一個陣列。
2. `b = np.array([5, 4, 3, 2])`：建立第二個陣列。
3. `print("a + b =", a + b)`：元素級相加。
4. `print("a - b =", a - b)`：元素級相減。
5. `print("a * b =", a * b)`：元素級相乘（非矩陣乘法）。
6. `print("a / b =", a / b)`：元素級相除。
7. `print("a // b =", a // b)`：元素級整數除法。
8. `print("a % b =", a % b)`：元素級取餘數。
9. `print("a ** b =", a ** b)`：元素級冪運算。

**🎯 重點摘要：**

- **核心功能**：所有算術運算子都進行元素級操作。
- **潛在問題**：容易與矩陣乘法混淆，NumPy 的 * 是元素積而非點積。
- **最佳使用情境**：需要對應元素間運算的數值計算，如調整亮度、縮放座標。

## <a id="廣播"></a>廣播
💡 **實際應用情境：** 廣播允許不同形狀陣列的運算，在機器學習中常見於批次處理和特徵縮放。

### 廣播規則

```python
h = np.arange(5).reshape(1, 1, 5)
print("h.shape =", h.shape)

result = h + [10, 20, 30, 40, 50]
print("broadcast result =", result)
```

**✅ 程式碼逐行解析：**

1. `h = np.arange(5).reshape(1, 1, 5)`：建立形狀為 (1, 1, 5) 的陣列。
2. `print("h.shape =", h.shape)`：輸出 h 的形狀。
3. `result = h + [10, 20, 30, 40, 50]`：與一維陣列相加，觸發廣播。
4. `print("broadcast result =", result)`：輸出廣播結果。

**🎯 重點摘要：**

- **核心功能**：自動擴展較小陣列以匹配較大陣列的形狀。
- **潛在問題**：形狀不相容會引發錯誤，需理解規則避免。
- **最佳使用情境**：批次運算，如將向量加到每個矩陣行。

### 更多廣播範例

```python
k = np.arange(6).reshape(2, 3)
print("k =", k)

result2 = k + [[100], [200]]
print("k + [[100], [200]] =", result2)

result3 = k + [100, 200, 300]
print("k + [100, 200, 300] =", result3)

result4 = k + 1000
print("k + 1000 =", result4)
```

**✅ 程式碼逐行解析：**

1. `k = np.arange(6).reshape(2, 3)`：建立 2x3 陣列。
2. `print("k =", k)`：輸出 k。
3. `result2 = k + [[100], [200]]`：每行加不同值。
4. `print("k + [[100], [200]] =", result2)`：輸出結果。
5. `result3 = k + [100, 200, 300]`：每列加不同值。
6. `print("k + [100, 200, 300] =", result3)`：輸出結果。
7. `result4 = k + 1000`：所有元素加同一值。
8. `print("k + 1000 =", result4)`：輸出結果。

**🎯 重點摘要：**

- **核心功能**：廣播適用於多種形狀組合，自動處理維度擴展。
- **潛在問題**：複雜廣播可能難以理解，建議用簡單規則。
- **最佳使用情境**：特徵標準化、批次偏移等需要一致運算的情況。

### 類型提升

```python
k1 = np.arange(0, 5, dtype=np.uint8)
k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)
print("k1.dtype =", k1.dtype, "k2.dtype =", k2.dtype)

k3 = k1 + 1.5
print("k3.dtype =", k3.dtype)
```

**✅ 程式碼逐行解析：**

1. `k1 = np.arange(0, 5, dtype=np.uint8)`：建立 uint8 陣列。
2. `k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)`：與 int8 陣列相加，類型提升。
3. `print("k1.dtype =", k1.dtype, "k2.dtype =", k2.dtype)`：輸出類型變化。
4. `k3 = k1 + 1.5`：與浮點數相加，進一步提升。
5. `print("k3.dtype =", k3.dtype)`：輸出最終類型。

**🎯 重點摘要：**

- **核心功能**：自動類型提升確保運算正確，但可能增加記憶體使用。
- **潛在問題**：意外提升導致效能下降或精確度問題。
- **最佳使用情境**：混合類型運算時，NumPy 會自動選擇合適類型。

## <a id="條件運算子"></a>條件運算子
💡 **實際應用情境：** 條件運算是資料過濾和布林索引的基礎，在資料分析中用於選擇滿足條件的值。

```python
m = np.array([20, -5, 30, 40])
mask = m < [15, 16, 35, 36]
print("m =", m)
print("m < [15, 16, 35, 36] =", mask)

mask2 = m < 25
print("m < 25 =", mask2)

filtered = m[m < 25]
print("m[m < 25] =", filtered)
```

**✅ 程式碼逐行解析：**

1. `m = np.array([20, -5, 30, 40])`：建立測試陣列。
2. `mask = m < [15, 16, 35, 36]`：元素級比較，返回布林陣列。
3. `print("m =", m)`：輸出原始陣列。
4. `print("m < [15, 16, 35, 36] =", mask)`：輸出比較結果。
5. `mask2 = m < 25`：與純量比較，觸發廣播。
6. `print("m < 25 =", mask2)`：輸出布林遮罩。
7. `filtered = m[m < 25]`：使用布林索引過濾元素。
8. `print("m[m < 25] =", filtered)`：輸出過濾結果。

**🎯 重點摘要：**

- **核心功能**：條件運算返回布林陣列，可用於索引和過濾。
- **潛在問題**：布林索引返回一維陣列，失去原形狀。
- **最佳使用情境**：資料清理、條件選擇、統計過濾。

## <a id="數學和統計函數"></a>數學和統計函數
💡 **實際應用情境：** 這些函數在資料分析和統計計算中不可或缺，如計算平均值、標準差等。

### 陣列方法

```python
a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
print("a =", a)
print("mean =", a.mean())
print("max =", a.max())
print("sum =", a.sum())
print("std =", a.std())
print("var =", a.var())
```

**✅ 程式碼逐行解析：**

1. `a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])`：建立二維陣列。
2. `print("a =", a)`：輸出陣列。
3. `print("mean =", a.mean())`：計算所有元素的平均值。
4. `print("max =", a.max())`：找到最大值。
5. `print("sum =", a.sum())`：計算總和。
6. `print("std =", a.std())`：計算標準差。
7. `print("var =", a.var())`：計算變異數。

**🎯 重點摘要：**

- **核心功能**：陣列方法提供常見統計運算，預設處理所有元素。
- **潛在問題**：不指定 axis 會展平陣列進行運算。
- **最佳使用情境**：整體統計分析，如資料集的平均值。

### 指定軸運算

```python
c = np.arange(24).reshape(2, 3, 4)
print("c.shape =", c.shape)
print("sum axis 0 =", c.sum(axis=0))
print("sum axis 1 =", c.sum(axis=1))
print("sum axis (0,2) =", c.sum(axis=(0, 2)))
```

**✅ 程式碼逐行解析：**

1. `c = np.arange(24).reshape(2, 3, 4)`：建立 2x3x4 三維陣列。
2. `print("c.shape =", c.shape)`：輸出形狀。
3. `print("sum axis 0 =", c.sum(axis=0))`：沿第 0 軸求和。
4. `print("sum axis 1 =", c.sum(axis=1))`：沿第 1 軸求和。
5. `print("sum axis (0,2) =", c.sum(axis=(0, 2)))`：沿多個軸求和。

**🎯 重點摘要：**

- **核心功能**：axis 參數控制運算維度，提供靈活的統計計算。
- **潛在問題**：軸索引從 0 開始，負數表示從後往前。
- **最佳使用情境**：多維資料分析，如計算每行/列的統計值。

### 通用函數

```python
a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
print("Original array:")
print(a)
print("np.square(a) =", np.square(a))
print("np.abs(a) =", np.abs(a))
print("np.exp(a) =", np.exp(a))
print("np.log(a) =", np.log(a))
print("np.cos(a) =", np.cos(a))
```

**✅ 程式碼逐行解析：**

1. `a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])`：建立測試陣列。
2. `print("Original array:")`：標記原始陣列。
3. `print(a)`：輸出陣列。
4. `print("np.square(a) =", np.square(a))`：計算平方。
5. `print("np.abs(a) =", np.abs(a))`：計算絕對值。
6. `print("np.exp(a) =", np.exp(a))`：計算指數。
7. `print("np.log(a) =", np.log(a))`：計算對數（負數有警告）。
8. `print("np.cos(a) =", np.cos(a))`：計算餘弦。

**🎯 重點摘要：**

- **核心功能**：ufunc 提供向量化數學函數，效能優於迴圈。
- **潛在問題**：某些函數如 log 在無效輸入時返回 nan。
- **最佳使用情境**：數值計算、訊號處理、科學模擬。

### 二元通用函數

```python
a = np.array([1, -2, 3, 4])
b = np.array([2, 8, -1, 7])
print("a =", a)
print("b =", b)
print("np.add(a, b) =", np.add(a, b))
print("np.greater(a, b) =", np.greater(a, b))
print("np.maximum(a, b) =", np.maximum(a, b))
```

**✅ 程式碼逐行解析：**

1. `a = np.array([1, -2, 3, 4])`：建立第一個陣列。
2. `b = np.array([2, 8, -1, 7])`：建立第二個陣列。
3. `print("a =", a)`：輸出 a。
4. `print("b =", b)`：輸出 b。
5. `print("np.add(a, b) =", np.add(a, b))`：元素級相加。
6. `print("np.greater(a, b) =", np.greater(a, b))`：元素級大於比較。
7. `print("np.maximum(a, b) =", np.maximum(a, b))`：元素級最大值。

**🎯 重點摘要：**

- **核心功能**：二元 ufunc 進行元素級二元運算，支援廣播。
- **潛在問題**：運算子重載可能導致混淆，如 + 與 np.add。
- **最佳使用情境**：需要元素級比較或組合的進階運算。

## <a id="陣列索引"></a>陣列索引
💡 **實際應用情境：** 索引是存取和修改陣列元素的關鍵，在資料選擇和特徵工程中廣泛應用。

### 一維陣列

```python
a = np.array([1, 5, 3, 19, 13, 7, 3])
print("a =", a)
print("a[3] =", a[3])
print("a[2:5] =", a[2:5])
print("a[2:-1] =", a[2:-1])
print("a[:2] =", a[:2])
print("a[2::2] =", a[2::2])
print("a[::-1] =", a[::-1])

a[3] = 999
print("a after modification =", a)

a[2:5] = [997, 998, 999]
print("a after slice assignment =", a)
```

**✅ 程式碼逐行解析：**

1. `a = np.array([1, 5, 3, 19, 13, 7, 3])`：建立一維陣列。
2. `print("a =", a)`：輸出完整陣列。
3. `print("a[3] =", a[3])`：存取第 4 個元素（索引從 0 開始）。
4. `print("a[2:5] =", a[2:5])`：切片索引 2 到 4。
5. `print("a[2:-1] =", a[2:-1])`：從索引 2 到倒數第 2 個。
6. `print("a[:2] =", a[:2])`：前 2 個元素。
7. `print("a[2::2] =", a[2::2])`：從索引 2 開始步長 2。
8. `print("a[::-1] =", a[::-1])`：反轉陣列。
9. `a[3] = 999`：修改單一元素。
10. `print("a after modification =", a)`：輸出修改後陣列。
11. `a[2:5] = [997, 998, 999]`：切片賦值。
12. `print("a after slice assignment =", a)`：輸出切片賦值後陣列。

**🎯 重點摘要：**

- **核心功能**：支援標準 Python 索引語法，包括負索引和切片。
- **潛在問題**：切片賦值會廣播單一值到整個切片。
- **最佳使用情境**：資料存取、修改和子集選擇。

### 多維陣列

```python
b = np.arange(48).reshape(4, 12)
print("b.shape =", b.shape)
print("b[1, 2] =", b[1, 2])
print("b[1, :] =", b[1, :])
print("b[:, 1] =", b[:, 1])
```

**✅ 程式碼逐行解析：**

1. `b = np.arange(48).reshape(4, 12)`：建立 4x12 陣列。
2. `print("b.shape =", b.shape)`：輸出形狀。
3. `print("b[1, 2] =", b[1, 2])`：存取第 2 行第 3 列元素。
4. `print("b[1, :] =", b[1, :])`：第 2 行所有列。
5. `print("b[:, 1] =", b[:, 1])`：所有行第 2 列。

**🎯 重點摘要：**

- **核心功能**：多維索引使用逗號分隔的索引或切片。
- **潛在問題**：切片返回不同維度的視圖。
- **最佳使用情境**：矩陣和張量操作，如影像處理。

### 花式索引

```python
print("b[(0,2), 2:5] =", b[(0,2), 2:5])
print("b[:, (-1, 2, -1)] =", b[:, (-1, 2, -1)])
print("b[(-1, 2, -1, 2), (5, 9, 1, 9)] =", b[(-1, 2, -1, 2), (5, 9, 1, 9)])
```

**✅ 程式碼逐行解析：**

1. `print("b[(0,2), 2:5] =", b[(0,2), 2:5])`：選擇第 1 和 3 行，第 3 到 5 列。
2. `print("b[:, (-1, 2, -1)] =", b[:, (-1, 2, -1)])`：所有行，第 -1、2、-1 列。
3. `print("b[(-1, 2, -1, 2), (5, 9, 1, 9)] =", b[(-1, 2, -1, 2), (5, 9, 1, 9)])`：指定座標的元素。

**🎯 重點摘要：**

- **核心功能**：花式索引允許任意元素選擇，返回一維陣列。
- **潛在問題**：索引陣列長度必須相同。
- **最佳使用情境**：複雜資料選擇，如非連續元素存取。

### 布林索引

```python
rows_on = np.array([True, False, True, False])
print("b[rows_on, :] =", b[rows_on, :])

cols_on = np.array([False, True, False] * 4)
print("b[:, cols_on] =", b[:, cols_on])

print("b[b % 3 == 1] =", b[b % 3 == 1])
```

**✅ 程式碼逐行解析：**

1. `rows_on = np.array([True, False, True, False])`：建立行遮罩。
2. `print("b[rows_on, :] =", b[rows_on, :])`：選擇特定行。
3. `cols_on = np.array([False, True, False] * 4)`：建立列遮罩。
4. `print("b[:, cols_on] =", b[:, cols_on])`：選擇特定列。
5. `print("b[b % 3 == 1] =", b[b % 3 == 1])`：選擇除以 3 餘 1 的元素。

**🎯 重點摘要：**

- **核心功能**：布林索引根據條件選擇元素。
- **潛在問題**：遮罩形狀必須與對應維度匹配。
- **最佳使用情境**：條件過濾，如選擇大於閾值的元素。

## <a id="疊加陣列"></a>疊加陣列
💡 **實際應用情境：** 陣列疊加在資料合併和批次處理中常見，如合併多個資料集或建立三維張量。

```python
q1 = np.full((3,4), 1.0)
q2 = np.full((4,4), 2.0)
q3 = np.full((3,4), 3.0)

q4 = np.vstack((q1, q2, q3))
print("q4.shape =", q4.shape)

q5 = np.hstack((q1, q3))
print("q5.shape =", q5.shape)

q7 = np.concatenate((q1, q2, q3), axis=0)
print("q7.shape =", q7.shape)

q8 = np.stack((q1, q3))
print("q8.shape =", q8.shape)
```

**✅ 程式碼逐行解析：**

1. `q1 = np.full((3,4), 1.0)`：建立 3x4 的全 1 陣列。
2. `q2 = np.full((4,4), 2.0)`：建立 4x4 的全 2 陣列。
3. `q3 = np.full((3,4), 3.0)`：建立 3x4 的全 3 陣列。
4. `q4 = np.vstack((q1, q2, q3))`：垂直疊加，形狀必須在非堆疊軸匹配。
5. `print("q4.shape =", q4.shape)`：輸出疊加後形狀。
6. `q5 = np.hstack((q1, q3))`：水平疊加。
7. `print("q5.shape =", q5.shape)`：輸出水平疊加形狀。
8. `q7 = np.concatenate((q1, q2, q3), axis=0)`：沿軸 0 串接，等同 vstack。
9. `print("q7.shape =", q7.shape)`：輸出串接形狀。
10. `q8 = np.stack((q1, q3))`：沿新軸疊加，形狀必須完全相同。
11. `print("q8.shape =", q8.shape)`：輸出堆疊形狀。

**🎯 重點摘要：**

- **核心功能**：vstack 垂直、hstack 水平、concatenate 指定軸、stack 新軸。
- **潛在問題**：形狀不匹配會引發錯誤。
- **最佳使用情境**：資料合併，如合併批次或特徵。

## <a id="分割陣列"></a>分割陣列
💡 **實際應用情境：** 分割用於資料分批處理，如將大資料集分成小批次進行訓練。

```python
r = np.arange(24).reshape(6,4)
print("r =", r)

r1, r2, r3 = np.vsplit(r, 3)
print("r1 =", r1)
print("r2 =", r2)
print("r3 =", r3)

r4, r5 = np.hsplit(r, 2)
print("r4 =", r4)
print("r5 =", r5)
```

**✅ 程式碼逐行解析：**

1. `r = np.arange(24).reshape(6,4)`：建立 6x4 陣列。
2. `print("r =", r)`：輸出原始陣列。
3. `r1, r2, r3 = np.vsplit(r, 3)`：垂直分割成 3 等份。
4. `print("r1 =", r1)`：輸出第一份。
5. `print("r2 =", r2)`：輸出第二份。
6. `print("r3 =", r3)`：輸出第三份。
7. `r4, r5 = np.hsplit(r, 2)`：水平分割成 2 等份。
8. `print("r4 =", r4)`：輸出左半。
9. `print("r5 =", r5)`：輸出右半。

**🎯 重點摘要：**

- **核心功能**：vsplit 垂直分割，hsplit 水平分割。
- **潛在問題**：分割數必須能整除對應維度。
- **最佳使用情境**：資料分批，如交叉驗證或記憶體管理。

## <a id="轉置陣列"></a>轉置陣列
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
4. `print("t1.shape =", t1.shape)`：輸出轉置後形狀。
5. `t2 = t.transpose()`：預設反轉軸順序。
6. `print("t2.shape =", t2.shape)`：輸出預設轉置形狀。
7. `t3 = t.swapaxes(0,1)`：交換指定兩軸。
8. `print("t3.shape =", t3.shape)`：輸出軸交換後形狀。

**🎯 重點摘要：**

- **核心功能**：transpose 重新排列軸，swapaxes 交換兩軸。
- **潛在問題**：軸索引超出範圍會錯誤。
- **最佳使用情境**：矩陣運算、資料重塑。

## <a id="線性代數"></a>線性代數
💡 **實際應用情境：** 線性代數運算是機器學習和科學計算的核心，如解線性方程組和特徵分解。

### 矩陣運算

```python
m1 = np.arange(10).reshape(2,5)
print("m1 =", m1)
print("m1.T =", m1.T)

n1 = np.arange(10).reshape(2, 5)
n2 = np.arange(15).reshape(5,3)
print("n1.dot(n2) =", n1.dot(n2))
```

**✅ 程式碼逐行解析：**

1. `m1 = np.arange(10).reshape(2,5)`：建立 2x5 矩陣。
2. `print("m1 =", m1)`：輸出矩陣。
3. `print("m1.T =", m1.T)`：輸出轉置矩陣。
4. `n1 = np.arange(10).reshape(2, 5)`：建立第一個矩陣。
5. `n2 = np.arange(15).reshape(5,3)`：建立第二個矩陣。
6. `print("n1.dot(n2) =", n1.dot(n2))`：計算矩陣乘法。

**🎯 重點摘要：**

- **核心功能**：T 屬性轉置，dot 方法矩陣乘法。
- **潛在問題**：維度不匹配會錯誤，* 是元素積而非矩陣積。
- **最佳使用情境**：線性變換、神經網路層。

### 矩陣求逆和分解

```python
import numpy.linalg as linalg

m3 = np.array([[1,2,3],[5,7,11],[21,29,31]])
print("m3 =", m3)
print("linalg.inv(m3) =", linalg.inv(m3))
print("linalg.det(m3) =", linalg.det(m3))

eigenvalues, eigenvectors = linalg.eig(m3)
print("eigenvalues =", eigenvalues)
print("eigenvectors =", eigenvectors)
```

**✅ 程式碼逐行解析：**

1. `import numpy.linalg as linalg`：匯入線性代數模組。
2. `m3 = np.array([[1,2,3],[5,7,11],[21,29,31]])`：建立 3x3 矩陣。
3. `print("m3 =", m3)`：輸出矩陣。
4. `print("linalg.inv(m3) =", linalg.inv(m3))`：計算逆矩陣。
5. `print("linalg.det(m3) =", linalg.det(m3))`：計算行列式。
6. `eigenvalues, eigenvectors = linalg.eig(m3)`：計算特徵值和特徵向量。
7. `print("eigenvalues =", eigenvalues)`：輸出特徵值。
8. `print("eigenvectors =", eigenvectors)`：輸出特徵向量。

**🎯 重點摘要：**

- **核心功能**：inv 求逆，det 行列式，eig 特徵分解。
- **潛在問題**：奇異矩陣無逆矩陣。
- **最佳使用情境**：解方程組、主成分分析。

### 奇異值分解

```python
m4 = np.array([[1,0,0,0,2], [0,0,3,0,0], [0,0,0,0,0], [0,2,0,0,0]])
U, S_diag, V = linalg.svd(m4)
print("U =", U)
print("S_diag =", S_diag)
print("V =", V)
```

**✅ 程式碼逐行解析：**

1. `m4 = np.array([[1,0,0,0,2], [0,0,3,0,0], [0,0,0,0,0], [0,2,0,0,0]])`：建立稀疏矩陣。
2. `U, S_diag, V = linalg.svd(m4)`：進行 SVD 分解。
3. `print("U =", U)`：輸出 U 矩陣。
4. `print("S_diag =", S_diag)`：輸出奇異值對角線。
5. `print("V =", V)`：輸出 V 矩陣。

**🎯 重點摘要：**

- **核心功能**：SVD 分解矩陣為 U Σ V^T。
- **潛在問題**：計算成本高於其他分解。
- **最佳使用情境**：資料壓縮、降維。

## <a id="向量化"></a>向量化
💡 **實際應用情境：** 向量化是 NumPy 效能優化的關鍵，避免 Python 迴圈的低效率。

```python
import math

# 低效率方式
data_slow = np.empty((768, 1024))
for y in range(768):
    for x in range(1024):
        data_slow[y, x] = math.sin(x * y / 40.5)

# 高效率方式
x_coords = np.arange(0, 1024)
y_coords = np.arange(0, 768)
X, Y = np.meshgrid(x_coords, y_coords)
data_fast = np.sin(X * Y / 40.5)

print("Shapes match:", data_slow.shape == data_fast.shape)
print("Results match:", np.allclose(data_slow, data_fast))
```

**✅ 程式碼逐行解析：**

1. `import math`：匯入數學模組。
2. `data_slow = np.empty((768, 1024))`：建立空陣列。
3. 迴圈賦值：使用巢狀迴圈計算每個元素（效率低）。
4. `x_coords = np.arange(0, 1024)`：建立 x 座標陣列。
5. `y_coords = np.arange(0, 768)`：建立 y 座標陣列。
6. `X, Y = np.meshgrid(x_coords, y_coords)`：建立座標網格。
7. `data_fast = np.sin(X * Y / 40.5)`：向量化計算。
8. 比較結果：確認兩種方法結果相同。

**🎯 重點摘要：**

- **核心功能**：meshgrid 建立座標網格，向量化運算避免迴圈。
- **潛在問題**：記憶體使用量大於迴圈方式。
- **最佳使用情境**：數值計算、影像處理。

## <a id="儲存和載入"></a>儲存和載入
💡 **實際應用情境：** 資料持久化在機器學習工作流程中不可或缺，如儲存模型權重或中間結果。

### 二進位格式

```python
a = np.random.rand(2,3)
print("a =", a)
np.save("my_array", a)

a_loaded = np.load("my_array.npy")
print("a_loaded =", a_loaded)
print("Arrays equal:", np.array_equal(a, a_loaded))
```

**✅ 程式碼逐行解析：**

1. `a = np.random.rand(2,3)`：建立隨機陣列。
2. `print("a =", a)`：輸出原始陣列。
3. `np.save("my_array", a)`：儲存為 .npy 檔案。
4. `a_loaded = np.load("my_array.npy")`：載入檔案。
5. `print("a_loaded =", a_loaded)`：輸出載入陣列。
6. `print("Arrays equal:", np.array_equal(a, a_loaded))`：檢查是否相同。

**🎯 重點摘要：**

- **核心功能**：save/load 以二進位格式儲存單一陣列。
- **潛在問題**：檔案覆蓋無警告。
- **最佳使用情境**：單一陣列持久化。

### 文字格式

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

### 壓縮格式

```python
b = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
np.savez("my_arrays", my_a=a, my_b=b)

my_arrays = np.load("my_arrays.npz")
print("Keys:", list(my_arrays.keys()))
print("my_a =", my_arrays["my_a"])
```

**✅ 程式碼逐行解析：**

1. `b = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)`：建立另一個陣列。
2. `np.savez("my_arrays", my_a=a, my_b=b)`：儲存多個陣列到壓縮檔案。
3. `my_arrays = np.load("my_arrays.npz")`：載入壓縮檔案。
4. `print("Keys:", list(my_arrays.keys()))`：輸出鍵名。
5. `print("my_a =", my_arrays["my_a"])`：存取特定陣列。

**🎯 重點摘要：**

- **核心功能**：savez 儲存多個陣列到單一壓縮檔案。
- **潛在問題**：鍵名管理複雜。
- **最佳使用情境**：儲存相關陣列集合。

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

## 🏷️ 最佳實踐

- **效能優化**：優先使用向量化運算，避免 Python 迴圈
- **記憶體管理**：選擇最小適當資料類型，使用 in-place 操作
- **程式碼清晰**：使用有意義的變數名，添加註釋
- **錯誤處理**：檢查陣列形狀相容性，處理數值異常
- **測試驗證**：使用 np.allclose 比較浮點數結果
- **文件記錄**：記錄陣列形狀和資料類型資訊

## 🏷️ 推薦標籤 (Suggested Hashtags)

#Python #程式設計 #教學 #NumPy #陣列 #線性代數 #科學計算 #編程 #學習筆記 #程式開發者 #軟體工程
