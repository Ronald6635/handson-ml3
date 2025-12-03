"""
NumPy 工具教學模組

此模組示範 NumPy 的核心功能，從陣列建立到線性代數。

關鍵功能：
- 陣列建立與操作
- 數學運算與廣播
- 索引、切片與重塑
- 統計函數與線性代數
- 資料儲存與載入

範例基於 tools_numpy.md 文件
"""

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import font_manager

# =============================================================================
# 中文字型設定 (CHINESE FONT CONFIGURATION)
# =============================================================================

# 候選中文字型清單 (嘗試找到系統可用的字型)
CJK_FONT_CANDIDATES = [
    "Microsoft JhengHei",  # Windows 繁體中文
    "Microsoft YaHei",     # Windows 簡體中文
    "SimHei",              # Windows 簡體黑體
    "Arial Unicode MS",    # 跨平台 Unicode 字型
    "Heiti TC",            # macOS 繁體中文
]

# 檢查可用字型
available_fonts = []
for font_name in CJK_FONT_CANDIDATES:
    if font_name in [f.name for f in font_manager.fontManager.ttflist]:
        available_fonts.append(font_name)

# 若無可用中文字型，使用系統預設
if not available_fonts:
    print("Warning: No Chinese font found, using default font")
    available_fonts = ["sans-serif"]

# 設定 Matplotlib 使用中文字型
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = available_fonts
plt.rcParams["axes.unicode_minus"] = False  # 正確顯示負號

# =============================================================================
# 範例 1：建立陣列 (ARRAY CREATION)
# =============================================================================

print("=" * 60)
print("範例 1：建立陣列")
print("=" * 60)

# 匯入 NumPy 模組並簡稱為 np (標準慣例)
import numpy as np

# 建立一個包含 6 個 0 的一維陣列
a = np.zeros(6)
print("a =", a)

# 建立 3x5 矩陣 (3 列 5 行)，所有元素為 0
b = np.zeros((3, 5))
print("b =", b)
print("b.shape =", b.shape)    # 輸出形狀 (3, 5)
print("b.ndim =", b.ndim)      # 輸出維度數 2
print("b.size =", b.size)      # 輸出總元素數 15

# 建立全 1 陣列 (形狀 3x8)
ones = np.ones((3, 8))
print("ones =", ones)

# 建立指定值的陣列 (形狀 3x3，所有元素為 π)
full = np.full((3, 3), np.pi)
print("full =", full)

# 建立未初始化的陣列 (內容不可預測，僅用於立即覆寫的場景)
empty = np.empty((2, 3))
print("empty =", empty)

# 1. 使用 arange 建立序列 (類似 Python range)
# 整數序列：[1, 2, 3, 4, 5] (不包含 6，左閉右開區間)
arange_int = np.arange(1, 6)
print("arange_int =", arange_int)

# 浮點數序列：[1., 2., 3., 4., 5.]
arange_float = np.arange(1.0, 6.0)
print("arange_float =", arange_float)

# 指定步長 0.5：[1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5]
arange_step = np.arange(1, 6, 0.5)
print("arange_step =", arange_step)

# 2. arange vs linspace 比較
# arange 對浮點數的終點判定較不穩定 (不建議用於浮點數範圍)
arange_like_linspace = np.arange(0, 10/3, (10/3)/5)
print("arange_like_linspace =", arange_like_linspace)

# 3. 使用 linspace 建立等間距點 (推薦用於浮點數範圍)
# 在 0 到 10/3 之間建立 6 個等間距點 (包含終點)
linspace = np.linspace(0, 10/3, 6)
print("linspace =", linspace)

# 4. 從 Python 列表轉換為 NumPy 陣列
# 巢狀列表 [[...], [...]] 轉換為二維陣列
array_from_list = np.array([[1, 2, 3, 4], [6, 7, 8, 9]])
print("array_from_list =\n", array_from_list)
print("array_from_list.shape =", array_from_list.shape)  # (2, 4): 2 列 4 行

# 建立均勻分佈的隨機數陣列 (範圍 [0, 1))
rand = np.random.rand(3, 5)
print("rand =", rand)

# 建立常態分佈的隨機數陣列 (均值 0，標準差 1)
randn = np.random.randn(3, 5)
print("randn =", randn)

# 視覺化隨機分佈 (繪製直方圖比較均勻分佈與常態分佈)
plt.hist(np.random.rand(100000), density=True, bins=100, histtype="step", 
         color="blue", label="均勻分佈 (rand)")
plt.hist(np.random.randn(100000), density=True, bins=100, histtype="step", 
         color="red", label="常態分佈 (randn)")
plt.axis((-2.5, 2.5, 0, 1.1))
plt.legend(loc="upper left")
plt.title("隨機分佈比較")
plt.xlabel("數值")
plt.ylabel("密度")
plt.show()

# 使用函數建立陣列 (fromfunction)
# 定義計算規則：依據座標 (z, y, x) 計算元素值
def my_function(z, y, x):
    return x + 10 * y + 100 * z  # 個位數=x, 十位數=y, 百位數=z

# 建立 (3, 2, 10) 的三維陣列，元素值由座標計算
print("np.fromfunction(my_function, (3, 2, 10)) =")
print(np.fromfunction(my_function, (3, 2, 10)))

print("\n=== 範例 1 完成 ===\n")

# =============================================================================
# 範例 2：陣列資料類型 (DATA TYPES)
# =============================================================================

print("=" * 60)
print("範例 2：陣列資料類型")
print("=" * 60)

# 建立整數陣列 (預設為 int32 或 int64)
c = np.arange(1, 5)
print("c.dtype =", c.dtype, "c =", c)

# 建立浮點數陣列 (自動推斷為 float64)
c = np.arange(1.0, 5.0)
print("c_float.dtype =", c.dtype, "c =", c)

# 明確指定複數類型 (complex64)
d = np.arange(1, 5, dtype=np.complex64)
print("d.dtype =", d.dtype, "d =", d)

# 檢查每個元素的位元組大小 (complex64 為 8 位元組)
e = np.arange(1, 5, dtype=np.complex64)
print("e.itemsize =", e.itemsize)

# 建立 int32 二維陣列
f = np.array([[1, 2], [1000, 2000]], dtype=np.int32)
print("f.data =", f.data)

# 存取原始位元組資料 (謹慎使用，直接操作可能導致資料損壞)
if hasattr(f.data, "tobytes"):
    data_bytes = f.data.tobytes()  # 新版 NumPy
else:
    data_bytes = f.data  # 舊版 NumPy
print("data_bytes =", data_bytes)

print("\n=== 範例 2 完成 ===\n")

# =============================================================================
# 範例 3：重塑陣列 (RESHAPING ARRAYS)
# =============================================================================

print("=" * 60)
print("範例 3：重塑陣列")
print("=" * 60)

# 建立包含 0 到 23 的一維陣列
g = np.arange(24)
print("g =", g)
print("Rank:", g.ndim)  # 維度數 1

# 就地重塑：直接修改 shape 屬性 (元素數必須相同)
g.shape = (6, 4)
print("g 重新塑形後 =", g)
print("Rank:", g.ndim)  # 維度數 2

# 進一步重塑為三維
g.shape = (2, 3, 4)
print("g 三維 =", g)

# 使用 reshape 方法建立新視圖 (與原陣列共享資料)
g2 = g.reshape(4, 6)
print("g2 =", g2)
print("Rank:", g2.ndim)

# 修改 g2 會影響原始陣列 g (因為共享資料)
g2[1, 2] = 999
print("g2 修改後 =", g2)
print("原始 g =", g)  # g 也被修改

# 使用 ravel 展平陣列 (通常返回視圖，速度快)
print("g.ravel() =", g.ravel())

print("\n=== 範例 3 完成 ===\n")

# =============================================================================
# 範例 4：算術運算 (ARITHMETIC OPERATIONS)
# =============================================================================

print("=" * 60)
print("範例 4：算術運算")
print("=" * 60)

# 建立兩個形狀相同的陣列
a = np.array([14, 23, 32, 41])
b = np.array([5, 4, 3, 2])

# 元素級運算 (element-wise operations)
print("a + b  =", a + b)   # 元素對應相加
print("a - b  =", a - b)   # 元素對應相減
print("a * b  =", a * b)   # 元素對應相乘 (非矩陣乘法)
print("a / b  =", a / b)   # 元素對應除法 (結果為浮點數)
print("a // b  =", a // b) # 整數除法 (地板除法)
print("a % b  =", a % b)   # 取餘數
print("a ** b =", a ** b)  # 指數運算

print("\n=== 範例 4 完成 ===\n")

# =============================================================================
# 範例 5：廣播 (BROADCASTING)
# =============================================================================

print("=" * 60)
print("範例 5：廣播")
print("=" * 60)

# 建立形狀為 (1, 1, 5) 的三維陣列
h = np.arange(5).reshape(1, 1, 5)
print("h =", h)

# 與一維陣列相加 (觸發廣播)
print("h + [10, 20, 30, 40, 50] =", h + [10, 20, 30, 40, 50])

# 建立 2x3 陣列
k = np.arange(6).reshape(2, 3)
print("k =", k)

# 列向量廣播 (形狀 (2, 1)，沿 Axis 1 擴展)
print("k + [[100], [200]] =", k + [[100], [200]])

# 行向量廣播 (形狀 (3,)，沿 Axis 0 擴展)
print("k + [100, 200, 300] =", k + [100, 200, 300])

# 純量廣播 (所有元素加同一值)
print("k + 1000 =", k + 1000)

# 嘗試不相容的廣播 (會引發 ValueError)
try:
    k + np.array([33, 44])
except ValueError as e:
    print("廣播錯誤:", e)

# 型別提升範例
k1 = np.arange(0, 5, dtype=np.uint8)  # 8 位元無符號整數
print("k1.dtype =", k1.dtype, "k1 =", k1)

# 與 int8 陣列相加 (觸發型別提升)
k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)
print("k2.dtype =", k2.dtype, "k2 =", k2)

# 與浮點數相加 (提升為 float64)
k3 = k1 + 1.5
print("k3.dtype =", k3.dtype, "k3 =", k3)

print("\n=== 範例 5 完成 ===\n")

# =============================================================================
# 範例 6：條件運算子 (CONDITIONAL OPERATORS)
# =============================================================================

print("=" * 60)
print("範例 6：條件運算子")
print("=" * 60)

# 建立測試陣列
m = np.array([20, -5, 30, 40])
print("m =", m)

# 元素級比較 (返回布林陣列)
print("m < [15, 16, 35, 36] =", m < [15, 16, 35, 36])

# 純量比較 (觸發廣播)
print("m < 25 =", m < 25)

# 布林索引 (選擇滿足條件的元素)
print("m[m < 25] =", m[m < 25])

print("\n=== 範例 6 完成 ===\n")

# =============================================================================
# 範例 7：數學和統計函數 (MATH AND STATISTICAL FUNCTIONS)
# =============================================================================

print("=" * 60)
print("範例 7：數學和統計函數")
print("=" * 60)

# 建立二維陣列
a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
print("a =", a)

# 計算所有元素的平均值
print("mean =", a.mean())

# 各種統計函數
for func in (a.min, a.max, a.sum, a.prod, a.std, a.var):
    print(func.__name__, "=", func())

# 沿指定軸運算
c = np.arange(24).reshape(2, 3, 4)
print("c =", c)
print("c.sum(axis=0) =", c.sum(axis=0))  # 沿 Axis 0 求和 (結果形狀 (3, 4))
print("c.sum(axis=1) =", c.sum(axis=1))  # 沿 Axis 1 求和 (結果形狀 (2, 4))
print("c.sum(axis=(0,2)) =", c.sum(axis=(0, 2)))  # 沿 Axis 0 和 2 求和 (結果形狀 (3,))

# 驗證計算 (手動計算 axis=(0,2) 的結果)
print("驗證計算:", 0+1+2+3 + 12+13+14+15, 4+5+6+7 + 16+17+18+19, 8+9+10+11 + 20+21+22+23)

# 通用函數 (Universal Functions / ufunc)
a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
print("np.square(a) =", np.square(a))  # 元素平方

print("Original ndarray")
print(a)
# 各種數學函數
for func in (np.abs, np.sqrt, np.exp, np.log, np.sign, np.ceil, np.modf, np.isnan, np.cos):
    print("\n", func.__name__)
    print(func(a))

# 二元通用函數 (Binary ufuncs)
a = np.array([1, -2, 3, 4])
b = np.array([2, 8, -1, 7])
print("np.add(a, b) =", np.add(a, b))      # 元素級相加
print("np.greater(a, b) =", np.greater(a, b))  # 元素級比較
print("np.maximum(a, b) =", np.maximum(a, b))  # 逐元素取最大值
print("np.copysign(a, b) =", np.copysign(a, b))  # 複製符號

print("\n=== 範例 7 完成 ===\n")

# =============================================================================
# 範例 8：陣列索引 (ARRAY INDEXING)
# =============================================================================

print("=" * 60)
print("範例 8：陣列索引")
print("=" * 60)

# 建立一維陣列
a = np.array([1, 5, 3, 19, 13, 7, 3])
print("a =", a)

# 基本索引 (索引從 0 開始)
print("a[3] =", a[3])      # 第 4 個元素
print("a[2:5] =", a[2:5])  # 切片 [3, 5) (不含 5)
print("a[2:-1] =", a[2:-1]) # 從索引 2 到倒數第 2 個
print("a[:2] =", a[:2])    # 前 2 個元素
print("a[2::2] =", a[2::2]) # 從索引 2 開始，步長 2
print("a[::-1] =", a[::-1]) # 反轉陣列

# 修改單個元素
a[3] = 999
print("a 修改後 =", a)

# 切片賦值 (長度必須匹配)
a[2:5] = [997, 998, 999]
print("a 切片賦值後 =", a)

# 切片賦值長度不匹配會報錯
try:
    a[2:5] = [1, 2]
except ValueError as e:
    print("切片賦值錯誤:", e)

# 步長切片賦值長度不匹配會報錯
try:
    a[2:5:2] = [1, 2]
except ValueError as e:
    print("步長切片賦值錯誤:", e)

# 切片返回視圖 (修改視圖會影響原陣列)
a_slice = a[2:6]
a_slice[1] = 1000
print("a_slice 修改後的 a =", a)

# 修改原陣列會影響視圖
a[3] = 2000
print("a 修改後的 a_slice =", a_slice)

# 使用 copy() 建立獨立副本
another_slice = a[2:6].copy()
another_slice[1] = 3000
print("複製切片修改後的 a =", a)

a[3] = 4000
print("a 修改後的複製切片 =", another_slice)

# 多維陣列索引
b = np.arange(48).reshape(4, 12)
print("b =", b)
print("b[1, 2] =", b[1, 2])  # 第 2 列第 3 行 (索引從 0 開始)
print("b[1, :] =", b[1, :])  # 第 2 列所有行
print("b[:, 1] =", b[:, 1])  # 所有列的第 2 行

# 單索引 vs 切片的差異
print("b[1, :] =", b[1, :])    # 返回一維陣列
print("b[1:2, :] =", b[1:2, :]) # 返回二維陣列 (保留維度)

# 花式索引 (Fancy Indexing)
print("b[(0,2), 2:5] =", b[(0,2), 2:5])  # 選取第 1 和 3 列的第 3-5 行
print("b[:, (-1, 2, -1)] =", b[:, (-1, 2, -1)])  # 重排行順序
print("b[(-1, 2, -1, 2), (5, 9, 1, 9)] =", b[(-1, 2, -1, 2), (5, 9, 1, 9)])  # 點對點選取

# 高維陣列索引
c = b.reshape(4, 2, 6)
print("c =", c)
print("c[2, 1, 4] =", c[2, 1, 4])  # 三維索引
print("c[2, :, 3] =", c[2, :, 3])  # 部分索引
print("c[2, 1] =", c[2, 1])        # 省略最後一維

# 使用省略號 (Ellipsis)
print("c[2, ...] =", c[2, ...])      # 等同於 c[2, :, :]
print("c[2, 1, ...] =", c[2, 1, ...]) # 等同於 c[2, 1, :]
print("c[2, ..., 3] =", c[2, ..., 3]) # 等同於 c[2, :, 3]
print("c[..., 3] =", c[..., 3])      # 等同於 c[:, :, 3]

# 布林索引
b = np.arange(48).reshape(4, 12)
rows_on = np.array([True, False, True, False])
print("b[rows_on, :] =", b[rows_on, :])  # 選取第 1 和 3 列

cols_on = np.array([False, True, False] * 4)
print("b[:, cols_on] =", b[:, cols_on])  # 選取第 2, 5, 8, 11 行

# 使用 np.ix_ 進行布林索引的交叉選取
print("b[np.ix_(rows_on, cols_on)] =", b[np.ix_(rows_on, cols_on)])
print("np.ix_(rows_on, cols_on) =", np.ix_(rows_on, cols_on))

# 條件式布林索引 (返回一維陣列)
print("b[b % 3 == 1] =", b[b % 3 == 1])

# 迭代陣列
c = np.arange(24).reshape(2, 3, 4)
print("c =", c)

# 迭代第一個軸
for m in c:
    print("sub-array:\n", m)

# 使用 range 迭代
for i in range(len(c)):
    print("index", i, ":\n", c[i])

# 使用 flat 迭代所有元素
for i in c.flat:
    print("element:", i)

print("\n=== 範例 8 完成 ===\n")

# =============================================================================
# 範例 9：疊加陣列 (STACKING ARRAYS)
# =============================================================================

print("=" * 60)
print("範例 9：疊加陣列")
print("=" * 60)

# 建立測試陣列
q1 = np.full((3, 4), 1.0)
print("q1 =", q1)

q2 = np.full((4, 4), 2.0)
print("q2 =", q2)

q3 = np.full((3, 4), 3.0)
print("q3 =", q3)

# 垂直疊加 (Vertical Stack) - 沿 Axis 0
q4 = np.vstack((q1, q2, q3))
print("q4 =", q4)
print("q4.shape =", q4.shape)  # (3+4+3, 4) = (10, 4)

# 水平疊加 (Horizontal Stack) - 沿 Axis 1
q5 = np.hstack((q1, q3))
print("q5 =", q5)
print("q5.shape =", q5.shape)  # (3, 4+4) = (3, 8)

# hstack 要求 Axis 0 長度必須相同
try:
    np.hstack((q1, q2))
except ValueError as e:
    print("hstack 錯誤:", e)

# 通用串接 (Concatenate) - 指定軸
q7 = np.concatenate((q1, q2, q3), axis=0)
print("q7 =", q7)
print("q7.shape =", q7.shape)

# 堆疊 (Stack) - 增加新維度
q8 = np.stack((q1, q3))
print("q8 =", q8)
print("q8.shape =", q8.shape)  # (2, 3, 4)

print("\n=== 範例 9 完成 ===\n")

# =============================================================================
# 範例 10：分割陣列 (SPLITTING ARRAYS)
# =============================================================================

print("=" * 60)
print("範例 10：分割陣列")
print("=" * 60)

# 建立 6x4 陣列
r = np.arange(24).reshape(6, 4)
print("r =", r)

# 垂直分割成 3 等份
r1, r2, r3 = np.vsplit(r, 3)
print("r1 =", r1)
print("r2 =", r2)
print("r3 =", r3)

# 水平分割成 2 等份
r4, r5 = np.hsplit(r, 2)
print("r4 =", r4)
print("r5 =", r5)

print("\n=== 範例 10 完成 ===\n")

# =============================================================================
# 範例 11：轉置陣列 (TRANSPOSING ARRAYS)
# =============================================================================

print("=" * 60)
print("範例 11：轉置陣列")
print("=" * 60)

# 建立 4x2x3 三維陣列
t = np.arange(24).reshape(4, 2, 3)
print("t =", t)

# 重新排列軸 (1,2,0)
t1 = t.transpose((1, 2, 0))
print("t1 =", t1)
print("t1.shape =", t1.shape)  # (2, 3, 4)

# 預設反轉軸順序
t2 = t.transpose()
print("t2 =", t2)
print("t2.shape =", t2.shape)  # (3, 2, 4)

# 交換兩個軸
t3 = t.swapaxes(0, 1)
print("t3 =", t3)
print("t3.shape =", t3.shape)  # (2, 4, 3)

print("\n=== 範例 11 完成 ===\n")

# =============================================================================
# 範例 12：線性代數 (LINEAR ALGEBRA)
# =============================================================================

print("=" * 60)
print("範例 12：線性代數")
print("=" * 60)

# 矩陣轉置
m1 = np.arange(10).reshape(2, 5)
print("m1 =", m1)
print("m1.T =", m1.T)  # 轉置矩陣

# 一維陣列轉置不變
m2 = np.arange(5)
print("m2 =", m2)
print("m2.T =", m2.T)

# 行向量 (2D) 轉置變列向量
m2r = m2.reshape(1, 5)
print("m2r =", m2r)
print("m2r.T =", m2r.T)

# 矩陣乘法
n1 = np.arange(10).reshape(2, 5)
n2 = np.arange(15).reshape(5, 3)
print("n1 =", n1)
print("n2 =", n2)
print("n1.dot(n2) =", n1.dot(n2))  # 或使用 n1 @ n2

# 逆矩陣
m3 = np.array([[1, 2, 3], [5, 7, 11], [21, 29, 31]])
print("m3 =", m3)
print("linalg.inv(m3) =", linalg.inv(m3))  # 逆矩陣
print("linalg.pinv(m3) =", linalg.pinv(m3))  # 偽逆矩陣
print("m3.dot(linalg.inv(m3)) =", m3.dot(linalg.inv(m3)))  # 驗證 (應接近單位矩陣)
print("np.eye(3) =", np.eye(3))  # 單位矩陣

# QR 分解
q, r = linalg.qr(m3)
print("q =", q)
print("r =", r)
print("q.dot(r) =", q.dot(r))  # 驗證

# 行列式
print("linalg.det(m3) =", linalg.det(m3))

# 特徵值與特徵向量
eigenvalues, eigenvectors = linalg.eig(m3)
print("eigenvalues =", eigenvalues)
print("eigenvectors =", eigenvectors)
print("m3.dot(eigenvectors) - eigenvalues * eigenvectors =", 
      m3.dot(eigenvectors) - eigenvalues * eigenvectors)  # 驗證 (應接近 0)

# 奇異值分解 (SVD)
m4 = np.array([[1, 0, 0, 0, 2], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0], [0, 2, 0, 0, 0]])
print("m4 =", m4)

U, S_diag, V = linalg.svd(m4)
print("U =", U)
print("S_diag =", S_diag)

# 建立完整的 Sigma 矩陣 (形狀與 m4 相同)
S = np.zeros((4, 5))
S[np.diag_indices(4)] = S_diag
print("S =", S)
print("V =", V)
print("U.dot(S).dot(V) =", U.dot(S).dot(V))  # 驗證重建

# 對角線與跡數
print("np.diag(m3) =", np.diag(m3))  # 主對角線元素
print("m3.trace() =", m3.trace())    # 跡數 (對角線和)

# 解線性方程組
coeffs = np.array([[2, 6], [5, 3]])
depvars = np.array([6, -9])
solution = linalg.solve(coeffs, depvars)
print("solution =", solution)
print("coeffs.dot(solution) =", coeffs.dot(solution))
print("depvars =", depvars)
print("np.allclose(coeffs.dot(solution), depvars) =", 
      np.allclose(coeffs.dot(solution), depvars))

print("\n=== 範例 12 完成 ===\n")

# =============================================================================
# 範例 13：向量化 (VECTORIZATION)
# =============================================================================

print("=" * 60)
print("範例 13：向量化")
print("=" * 60)

import math

# 低效率方式：使用 Python 迴圈
data_loop = np.empty((768, 1024))
for y in range(768):
    for x in range(1024):
        data_loop[y, x] = math.tan(x + y)

# 高效率方式：使用 NumPy 向量化
x_coords = np.arange(0, 1024)
y_coords = np.arange(0, 768)
X, Y = np.meshgrid(x_coords, y_coords)  # 建立座標網格
data_vec = np.tan(X + Y)  # 向量化計算

print("Shapes match:", data_loop.shape == data_vec.shape)
print("Results match:", np.allclose(data_loop, data_vec))

# 視覺化結果
fig = plt.figure(1, figsize=(8, 6))
plt.imshow(data_vec, cmap="autumn")
plt.title("向量化運算結果")
plt.colorbar()
plt.show()

print("\n=== 範例 13 完成 ===\n")

# =============================================================================
# 範例 14：儲存和載入 (SAVING AND LOADING)
# =============================================================================

print("=" * 60)
print("範例 14：儲存和載入")
print("=" * 60)

# 建立隨機陣列
a = np.random.rand(3, 3) * 2 - 1
print("a =", a)

# 二進位格式儲存 (.npy)
np.save("my_array", a)

# 載入二進位格式
a_loaded = np.load("my_array.npy")
print("a_loaded =", a_loaded)
print("Arrays equal:", np.array_equal(a, a_loaded))

# 文字格式儲存 (.csv)
np.savetxt("my_array.csv", a, delimiter=",")

# 讀取 CSV 內容
with open("my_array.csv", "rt") as f:
    print("CSV content:")
    print(f.read())

# 載入文字格式
a_loaded_txt = np.loadtxt("my_array.csv", delimiter=",")
print("Loaded from CSV:", a_loaded_txt)
print("CSV arrays equal:", np.allclose(a, a_loaded_txt))

# 壓縮格式儲存 (.npz)
b = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
np.savez("my_arrays", my_a=a, my_b=b)

# 載入壓縮格式
my_arrays = np.load("my_arrays.npz")
print("Keys:", list(my_arrays.keys()))
print("my_a =", my_arrays["my_a"])

print("\n=== 範例 14 完成 ===\n")

print("=" * 60)
print("所有範例執行完畢")
print("=" * 60)