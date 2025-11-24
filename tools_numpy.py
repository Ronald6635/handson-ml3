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

import string
import numpy as np
import numpy.linalg as linalg
import math
import matplotlib.pyplot as plt
from matplotlib import font_manager

CJK_FONT_CANDIDATES = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "Heiti TC",
]

available_fonts = []
for font_name in CJK_FONT_CANDIDATES:
    try:
        font_manager.findfont(font_name, fallback_to_default=False)
    except ValueError:
        continue
    available_fonts.append(font_name)

if not available_fonts:
    available_fonts.append("DejaVu Sans")  # guaranteed by Matplotlib

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = available_fonts
plt.rcParams["axes.unicode_minus"] = False

# =============================================================================
# 範例 1：建立陣列
# =============================================================================

print("=== 範例 1：建立陣列 ===")

# 匯入 NumPy
import numpy as np

# 建立全零陣列
print("np.zeros(5) =", np.zeros(5))

# 建立 3x4 全零矩陣
print("np.zeros((3,4)) =")
a = np.zeros((3,4))
print(a)

# 檢查陣列屬性
print("a.shape =", a.shape)
print("a.ndim =", a.ndim)
print("a.size =", a.size)

# 建立三維陣列
print("np.zeros((2,3,4)) =")
print(np.zeros((2,3,4)))

# 檢查陣列類型
print("type(np.zeros((3,4))) =", type(np.zeros((3,4))))

# 全一陣列
print("np.ones((3,4)) =")
print(np.ones((3,4)))

# 指定值陣列
print("np.full((3,4), np.pi) =")
print(np.full((3,4), np.pi))

# 未初始化陣列
print("np.empty((2,3)) =")
print(np.empty((2,3)))

# 從列表建立陣列
print("np.array([[1,2,3,4], [10, 20, 30, 40]]) =")
print(np.array([[1,2,3,4], [10, 20, 30, 40]]))

# arange 函數
print("np.arange(1, 5) =", np.arange(1, 5))
print("np.arange(1.0, 5.0) =", np.arange(1.0, 5.0))
print("np.arange(1, 5, 0.5) =", np.arange(1, 5, 0.5))

# arange 浮點數問題
print("np.arange(0, 5/3, 1/3) =", np.arange(0, 5/3, 1/3))
print("np.arange(0, 5/3, 0.333333333) =", np.arange(0, 5/3, 0.333333333))
print("np.arange(0, 5/3, 0.333333334) =", np.arange(0, 5/3, 0.333333334))

# linspace 函數
print("np.linspace(0, 5/3, 6) =", np.linspace(0, 5/3, 6))

# 隨機陣列
print("np.random.rand(3,4) =")
print(np.random.rand(3,4))

print("np.random.randn(3,4) =")
print(np.random.randn(3,4))

# 繪製隨機分佈直方圖
plt.hist(np.random.rand(100000), density=True, bins=100, histtype="step", color="blue", label="rand")
plt.hist(np.random.randn(100000), density=True, bins=100, histtype="step", color="red", label="randn")
plt.axis((-2.5, 2.5, 0, 1.1))
plt.legend(loc = "upper left")
plt.title("隨機分佈")
plt.xlabel("數值")
plt.ylabel("密度")
plt.show()

# 使用函數建立陣列
def my_function(z, y, x):
    return x + 10 * y + 100 * z

print("np.fromfunction(my_function, (3, 2, 10)) =")
print(np.fromfunction(my_function, (3, 2, 10)))

print("\n=== 範例 1 完成 ===")

# =============================================================================
# 範例 2：陣列資料類型
# =============================================================================

print("\n=== 範例 2：陣列資料類型 ===")

c = np.arange(1, 5)
print("c.dtype =", c.dtype, "c =", c)

c = np.arange(1.0, 5.0)
print("c_float.dtype =", c.dtype, "c =", c)

d = np.arange(1, 5, dtype=np.complex64)
print("d.dtype =", d.dtype, "d =", d)

e = np.arange(1, 5, dtype=np.complex64)
print("e.itemsize =", e.itemsize)

f = np.array([[1,2],[1000, 2000]], dtype=np.int32)
print("f.data =", f.data)

if hasattr(f.data, "tobytes"):
    data_bytes = f.data.tobytes()
else:
    data_bytes = memoryview(f.data).tobytes()
print("data_bytes =", data_bytes)

print("\n=== 範例 2 完成 ===")

# =============================================================================
# 範例 3：重塑陣列
# =============================================================================

print("\n=== 範例 3：重塑陣列 ===")

g = np.arange(24)
print("g =", g)
print("Rank:", g.ndim)

g.shape = (6, 4)
print("g 重新塑形後 =", g)
print("Rank:", g.ndim)

g.shape = (2, 3, 4)
print("g 三維 =", g)

g2 = g.reshape(4,6)
print("g2 =", g2)
print("Rank:", g2.ndim)

g2[1, 2] = 999
print("g2 修改後 =", g2)
print("原始 g =", g)

print("g.ravel() =", g.ravel())

print("\n=== 範例 3 完成 ===")

# =============================================================================
# 範例 4：算術運算
# =============================================================================

print("\n=== 範例 4：算術運算 ===")

a = np.array([14, 23, 32, 41])
b = np.array([5,  4,  3,  2])
print("a + b  =", a + b)
print("a - b  =", a - b)
print("a * b  =", a * b)
print("a / b  =", a / b)
print("a // b  =", a // b)
print("a % b  =", a % b)
print("a ** b =", a ** b)

print("\n=== 範例 4 完成 ===")

# =============================================================================
# 範例 5：廣播
# =============================================================================

print("\n=== 範例 5：廣播 ===")

h = np.arange(5).reshape(1, 1, 5)
print("h =", h)
print("h + [10, 20, 30, 40, 50] =", h + [10, 20, 30, 40, 50])

k = np.arange(6).reshape(2, 3)
print("k =", k)
print("k + [[100], [200]] =", k + [[100], [200]])
print("k + [100, 200, 300] =", k + [100, 200, 300])
print("k + 1000 =", k + 1000)

try:
    print("k + [33, 44] =", k + [33, 44])
except ValueError as e:
    print("錯誤:", e)

k1 = np.arange(0, 5, dtype=np.uint8)
print("k1.dtype =", k1.dtype, "k1 =", k1)

k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)
print("k2.dtype =", k2.dtype, "k2 =", k2)

k3 = k1 + 1.5
print("k3.dtype =", k3.dtype, "k3 =", k3)

print("\n=== 範例 5 完成 ===")

# =============================================================================
# 範例 6：條件運算子
# =============================================================================

print("\n=== 範例 6：條件運算子 ===")

m = np.array([20, -5, 30, 40])
print("m =", m)
print("m < [15, 16, 35, 36] =", m < [15, 16, 35, 36])
print("m < 25 =", m < 25)
print("m[m < 25] =", m[m < 25])

print("\n=== 範例 6 完成 ===")

# =============================================================================
# 範例 7：數學和統計函數
# =============================================================================

print("\n=== 範例 7：數學和統計函數 ===")

a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
print("a =", a)
print("mean =", a.mean())

for func in (a.min, a.max, a.sum, a.prod, a.std, a.var):
    print(func.__name__, "=", func())

c = np.arange(24).reshape(2,3,4)
print("c =", c)
print("c.sum(axis=0) =", c.sum(axis=0))
print("c.sum(axis=1) =", c.sum(axis=1))
print("c.sum(axis=(0,2)) =", c.sum(axis=(0,2)))

print("驗證計算:", 0+1+2+3 + 12+13+14+15, 4+5+6+7 + 16+17+18+19, 8+9+10+11 + 20+21+22+23)

a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
print("np.square(a) =", np.square(a))

print("Original ndarray")
print(a)
for func in (np.abs, np.sqrt, np.exp, np.log, np.sign, np.ceil, np.modf, np.isnan, np.cos):
    print("\n", func.__name__)
    print(func(a))

a = np.array([1, -2, 3, 4])
b = np.array([2, 8, -1, 7])
print("np.add(a, b) =", np.add(a, b))
print("np.greater(a, b) =", np.greater(a, b))
print("np.maximum(a, b) =", np.maximum(a, b))
print("np.copysign(a, b) =", np.copysign(a, b))

print("\n=== 範例 7 完成 ===")

# =============================================================================
# 範例 8：陣列索引
# =============================================================================

print("\n=== 範例 8：陣列索引 ===")

a = np.array([1, 5, 3, 19, 13, 7, 3])
print("a =", a)
print("a[3] =", a[3])
print("a[2:5] =", a[2:5])
print("a[2:-1] =", a[2:-1])
print("a[:2] =", a[:2])
print("a[2::2] =", a[2::2])
print("a[::-1] =", a[::-1])

a[3] = 999
print("a 修改後 =", a)

a[2:5] = [997, 998, 999]
print("a 切片賦值後 =", a)

try:
    a[2:5] = [1,2,3,4,5,6]
except ValueError as e:
    print("錯誤:", e)

try:
    del a[2:5]
except ValueError as e:
    print("錯誤:", e)

a_slice = a[2:6]
a_slice[1] = 1000
print("a_slice 修改後的 a =", a)

a[3] = 2000
print("a 修改後的 a_slice =", a_slice)

another_slice = a[2:6].copy()
another_slice[1] = 3000
print("複製切片修改後的 a =", a)

a[3] = 4000
print("a 修改後的複製切片 =", another_slice)

b = np.arange(48).reshape(4, 12)
print("b =", b)
print("b[1, 2] =", b[1, 2])
print("b[1, :] =", b[1, :])
print("b[:, 1] =", b[:, 1])

print("b[1, :] =", b[1, :])
print("b[1:2, :] =", b[1:2, :])

print("b[(0,2), 2:5] =", b[(0,2), 2:5])
print("b[:, (-1, 2, -1)] =", b[:, (-1, 2, -1)])
print("b[(-1, 2, -1, 2), (5, 9, 1, 9)] =", b[(-1, 2, -1, 2), (5, 9, 1, 9)])

c = b.reshape(4,2,6)
print("c =", c)
print("c[2, 1, 4] =", c[2, 1, 4])
print("c[2, :, 3] =", c[2, :, 3])
print("c[2, 1] =", c[2, 1])
print("c[2, ...] =", c[2, ...])
print("c[2, 1, ...] =", c[2, 1, ...])
print("c[2, ..., 3] =", c[2, ..., 3])
print("c[..., 3] =", c[..., 3])

b = np.arange(48).reshape(4, 12)
rows_on = np.array([True, False, True, False])
print("b[rows_on, :] =", b[rows_on, :])

cols_on = np.array([False, True, False] * 4)
print("b[:, cols_on] =", b[:, cols_on])

print("b[np.ix_(rows_on, cols_on)] =", b[np.ix_(rows_on, cols_on)])
print("np.ix_(rows_on, cols_on) =", np.ix_(rows_on, cols_on))

print("b[b % 3 == 1] =", b[b % 3 == 1])

c = np.arange(24).reshape(2, 3, 4)
print("c =", c)

for m in c:
    print("Item:")
    print(m)

for i in range(len(c)):
    print("Item:")
    print(c[i])

for i in c.flat:
    print("Item:", i)

print("\n=== 範例 8 完成 ===")

# =============================================================================
# 範例 9：疊加陣列
# =============================================================================

print("\n=== 範例 9：疊加陣列 ===")

q1 = np.full((3,4), 1.0)
print("q1 =", q1)

q2 = np.full((4,4), 2.0)
print("q2 =", q2)

q3 = np.full((3,4), 3.0)
print("q3 =", q3)

q4 = np.vstack((q1, q2, q3))
print("q4 =", q4)
print("q4.shape =", q4.shape)

q5 = np.hstack((q1, q3))
print("q5 =", q5)
print("q5.shape =", q5.shape)

try:
    q5 = np.hstack((q1, q2, q3))
except ValueError as e:
    print("錯誤:", e)

q7 = np.concatenate((q1, q2, q3), axis=0)
print("q7 =", q7)
print("q7.shape =", q7.shape)

q8 = np.stack((q1, q3))
print("q8 =", q8)
print("q8.shape =", q8.shape)

print("\n=== 範例 9 完成 ===")

# =============================================================================
# 範例 10：分割陣列
# =============================================================================

print("\n=== 範例 10：分割陣列 ===")

r = np.arange(24).reshape(6,4)
print("r =", r)

r1, r2, r3 = np.vsplit(r, 3)
print("r1 =", r1)
print("r2 =", r2)
print("r3 =", r3)

r4, r5 = np.hsplit(r, 2)
print("r4 =", r4)
print("r5 =", r5)

print("\n=== 範例 10 完成 ===")

# =============================================================================
# 範例 11：轉置陣列
# =============================================================================

print("\n=== 範例 11：轉置陣列 ===")

t = np.arange(24).reshape(4,2,3)
print("t =", t)

t1 = t.transpose((1,2,0))
print("t1 =", t1)
print("t1.shape =", t1.shape)

t2 = t.transpose()
print("t2 =", t2)
print("t2.shape =", t2.shape)

t3 = t.swapaxes(0,1)
print("t3 =", t3)
print("t3.shape =", t3.shape)

print("\n=== 範例 11 完成 ===")

# =============================================================================
# 範例 12：線性代數
# =============================================================================

print("\n=== 範例 12：線性代數 ===")

m1 = np.arange(10).reshape(2,5)
print("m1 =", m1)
print("m1.T =", m1.T)

m2 = np.arange(5)
print("m2 =", m2)
print("m2.T =", m2.T)

m2r = m2.reshape(1,5)
print("m2r =", m2r)
print("m2r.T =", m2r.T)

n1 = np.arange(10).reshape(2, 5)
n2 = np.arange(15).reshape(5,3)
print("n1 =", n1)
print("n2 =", n2)
print("n1.dot(n2) =", n1.dot(n2))

m3 = np.array([[1,2,3],[5,7,11],[21,29,31]])
print("m3 =", m3)
print("linalg.inv(m3) =", linalg.inv(m3))
print("linalg.pinv(m3) =", linalg.pinv(m3))
print("m3.dot(linalg.inv(m3)) =", m3.dot(linalg.inv(m3)))
print("np.eye(3) =", np.eye(3))

q, r = linalg.qr(m3)
print("q =", q)
print("r =", r)
print("q.dot(r) =", q.dot(r))

print("linalg.det(m3) =", linalg.det(m3))

eigenvalues, eigenvectors = linalg.eig(m3)
print("eigenvalues =", eigenvalues)
print("eigenvectors =", eigenvectors)
print("m3.dot(eigenvectors) - eigenvalues * eigenvectors =", m3.dot(eigenvectors) - eigenvalues * eigenvectors)

m4 = np.array([[1,0,0,0,2], [0,0,3,0,0], [0,0,0,0,0], [0,2,0,0,0]])
print("m4 =", m4)

U, S_diag, V = linalg.svd(m4)
print("U =", U)
print("S_diag =", S_diag)

S = np.zeros((4, 5))
S[np.diag_indices(4)] = S_diag
print("S =", S)
print("V =", V)
print("U.dot(S).dot(V) =", U.dot(S).dot(V))

print("np.diag(m3) =", np.diag(m3))
print("m3.trace() =", m3.trace())

coeffs = np.array([[2, 6], [5, 3]])
depvars = np.array([6, -9])
solution = linalg.solve(coeffs, depvars)
print("solution =", solution)
print("coeffs.dot(solution) =", coeffs.dot(solution))
print("depvars =", depvars)
print("np.allclose(coeffs.dot(solution), depvars) =", np.allclose(coeffs.dot(solution), depvars))

print("\n=== 範例 12 完成 ===")

# =============================================================================
# 範例 13：向量化
# =============================================================================

print("\n=== 範例 13：向量化 ===")

# 低效方式
data_slow = np.empty((768, 1024))
for y in range(768):
    for x in range(1024):
        data_slow[y, x] = math.sin(x * y / 40.5)

# 高效方式
x_coords = np.arange(0, 1024)
y_coords = np.arange(0, 768)
X, Y = np.meshgrid(x_coords, y_coords)
data_fast = np.sin(X * Y / 40.5)

print("Shapes match:", data_slow.shape == data_fast.shape)
print("Results match:", np.allclose(data_slow, data_fast))

# 繪製影像
fig = plt.figure(1, figsize=(7, 6))
plt.imshow(data_fast, cmap="hot")
plt.show()

print("\n=== 範例 13 完成 ===")

# =============================================================================
# 範例 14：儲存和載入
# =============================================================================

print("\n=== 範例 14：儲存和載入 ===")

a = np.random.rand(2,3)
print("a =", a)

np.save("my_array", a)

a_loaded = np.load("my_array.npy")
print("a_loaded =", a_loaded)
print("Arrays equal:", np.array_equal(a, a_loaded))

np.savetxt("my_array.csv", a, delimiter=",")

with open("my_array.csv", "rt") as f:
    print("CSV content:")
    print(f.read())

a_loaded_txt = np.loadtxt("my_array.csv", delimiter=",")
print("Loaded from CSV:", a_loaded_txt)
print("CSV arrays equal:", np.allclose(a, a_loaded_txt))

b = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
print("b =", b)

np.savez("my_arrays", my_a=a, my_b=b)

my_arrays = np.load("my_arrays.npz")
print("my_arrays =", my_arrays)
print("Keys:", list(my_arrays.keys()))
print("my_a =", my_arrays["my_a"])

print("\n=== 範例 14 完成 ===")

print("\n=== 所有範例完成 ===")