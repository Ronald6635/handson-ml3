import matplotlib.pyplot as plt
import numpy as np

# 1. 設定字型系列，優先使用英文字型，後備使用中文字型
#    請將 'Microsoft JhengHei' 換成您系統上安裝的可用中文字型
plt.rc('font', family=['Arial', 'Microsoft JhengHei'])

# 2. 解決負號顯示問題
plt.rc('axes', unicode_minus=False)

# --- 繪製範例圖表 ---
x = np.linspace(-10, 10, 100)
y = -x**2 + 5

plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.title("中英文標題 (Title with Chinese & English)")
plt.xlabel("X 軸 (X-axis)")
plt.ylabel("Y 軸 (Y-axis)")
plt.grid(True)
plt.show()