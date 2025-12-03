"""
集成學習與隨機森林模組

此模組示範集成學習技術，包括投票分類器、裝袋、隨機森林、提升和堆疊。
涵蓋從基本概念到進階實作的完整範例。

關鍵特點:
- 投票分類器實作
- 裝袋與粘貼方法
- 隨機森林與特徵重要性
- AdaBoost與梯度提升
- 堆疊集成
- MNIST與住房資料集應用

範例基於markdown文件07_ensemble_learning_and_random_forests.md
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from packaging import version
import sklearn
from pathlib import Path
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import tarfile
import urllib.request
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml

# =============================================================================
# 設定與環境檢查
# =============================================================================

print("=== 設定與環境檢查 ===")

# 檢查Python版本
assert sys.version_info >= (3, 7)
print("Python版本檢查通過")

# 檢查sklearn版本
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
print("Scikit-Learn版本檢查通過")

# 設定matplotlib
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# 建立圖片目錄
IMAGES_PATH = Path() / "images" / "ensembles"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    # 儲存圖表到檔案
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

print("設定完成")

# =============================================================================
# 大數法則示範
# =============================================================================

print("\n=== 大數法則示範 ===")

# 模擬大數法則
heads_proba = 0.51
np.random.seed(42)
coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
cumulative_heads = coin_tosses.cumsum(axis=0)
cumulative_heads_ratio = cumulative_heads / np.arange(1, 10001).reshape(-1, 1)

plt.figure(figsize=(8, 3.5))
plt.plot(cumulative_heads_ratio)
plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
plt.xlabel("Number of coin tosses")
plt.ylabel("Heads ratio")
plt.legend(loc="lower right")
plt.axis([0, 10000, 0.42, 0.58])
plt.grid()
save_fig("law_of_large_numbers_plot")
plt.show()

print("大數法則圖表已儲存")

# =============================================================================
# 投票分類器
# =============================================================================

print("\n=== 投票分類器 ===")

# 建立資料集
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 建立投票分類器
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)

print("投票分類器已訓練")
print("估計器:", voting_clf.estimators_)
print("具名估計器:", list(voting_clf.named_estimators_.keys()))

# 評估個別分類器
for name, clf in voting_clf.named_estimators_.items():
    score = clf.score(X_test, y_test)
    print(f"{name} 準確度: {score}")

# 測試預測
X_test_sample = X_test[:1]
ensemble_pred = voting_clf.predict(X_test_sample)
individual_preds = [clf.predict(X_test_sample) for clf in voting_clf.estimators_]
print(f"集成預測: {ensemble_pred}")
print(f"個別預測: {individual_preds}")

# 整體準確度
ensemble_score = voting_clf.score(X_test, y_test)
print(f"集成準確度: {ensemble_score}")

# 切換到軟投票
voting_clf.voting = "soft"
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(X_train, y_train)
soft_score = voting_clf.score(X_test, y_test)
print(f"軟投票準確度: {soft_score}")

# =============================================================================
# 裝袋與粘貼
# =============================================================================

print("\n=== 裝袋與粘貼 ===")

# 建立裝袋分類器
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)

print("裝袋分類器已訓練")

# 定義決策邊界繪圖函數
def plot_decision_boundary(clf, X, y, alpha=1.0):
    # 繪製分類器決策邊界
    axes=[-1.5, 2.4, -1, 1.5]
    x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                         np.linspace(axes[2], axes[3], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    
    plt.contourf(x1, x2, y_pred, alpha=0.3 * alpha, cmap='Wistia')
    plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8 * alpha)
    colors = ["#78785c", "#c47b27"]
    markers = ("o", "^")
    for idx in (0, 1):
        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                 color=colors[idx], marker=markers[idx], linestyle="none")
    plt.axis(axes)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation=0)

# 比較單一樹與裝袋
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(tree_clf, X_train, y_train)
plt.title("Decision Tree")
plt.sca(axes[1])
plot_decision_boundary(bag_clf, X_train, y_train)
plt.title("Decision Trees with Bagging")
plt.ylabel("")
save_fig("decision_tree_without_and_with_bagging_plot")
plt.show()

print("決策邊界比較圖表已儲存")

# 袋外評估
bag_clf_oob = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                                oob_score=True, n_jobs=-1, random_state=42)
bag_clf_oob.fit(X_train, y_train)
oob_score = bag_clf_oob.oob_score_
print(f"OOB分數: {oob_score}")

# OOB決策函數
oob_decision = bag_clf_oob.oob_decision_function_[:3]
print(f"前3個樣本OOB機率: {oob_decision}")

# 測試準確度
y_pred_bag = bag_clf_oob.predict(X_test)
bag_accuracy = accuracy_score(y_test, y_pred_bag)
print(f"裝袋測試準確度: {bag_accuracy}")

# =============================================================================
# 隨機森林
# =============================================================================

print("\n=== 隨機森林 ===")

# 建立隨機森林
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

print("隨機森林已訓練")

# 等效裝袋實作
bag_clf_rf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500, n_jobs=-1, random_state=42)

bag_clf_rf.fit(X_train, y_train)
y_pred_bag_rf = bag_clf_rf.predict(X_test)
predictions_match = np.all(y_pred_bag_rf == y_pred_rf)
print(f"預測是否相同: {predictions_match}")

# 鳶尾花特徵重要性
iris = load_iris(as_frame=True)
rnd_clf_iris = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf_iris.fit(iris.data, iris.target)

print("鳶尾花特徵重要性:")
for score, name in zip(rnd_clf_iris.feature_importances_, iris.data.columns):
    print(f"重要性: {round(score, 2)}, 特徵: {name}")

# MNIST特徵重要性
print("\n載入MNIST資料集...")
X_mnist, y_mnist = fetch_openml('mnist_784', return_X_y=True, as_frame=False,
                                parser='auto')

rnd_clf_mnist = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf_mnist.fit(X_mnist, y_mnist)

heatmap_image = rnd_clf_mnist.feature_importances_.reshape(28, 28)
plt.imshow(heatmap_image, cmap="hot")
cbar = plt.colorbar(ticks=[rnd_clf_mnist.feature_importances_.min(),
                           rnd_clf_mnist.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'], fontsize=14)
plt.axis("off")
save_fig("mnist_feature_importance_plot")
plt.show()

print("MNIST特徵重要性熱圖已儲存")

# =============================================================================
# AdaBoost
# =============================================================================

print("\n=== AdaBoost ===")

# 手動AdaBoost模擬
m = len(X_train)
fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)

sample_weights = np.ones(m) / m
for subplot, learning_rate in ((0, 1), (1, 0.5)):
    plt.sca(axes[subplot])
    for i in range(5):
        svm_clf = SVC(C=0.2, gamma=0.6, random_state=42)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights * m)
        y_pred = svm_clf.predict(X_train)

        error_weights = sample_weights[y_pred != y_train].sum()
        r = error_weights / sample_weights.sum()
        alpha = learning_rate * np.log((1 - r) / r)
        sample_weights[y_pred != y_train] *= np.exp(alpha)
        sample_weights /= sample_weights.sum()

        plot_decision_boundary(svm_clf, X_train, y_train, alpha=0.4)
        plt.title(f"learning_rate = {learning_rate}")
    if subplot == 0:
        plt.text(-0.75, -0.95, "1", fontsize=16)
        plt.text(-1.05, -0.95, "2", fontsize=16)
        plt.text(1.0, -0.95, "3", fontsize=16)
        plt.text(-1.45, -0.5, "4", fontsize=16)
        plt.text(1.36,  -0.95, "5", fontsize=16)
    else:
        plt.ylabel("")

save_fig("boosting_plot")
plt.show()

print("AdaBoost過程圖表已儲存")

# 顯示樣本權重
plt.figure()
plt.plot(sample_weights)
plt.title("Sample Weights")
plt.show()

# Scikit-Learn AdaBoost
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=30,
    learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)

plot_decision_boundary(ada_clf, X_train, y_train)
plt.show()

print("AdaBoost分類器已訓練")

# SVM作為基礎估計器
svm_ada_clf = AdaBoostClassifier(
    SVC(probability=True), n_estimators=30,
    learning_rate=0.5, random_state=42)
svm_ada_clf.fit(X_train, y_train)
plot_decision_boundary(svm_ada_clf, X_train, y_train)
plt.show()

print("SVM AdaBoost已訓練")

# =============================================================================
# 梯度提升
# =============================================================================

print("\n=== 梯度提升 ===")

# 建立二次資料集
np.random.seed(42)
X_quad = np.random.rand(100, 1) - 0.5
y_quad = 3 * X_quad[:, 0] ** 2 + 0.05 * np.random.randn(100)

# 訓練決策樹回歸器
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X_quad, y_quad)

y2 = y_quad - tree_reg1.predict(X_quad)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X_quad, y2)

y3 = y2 - tree_reg2.predict(X_quad)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X_quad, y3)

X_new_quad = np.array([[-0.4], [0.], [0.5]])
ensemble_pred = sum(tree.predict(X_new_quad) for tree in (tree_reg1, tree_reg2, tree_reg3))
print(f"集成預測: {ensemble_pred}")

y_new_quad = 3 * X_new_quad ** 2
print(f"真實值: {y_new_quad}")

# 繪圖函數
def plot_predictions(regressors, X, y, axes, style,
                     label=None, data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1))
                 for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center")
    plt.axis(axes)

# 繪製梯度提升過程
plt.figure(figsize=(11, 11))

plt.subplot(3, 2, 1)
plot_predictions([tree_reg1], X_quad, y_quad, axes=[-0.5, 0.5, -0.2, 0.8], style="g-",
                 label="$h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$  ", rotation=0)
plt.title("Residuals and tree predictions")

plt.subplot(3, 2, 2)
plot_predictions([tree_reg1], X_quad, y_quad, axes=[-0.5, 0.5, -0.2, 0.8], style="r-",
                 label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.title("Ensemble predictions")

plt.subplot(3, 2, 3)
plot_predictions([tree_reg2], X_quad, y2, axes=[-0.5, 0.5, -0.4, 0.6], style="g-",
                 label="$h_2(x_1)$", data_style="k+",
                 data_label="Residuals: $y - h_1(x_1)$")
plt.ylabel("$y$  ", rotation=0)

plt.subplot(3, 2, 4)
plot_predictions([tree_reg1, tree_reg2], X_quad, y_quad, axes=[-0.5, 0.5, -0.2, 0.8],
                  style="r-", label="$h(x_1) = h_1(x_1) + h_2(x_1)$")

plt.subplot(3, 2, 5)
plot_predictions([tree_reg3], X_quad, y3, axes=[-0.5, 0.5, -0.4, 0.6], style="g-",
                 label="$h_3(x_1)$", data_style="k+",
                 data_label="Residuals: $y - h_1(x_1) - h_2(x_1)$")
plt.xlabel("$x_1$")
plt.ylabel("$y$  ", rotation=0)

plt.subplot(3, 2, 6)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X_quad, y_quad,
                 axes=[-0.5, 0.5, -0.2, 0.8], style="r-",
                 label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$")

save_fig("gradient_boosting_plot")
plt.show()

print("梯度提升過程圖表已儲存")

# Scikit-Learn梯度提升
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3,
                                 learning_rate=1.0, random_state=42)
gbrt.fit(X_quad, y_quad)

gbrt_best = GradientBoostingRegressor(
    max_depth=2, learning_rate=0.05, n_estimators=500,
    n_iter_no_change=10, random_state=42)
gbrt_best.fit(X_quad, y_quad)

print(f"最佳模型估計器數量: {gbrt_best.n_estimators_}")

# 比較學習率
fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)

plt.sca(axes[0])
plot_predictions([gbrt], X_quad, y_quad, axes=[-0.5, 0.5, -0.1, 0.8], style="r-",
                 label="Ensemble predictions")
plt.title(f"learning_rate={gbrt.learning_rate}, "
          f"n_estimators={gbrt.n_estimators_}")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)

plt.sca(axes[1])
plot_predictions([gbrt_best], X_quad, y_quad, axes=[-0.5, 0.5, -0.1, 0.8], style="r-")
plt.title(f"learning_rate={gbrt_best.learning_rate}, "
          f"n_estimators={gbrt_best.n_estimators_}")
plt.xlabel("$x_1$")

save_fig("gbrt_learning_rate_plot")
plt.show()

print("學習率比較圖表已儲存")

# =============================================================================
# 住房資料集與直方圖梯度提升
# =============================================================================

print("\n=== 住房資料集與直方圖梯度提升 ===")

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
housing_labels = train_set["median_house_value"]
housing_prepared = train_set.drop("median_house_value", axis=1)

print("住房資料集載入完成")
print(f"資料形狀: {housing_prepared.shape}")

# 建立直方圖梯度提升pipeline
hgb_reg = make_pipeline(
    make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
                            remainder="passthrough"),
    HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
)
hgb_reg.fit(housing_prepared, housing_labels)

print("直方圖梯度提升模型已訓練")

# 交叉驗證評估
hgb_rmses = -cross_val_score(hgb_reg, housing_prepared, housing_labels,
                             scoring="neg_root_mean_squared_error", cv=10)
print("RMSE統計:")
print(hgb_rmses.describe())

# =============================================================================
# 堆疊
# =============================================================================

print("\n=== 堆疊 ===")

# 建立堆疊分類器
stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5
)
stacking_clf.fit(X_train, y_train)

stacking_score = stacking_clf.score(X_test, y_test)
print(f"堆疊分類器準確度: {stacking_score}")

# 比較個別分類器
for name, clf in stacking_clf.named_estimators_.items():
    score = clf.score(X_test, y_test)
    print(f"{name} 準確度: {score}")

print(f"最終估計器特徵數: {stacking_clf.final_estimator_.n_features_in_}")

# =============================================================================
# MNIST練習
# =============================================================================

print("\n=== MNIST練習 ===")

# 分割MNIST資料
X_train_mnist, y_train_mnist = X_mnist[:50_000], y_mnist[:50_000]
X_valid_mnist, y_valid_mnist = X_mnist[50_000:60_000], y_mnist[50_000:60_000]
X_test_mnist, y_test_mnist = X_mnist[60_000:], y_mnist[60_000:]

print("MNIST資料分割完成")

# 訓練分類器
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20, dual=True, random_state=42)
mlp_clf = MLPClassifier(random_state=42)

estimators_mnist = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]

for estimator in estimators_mnist:
    print(f"訓練 {estimator}")
    estimator.fit(X_train_mnist, y_train_mnist)

# 驗證準確度
valid_scores = [estimator.score(X_valid_mnist, y_valid_mnist) for estimator in estimators_mnist]
print(f"驗證準確度: {valid_scores}")

# 投票分類器
named_estimators_mnist = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]

voting_clf_mnist = VotingClassifier(named_estimators_mnist)
voting_clf_mnist.fit(X_train_mnist, y_train_mnist)
voting_valid_score = voting_clf_mnist.score(X_valid_mnist, y_valid_mnist)
print(f"投票分類器驗證準確度: {voting_valid_score}")

# 移除SVM
voting_clf_mnist.set_params(svm_clf="drop")
svm_clf_trained = voting_clf_mnist.named_estimators_.pop("svm_clf")
voting_clf_mnist.estimators_.remove(svm_clf_trained)

voting_valid_score_no_svm = voting_clf_mnist.score(X_valid_mnist, y_valid_mnist)
print(f"移除SVM後驗證準確度: {voting_valid_score_no_svm}")

# 測試準確度
voting_test_score = voting_clf_mnist.score(X_test_mnist, y_test_mnist)
print(f"投票分類器測試準確度: {voting_test_score}")

individual_test_scores = [estimator.score(X_test_mnist, y_test_mnist.astype(np.int64))
                         for estimator in voting_clf_mnist.estimators_]
print(f"個別測試準確度: {individual_test_scores}")

# 堆疊練習
X_valid_predictions = np.empty((len(X_valid_mnist), len(estimators_mnist)), dtype=object)
for index, estimator in enumerate(estimators_mnist):
    X_valid_predictions[:, index] = estimator.predict(X_valid_mnist)

rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True,
                                            random_state=42)
rnd_forest_blender.fit(X_valid_predictions, y_valid_mnist)

print(f"混成器OOB分數: {rnd_forest_blender.oob_score_}")

X_test_predictions = np.empty((len(X_test_mnist), len(estimators_mnist)), dtype=object)
for index, estimator in enumerate(estimators_mnist):
    X_test_predictions[:, index] = estimator.predict(X_test_mnist)

y_pred_stacking = rnd_forest_blender.predict(X_test_predictions)
stacking_accuracy = accuracy_score(y_test_mnist, y_pred_stacking)
print(f"自訂堆疊準確度: {stacking_accuracy}")

# Scikit-Learn堆疊
X_train_full, y_train_full = X_mnist[:60_000], y_mnist[:60_000]

stack_clf_mnist = StackingClassifier(named_estimators_mnist,
                                     final_estimator=rnd_forest_blender)
stack_clf_mnist.fit(X_train_full, y_train_full)

stack_test_score = stack_clf_mnist.score(X_test_mnist, y_test_mnist)
print(f"Scikit-Learn堆疊測試準確度: {stack_test_score}")

print("\n=== 所有範例完成 ===")