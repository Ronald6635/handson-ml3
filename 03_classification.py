
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# **Chapter 3 – Classification**

# %% [markdown]
# _This notebook contains all the sample code and solutions to the exercises in chapter 3._

# %%
import sys

assert sys.version_info >= (3, 7)


# %%
from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")


# %%
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)


# %%
from pathlib import Path

IMAGES_PATH = Path() / "images" / "classification"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# %% [markdown]
# # MNIST

# %%
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)


# %%
X, y = mnist.data, mnist.target
X


# %%
X.shape


# %%
y


# %%
y.shape


# %%
import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()


# %%
y[0]


# %%
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# %% [markdown]
# # Training a Binary Classifier

# %%
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')


# %%
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


# %%
sgd_clf.predict([some_digit])

# %% [markdown]
# # Performance Measures
# %% [markdown]
# ## Measuring Accuracy Using Cross-Validation

# %%
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# %%
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))


# %%
cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# %% [markdown]
# ## Confusion Matrix

# %%
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


# %%
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5, y_train_pred)
cm


# %%
y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)

# %% [markdown]
# ## Precision and Recall

# %%
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)


# %%
recall_score(y_train_5, y_train_pred)


# %%
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)

# %% [markdown]
# ## Precision/Recall Trade-off

# %%
y_scores = sgd_clf.decision_function([some_digit])
y_scores


# %%
threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# %%
threshold = 3000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# %%
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")


# %%
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# %%
plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc="center right")
plt.show()


# %%
import matplotlib.patches as patches

plt.figure(figsize=(6, 5))

plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
plt.show()


# %%
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision


# %%
y_train_pred_90 = (y_scores >= threshold_for_90_precision)


# %%
precision_score(y_train_5, y_train_pred_90)


# %%
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
recall_at_90_precision

# %% [markdown]
# ## The ROC Curve

# %%
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# %%
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=13)
plt.show()


# %%
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)


# %%
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)


# %%
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")


# %%
y_probas_forest[:2]


# %%
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)


# %%
plt.figure(figsize=(6, 5))

plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "g--", linewidth=2, label="SGD")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
plt.show()


# %%
y_train_pred_forest = y_probas_forest[:, 1] >= 0.5
f1_score(y_train_5, y_train_pred_forest)


# %%
roc_auc_score(y_train_5, y_scores_forest)


# %%
precision_score(y_train_5, y_train_pred_forest)


# %%
recall_score(y_train_5, y_train_pred_forest)

# %% [markdown]
# # Multiclass Classification

# %%
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])


# %%
svm_clf.predict([some_digit])


# %%
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores.round(2)


# %%
class_id = some_digit_scores.argmax()
class_id


# %%
svm_clf.classes_


# %%
svm_clf.classes_[class_id]


# %%
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])


# %%
ovr_clf.predict([some_digit])


# %%
len(ovr_clf.estimators_)


# %%
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])


# %%
sgd_clf.decision_function([some_digit]).round()


# %%
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")


# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

# %% [markdown]
# # Error Analysis

# %%
from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
plt.rc('font', size=9)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()


# %%
plt.rc('font', size=10)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        normalize="true", values_format=".0%")
plt.show()


# %%
sample_weight = (y_train_pred != y_train)
plt.rc('font', size=10)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="true", values_format=".0%")
plt.show()


# %%
cl_a, cl_b = '3', '5'
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

# %% [markdown]
# # Multilabel Classification

# %%
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train.astype('int8') >= 7)
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)


# %%
knn_clf.predict([some_digit])


# %%
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")

# %% [markdown]
# # Multioutput Classification

# %%
np.random.seed(42)
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test


# %%
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
plt.show()
