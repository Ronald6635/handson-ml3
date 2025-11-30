"""
Decision Trees Module

This module demonstrates decision tree algorithms in machine learning.

Key features:
- Decision tree classification and regression
- Model visualization and interpretation
- Hyperparameter tuning and regularization
- Handling axis orientation sensitivity
- Ensemble methods basics

Examples are based on the markdown documentation 06_decision_trees.md
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from packaging import version
import sklearn

# =============================================================================
# SETUP AND ENVIRONMENT CHECK
# =============================================================================

print("=== Setup and Environment Check ===")

# Check Python version
assert sys.version_info >= (3, 7)
print("✓ Python version check passed")

# Check Scikit-Learn version
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
print("✓ Scikit-Learn version check passed")

# Configure matplotlib
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# Create images directory
IMAGES_PATH = Path() / "images" / "decision_trees"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

print("✓ Environment setup complete")

# =============================================================================
# EXAMPLE 1: TRAINING AND VISUALIZING A DECISION TREE
# =============================================================================

print("\n=== Example 1: Training and Visualizing a Decision Tree ===")

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load iris dataset
iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

# Create and train decision tree
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

print("✓ Decision tree trained on iris dataset")

# Export tree visualization
from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file=str(IMAGES_PATH / "iris_tree.dot"),
    feature_names=["petal length (cm)", "petal width (cm)"],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

print("✓ Tree visualization exported to iris_tree.dot")

# =============================================================================
# EXAMPLE 2: MAKING PREDICTIONS
# =============================================================================

print("\n=== Example 2: Making Predictions ===")

# Make predictions
prediction = tree_clf.predict([[5, 1.5]])
print(f"Prediction for [5, 1.5]: {prediction}")

probability = tree_clf.predict_proba([[5, 1.5]]).round(3)
print(f"Class probabilities: {probability}")

# =============================================================================
# EXAMPLE 3: REGULARIZATION HYPERPARAMETERS
# =============================================================================

print("\n=== Example 3: Regularization Hyperparameters ===")

from sklearn.datasets import make_moons

# Generate moons dataset
X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

# Create trees with different regularization
tree_clf1 = DecisionTreeClassifier(random_state=42)
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
tree_clf1.fit(X_moons, y_moons)
tree_clf2.fit(X_moons, y_moons)

print("✓ Trained decision trees with different regularization")

# =============================================================================
# EXAMPLE 4: DECISION TREE REGRESSION
# =============================================================================

print("\n=== Example 4: Decision Tree Regression ===")

from sklearn.tree import DecisionTreeRegressor

# Generate quadratic dataset
np.random.seed(42)
X_quad = np.random.rand(200, 1) - 0.5
y_quad = X_quad ** 2 + 0.025 * np.random.randn(200, 1)

# Train regression tree
tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)

print("✓ Decision tree regressor trained")

# =============================================================================
# EXAMPLE 5: SENSITIVITY TO AXIS ORIENTATION
# =============================================================================

print("\n=== Example 5: Sensitivity to Axis Orientation ===")

# Generate square dataset
np.random.seed(6)
X_square = np.random.rand(100, 2) - 0.5
y_square = (X_square[:, 0] > 0).astype(np.int64)

# Rotate the dataset
angle = np.pi / 4  # 45 degrees
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
X_rotated_square = X_square.dot(rotation_matrix)

# Train trees on original and rotated data
tree_clf_square = DecisionTreeClassifier(random_state=42)
tree_clf_square.fit(X_square, y_square)
tree_clf_rotated_square = DecisionTreeClassifier(random_state=42)
tree_clf_rotated_square.fit(X_rotated_square, y_square)

print("✓ Demonstrated axis orientation sensitivity")

# =============================================================================
# EXAMPLE 6: HIGH VARIANCE OF DECISION TREES
# =============================================================================

print("\n=== Example 6: High Variance of Decision Trees ===")

# Train tree with different random state
tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)
tree_clf_tweaked.fit(X_iris, y_iris)

print("✓ Demonstrated high variance with different random state")

# =============================================================================
# EXAMPLE 7: ACCESSING TREE STRUCTURE
# =============================================================================

print("\n=== Example 7: Accessing Tree Structure ===")

tree = tree_clf.tree_
print(f"Total nodes: {tree.node_count}")
print(f"Maximum depth: {tree.max_depth}")
print(f"Number of leaves: {tree.n_leaves}")

# =============================================================================
# EXAMPLE 8: EXERCISE - TUNING DECISION TREE
# =============================================================================

print("\n=== Example 8: Exercise - Tuning Decision Tree ===")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Generate larger moons dataset
X_moons_large, y_moons_large = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_moons_large, y_moons_large,
                                                    test_size=0.2, random_state=42)

# Grid search for best parameters
params = {
    'max_leaf_nodes': list(range(2, 100)),
    'max_depth': list(range(1, 7)),
    'min_samples_split': [2, 3, 4]
}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42),
                              params, cv=3)
grid_search_cv.fit(X_train, y_train)

print(f"Best parameters: {grid_search_cv.best_params_}")

# Evaluate on test set
y_pred = grid_search_cv.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")

print("\n=== All Examples Completed ===")