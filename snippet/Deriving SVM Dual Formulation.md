# SVM: Primal to Dual Formulation

This document details the mathematical derivation of the Support Vector Machine (SVM) dual problem. It begins with the primal problem for both hard and soft margins and proceeds step-by-step to the dual formulation using Lagrange multipliers.

---

## 1. The Primal Problem

The primal problem seeks to minimize the weights (maximize margin) while satisfying the classification constraints.

### A. Hard Margin (Linearly Separable)

If the data is perfectly separable, the constraints require every point to be on the correct side of the margin (at least distance 1).

* **Objective:** $$\min_{w, b} \frac{1}{2} ||w||^2$$
* **Constraints:**
    $$y_i(w^T x_i + b) \ge 1 \quad \forall i=1,\dots,N$$

### B. Soft Margin (Non-Linearly Separable)

This is the more general form used in practice. We introduce **slack variables** $\xi_i$ to allow some misclassification.

* **Objective:** $$\min_{w, b, \xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{N} \xi_i$$
* **Constraints:**
    1.  $$y_i(w^T x_i + b) \ge 1 - \xi_i \quad \forall i$$
    2.  $$\xi_i \ge 0 \quad \forall i$$

*(Note: The derivation below uses the Hard Margin constraints for mathematical clarity, as the Soft Margin derivation leads to the exact same dual form, just with an upper bound constraint on $\alpha$.)*

---

### 2. Forming the Lagrangian

To use Lagrange multipliers, we rewrite the constraints in the form $g(x) \le 0$.
$$1 - y_i(w^T x_i + b) \le 0$$

We introduce Lagrange multipliers $\alpha_i \ge 0$ and add the constraints to the objective:

$$L(w, b, \alpha) = \frac{1}{2} ||w||^2 + \sum_{i=1}^{N} \alpha_i \left[ 1 - y_i(w^T x_i + b) \right]$$

Which expands to:
$$L(w, b, \alpha) = \frac{1}{2} ||w||^2 + \sum_{i=1}^{N} \alpha_i - \sum_{i=1}^{N} \alpha_i y_i w^T x_i - \sum_{i=1}^{N} \alpha_i y_i b$$

---

### 3. KKT Conditions (Minimizing Primal Variables)

We find the minimum with respect to $w$ and $b$ by setting the partial derivatives to zero.

**Gradient w.r.t. $w$:**
$$\frac{\partial L}{\partial w} = w - \sum_{i=1}^{N} \alpha_i y_i x_i = 0 \implies \mathbf{w = \sum_{i=1}^{N} \alpha_i y_i x_i}$$

**Gradient w.r.t. $b$:**
$$\frac{\partial L}{\partial b} = - \sum_{i=1}^{N} \alpha_i y_i = 0 \implies \mathbf{\sum_{i=1}^{N} \alpha_i y_i = 0}$$

---

### 4. Substituting Back (The Dual)

We plug the optimal $w$ and the constraint on $b$ back into the Lagrangian to eliminate them.

**Step A: Substitute $w$ into the term $\frac{1}{2} ||w||^2$**
$$\frac{1}{2} w^T w = \frac{1}{2} \left( \sum_{i} \alpha_i y_i x_i \right)^T \left( \sum_{j} \alpha_j y_j x_j \right) = \frac{1}{2} \sum_{i} \sum_{j} \alpha_i \alpha_j y_i y_j (x_i^T x_j)$$

**Step B: Substitute $w$ into the term $-\sum \alpha_i y_i w^T x_i$**
$$- \sum_{i} \alpha_i y_i \left( \sum_{j} \alpha_j y_j x_j \right)^T x_i = - \sum_{i} \sum_{j} \alpha_i \alpha_j y_i y_j (x_i^T x_j)$$

**Step C: Handle the bias term**
Since $\sum \alpha_i y_i = 0$, the term involving $b$ becomes zero:
$$- b \sum_{i} \alpha_i y_i = -b(0) = 0$$

**Step D: Combine everything**
$$L(\alpha) = \underbrace{\sum_{i} \alpha_i}_{\text{from expansion}} + \underbrace{\frac{1}{2} \sum \sum \dots}_{\text{from Step A}} - \underbrace{\sum \sum \dots}_{\text{from Step B}}$$

$$L(\alpha) = \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i^T x_j)$$

---

### 5. Final Dual Problem

We maximize this function with respect to $\alpha$, subject to the constraints.

**Objective:**
$$\max_{\alpha} \left( \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i^T x_j) \right)$$

**Subject to:**
1.  $\sum_{i=1}^{N} \alpha_i y_i = 0$
2.  **Constraints on $\alpha$:**
    * **Hard Margin:** $\alpha_i \ge 0$
    * **Soft Margin:** $0 \le \alpha_i \le C$ (This is the only difference in the final dual form).

Would you like me to calculate a simple numerical example (e.g., with XOR logic or 1D points) to show how the $\alpha$ values are actually found?

... [Derive the Dual Formulation for Support Vector Machines](https://www.youtube.com/watch?v=4TIbsTa3r24) ...

This video is highly relevant as it provides a whiteboard walk-through of the specific mathematical derivation from the Lagrangian to the Dual form shown above.
