# Deriving the Dual Form of the Support Vector Machine (SVM)

The primal SVM optimization problem (hard-margin for linearly separable data) is:

$$
\begin{align}
\min_{w, b} \quad & \frac{1}{2} \|w\|^2 \\
\text{s.t.} \quad & y_i (w^T x_i + b) \geq 1 \quad \forall i = 1,\dots,m
\end{align}
$$

where $y_i \in \{-1, +1\}$, it's the label for data point $x_i$.

We will derive the **dual Lagrangian** form step by step.

## Step 1: Write the Lagrangian

To use Lagrange multipliers, we first rewrite the constraints in the form $g_i(w,b) \geq 0$.
    $$y_i (w^T x_i + b) - 1 \geq 0$$

Introduce non-negative Lagrange multipliers $\alpha_i \geq 0$ for each inequality constraint:

$$
\mathcal{L}(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^m \alpha_i \left[ y_i (w^T x_i + b) - 1 \right]
$$

(Note the minus sign because the constraint is $\geq 1$; this is the standard KKT form.)

## Step 2: Take partial derivatives and set them to zero (for minimality w.r.t. primal variables)

We find the stationary points by taking derivatives w.r.t. $w$ and $b$:

$$
\frac{\partial \mathcal{L}}{\partial w} = 0 \quad \Rightarrow \quad w = \sum_{i=1}^m \alpha_i y_i x_i
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = 0 \quad \Rightarrow \quad \sum_{i=1}^m \alpha_i y_i = 0
$$

These are the two key stationarity conditions.

## Step 3: Substitute the relations back into the Lagrangian

Plug $w = \sum_j \alpha_j y_j x_j$ into $\mathcal{L}$:

$$
\begin{align}
\mathcal{L} &= \frac{1}{2} \left\|\sum_j \alpha_j y_j x_j\right\|^2 - \sum_i \alpha_i y_i \left( w^T x_i + b \right) + \sum_i \alpha_i \\
&= \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^T x_j) - \sum_i \alpha_i y_i \left( \left(\sum_j \alpha_j y_j x_j\right)^T x_i + b \right) + \sum_i \alpha_i
\end{align}
$$

Now expand the middle term:

$$
\sum_i \alpha_i y_i \left( \sum_j \alpha_j y_j (x_j^T x_i) \right) + \sum_i \alpha_i y_i b = \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^T x_j) + b \sum_i \alpha_i y_i
$$

Because of the second stationarity condition $\sum_i \alpha_i y_i = 0$, the $b$-term vanishes.

So the Lagrangian simplifies to:

$$
\mathcal{L} = \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^T x_j) - \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^T x_j) + \sum_i \alpha_i
$$

$$
= \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^T x_j)
$$

## Step 4: The dual optimization problem

The dual function is $g(\alpha) = \min_{w,b} \mathcal{L}(w,b,\alpha)$, which after substitution is exactly the expression above. Therefore we **maximize** it subject to the dual constraints:

$$
\begin{align}
\max_{\alpha} \quad & \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j (x_i^T x_j) \\
\text{s.t.} \quad & \alpha_i \geq 0 \quad \forall i \\
& \sum_{i=1}^m \alpha_i y_i = 0
\end{align}
$$

This is usually written as a minimization by flipping the sign:

$$
\begin{align}
\min_{\alpha} \quad & \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle - \sum_i \alpha_i \\
\text{s.t.} \quad & \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0
\end{align}
$$

or more compactly in matrix form:

$$
\begin{align}
\min_{\alpha} \quad & \frac{1}{2} \alpha^T Q \alpha - 1^T \alpha \\
\text{s.t.} \quad & y^T \alpha = 0, \quad \alpha \succeq 0
\end{align}
$$

where $Q_{ij} = y_i y_j \langle x_i, x_j \rangle$ and $1$ is the all-ones vector.

## Step 5: Kernel trick (moving to feature space)

Replace the dot product with a kernel:

$$
\langle x_i, x_j \rangle \;\to\; k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle
$$

The dual becomes identical except $Q_{ij} = y_i y_j k(x_i, x_j)$.

The decision function is then:

$$
f(x) = \operatorname{sign}\left( \sum_{i:\alpha_i>0} \alpha_i y_i k(x_i, x) + b \right)
$$

where only support vectors ($\alpha_i > 0$) appear.

At its core, the decision function calculates a weighted sum of kernel evaluations between the input $x$ and the support vectors, plus the bias term $b$.

## Summary of the final dual problem (most common form seen in textbooks)

$$
\begin{align}
\max_{\alpha} \quad & \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \, k(x_i, x_j) \\
\text{s.t.} \quad & \alpha_i \geq 0 \quad \forall i \\
& \sum_{i=1}^m \alpha_i y_i = 0
\end{align}
$$

This is the standard dual SVM formulation that is actually solved in practice (e.g., via SMO, libsvm, etc.), and it naturally enables the kernel trick for non-linear classification.