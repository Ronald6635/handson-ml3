# Derivation of the SVM Decision Function

The derivation of the SVM decision function is a direct consequence of the optimization problem that defines the support vector classifier. Here is a step-by-step explanation of how we arrive at the final form.

## The Starting Point: The Primal Decision Function

The goal of a linear classifier is to find a hyperplane that separates the data. The equation of this hyperplane is defined by a weight vector $w$ and a bias term $b$. For any new point $x$, we classify it based on which side of the hyperplane it falls on. This gives us our initial, primal decision function:

$$
f(x) = \operatorname{sign}(w^T x + b)
$$

The challenge is that we don't know $w$ directly. The SVM optimization process solves for the Lagrange multipliers ($\alpha_i$), not $w$. So, we need to express $w$ in terms of the variables we *do* solve for.

### Step 1: Expressing the Weight Vector $w$

During the derivation of the dual problem, we took the partial derivative of the Lagrangian function $L(w, b, \alpha)$ with respect to $w$ and set it to zero to find the minimum. This gave us a crucial relationship for the optimal weight vector $w$:
$$
\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{i=1}^{m} \alpha_i y_i x_i = 0
$$

Solving for $w$, we get:

$$
\mathbf{w = \sum_{i=1}^{m} \alpha_i y_i x_i}
$$

This equation is fundamental. It states that the optimal weight vector $w$ (which defines the separating hyperplane) is simply a **linear combination of the training data points $x_i$**. The $\alpha_i$ values act as the weights for this combination.

### Step 2: Substituting $w$ into the Decision Function

Now we can substitute this expression for $w$ back into our original decision function:

$$
f(x) = \operatorname{sign}\left( \left( \sum_{i=1}^{m} \alpha_i y_i x_i \right)^T x + b \right)
$$

By rearranging the terms (specifically, moving the transpose and the input vector $x$ inside the summation), we get:

$$
f(x) = \operatorname{sign}\left( \sum_{i=1}^{m} \alpha_i y_i (x_i^T x) + b \right)
$$

This form is already powerful because it shows that the prediction depends on the dot product between the training points $x_i$ and the new point $x$.

### Step 3: The Role of Support Vectors (Why only some $\alpha_i$ matter)

This is the most critical step. The Karush-Kuhn-Tucker (KKT) conditions of the SVM optimization problem include a "complementary slackness" condition:

$$
\alpha_i \left[ y_i (w^T x_i + b) - 1 \right] = 0
$$

This equation tells us that for any training point $x_i$, at least one of the two factors must be zero:

1.  $\alpha_i = 0$, which means the point does not contribute to defining the hyperplane.
2.  $y_i (w^T x_i + b) - 1 = 0$, which means the point lies exactly on the margin.

This has a profound implication: **If a training point $x_i$ is correctly classified and lies outside the margin, its corresponding $\alpha_i$ must be zero.**

Therefore, the only points that can have a non-zero $\alpha_i$ are the ones that lie exactly on the margin or are misclassified (in the soft-margin case). These points are the **support vectors**.

Since any term in the summation where $\alpha_i = 0$ vanishes, we can simplify the sum to only include the support vectors:

$$
f(x) = \operatorname{sign}\left( \sum_{i \in \text{Support Vectors}} \alpha_i y_i (x_i^T x) + b \right)
$$

This is often written more compactly as summing over indices where $\alpha_i > 0$.

### Step 4: Introducing the Kernel Trick

The final step is to generalize the function for non-linear classification. The "kernel trick" is the observation that the decision function only depends on the dot product $x_i^T x$. We can replace this dot product with a more general **kernel function** $k(x_i, x)$, which computes the similarity between $x_i$ and $x$ in a higher-dimensional feature space without ever explicitly calculating the transformation.

By replacing the dot product with the kernel function, we arrive at the final form of the SVM decision function:

$$
f(x) = \operatorname{sign}\left( \sum_{i:\alpha_i>0} \alpha_i y_i k(x_i, x) + b \right)
$$

This derivation shows how the elegant final decision rule emerges directly from the mathematical foundations of the SVM optimization problem, revealing its dependency on only the most critical points in the dataset—the support vectors.
