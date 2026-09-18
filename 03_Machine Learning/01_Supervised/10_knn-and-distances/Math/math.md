# K-Nearest Neighbors (KNN) 

## 1. The Idea

KNN classifies a new point by looking at the $k$ closest points in the training data (measured by distance) and taking a majority vote of their labels.

## 2. Example Dataset

Suppose we want to classify fruits as **Apple (A)** or **Orange (O)** based on two features: *Weight (g)* and *Sweetness (1–10)*.

| Point | Weight ($x_1$) | Sweetness ($x_2$) | Label |
|-------|------|------|-------|
| $P_1$ | 150 | 7 | A |
| $P_2$ | 170 | 6 | A |
| $P_3$ | 140 | 8 | A |
| $P_4$ | 130 | 3 | O |
| $P_5$ | 120 | 2 | O |
| $P_6$ | 200 | 1 | O |

We want to classify a new point:

$$
Q = (160, 6)
$$

## 3. Distance Formula (Euclidean)

$$
d(Q, P_i) = \sqrt{(x_1^{(Q)} - x_1^{(P_i)})^2 + (x_2^{(Q)} - x_2^{(P_i)})^2}
$$

## 4. Compute Distances

$$
d(Q, P_1) = \sqrt{(160-150)^2 + (6-7)^2} = \sqrt{100 + 1} = \sqrt{101} \approx 10.05
$$

$$
d(Q, P_2) = \sqrt{(160-170)^2 + (6-6)^2} = \sqrt{100 + 0} = \sqrt{100} = 10.00
$$

$$
d(Q, P_3) = \sqrt{(160-140)^2 + (6-8)^2} = \sqrt{400 + 4} = \sqrt{404} \approx 20.10
$$

$$
d(Q, P_4) = \sqrt{(160-130)^2 + (6-3)^2} = \sqrt{900 + 9} = \sqrt{909} \approx 30.15
$$

$$
d(Q, P_5) = \sqrt{(160-120)^2 + (6-2)^2} = \sqrt{1600 + 16} = \sqrt{1616} \approx 40.20
$$

$$
d(Q, P_6) = \sqrt{(160-200)^2 + (6-1)^2} = \sqrt{1600 + 25} = \sqrt{1625} \approx 40.31
$$

## 5. Rank Distances (Ascending)

$$
d(Q,P_2) \approx 10.00 < d(Q,P_1) \approx 10.05 < d(Q,P_3) \approx 20.10 < \dots
$$

## 6. Choose $k$ and Vote

If $k = 3$, the nearest neighbors are:

$$
P_2 \ (A), \quad P_1 \ (A), \quad P_3 \ (A)
$$

Majority vote:

$$
\hat{y}(Q) = \arg\max_{c \in \{A, O\}} \sum_{i \in N_k(Q)} \mathbb{1}(y_i = c)
$$

Here, votes are $A: 3$, $O: 0$, so:

$$
\hat{y}(Q) = \text{Apple}
$$

## 7. General KNN Classification Formula

$$
\hat{y}(x) = \text{mode}(\{y_i : i \in N_k(x)\})
$$

where $N_k(x)$ is the set of indices of the $k$ training points closest to $x$.

For **regression**, instead of a majority vote, you average the neighbors' target values:

$$
\hat{y}(x) = \frac{1}{k} \sum_{i \in N_k(x)} y_i
$$
