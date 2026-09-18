# Random Forest — (Play Tennis Dataset)

This is the classic **"Play Tennis"** dataset (originally used by Quinlan for ID3/C4.5 decision trees, and widely reused as a standard teaching example for tree-based methods including Random Forest).

## 1. The Real Dataset

Predict whether a person will **play tennis** based on weather conditions.

| Day | Outlook | Temperature | Humidity | Windy | Play Tennis ($y$) |
|-----|---------|-------------|----------|-------|--------------------|
| $D_1$ | Sunny | Hot | High | False | No |
| $D_2$ | Sunny | Hot | High | True | No |
| $D_3$ | Overcast | Hot | High | False | Yes |
| $D_4$ | Rain | Mild | High | False | Yes |
| $D_5$ | Rain | Cool | Normal | False | Yes |
| $D_6$ | Rain | Cool | Normal | True | No |
| $D_7$ | Overcast | Cool | Normal | True | Yes |
| $D_8$ | Sunny | Mild | High | False | No |
| $D_9$ | Sunny | Cool | Normal | False | Yes |
| $D_{10}$ | Rain | Mild | Normal | False | Yes |
| $D_{11}$ | Sunny | Mild | Normal | True | Yes |
| $D_{12}$ | Overcast | Mild | High | True | Yes |
| $D_{13}$ | Overcast | Hot | Normal | False | Yes |
| $D_{14}$ | Rain | Mild | High | True | No |

We want to predict for a new day:

$$
Q = (\text{Outlook} = \text{Sunny}, \ \text{Temperature} = \text{Cool}, \ \text{Humidity} = \text{High}, \ \text{Windy} = \text{False})
$$

## 2. Baseline: Class Distribution and Entropy

Out of 14 days: **9 Yes**, **5 No**.

$$
p_{\text{Yes}} = \frac{9}{14} = 0.643, \qquad p_{\text{No}} = \frac{5}{14} = 0.357
$$

Root entropy:

$$
H(S) = -p_{\text{Yes}} \log_2 p_{\text{Yes}} - p_{\text{No}} \log_2 p_{\text{No}}
$$

$$
H(S) = -(0.643)\log_2(0.643) - (0.357)\log_2(0.357) = 0.410 + 0.530 = 0.940
$$

## 3. Step 1 — Bootstrap Sampling (Bagging)

Random Forest draws random samples **with replacement** from the 14 days to build each tree. Suppose we grow $T = 3$ trees:

$$
D_1 = \{D_1, D_2, D_3, D_3, D_5, D_7, D_8, D_9, D_9, D_{10}, D_{11}, D_{12}, D_{13}, D_{14}\}
$$

$$
D_2 = \{D_1, D_2, D_4, D_4, D_5, D_6, D_7, D_8, D_{10}, D_{11}, D_{11}, D_{12}, D_{13}, D_{14}\}
$$

$$
D_3 = \{D_2, D_3, D_3, D_4, D_5, D_6, D_7, D_9, D_9, D_{10}, D_{11}, D_{12}, D_{13}, D_{13}\}
$$

## 4. Step 2 — Random Feature Subsets per Split

With $p = 4$ features (Outlook, Temperature, Humidity, Windy), a common default is:

$$
m = \sqrt{p} = \sqrt{4} = 2
$$

So at each split, only 2 randomly chosen features are considered — not all 4.

## 5. Step 3 — Information Gain for the Real Split on "Outlook"

Splitting the full 14-day dataset on **Outlook** (a strong, real signal in this dataset):

- **Sunny** (5 days: $D_1,D_2,D_8,D_9,D_{11}$): 2 Yes, 3 No
- **Overcast** (4 days: $D_3,D_7,D_{12},D_{13}$): 4 Yes, 0 No
- **Rain** (5 days: $D_4,D_5,D_6,D_{10},D_{14}$): 3 Yes, 2 No

$$
H(\text{Sunny}) = -\tfrac{2}{5}\log_2\tfrac{2}{5} - \tfrac{3}{5}\log_2\tfrac{3}{5} = 0.971
$$

$$
H(\text{Overcast}) = -\tfrac{4}{4}\log_2\tfrac{4}{4} - 0 = 0 \quad (\text{pure node — always Yes})
$$

$$
H(\text{Rain}) = -\tfrac{3}{5}\log_2\tfrac{3}{5} - \tfrac{2}{5}\log_2\tfrac{2}{5} = 0.971
$$

Weighted entropy after the split:

$$
H(S \mid \text{Outlook}) = \tfrac{5}{14}(0.971) + \tfrac{4}{14}(0) + \tfrac{5}{14}(0.971) = 0.693
$$

Information gain:

$$
IG(\text{Outlook}) = H(S) - H(S \mid \text{Outlook}) = 0.940 - 0.693 = 0.247
$$

This is the same split real decision trees pick first on this dataset — **Outlook** is the strongest feature, exactly what Random Forest trees will tend to select often too.

## 6. Step 4 — Simplified Trees Actually Grown

**Tree 1** (trained on $D_1$, features Outlook + Humidity available):

$$
\text{Outlook} = \text{Overcast} \Rightarrow \text{Yes}
$$
$$
\text{Outlook} = \text{Sunny}, \ \text{Humidity} = \text{High} \Rightarrow \text{No}
$$
$$
\text{Outlook} = \text{Sunny}, \ \text{Humidity} = \text{Normal} \Rightarrow \text{Yes}
$$
$$
\text{Outlook} = \text{Rain} \Rightarrow \text{Yes}
$$

**Tree 2** (trained on $D_2$, features Outlook + Windy available):

$$
\text{Outlook} = \text{Overcast} \Rightarrow \text{Yes}
$$
$$
\text{Outlook} = \text{Sunny} \Rightarrow \text{No}
$$
$$
\text{Outlook} = \text{Rain}, \ \text{Windy} = \text{False} \Rightarrow \text{Yes}
$$
$$
\text{Outlook} = \text{Rain}, \ \text{Windy} = \text{True} \Rightarrow \text{No}
$$

**Tree 3** (trained on $D_3$, features Humidity + Windy available):

$$
\text{Humidity} = \text{Normal} \Rightarrow \text{Yes}
$$
$$
\text{Humidity} = \text{High}, \ \text{Windy} = \text{False} \Rightarrow \text{Yes}
$$
$$
\text{Humidity} = \text{High}, \ \text{Windy} = \text{True} \Rightarrow \text{No}
$$

## 7. Step 5 — Predict $Q$ with Each Tree

Recall:

$$
Q = (\text{Sunny}, \text{Cool}, \text{High}, \text{False})
$$

$$
\text{Tree 1: Outlook=Sunny, Humidity=High} \Rightarrow \text{No}
$$

$$
\text{Tree 2: Outlook=Sunny} \Rightarrow \text{No}
$$

$$
\text{Tree 3: Humidity=High, Windy=False} \Rightarrow \text{Yes}
$$

## 8. Step 6 — Aggregate (Majority Vote)

$$
\hat{y}(Q) = \text{mode}(\text{No}, \text{No}, \text{Yes}) = \text{No}
$$

Vote tally: **No: 2, Yes: 1** → the forest predicts **No, don't play tennis**.

This matches real intuition: Sunny + High Humidity is historically the strongest "No" signal in this exact dataset (see $D_1, D_2, D_8$, all Sunny+High → No), and 2 of 3 trees captured that pattern even though they were trained on different random samples and feature subsets.

## 9. General Formula Used

$$
\hat{y}(x) = \text{mode}\big(\{\hat{y}_t(x) \ : \ t = 1, \dots, T\}\big)
$$

## 10. Real Feature Importance (Qualitative)

Because **Outlook** produced the single largest information gain ($IG = 0.247$) at the root, it is used in every one of the 3 trees' early splits whenever it's included in that tree's random feature subset — consistent with real Random Forest behavior, where genuinely predictive features tend to dominate importance rankings even under feature-subsampling randomness.
