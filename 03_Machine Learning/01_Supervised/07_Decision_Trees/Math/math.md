# Decision Tree — (Play Tennis Dataset, ID3 Algorithm)

Same real dataset as before — the classic **Play Tennis** dataset (Quinlan, used to introduce ID3/C4.5 decision trees).

## 1. The Real Dataset

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

Target: predict $y$ for a new day $Q = (\text{Sunny}, \text{Cool}, \text{High}, \text{False})$.

## 2. Entropy Formula

$$
H(S) = -\sum_{k=1}^{K} p_k \log_2 p_k
$$

## 3. Root Entropy

9 Yes, 5 No out of 14:

$$
H(S) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14} = 0.410 + 0.530 = 0.940
$$

## 4. Information Gain Formula

$$
IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
$$

We compute $IG$ for **all four** features to find the best root split.

## 5. Information Gain — Outlook

- Sunny (5): 2 Yes, 3 No → $H = 0.971$
- Overcast (4): 4 Yes, 0 No → $H = 0$
- Rain (5): 3 Yes, 2 No → $H = 0.971$

$$
H(S|\text{Outlook}) = \frac{5}{14}(0.971) + \frac{4}{14}(0) + \frac{5}{14}(0.971) = 0.693
$$

$$
IG(\text{Outlook}) = 0.940 - 0.693 = 0.247
$$

## 6. Information Gain — Humidity

- High (7): 3 Yes, 4 No → $H = -\frac{3}{7}\log_2\frac{3}{7} - \frac{4}{7}\log_2\frac{4}{7} = 0.985$
- Normal (7): 6 Yes, 1 No → $H = -\frac{6}{7}\log_2\frac{6}{7} - \frac{1}{7}\log_2\frac{1}{7} = 0.592$

$$
H(S|\text{Humidity}) = \frac{7}{14}(0.985) + \frac{7}{14}(0.592) = 0.789
$$

$$
IG(\text{Humidity}) = 0.940 - 0.789 = 0.151
$$

## 7. Information Gain — Windy

- False (8): 6 Yes, 2 No → $H = -\frac{6}{8}\log_2\frac{6}{8} - \frac{2}{8}\log_2\frac{2}{8} = 0.811$
- True (6): 3 Yes, 3 No → $H = 1.000$

$$
H(S|\text{Windy}) = \frac{8}{14}(0.811) + \frac{6}{14}(1.000) = 0.892
$$

$$
IG(\text{Windy}) = 0.940 - 0.892 = 0.048
$$

## 8. Information Gain — Temperature

- Hot (4): 2 Yes, 2 No → $H = 1.000$
- Mild (6): 4 Yes, 2 No → $H = 0.918$
- Cool (4): 3 Yes, 1 No → $H = 0.811$

$$
H(S|\text{Temp}) = \frac{4}{14}(1.000) + \frac{6}{14}(0.918) + \frac{4}{14}(0.811) = 0.911
$$

$$
IG(\text{Temperature}) = 0.940 - 0.911 = 0.029
$$

## 9. Choosing the Root

$$
IG(\text{Outlook}) = 0.247 \ \gg \ IG(\text{Humidity}) = 0.151 \ > \ IG(\text{Windy}) = 0.048 \ > \ IG(\text{Temp}) = 0.029
$$

**Outlook wins** → it becomes the root node.

## 10. Expand the "Overcast" Branch

All 4 Overcast days ($D_3, D_7, D_{12}, D_{13}$) are **Yes** → $H = 0$ → this branch is already **pure**. It becomes a leaf:

$$
\text{Outlook} = \text{Overcast} \ \Rightarrow \ \text{Leaf: Yes}
$$

## 11. Expand the "Sunny" Branch

Subset (5 days): $D_1, D_2, D_8, D_9, D_{11}$ → 2 Yes, 3 No, $H = 0.971$

Compute $IG$ within this subset for remaining features:

**Humidity** within Sunny:
- High ($D_1, D_2, D_8$): 0 Yes, 3 No → $H = 0$
- Normal ($D_9, D_{11}$): 2 Yes, 0 No → $H = 0$

$$
H(\text{Sunny}|\text{Humidity}) = \frac{3}{5}(0) + \frac{2}{5}(0) = 0
$$

$$
IG(\text{Humidity} \mid \text{Sunny}) = 0.971 - 0 = 0.971 \quad (\text{perfect split!})
$$

So within Sunny, **Humidity** perfectly separates the classes:

$$
\text{Outlook} = \text{Sunny}, \text{Humidity} = \text{High} \ \Rightarrow \ \text{Leaf: No}
$$

$$
\text{Outlook} = \text{Sunny}, \text{Humidity} = \text{Normal} \ \Rightarrow \ \text{Leaf: Yes}
$$

## 12. Expand the "Rain" Branch

Subset (5 days): $D_4, D_5, D_6, D_{10}, D_{14}$ → 3 Yes, 2 No, $H = 0.971$

**Windy** within Rain:
- False ($D_4, D_5, D_{10}$): 3 Yes, 0 No → $H = 0$
- True ($D_6, D_{14}$): 0 Yes, 2 No → $H = 0$

$$
H(\text{Rain}|\text{Windy}) = \frac{3}{5}(0) + \frac{2}{5}(0) = 0
$$

$$
IG(\text{Windy} \mid \text{Rain}) = 0.971 - 0 = 0.971 \quad (\text{perfect split!})
$$

$$
\text{Outlook} = \text{Rain}, \text{Windy} = \text{False} \ \Rightarrow \ \text{Leaf: Yes}
$$

$$
\text{Outlook} = \text{Rain}, \text{Windy} = \text{True} \ \Rightarrow \ \text{Leaf: No}
$$

## 13. The Final Tree

$$
\text{Outlook}
$$
$$
\begin{aligned}
&\vdash \text{Overcast} \rightarrow \text{Yes} \\
&\vdash \text{Sunny} \rightarrow \text{Humidity} \begin{cases} \text{High} \rightarrow \text{No} \\ \text{Normal} \rightarrow \text{Yes} \end{cases} \\
&\vdash \text{Rain} \rightarrow \text{Windy} \begin{cases} \text{False} \rightarrow \text{Yes} \\ \text{True} \rightarrow \text{No} \end{cases}
\end{aligned}
$$

Note: **Temperature never gets used anywhere in the tree** — real ID3 output on this dataset, because Outlook, Humidity, and Windy alone perfectly classify all 14 training examples.

## 14. Classify the New Example

$$
Q = (\text{Outlook}=\text{Sunny}, \text{Temperature}=\text{Cool}, \text{Humidity}=\text{High}, \text{Windy}=\text{False})
$$

Trace the tree:

$$
\text{Outlook} = \text{Sunny} \ \rightarrow \ \text{Humidity} = \text{High} \ \rightarrow \ \boxed{\hat{y}(Q) = \text{No}}
$$

## 15. General Decision Tree Prediction Rule

$$
\hat{y}(x) = \text{label of the leaf reached by following } x \text{'s feature values down the tree}
$$

## 16. Gini Impurity (Alternative to Entropy, Used by CART)

Instead of entropy, CART-style trees (e.g., scikit-learn's default) use **Gini impurity**:

$$
G(S) = 1 - \sum_{k=1}^{K} p_k^2
$$

For the root node (9 Yes, 5 No):

$$
G(S) = 1 - \left[\left(\frac{9}{14}\right)^2 + \left(\frac{5}{14}\right)^2\right] = 1 - [0.413 + 0.128] = 0.459
$$

The split selection logic is identical to entropy/information gain — just swap $H$ for $G$ throughout; Outlook still wins as the best root split on this real dataset either way.
