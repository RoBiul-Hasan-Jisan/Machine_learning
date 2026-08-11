# K-Nearest Neighbors

> **The one-sentence version:** to guess what something is, look at the few things most like it and go with the majority.


---

## 1. The Idea, in Plain Words

Imagine you move to a new neighborhood and want to guess whether your new neighbors are more likely to be dog people or cat people. A reasonable approach: look at the five closest houses, count how many have dogs vs. cats, and go with whichever is more common.

That's K-nearest neighbors. Nothing more sophisticated than that.

**What makes it unusual** compared to almost every other machine learning method: there's no "training." You don't fit a line, adjust weights, or minimize any loss. You just keep all your data around, and when a new question comes in, you compare it against everything you have and let the closest matches decide the answer.

- To **classify** something (dog person or cat person?) → take a vote among the nearest neighbors.
- To **predict a number** (how much will this house sell for?) → average the nearest neighbors' values.

---

## 2. What "Nearest" Actually Means

"Closest" isn't a single fixed idea — it depends on how you measure distance, and the choice matters more than people expect.

**Euclidean distance (the default)** — straight-line distance, like measuring with a ruler:

```
distance = sqrt( (a1-b1)² + (a2-b2)² + ... )
```

This is intuitive, but it has a trap: **it's sensitive to scale.** If one feature is "income in dollars" (ranging into the tens of thousands) and another is "age" (ranging 0–100), income will completely dominate the distance calculation just because its numbers are bigger — not because it's actually more important. The fix is always the same: standardize your features (rescale everything to a comparable range) before measuring distance.

**Manhattan distance** — like walking city blocks instead of cutting diagonally:

```
distance = |a1-b1| + |a2-b2| + ...
```

It doesn't square the differences, so one huge outlier gap doesn't dominate the total the way it does with Euclidean distance. Good when your data has noisy outliers.

**Cosine distance** — ignores *how big* the vectors are and only cares about the *direction* they point:

```
distance = 1 - (a · b) / (|a| * |b|)
```

This is the standard choice for text and embeddings. Two documents about the same topic can be different lengths, but if they "point the same way," cosine distance says they're similar — Euclidean distance would get confused by the length difference.

**Minkowski distance** is just a knob that can turn into any of the above: set its parameter `p=1` and you get Manhattan, `p=2` gives Euclidean, and `p→∞` gives "just look at the single biggest difference" (Chebyshev distance).

**Quick guide to picking one:**

| Your data | Use |
|---|---|
| Normal numeric features, similar scale | Euclidean (L2) |
| Numeric features with outliers | Manhattan (L1) |
| Text / embeddings | Cosine |
| High-dimensional and sparse | Cosine or Manhattan (Euclidean struggles here — see Section 6) |

---

## 3. Picking K

K is the only real decision you make with this algorithm, and it trades off two failure modes:

- **K too small (e.g. K=1):** your prediction depends entirely on whichever single point happens to be nearest — including any mislabeled or noisy points. This overfits: it "memorizes" the training data instead of learning a general pattern.
- **K too large (e.g. K = the whole dataset):** every prediction just becomes "the most common label overall," ignoring the query entirely. This underfits.

Somewhere in between is a boundary that's smooth enough to generalize but still responsive to real local patterns. A common starting point is **K ≈ √(number of training points)**, and for two-class problems, pick an odd K so votes can't tie.

---

## 4. Giving Closer Neighbors More Say

Plain KNN treats all K neighbors equally — a neighbor that's almost exactly on top of your query point gets the same one vote as a neighbor way out at the edge of the K-nearest group. That doesn't feel right, and there's a simple fix.

**Distance-weighted KNN** gives each neighbor a vote proportional to `1 / distance`, so very close points matter a lot and far points barely matter:

```
weight = 1 / (distance + tiny_number)
```

(The tiny number just avoids dividing by zero if a training point exactly matches the query.)

A nice side effect: weighted KNN is much less sensitive to your exact choice of K, since a neighbor far out in the "K-th place" spot contributes almost nothing anyway.

---

## 5. KNN for Predicting Numbers

Everything above was about classifying into categories. For predicting a continuous number (a price, a temperature, a score), swap "vote" for "average":

```
prediction = average of the K nearest neighbors' target values

or, weighted:
prediction = (sum of weight × value) / (sum of weights)
```

One quirk worth knowing: **KNN regression can never extrapolate.** If every house in your training data sold for between $100k and $900k, KNN will never predict $1.2 million for a new house, no matter how big or fancy it is — it can only ever output some kind of average of values it has already seen.

---

## 6. Why KNN Falls Apart in High Dimensions

This is the part that surprises people, and it's not a vague warning — it's a real mathematical effect called **the curse of dimensionality.**

**The core problem: distances stop being meaningful.** Picture scattering random points in a space with more and more dimensions. As the dimension count grows, something strange happens: the distance to the *closest* point and the distance to the *farthest* point start to converge — everything becomes almost equally "far away."

```
Ratio of farthest to nearest distance, for random points:

  2 dimensions:     varies a lot (this is what your intuition expects)
  100 dimensions:   ratio ≈ 1.01
  1000 dimensions:  ratio ≈ 1.001
```

If "nearest" and "farthest" are basically the same number, the whole idea of "nearest neighbor" loses its meaning.

**Why this happens, intuitively:** to capture the same fraction of your data as neighbors, your search radius has to expand dramatically as dimensions increase — the "neighborhood" balloons until it covers most of the space. And in a high-dimensional cube, almost all the volume sits out near the corners, not clustered near the center the way your 2D/3D intuition expects.

**The practical takeaway:** KNN works well up to roughly 20–50 features. Past that, either shrink the dimensionality first (PCA, UMAP) or accept that distance-based methods need help.

---

## 7. Making It Fast: Smarter Search

The naive way to find nearest neighbors is to compare the query against *every single point* — that's `O(n × d)` work per prediction, which gets painfully slow as your dataset grows.

**KD-trees** speed this up by organizing the data ahead of time: recursively slice the space in half along one feature at a time (like a 20-questions game — "is x1 above or below 5?", then "is x2 above or below 3?", and so on). To find a neighbor, you walk straight down to the right region, then only backtrack and check nearby regions if they *could* possibly contain something closer.

This gets you down to roughly `O(log n)` per query — but only in low dimensions. The catch is the same curse of dimensionality from Section 6: past about 20 dimensions, the "could possibly contain something closer" check almost always says yes, so you end up backtracking through nearly everything anyway, and the speed advantage disappears.

**Ball trees** are a variant that groups points into nested spheres instead of axis-aligned boxes. They tend to hold up a bit better than KD-trees in moderately high dimensions (up to ~50), because a sphere can bound a cluster of points more tightly than a box can.

**Beyond that**, at real production scale — millions of points, hundreds of dimensions, like a vector database — even ball trees aren't enough. That's when people switch to *approximate* nearest neighbor methods (HNSW, IVF, and similar), which trade a small amount of accuracy for a large amount of speed.

---

## 8. Lazy vs. Eager Learning

KNN belongs to a category called **lazy learners**: all the real work happens at prediction time, not training time.

| | KNN (lazy) | Most other models — SVMs, neural nets (eager) |
|---|---|---|
| "Training" | Just store the data — instant | Real computation: gradient descent, epochs |
| Prediction | Compare against all stored data — can be slow | Fast — just apply the already-learned parameters |
| Adding new data | Instant — just add the point | Requires retraining |
| What gets stored | The entire training set | A compact set of learned parameters |

This tradeoff is genuinely useful in some situations: if your data changes constantly and retraining a model every time would be a hassle, KNN lets you just add new points on the fly.

---

## 9. Where KNN Shows Up in Real Systems

KNN feels like a toy algorithm, but the same core idea — "find the closest things and use them" — powers a surprising amount of modern AI infrastructure, just at bigger scale and under different names:

- **Vector databases** run nearest-neighbor search over embeddings.
- **Retrieval-augmented generation (RAG)** finds the K most relevant document chunks to feed into a language model.
- **Recommendation systems** find users or items similar to what you're already interested in.

The algorithm is the same; only the scale and the search structure (approximate methods instead of brute force or KD-trees) changes.

---

## 10. Trying It Yourself

Always scale your features first — this is the single most important step, since KNN's whole notion of "close" depends directly on your numbers being comparable.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
])
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

Scikit-learn will automatically choose a KD-tree, ball tree, or brute force depending on your dataset size and dimensionality — you can also force this with the `algorithm` parameter.

For large-scale nearest-neighbor search (millions of vectors), reach for a dedicated library instead:

```python
import faiss

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
distances, indices = index.search(query_vectors, k=5)
```

---

## 11. Glossary

| Term | Plain-English meaning |
|---|---|
| **K** | The number of neighbors you look at. Your one real hyperparameter. |
| **Nearest neighbor** | A stored point that's close to your query, by whatever distance function you chose. |
| **Majority vote** | Classification method: whichever class appears most often among the K neighbors wins. |
| **Distance-weighted KNN** | A version where closer neighbors count more than farther ones. |
| **Lazy learning** | No work at training time — everything happens when a prediction is requested. KNN is the classic example. |
| **Eager learning** | The opposite — heavy computation up front (training), fast predictions after. Most other ML models. |
| **Curse of dimensionality** | The effect where distances stop being meaningful once you have too many features. |
| **KD-tree** | A data structure that organizes points to make nearest-neighbor search much faster in low dimensions. |
| **Ball tree** | Similar to a KD-tree, but using nested spheres instead of boxes — holds up better in moderate dimensions. |
| **Approximate nearest neighbor (ANN)** | Search methods (like HNSW) that sacrifice a little accuracy for a lot of speed at massive scale. |

---

## 12. Common Mistakes

- **Forgetting to scale features** — the single most common KNN mistake. One large-range feature silently dominates every distance calculation.
- **Using Euclidean distance on high-dimensional or sparse data** (like text) — cosine distance is almost always a better fit.
- **Picking K=1 "because it's simplest"** — this just memorizes the training data and is very sensitive to noisy or mislabeled points.
- **Expecting KNN regression to extrapolate** — it can only output values within the range it has already seen.
- **Using brute-force search on a huge dataset** — past a certain size, you need a KD-tree, ball tree, or approximate method, not a linear scan over every point.
- **Using a KD-tree on high-dimensional data and expecting a speedup** — past ~20-50 dimensions it often performs no better than brute force, because of the curse of dimensionality.
