# Solutions — 02 Interpretable Models

1. Exact numbers depend on the run but expect the decision tree (`max_depth=3`) around ~93-95% accuracy; top logistic regression features are typically things like `worst concave points`, `worst radius`, `worst perimeter` (largest |coefficient|).

2. `max_depth=1` (a "decision stump") will have noticeably lower accuracy since it can only make one split — but it's trivially human-readable in a single sentence. `max_depth=6` usually improves accuracy modestly but the tree has up to 64 leaves, which is difficult for a person to read at a glance. Interpretability degrades well before depth 6.

3. A positive coefficient means increasing that feature pushes the prediction *toward* class 1 (Benign) — i.e. it's associated with less severe cancer indicators. A negative coefficient means increasing that feature pushes toward class 0 (Malignant) — larger/more irregular tumor measurements typically have negative coefficients.

4. They often disagree, because: (a) decision trees pick features that best split remaining impurity greedily and can miss features that are individually weak but linearly predictive; (b) logistic regression coefficients depend on feature scale and assume a linear, additive relationship, while tree importance captures any nonlinear split usefulness. Correlated features can also "share" importance differently between the two model types.

5. Yes — coefficient magnitudes typically change substantially after standardization, because raw coefficients are also a function of each feature's original scale (a feature measured in the thousands will naturally get a tiny coefficient even if it's important). After scaling, coefficients become directly comparable in units of "per standard deviation," which is why scaling matters for coefficient-based interpretation but is irrelevant to tree splits (which are scale-invariant thresholds).
