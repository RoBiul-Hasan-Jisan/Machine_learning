# Exercises — 07 Causal & Concept-Based Explanations

1. **Correlation vs causation.** Using the ice cream/drowning example from the theory, construct your own analogous example from a domain you know (e.g. tech, sports, retail) where two variables correlate but neither causes the other.
2. **Do-calculus intuition.** Explain in plain English the difference between `P(Y|X=x)` and `P(Y|do(X=x))`. Why does a black-box classifier trained on observational data only ever give you the former?
3. **TCAV thought experiment.** The theory describes a pneumonia classifier that relies on "medical mask" pixels. Design a TCAV-style test (in words, not code) you would run to check whether an image classifier for "wolf vs. husky" is actually keying on snow in the background rather than the animal.
4. **Mutual information.** In your own words, explain what it means for `I(feature_i; target) > 0`, and why a feature could have `I > 0` on its own but drop to near 0 once you condition on other features (i.e. `Relevance` from the formula goes to ~0).
5. **SCM sketch.** Draw (describe in words or as a simple diagram) a Structural Causal Model with 3 nodes for a scenario of your choosing (e.g. Exercise → Fitness → Injury Risk), and identify which arrows represent `f_X` and `f_Y` from the SCM equations in the theory.
