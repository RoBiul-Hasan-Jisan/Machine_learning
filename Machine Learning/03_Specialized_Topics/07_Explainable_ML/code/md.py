import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import shap
import lime
import lime.lime_tabular

# Create the plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

pd.set_option("display.width", 120)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
print(f"Model test accuracy: {model.score(X_test, y_test):.4f}\n")


print("1) FEATURE IMPORTANCE (built-in, tree-based)")


importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 features by built-in (impurity-based) importance:")
print(importances.head(10))

plt.figure(figsize=(8, 5))
importances.head(10).plot.barh()
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (built-in)")
plt.tight_layout()
plt.savefig("plots/25_feature_importance.png", bbox_inches="tight")
plt.close()


print("2) PERMUTATION IMPORTANCE (model-agnostic, computed on held-out data)")


perm_result = permutation_importance(
    model, X_test, y_test, n_repeats=15, random_state=42, scoring="roc_auc"
)
perm_importances = pd.Series(perm_result.importances_mean, index=X.columns).sort_values(ascending=False)
perm_std = pd.Series(perm_result.importances_std, index=X.columns)

print("Top 10 features by permutation importance (drop in ROC-AUC when shuffled):")
for feat in perm_importances.head(10).index:
    print(f"  {feat:28s} {perm_importances[feat]:.4f} +/- {perm_std[feat]:.4f}")

print("\nComparison — built-in vs permutation top 5:")
print("Built-in top 5     :", list(importances.head(5).index))
print("Permutation top 5  :", list(perm_importances.head(5).index))


print("3) SHAP — SHapley Additive exPlanations")


explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)  # new SHAP API returns an Explanation object

# For binary classification, shap_values has shape (n_samples, n_features, n_classes)
# We'll focus on the positive class (index 1)
if len(shap_values.shape) == 3:
    shap_values_pos = shap_values[:, :, 1]
else:
    shap_values_pos = shap_values

print("SHAP base value (average model output):", round(float(np.array(shap_values_pos.base_values).mean()), 4))
print("\nSHAP values for first test instance (top 5 by |impact|):")
instance_shap = pd.Series(shap_values_pos.values[0], index=X.columns).sort_values(key=abs, ascending=False)
print(instance_shap.head(5))

# Global feature importance via mean |SHAP value|
mean_abs_shap = pd.Series(np.abs(shap_values_pos.values).mean(axis=0), index=X.columns).sort_values(ascending=False)
print("\nGlobal feature importance via mean |SHAP value| (top 10):")
print(mean_abs_shap.head(10))

# Save summary (beeswarm) plot
plt.figure()
shap.summary_plot(shap_values_pos, X_test, show=False)
plt.tight_layout()
plt.savefig("plots/25_shap_summary.png", bbox_inches="tight")
plt.close()
print("\nSaved SHAP summary plot to plots/25_shap_summary.png")

# Save a waterfall plot for a single instance
plt.figure()
shap.plots.waterfall(shap_values_pos[0], show=False)
plt.tight_layout()
plt.savefig("plots/25_shap_waterfall.png", bbox_inches="tight")
plt.close()
print("Saved SHAP waterfall plot (instance 0) to plots/25_shap_waterfall.png")


print("4) LIME — Local Interpretable Model-agnostic Explanations")


lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=["malignant", "benign"],
    mode="classification",
    random_state=42
)

instance_idx = 0
predict_fn = lambda arr: model.predict_proba(pd.DataFrame(arr, columns=X.columns))
exp = lime_explainer.explain_instance(
    X_test.iloc[instance_idx].values,
    predict_fn,
    num_features=8
)
print(f"LIME explanation for test instance {instance_idx} "
      f"(true label: {data.target_names[y_test[instance_idx]]}):")
for feature, weight in exp.as_list():
    print(f"  {feature:40s} {weight:+.4f}")

exp.save_to_file("plots/25_lime_explanation.html")
print("\nSaved interactive LIME explanation to plots/25_lime_explanation.html")

print("\n--- SHAP vs LIME on the same instance ---")
print("SHAP top contributors:")
print(instance_shap.head(5))
print("\nLIME top contributors (feature-condition, weight):")
for feature, weight in exp.as_list()[:5]:
    print(f"  {feature:40s} {weight:+.4f}")


print("5) PARTIAL DEPENDENCE (PDP)")


top_feature = importances.index[0]
print(f"Plotting PDP for the most important feature: '{top_feature}'")

fig, ax = plt.subplots(figsize=(7, 5))
PartialDependenceDisplay.from_estimator(model, X_train, features=[top_feature], ax=ax)
plt.title(f"Partial Dependence — {top_feature}")
plt.tight_layout()
plt.savefig("plots/25_pdp.png", bbox_inches="tight")
plt.close()
print("Saved PDP plot to plots/25_pdp.png")

# 2-feature interaction PDP
second_feature = importances.index[1]
fig, ax = plt.subplots(figsize=(7, 5))
PartialDependenceDisplay.from_estimator(model, X_train, features=[(top_feature, second_feature)], ax=ax)
plt.title(f"2D Partial Dependence — {top_feature} x {second_feature}")
plt.tight_layout()
plt.savefig("plots/25_pdp_2d.png", bbox_inches="tight")
plt.close()
print("Saved 2D interaction PDP plot to plots/25_pdp_2d.png")


print("6) INDIVIDUAL CONDITIONAL EXPECTATION (ICE)")


fig, ax = plt.subplots(figsize=(7, 5))
PartialDependenceDisplay.from_estimator(
    model, X_train, features=[top_feature], kind="both", ax=ax  # "both" = ICE lines + PDP average overlay
)
plt.title(f"ICE + PDP overlay — {top_feature}")
plt.tight_layout()
plt.savefig("plots/25_ice.png", bbox_inches="tight")
plt.close()
print("Saved ICE + PDP overlay plot to plots/25_ice.png")
print("(Individual thin lines = ICE per-instance curves; thick line = PDP average.")
print(" Fanning/crossing lines would reveal heterogeneous effects PDP alone hides.)")

print("\nDone.")