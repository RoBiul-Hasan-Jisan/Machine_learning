"""
Identifiability -- Gaussian latents defeat classical PCA-based recovery
(any rotation of an isotropic-in-the-wrong-basis Gaussian is equally
consistent with the data), but latents whose variances differ ACROSS
environments (the auxiliary variable U = environment) can be recovered, up to
scale and permutation, by joint diagonalization of the environments'
covariance matrices -- the linear analog of the iVAE-style identifiability
result in theory.md section 2.3.

Recovery is checked the meaningful way: project the observed X onto a
candidate "unmixing" direction and see whether that projection correlates
with one of the TRUE, generative sources Z -- exactly what "recovering a
latent factor" means.
"""
import numpy as np
from scipy.linalg import eigh

rng = np.random.default_rng(0)
A = np.array([[1.0, 0.5], [0.3, 1.2]])                 # true (unknown) linear mixing
env_variances = [(4.0, 1.0), (1.0, 4.0)]               # variances differ ACROSS environments
n = 200_000

def sample_env(var1, var2, n):
    Z = rng.normal(0, 1, (n, 2)) * np.sqrt([var1, var2])   # Gaussian, independent components
    return Z, Z @ A.T

Z1, X1 = sample_env(*env_variances[0], n)
Z2, X2 = sample_env(*env_variances[1], n)
Z_pooled, X_pooled = np.vstack([Z1, Z2]), np.vstack([X1, X2])

def best_recovery(direction, Z):
    """Correlation of the projection X@direction with whichever true source matches best."""
    proj = X_pooled @ direction
    return max(abs(np.corrcoef(proj, Z[:, 0])[0, 1]), abs(np.corrcoef(proj, Z[:, 1])[0, 1]))

print("=== Classical PCA directions on pooled data (single covariance matrix) ===")
cov = np.cov((X_pooled - X_pooled.mean(0)).T)
_, eigvec = np.linalg.eigh(cov)
for i in range(2):
    print(f"  PCA direction {i}: correlation with best-matching true source = {best_recovery(eigvec[:, i], Z_pooled):.3f}")

print("\n=== Joint diagonalization of TWO environments' covariances (uses U = environment) ===")
C1, C2 = np.cov(X1.T), np.cov(X2.T)
_, eigvec2 = eigh(C1, C2)                              # generalized eigenproblem C1 v = lambda C2 v
for i in range(2):
    print(f"  recovered direction {i}: correlation with best-matching true source = {best_recovery(eigvec2[:, i], Z_pooled):.3f}")

print("\nPCA on the pooled, single-covariance data cannot isolate either true source (moderate,")
print("not clean, correlations -- the rotation is genuinely ambiguous for Gaussian latents).")
print("Joint diagonalization using the auxiliary environment label breaks that ambiguity and")
print("recovers each true source almost perfectly, up to sign and scale.")
