"""
Exercise 01: Dimensionality Reduction with PCA and PCA+SVD
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# GIVEN DATASET
# ==========================================
# 2D dataset with N = 4 samples represented by matrix X in R^{2x4}
# Row 1: feature 1 (x1), Row 2: feature 2 (x2)
X = np.array([
    [2.0, 3.0, 5.0, 6.0],
    [1.0, 5.0, 4.0, 6.0]
])

N = X.shape[1] # Number of samples = 4
d = X.shape[0] # Dimension = 2

print("=" * 70)
print("EXERCISE 01: DIMENSIONALITY REDUCTION WITH PCA AND PCA+SVD")
print("=" * 70)
print("\nGiven Dataset X (shape 2x4):")
print(X)

# ==============================================================================
# Q1: STANDARD PCA VIA COVARIANCE MATRIX (8 STEPS)
# ==============================================================================
print("\n" + "=" * 70)
print("Q1: STANDARD PCA VIA COVARIANCE MATRIX (STEP-BY-STEP)")
print("=" * 70)

# Step 1: Calculate Mean Vector mu
mu_pca = np.mean(X, axis=1, keepdims=True)
print("\n[Step 1] Mean vector mu (2x1):")
print(mu_pca)

# Step 2: Mean Centering (Zero-mean data X_tilde)
X_tilde_pca = X - mu_pca
print("\n[Step 2] Centered Data X_tilde = X - mu (2x4):")
print(X_tilde_pca)

# Step 3: Compute Covariance Matrix C
# Using sample covariance: C = (1 / (N - 1)) * (X_tilde @ X_tilde.T)
C = (1 / (N - 1)) * (X_tilde_pca @ X_tilde_pca.T)
print("\n[Step 3] Covariance Matrix C (2x2):")
print(C)

# Step 4: Compute Eigenvalues and Eigenvectors of C
eigenvalues_pca, eigenvectors_pca = np.linalg.eigh(C)
print("\n[Step 4] Eigenvalues and Eigenvectors of Covariance Matrix:")
for i in range(len(eigenvalues_pca)):
    print(f"  Eigenvalue {i+1}: {eigenvalues_pca[i]:.4f}, Eigenvector {i+1}: {eigenvectors_pca[:, i]}")

# Step 5: Sort Eigenvalues in Descending Order and Select Leading Eigenvector (PC1)
idx_sorted = np.argsort(eigenvalues_pca)[::-1]
eigenvalues_pca = eigenvalues_pca[idx_sorted]
eigenvectors_pca = eigenvectors_pca[:, idx_sorted]

pc1_vector = eigenvectors_pca[:, 0:1] # Shape (2, 1)
# Ensure consistent orientation (e.g. positive first element if needed)
if pc1_vector[0, 0] < 0:
    pc1_vector = -pc1_vector

print("\n[Step 5] Sorted Eigenvalues and Selected PC1:")
print(f"  Sorted Eigenvalues: {eigenvalues_pca}")
print(f"  Leading Eigenvector (PC1, shape 2x1):\n{pc1_vector}")

# Step 6: Project Centered Data onto PC1 (1D Transformed Coordinates Z)
Z_pca = pc1_vector.T @ X_tilde_pca # Shape (1, 4)
print("\n[Step 6] 1D Projected Coordinates Z = v1.T @ X_tilde (1x4):")
print(Z_pca)

# Step 7: Reconstruct Data in Original 2D Space
X_reconstructed_pca = pc1_vector @ Z_pca + mu_pca # Shape (2, 4)
print("\n[Step 7] Reconstructed Data X_reconstructed = v1 @ Z + mu (2x4):")
print(X_reconstructed_pca)

# Step 8: Evaluation (Explained Variance & Reconstruction Error)
var_explained_pca = eigenvalues_pca[0] / np.sum(eigenvalues_pca) * 100
recon_error_pca = np.mean(np.sum((X - X_reconstructed_pca)**2, axis=0))
print("\n[Step 8] Numerical Evaluation:")
print(f"  Explained Variance by PC1: {var_explained_pca:.2f}%")
print(f"  Mean Squared Reconstruction Error (MSE): {recon_error_pca:.4f}")


# ==============================================================================
# Q2: PCA USING SVD (8 STEPS)
# ==============================================================================
print("\n" + "=" * 70)
print("Q2: PCA VIA SINGULAR VALUE DECOMPOSITION (SVD) (STEP-BY-STEP)")
print("=" * 70)

# Step 1: Calculate Mean Vector mu
mu_svd = np.mean(X, axis=1, keepdims=True)
print("\n[Step 1] Mean vector mu (2x1):")
print(mu_svd)

# Step 2: Mean Centering (Centered data matrix X_tilde)
X_tilde_svd = X - mu_svd
print("\n[Step 2] Centered Data Matrix X_tilde (2x4):")
print(X_tilde_svd)

# Step 3: Formulate SVD formulation
print("\n[Step 3] Formulate SVD: X_tilde = U * Sigma * V^T")
print("  where U is left singular vectors (columns are PCs), Sigma has singular values, V is right singular vectors.")

# Step 4: Perform SVD decomposition
U, S, Vt = np.linalg.svd(X_tilde_svd, full_matrices=False)
print("\n[Step 4] SVD Decomposition Results:")
print("  Matrix U (Left singular vectors, 2x2):")
print(U)
print(f"  Singular values S: {S}")
print("  Matrix V^T (Right singular vectors, 2x4):")
print(Vt)

# Step 5: Identify Principal Components and Equivalent Eigenvalues
# PC1 is the first column of U
u1 = U[:, 0:1] # Shape (2, 1)
# Consistent sign orientation with Q1
if np.dot(u1.flatten(), pc1_vector.flatten()) < 0:
    u1 = -u1
    Vt[0, :] = -Vt[0, :]

lambda_svd = (S**2) / (N - 1)
print("\n[Step 5] Principal Component PC1 and Associated Variance:")
print(f"  PC1 Vector from U[:, 0] (shape 2x1):\n{u1}")
print(f"  Equivalent Eigenvalues from S^2 / (N-1): {lambda_svd}")

# Step 6: Compute 1D Transformed Coordinates
# Z_svd = u1^T @ X_tilde or alternatively Z_svd = S[0] * Vt[0:1, :]
Z_svd = u1.T @ X_tilde_svd
print("\n[Step 6] 1D Projected Coordinates Z_svd = u1^T @ X_tilde (1x4):")
print(Z_svd)

# Step 7: Reconstruct Data
X_reconstructed_svd = u1 @ Z_svd + mu_svd
print("\n[Step 7] Reconstructed Data X_reconstructed_svd (2x4):")
print(X_reconstructed_svd)

# Step 8: Compare and Verify Equivalence with Q1
diff_recon = np.max(np.abs(X_reconstructed_pca - X_reconstructed_svd))
diff_coords = np.max(np.abs(Z_pca - Z_svd))
print("\n[Step 8] Verification of Equivalence with Q1 (Standard PCA):")
print(f"  Max absolute difference in Reconstructed Points: {diff_recon:.2e}")
print(f"  Max absolute difference in 1D Coordinates: {diff_coords:.2e}")
print("  => Conclusion: SVD produces mathematically identical results to Covariance PCA.")


# ==============================================================================
# Q3: VISUALIZATION
# ==============================================================================
print("\n" + "=" * 70)
print("Q3: VISUALIZATION (GENERATING PLOT)")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

def plot_pca_subplot(ax, X_orig, mu, X_recon, pc_vec, title):
    # Plot original points
    ax.scatter(X_orig[0, :], X_orig[1, :], color='blue', s=90, zorder=5, label='Original Points $X$')
    
    # Plot center mu
    ax.scatter(mu[0], mu[1], color='black', marker='X', s=150, zorder=6, label=r'Center $\mu$ (4, 4)')
    
    # Plot reconstructed points
    ax.scatter(X_recon[0, :], X_recon[1, :], color='green', s=90, marker='o', zorder=5, label='Reconstructed $X_{recon}$')
    
    # Draw red dashed PC1 axis passing through center mu
    t_vals = np.linspace(-4, 4, 100)
    pc1_line = mu + pc_vec * t_vals
    ax.plot(pc1_line[0, :], pc1_line[1, :], 'r--', linewidth=2, label='PC1 Axis (Red dashed)')
    
    # Draw dotted gray orthogonal projection lines connecting each X_i to its reconstructed point
    for i in range(X_orig.shape[1]):
        ax.plot([X_orig[0, i], X_recon[0, i]], 
                [X_orig[1, i], X_recon[1, i]], 
                color='gray', linestyle=':', linewidth=1.8,
                label='Orthogonal Projection' if i == 0 else "")
        # Add text labels for points
        ax.annotate(f'$X_{i+1}$({X_orig[0, i]:.0f}, {X_orig[1, i]:.0f})', 
                    (X_orig[0, i], X_orig[1, i]), 
                    textcoords="offset points", xytext=(5, 5), fontsize=10, color='darkblue')
        ax.annotate(f'$X\'_{i+1}$({X_recon[0, i]:.2f}, {X_recon[1, i]:.2f})', 
                    (X_recon[0, i], X_recon[1, i]), 
                    textcoords="offset points", xytext=(-25, -15), fontsize=9, color='darkgreen')
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axis('equal')
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.legend(loc='upper left', framealpha=0.9)

# Subplot 1: Standard PCA
plot_pca_subplot(axes[0], X, mu_pca, X_reconstructed_pca, pc1_vector, 'Subplot 1: Standard PCA via Covariance Matrix')

# Subplot 2: PCA + SVD
plot_pca_subplot(axes[1], X, mu_svd, X_reconstructed_svd, u1, 'Subplot 2: PCA using SVD')

plt.tight_layout()
plt.savefig('exercise_01_pca_visualization.png', dpi=300, bbox_inches='tight')
print("Figure saved to 'exercise_01_pca_visualization.png'.")
