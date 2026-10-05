"""
Exercise 02: Linear Discriminant Analysis (LDA)
Dataset: Sample 2 classes (setosa and versicolor) from the Iris dataset (N = 10, d = 4).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# GIVEN DATASET
# ==========================================
data = {
    'sepal length (cm)': [4.3, 5.1, 4.8, 4.8, 5.1, 6.1, 5.5, 5.5, 5.7, 5.8],
    'sepal width (cm)':  [3.0, 3.4, 3.1, 3.0, 3.5, 2.9, 2.5, 2.4, 3.0, 2.7],
    'petal length (cm)': [1.1, 1.5, 1.6, 1.4, 1.4, 4.7, 4.0, 3.8, 4.2, 4.1],
    'petal width (cm)':  [0.1, 0.2, 0.2, 0.3, 0.3, 1.4, 1.3, 1.1, 1.2, 1.0],
    'species': ['setosa'] * 5 + ['versicolor'] * 5
}

df = pd.DataFrame(data)
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Separate data into Class 1 (setosa) and Class 2 (versicolor)
# Represented as matrices where each row is a sample, shape (5, 4)
X_setosa = df[df['species'] == 'setosa'][feature_names].values
X_versicolor = df[df['species'] == 'versicolor'][feature_names].values
X_all = df[feature_names].values # shape (10, 4)

N1 = len(X_setosa)     # 5
N2 = len(X_versicolor) # 5
N = len(df)            # 10
d = len(feature_names) # 4

print("=" * 75)
print("EXERCISE 02: LINEAR DISCRIMINANT ANALYSIS (LDA) STEP-BY-STEP")
print("=" * 75)
print("\nDataset Table:")
print(df)

# ==============================================================================
# STEP 1: CLASS MEAN VECTORS & WITHIN-CLASS SCATTER MATRIX (S_W)
# ==============================================================================
print("\n" + "=" * 75)
print("[STEP 1] CLASS MEAN VECTORS & WITHIN-CLASS SCATTER MATRIX (S_W)")
print("=" * 75)

# Mean vector of each class (shape: (4, 1))
mu1 = np.mean(X_setosa, axis=0, keepdims=True).T      # Mean vector for setosa
mu2 = np.mean(X_versicolor, axis=0, keepdims=True).T  # Mean vector for versicolor

print("\nMean vector mu1 (Setosa, 4x1):")
print(mu1)
print("\nMean vector mu2 (Versicolor, 4x1):")
print(mu2)

# Within-class scatter matrix for Class 1 (S1)
# S1 = sum_{x in C1} (x - mu1)(x - mu1)^T
S1 = np.zeros((d, d))
for row in X_setosa:
    diff = row.reshape(-1, 1) - mu1
    S1 += diff @ diff.T

# Within-class scatter matrix for Class 2 (S2)
# S2 = sum_{x in C2} (x - mu2)(x - mu2)^T
S2 = np.zeros((d, d))
for row in X_versicolor:
    diff = row.reshape(-1, 1) - mu2
    S2 += diff @ diff.T

# Overall within-class scatter matrix S_W = S1 + S2
S_W = S1 + S2

print("\nWithin-class Scatter Matrix S1 (Setosa, 4x4):")
print(np.round(S1, 4))
print("\nWithin-class Scatter Matrix S2 (Versicolor, 4x4):")
print(np.round(S2, 4))
print("\nOverall Within-class Scatter Matrix S_W = S1 + S2 (4x4):")
print(np.round(S_W, 4))

# ==============================================================================
# STEP 2: GLOBAL MEAN VECTOR & BETWEEN-CLASS SCATTER MATRIX (S_B)
# ==============================================================================
print("\n" + "=" * 75)
print("[STEP 2] GLOBAL MEAN VECTOR & BETWEEN-CLASS SCATTER MATRIX (S_B)")
print("=" * 75)

# Global mean vector mu (shape: (4, 1))
mu = np.mean(X_all, axis=0, keepdims=True).T

# Between-class scatter matrix:
# S_B = N1 * (mu1 - mu)(mu1 - mu)^T + N2 * (mu2 - mu)(mu2 - mu)^T
diff1 = mu1 - mu
diff2 = mu2 - mu
S_B = N1 * (diff1 @ diff1.T) + N2 * (diff2 @ diff2.T)

print("\nGlobal Mean Vector mu (4x1):")
print(mu)
print("\nBetween-class Scatter Matrix S_B (4x4):")
print(np.round(S_B, 4))

# ==============================================================================
# STEP 3: SOLVE GENERALIZED EIGENVALUE PROBLEM FOR S_W^{-1} S_B
# ==============================================================================
print("\n" + "=" * 75)
print("[STEP 3] SOLVE GENERALIZED EIGENVALUE PROBLEM: (S_W^-1 S_B) w = lambda w")
print("=" * 75)

# Invert S_W and multiply with S_B
S_W_inv = np.linalg.pinv(S_W) if np.linalg.cond(S_W) > 1e12 else np.linalg.inv(S_W)
M = S_W_inv @ S_B

# Compute eigenvalues and eigenvectors of S_W^{-1} S_B
eigenvalues, eigenvectors = np.linalg.eig(M)

# Eigenvalues might have very tiny imaginary parts due to float precision, convert to real
eigenvalues = np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)

print("\nComputed Eigenvalues (lambda_i) and Eigenvectors (w_i):")
for i in range(d):
    print(f"  Eigenvalue {i+1} (lambda_{i+1}): {eigenvalues[i]:.6f}")
    print(f"  Eigenvector {i+1} (w_{i+1}): {np.round(eigenvectors[:, i], 4)}")

# ==============================================================================
# STEP 4: SELECT LEADING EIGENVECTOR AND COMPUTE 1D PROJECTION
# ==============================================================================
print("\n" + "=" * 75)
print("[STEP 4] SELECT LEADING EIGENVECTOR W AND COMPUTE 1D COORDINATES y = W^T x")
print("=" * 75)

# Sort eigenvalues in descending order
idx_sort = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[idx_sort]
eigenvectors_sorted = eigenvectors[:, idx_sort]

# Leading eigenvector (Projection vector W, normalized)
W = eigenvectors_sorted[:, 0:1] # shape (4, 1)
# Normalize W so that ||W|| = 1
W = W / np.linalg.norm(W)

# Choose positive direction for setosa / versicolor separation clarity
if (mu2.T @ W)[0, 0] < (mu1.T @ W)[0, 0]:
    W = -W

print("\nProjection Vector W (Leading Eigenvector, 4x1):")
print(np.round(W, 6))

# Compute 1D transformed coordinates: y_i = W^T x_i (shape: (10, 1))
y_all = (X_all @ W).flatten()

df_result = df.copy()
df_result['1D Transformed (y)'] = np.round(y_all, 4)

print("\n1D Transformed Coordinates y for all 10 samples:")
print(df_result[['species', '1D Transformed (y)']])

# ==============================================================================
# STEP 5: PLOT 1D PROJECTION RESULTS
# ==============================================================================
print("\n" + "=" * 75)
print("[STEP 5] PLOT 1D PROJECTION RESULTS")
print("=" * 75)

plt.figure(figsize=(12, 5.5))

y_setosa = y_all[:5]
y_versicolor = y_all[5:]

# 1D Dot Plot with jitter for clarity
jitter_setosa = np.zeros_like(y_setosa)
jitter_versicolor = np.zeros_like(y_versicolor)

plt.axhline(0, color='gray', linestyle='--', alpha=0.7, zorder=1)

plt.scatter(y_setosa, jitter_setosa, color='crimson', s=120, alpha=0.85, 
            edgecolors='darkred', linewidth=1.5, label='Setosa (Class 1)', zorder=4)

plt.scatter(y_versicolor, jitter_versicolor, color='royalblue', s=120, alpha=0.85, 
            edgecolors='darkblue', linewidth=1.5, label='Versicolor (Class 2)', zorder=4)

# Plot class means on 1D axis
mu1_proj = (mu1.T @ W)[0, 0]
mu2_proj = (mu2.T @ W)[0, 0]
plt.scatter([mu1_proj], [0], color='darkred', marker='X', s=180, label=r'Setosa Mean ($\mu_1^T W$)', zorder=5)
plt.scatter([mu2_proj], [0], color='midnightblue', marker='X', s=180, label=r'Versicolor Mean ($\mu_2^T W$)', zorder=5)

# Add sample index labels
for idx, (val, spec) in enumerate(zip(y_all, df['species'])):
    offset = 0.04 if idx % 2 == 0 else -0.04
    color = 'darkred' if spec == 'setosa' else 'navy'
    plt.annotate(f'Sample {idx}', (val, 0), textcoords="offset points", 
                 xytext=(0, 15 if offset > 0 else -20), ha='center', fontsize=9,
                 fontweight='bold', color=color,
                 arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

# Decision boundary midpoint
midpoint = (mu1_proj + mu2_proj) / 2
plt.axvline(midpoint, color='forestgreen', linestyle='-.', linewidth=1.8, 
            label=f'Midpoint Threshold ({midpoint:.2f})')

plt.title('Exercise 02: LDA 1D Projection (Setosa vs. Versicolor)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('1D Discriminant Coordinate $y = W^T x$', fontsize=12)
plt.yticks([]) # 1D axis has no y-dimension
plt.ylim(-0.1, 0.15)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', framealpha=0.95, fontsize=10)
plt.tight_layout()

plt.savefig('exercise_02_lda_visualization.png', dpi=300, bbox_inches='tight')
print("Figure saved to 'exercise_02_lda_visualization.png'.")
