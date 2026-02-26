import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_assets = 10
n_days = 252

# Simulate daily returns for 10 assets
returns = np.random.randn(n_days, n_assets) * 0.01 + 0.0003

# Covariance matrix
cov_matrix = np.cov(returns.T)
print("Covariance Matrix Shape:", cov_matrix.shape)
print("\nAnnualised Volatilities:")
for i in range(n_assets):
    vol = np.sqrt(cov_matrix[i,i] * 252)
    print(f"  Asset {i+1}: {vol:.2%}")

# Portfolio weights (equal weight)
weights = np.ones(n_assets) / n_assets

# Portfolio variance using matrix multiplication: wᵀ Σ w
port_variance = weights.T @ cov_matrix @ weights
port_vol = np.sqrt(port_variance * 252)
print(f"\nEqual-weight Portfolio Annualised Volatility: {port_vol:.2%}")

# PCA on asset returns
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

pca = PCA()
pca.fit(returns_scaled)

explained_var = np.cumsum(pca.explained_variance_ratio_)
print(f"\nVariance explained by first 3 components: {explained_var[2]:.1%}")
print(f"Variance explained by first 5 components: {explained_var[4]:.1%}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_assets+1), explained_var, 'bo-', linewidth=2)
plt.axhline(0.90, color='red', linestyle='--', label='90% threshold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: How Many Components Do We Need?')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()