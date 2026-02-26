import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
# Simulate 5 years of daily returns
returns = np.random.normal(loc=0.0005, scale=0.015, size=1260)

mean_r = np.mean(returns)
std_r = np.std(returns)
var_95 = np.percentile(returns, 5)   # 95% VaR
var_99 = np.percentile(returns, 1)   # 99% VaR
sharpe = (mean_r * 252) / (std_r * np.sqrt(252))  # Annualised

print("=" * 40)
print("RETURN DISTRIBUTION ANALYSIS")
print("=" * 40)
print(f"Mean Daily Return:     {mean_r:.4%}")
print(f"Daily Volatility:      {std_r:.4%}")
print(f"Annualised Return:     {mean_r*252:.2%}")
print(f"Annualised Volatility: {std_r*np.sqrt(252):.2%}")
print(f"Sharpe Ratio:          {sharpe:.2f}")
print(f"95% Daily VaR:         {var_95:.4%}")
print(f"99% Daily VaR:         {var_99:.4%}")
print(f"Skewness:              {stats.skew(returns):.3f}")
print(f"Kurtosis:              {stats.kurtosis(returns):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(returns, bins=60, density=True, alpha=0.7, color='steelblue', label='Returns')
x = np.linspace(returns.min(), returns.max(), 200)
axes[0].plot(x, stats.norm.pdf(x, mean_r, std_r), 'r-', lw=2, label='Normal fit')
axes[0].axvline(var_95, color='orange', linestyle='--', label=f'95% VaR: {var_95:.3%}')
axes[0].axvline(var_99, color='red', linestyle='--', label=f'99% VaR: {var_99:.3%}')
axes[0].set_title('Daily Return Distribution')
axes[0].set_xlabel('Daily Return')
axes[0].legend()

# Cumulative returns
cumulative = (1 + returns).cumprod()
axes[1].plot(cumulative, color='green', linewidth=1.5)
axes[1].set_title('Cumulative Portfolio Growth (£1 invested)')
axes[1].set_xlabel('Trading Days')
axes[1].set_ylabel('Portfolio Value')
axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('Stock_Return_Distribution_Analyser.png', dpi=150)
try:
    plt.show()
except KeyboardInterrupt:
    pass
print('Saved figure to Stock_Return_Distribution_Analyser.png')