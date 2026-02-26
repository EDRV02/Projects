import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART 1: Gradient descent on a simple function
# ============================================================
def loss(w):
    return (w - 3)**2 + 2  # Minimum at w=3, loss=2

def gradient(w):
    return 2 * (w - 3)

w = 10.0
learning_rate = 0.1
history_w = [w]
history_loss = [loss(w)]

for i in range(100):
    grad = gradient(w)
    w = w - learning_rate * grad
    history_w.append(w)
    history_loss.append(loss(w))

print(f"Final w: {w:.6f} (true minimum: 3.0)")
print(f"Final loss: {loss(w):.6f} (true minimum: 2.0)")

# ============================================================
# PART 2: Linear regression using gradient descent
# ============================================================
np.random.seed(42)
X = np.random.randn(100)
y = 2.5 * X + 1.2 + np.random.randn(100) * 0.5  # True: slope=2.5, intercept=1.2

m, b = 0.0, 0.0
lr = 0.01
losses = []

for epoch in range(1000):
    y_pred = m * X + b
    error = y_pred - y
    loss_val = np.mean(error**2)
    losses.append(loss_val)
    dm = (2/len(X)) * np.dot(error, X)
    db = (2/len(X)) * np.sum(error)
    m -= lr * dm
    b -= lr * db

print(f"\nLearned slope: {m:.3f} (true: 2.5)")
print(f"Learned intercept: {b:.3f} (true: 1.2)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(history_loss, color='steelblue')
axes[0].set_title('Loss Converging (Simple Function)')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss')

axes[1].scatter(X, y, alpha=0.5, label='Data')
axes[1].plot(X, m*X + b, color='red', linewidth=2, label=f'Fit: y={m:.2f}x+{b:.2f}')
axes[1].set_title('Linear Regression via Gradient Descent')
axes[1].legend()

axes[2].plot(losses, color='green')
axes[2].set_title('Training Loss Over Epochs')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('MSE Loss')

plt.tight_layout()
plt.show()