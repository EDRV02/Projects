import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 3*x - 5  # Change this to any function

x_values = np.linspace(-10, 10, 200)
y_values = f(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, color='steelblue', linewidth=2)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Function Plot: f(x) = x² + 3x − 5', fontsize=14)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.show()

# Find the approximate minimum
min_x = x_values[np.argmin(y_values)]
print(f"Approximate minimum at x = {min_x:.2f}, f(x) = {f(min_x):.2f}")