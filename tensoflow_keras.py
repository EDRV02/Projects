import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# BUILDING A NEURAL NETWORK — STEP BY STEP
# ============================================================

# Generate data
np.random.seed(42)
X = np.random.randn(2000, 15)
y = (X[:, 0]*2 + X[:, 3] - X[:, 7] + np.random.randn(2000)*0.5 > 0).astype(int)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = keras.Sequential([
    keras.layers.Input(shape=(15,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

# Callbacks — prevent overfitting, save best model
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=5, min_lr=1e-7)
]

# Train
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.3f}")
print(f"Test AUC: {test_auc:.3f}")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='Training')
axes[0].plot(history.history['val_loss'], label='Validation')
axes[0].set_title('Loss Curves')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Training')
axes[1].plot(history.history['val_accuracy'], label='Validation')
axes[1].set_title('Accuracy Curves')
axes[1].legend()
plt.show()