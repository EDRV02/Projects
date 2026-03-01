from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, mean_squared_error, r2_score)
from sklearn.datasets import fetch_california_housing, load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# COMPLETE ML PIPELINE — CLASSIFICATION
# ============================================================
data = load_breast_cancer()
X, y = data.data, data.target

# Always scale features for algorithms sensitive to magnitude
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CRITICAL: fit_transform on train, transform ONLY on test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=data.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Cross-validation — more reliable than single train/test split
cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:]  # Top 10
plt.barh(range(10), importances[indices])
plt.yticks(range(10), [data.feature_names[i] for i in indices])
plt.title('Top 10 Feature Importances')
plt.show()