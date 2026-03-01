from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (λ=1)': Ridge(alpha=1),
    'Ridge (λ=100)': Ridge(alpha=100),
    'Lasso (λ=0.01)': Lasso(alpha=0.01),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    results[name] = {'Train R²': train_r2, 'Test R²': test_r2, 'RMSE': test_rmse}
    print(f"{name:25} | Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f} | RMSE: {test_rmse:.3f}")

# Lasso feature selection
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
zero_features = np.sum(lasso.coef_ == 0)
print(f"\nLasso zeroed out {zero_features} of {len(feature_names)} features")