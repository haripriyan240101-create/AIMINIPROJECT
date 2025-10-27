# AIMINIPROJECT

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Initial model training
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
model = SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(X, y)

# Simulate wrong prediction and correction
test_X = np.array([[6]])
pred = model.predict(test_X)
print("Before correction:", pred)

# Detect error and locally adjust weights
error = 12 - pred  # true value = 12
model.partial_fit(test_X, [12])  # local update (no retraining)
new_pred = model.predict(test_X)
print("After correction:", new_pred)
