

from google.colab import files
uploaded = files.upload()


import pandas as pd
import numpy as np

data = pd.read_csv('xy_data.csv')
x_actual = data['x'].values
y_actual = data['y'].values
t_values = np.arange(6, 6 + len(data))


def parametric_curve(t, theta, M, X):
    x_model = t * np.cos(theta) - np.exp(M * np.abs(t)) * np.sin(0.3*t) * np.sin(theta) + X
    y_model = 42 + t * np.sin(theta) + np.exp(M * np.abs(t)) * np.sin(0.3*t) * np.cos(theta)
    return x_model, y_model


def loss(params):
    theta, M, X = params
    x_pred, y_pred = parametric_curve(t_values, theta, M, X)
    return np.sum(np.abs(x_actual - x_pred) + np.abs(y_actual - y_pred))


from scipy.optimize import minimize

initial_guess = [0.5, 0.0, 50]
bounds = [
    (np.deg2rad(0), np.deg2rad(50)), # theta in radians
    (-0.05, 0.05),                   # M
    (0, 100)                         # X
]

result = minimize(loss, initial_guess, bounds=bounds, method='L-BFGS-B')
theta_opt, M_opt, X_opt = result.x

print("Optimal θ (radians):", theta_opt)
print("Optimal M:", M_opt)
print("Optimal X:", X_opt)
print("Final L1 score:", loss([theta_opt, M_opt, X_opt]))



output:
\left(
  t\cos(0.826) - e^{0.0742|t|}\sin(0.3 t)\sin(0.826) + 11.5793,
  42 + t\sin(0.826) + e^{0.0742|t|}\sin(0.3 t)\cos(0.826)
\right)
Final L1 score: 1028196.8409790021

