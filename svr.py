# SVR — Support Vector Regression
# Predicting salaries from position levels using an RBF kernel

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def main():
    # ── Load dataset (relative to script location) ──────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Position_Salaries.csv")
    dataset = pd.read_csv(csv_path)

    X = dataset.iloc[:, 1:2].values  # Level (2D array)
    y = dataset.iloc[:, 2].values    # Salary (1D array)

    # ── Feature Scaling ─────────────────────────────────────────────────
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_scaled = sc_X.fit_transform(X)
    y_scaled = sc_y.fit_transform(y.reshape(-1, 1)).ravel()

    # ── Fit SVR (RBF kernel) ────────────────────────────────────────────
    regressor = SVR(kernel="rbf")
    regressor.fit(X_scaled, y_scaled)

    # ── Predict salary for level 6.5 ───────────────────────────────────
    level_input = np.array([[6.5]])
    y_pred_scaled = regressor.predict(sc_X.transform(level_input))
    y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    print(f"Predicted salary for level 6.5: ${y_pred[0]:,.0f}")

    # ── Visualise SVR results ───────────────────────────────────────────
    plt.scatter(X_scaled, y_scaled, color="red", label="Actual")
    plt.plot(X_scaled, regressor.predict(X_scaled), color="blue", label="SVR")
    plt.title("Truth or Bluff (SVR)")
    plt.xlabel("Position level (scaled)")
    plt.ylabel("Salary (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ── Higher-resolution smooth curve ──────────────────────────────────
    X_grid = np.arange(X_scaled.min(), X_scaled.max(), 0.01).reshape(-1, 1)
    plt.scatter(X_scaled, y_scaled, color="red", label="Actual")
    plt.plot(X_grid, regressor.predict(X_grid), color="blue", label="SVR")
    plt.title("Truth or Bluff (SVR — smooth)")
    plt.xlabel("Position level (scaled)")
    plt.ylabel("Salary (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
