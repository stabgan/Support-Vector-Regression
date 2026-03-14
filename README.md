# Support Vector Regression

Predicting salaries from job position levels using SVR with an RBF kernel — implemented in both Python and R.

## Overview

This project applies Support Vector Regression (SVR) to a small salary dataset (`Position_Salaries.csv`) containing 10 job positions mapped to salary values. The goal is to fit a non-linear regression model and predict salaries for intermediate position levels (e.g., level 6.5).

Both implementations use the RBF (Radial Basis Function) kernel and produce visualizations of the fitted curve against the actual data points.

## Dataset

| Position          | Level | Salary      |
|-------------------|-------|-------------|
| Business Analyst  | 1     | 45,000      |
| Junior Consultant | 2     | 50,000      |
| ...               | ...   | ...         |
| CEO               | 10    | 1,000,000   |

10 rows, 3 columns. The model uses `Level` as the feature and `Salary` as the target.

## Approach

1. Load the dataset and extract features (`Level`) and target (`Salary`)
2. Apply feature scaling (Python only — `StandardScaler` on both X and y)
3. Fit an SVR model with an RBF kernel
4. Predict salary for a new input (level 6.5)
5. Visualize results: scatter plot of actual data + fitted SVR curve
6. Generate a higher-resolution smooth curve for better visualization

### Kernel

The RBF (Gaussian) kernel is used in both implementations:

- **Python:** `SVR(kernel='rbf')` via scikit-learn
- **R:** `svm(type='eps-regression', kernel='radial')` via the `e1071` package

## Files

```
svr.py                  # Python implementation
svr.R                   # R implementation
Position_Salaries.csv   # Dataset
Rbf/                    # Kernel equation diagrams (PNG)
```

## Dependencies

### Python

- Python 3.x
- numpy
- matplotlib
- pandas
- scikit-learn

```bash
pip install numpy matplotlib pandas scikit-learn
```

### R

- e1071
- ggplot2

```r
install.packages(c("e1071", "ggplot2"))
```

## Running

### Python

```bash
python svr.py
```

### R

```bash
Rscript svr.R
```

## Known Issues

- **`svr.py` — deprecated `StandardScaler` usage on 1D arrays:** `sc_y.fit_transform(y)` passes a 1D array where a 2D array is expected. Modern scikit-learn will raise a warning or error. Fix: reshape `y` with `y.reshape(-1, 1)` before scaling.
- **`svr.py` — `regressor.predict(6.5)` passes a raw scalar:** Modern scikit-learn expects a 2D array. This will raise a `ValueError`. Should be `regressor.predict([[6.5]])` (and the value should be scaled first).
- **`svr.py` — `inverse_transform` on predict output:** The predict result also needs reshaping for newer scikit-learn versions.
- **`svr.py` — train/test split is commented out:** The entire dataset is used for both fitting and visualization. Fine for a demo, but not a real evaluation.
- **`svr.R` — `dataset[2:3]` drops the Position column but keeps Level and Salary:** This works, but the column subsetting is fragile and not self-documenting.

## License

MIT — Kaustabh Ganguly, 2018
