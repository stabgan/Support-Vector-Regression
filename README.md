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

## Methodology

1. Load the dataset and extract features (`Level`) and target (`Salary`)
2. Apply feature scaling (Python only — `StandardScaler` on both X and y)
3. Fit an SVR model with an RBF kernel
4. Predict salary for a new input (level 6.5)
5. Visualize results: scatter plot of actual data + fitted SVR curve
6. Generate a higher-resolution smooth curve for better visualization

### Kernel

The RBF (Gaussian) kernel maps inputs into a higher-dimensional space to capture non-linear relationships:

- **Python:** `SVR(kernel='rbf')` via scikit-learn
- **R:** `svm(type='eps-regression', kernel='radial')` via the `e1071` package

## Files

```
svr.py                  # Python implementation
svr.R                   # R implementation
Position_Salaries.csv   # Dataset
Rbf/                    # Kernel equation diagrams (PNG)
```

## Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| 🐍 Language     | Python 3.x, R                       |
| 📊 ML Library   | scikit-learn (`SVR`), e1071 (`svm`) |
| 📈 Visualization| matplotlib, ggplot2                 |
| 🧮 Data         | pandas, numpy                       |

## Dependencies

### Python

```bash
pip install numpy matplotlib pandas scikit-learn
```

### R

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

- The train/test split is commented out in both implementations — the entire dataset is used for fitting and visualization. Fine for a demo, but not suitable for real model evaluation.
- The R implementation uses positional column indexing (`dataset[2:3]`), which is fragile if the CSV structure changes.
- The dataset is very small (10 rows), so SVR performance here is illustrative rather than production-grade.

## License

MIT — Kaustabh Ganguly, 2018
