# Support Vector Regression

Predicting salaries from job position levels using SVR with an RBF kernel — implemented in Python and R.

## Overview

Applies Support Vector Regression to a small salary dataset (`Position_Salaries.csv`) with 10 job positions mapped to salary values. The model fits a non-linear regression curve and predicts salaries for intermediate position levels (e.g., level 6.5).

Both implementations use the RBF (Radial Basis Function) kernel and produce visualizations of the fitted curve against actual data points.

## Dataset

| Position          | Level | Salary    |
|-------------------|-------|-----------|
| Business Analyst  | 1     | 45,000    |
| Junior Consultant | 2     | 50,000    |
| …                 | …     | …         |
| CEO               | 10    | 1,000,000 |

10 rows, 3 columns. `Level` is the feature, `Salary` is the target.

## Methodology

1. Load dataset and extract features / target
2. Apply feature scaling (Python: `StandardScaler` on both X and y)
3. Fit SVR with RBF kernel
4. Predict salary for level 6.5
5. Visualize: scatter plot + fitted SVR curve
6. Generate a higher-resolution smooth curve

The RBF (Gaussian) kernel maps inputs into a higher-dimensional space to capture non-linear relationships:

- **Python** — `SVR(kernel='rbf')` via scikit-learn
- **R** — `svm(type='eps-regression', kernel='radial')` via `e1071`

## Files

```
svr.py                  # Python implementation
svr.R                   # R implementation
Position_Salaries.csv   # Dataset
requirements.txt        # Python dependencies
Rbf/                    # Kernel equation diagrams (PNG)
```

## 🛠 Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| 🐍 Language      | Python 3.8+, R                      |
| 📊 ML Library    | scikit-learn (`SVR`), e1071 (`svm`) |
| 📈 Visualization | matplotlib, ggplot2                 |
| 🧮 Data          | pandas, NumPy                       |

## Dependencies

### Python

```bash
pip install -r requirements.txt
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

## ⚠️ Known Issues

- Train/test split is commented out in both implementations — the full dataset is used for fitting. Fine for a demo, not for real evaluation.
- The R implementation uses positional column indexing (`dataset[2:3]`), which is fragile if the CSV structure changes.
- Dataset is very small (10 rows), so SVR performance is illustrative rather than production-grade.

## License

MIT
