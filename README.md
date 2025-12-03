# Comparative Stock Market Forecasting: S&P 500 vs NASDAQ-100 Using Machine Learning
---

## Team Members
* **Yug More** (NASDAQ-100: ^NDX analysis)
* **Srihan Cheemangunta** (S&P 500: ^GSPC analysis)

---

## Description of the Question and Research Topic
This project explores the **predictability of major stock market indices** using machine learning models. Specifically, we investigate whether the **S&P 500 (^GSPC)** or the **NASDAQ-100 (^NDX)** exhibits patterns that are easier for algorithms to forecast over **short-term (next-day and next-week) horizons**.

### Central Research Questions:
1.  **Can we predict next-day and next-week (5-day) index returns** using machine learning models?
2.  Are **next-week returns more predictable** than next-day returns for both indices?
3.  Do **complex models** (Random Forest, MLP/LSTM) **outperform simple models** (Linear Regression baseline)?
4.  Which **engineered financial features** (volatility, RSI, MACD, lagged returns) are the **most significant drivers** of predictive power?

---

## Methodology and Design

### Datasets
We use **25 years of daily historical price data (2000–2024)** retrieved consistently via the `yfinance` library:
* **NASDAQ-100 (NDX):** `^NDX`
* **S&P 500 (GSPC):** `^GSPC`

### Feature Engineering
A consistent set of financial indicators was engineered for both indices to model market behavior:
* **Lagged Returns:** `lag_1`, `lag_2`, `lag_5`
* **Moving Averages:** `MA_7`, `MA_14`, `MA_21`
* **Volatility:** Rolling Standard Deviation (7, 14 days)
* **Momentum:** 5-day percent change
* **Technical Indicators:** **RSI** (Relative Strength Index), **MACD** (Moving Average Convergence Divergence)

**Target Variables:** `next_day_return` and `next_week_return` (5-day forward return).

### Machine Learning Models
We trained and compared models across both short-term targets:
* **Baseline:** Linear Regression
* **Classical ML:** Random Forest Regressor
* **Advanced:** MLP Neural Network (Optional)
* **Advanced:** LSTM Time-Series Model (Optional)

### Evaluation Metrics
Model performance was assessed using standard regression and classification metrics:
* **Error:** MAE, MSE, RMSE
* **Classification:** Directional Accuracy (predicting the correct 'up' or 'down' movement)

---

## Key Findings and Insights

### Cross-Index Takeaways
* **Weekly vs. Daily:** Weekly returns were consistently **more predictable** than next-day returns for both the S&P 500 and the NASDAQ-100.
* **Model Performance:** The **Random Forest Regressor** consistently outperformed the simple Linear Regression baseline across both indices and prediction horizons.
* **Feature Importance:** Lagged returns, volatility, and RSI were found to be the most influential features.

### S&P 500 (^GSPC)
* Showed **more stable and smoother** price movements.
* Was **easier to predict** on a day-to-day basis, with slightly more consistent daily prediction models.
* Feature importance analysis favored **volatility and the 14-day Moving Average (`MA_14`)**.

### NASDAQ-100 (^NDX)
* Exhibited **higher volatility**, making daily predictions generally harder.
* Required **more complex models** (e.g., Random Forest/Neural Networks) to capture underlying patterns compared to the S&P 500.
* Feature importance highlighted **RSI and the one-day lagged return (`lag_1`)** as strong drivers.

---

## Project Outline/Plan

### Yug More (NASDAQ-100: ^NDX)
* **Objective:** Data collection, cleaning, and normalization for ^NDX. Creation of all engineered features. Model training (**LR, RF, MLP**) for next-day and next-week forecasting. Evaluation, visualization, and cross-index comparison.
* **Model Plans:** Linear Regression, Random Forest, MLP/LSTM.
* **Dataset:** Historical daily price data for NASDAQ-100 (`^NDX`) via `yfinance`.

### Srihan Cheemangunta (S&P 500: ^GSPC)
* **Objective:** Recreate the exact pipeline for S&P 500 (`^GSPC`). Model training and validation. Generation of performance metrics and plots. Contribution to the final cross-index analysis and presentation.
* **Model Plans:** Linear Regression, Random Forest, MLP/LSTM.
* **Dataset:** Historical daily price data for S&P 500 (`^GSPC`) via `yfinance`.

---

## Project Timeline

| Date | NASDAQ-100 Task (Yug More) | S&P 500 Task (Srihan Cheemangunta) | Shared/Comparison Task |
| :--- | :--- | :--- | :--- |
| **Nov 29** | Finalized ^NDX data and preprocessing setup. | Finalized ^GSPC data and preprocessing setup. | Defined project structure and variables. |
| **Nov 30** | Completed all feature engineering and technical indicators. | Completed all feature engineering and technical indicators. | Created target variables (`next_day_return`, `next_week_return`). |
| **Dec 1** | Trained Linear Regression and Random Forest models. | Trained Linear Regression and Random Forest models. | Evaluated next-day & next-week accuracy (MAE, RMSE, Directional). |
| **Dec 2** | Trained Neural Models (MLP/LSTM optional). Generated final plots (volatility, feature importance). | Trained Neural Models (MLP/LSTM optional). Generated final plots (volatility, feature importance). | **Generated Final Comparison and Insights.** |
| **Dec 3** | Cleaned and finalized notebooks. | Prepared presentation slides. | Completed `README.md` and project documentation. |
