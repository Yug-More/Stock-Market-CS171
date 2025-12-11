# Comparative Stock Market Forecasting: S&P 500 vs NASDAQ-100 Using Machine Learning

This project investigates the predictability of two major U.S. stock indices—the NASDAQ-100 (^NDX) and the S&P 500 (^GSPC)—using various machine learning models. We compare Linear Regression, Random Forest, and LSTM networks across next-day and next-week forecasting horizons.

---

## Team Members

* **Yug More** — Lead Researcher & Developer (NASDAQ-100: ^NDX)
* **Srihan Cheemangunta** — Supporting Developer (S&P 500: ^GSPC)

---

## Project Overview

### Research Goals

* Compare **next-day** versus **next-week** return prediction accuracy.
* Evaluate performance of **Linear Regression**, **Random Forest**, and **LSTM** models.
* Analyze how predictability changes with the time horizon.
* Determine the key differences in behavior and modeling between the **NASDAQ-100** and **S&P 500**.

### Key Findings

* **Next-day returns** are significantly **more predictable** than next-week returns.
* Advanced models (**RF, LSTM**) consistently outperform the baseline **Linear Regression**.
* The **S&P 500** is slightly **easier to predict** due to its lower volatility.
* The **LSTM** model generally yielded the best results for next-day forecasting on the high-volatility NASDAQ-100.

---

## Section A — NASDAQ-100 (^NDX)

**Lead Developer: Yug More**

### 1. Data Preprocessing (Notebook 1)

* **Dataset:** 25 years of daily NASDAQ-100 data (2000–2024), downloaded via `yfinance`.
* **Engineered Features:**
    * `log_return`
    * Lagged returns: `lag_1`, `lag_2`, `lag_5`
    * Moving Averages: `ma5`, `ma20`
    * `20-day rolling volatility`
    * Momentum indicator: `rsi` (Relative Strength Index)
* **Regression Targets:** `target_next_day_return`, `target_next_week_return`
* **Classification Target:** `target_direction`
* **Visualization:** An exploratory visualization of the NASDAQ-100 sector distribution was added for context. 

### 2. Model Construction (Notebook 2)

| Model | Summary | Performance Notes |
| :--- | :--- | :--- |
| **Linear Regression** | Baseline model. | Poor performance, predictions remained near zero. |
| **Random Forest** | Captures nonlinear behavior. | Better than LR, but output was flat on next-week predictions. |
| **LSTM Neural Network** | Learns short-term sequential patterns. | **Best for next-day predictions**, but struggled with weekly forecasting due to noise. |

### 3. Analysis & Findings (Notebook 3)

* **Research Question 1 (Next-day returns):** Predictions are noisy but limited predictability exists. **LSTM performs best**, followed by RF.
* **Research Question 2 (Advanced vs. LR):** **Yes, advanced models outperform LR**, but none can fully overcome the inherent randomness of financial returns.
* **Research Question 3 (Next-week returns):** **Yes, significantly harder.** Model error increases, RF output flattens, and LSTM becomes overly smoothed. Short-term indicators lose their predictive strength.

#### NASDAQ-100 Final Conclusion
Next-day returns are much more predictable than next-week returns. LSTM performs the best, but uncertainty increases rapidly over longer horizons.

### 4. Extra Notebook — Price Prediction Extension

* A simpler next-day **price prediction** extension was built for better interpretability.
* **Features:** Previous day's closing price, MA3, MA7, MA14.
* **Models Used:** Linear Regression, Gradient Boosting Regressor.
* **Result:** Produced a clean actual vs predicted price curve, enhancing project completeness and accessibility for non-technical readers.

---

## Section B — S&P 500 (^GSPC)

**Developer: Srihan Cheemangunta**

### Contributions

* **Preprocessing:** Recreated the full pipeline for the S&P 500, ensuring parallel feature engineering with the NASDAQ-100 dataset.
* **Modeling:** Trained and evaluated **Linear Regression** and **Random Forest** models.
* **Analysis:** Found the S&P 500 to be **easier to predict** than the NASDAQ-100 due to its **lower volatility**.

---

## Cross-Index Comparison (NASDAQ-100 vs S&P 500)

| Aspect | NASDAQ-100 (^NDX) | S&P 500 (^GSPC) |
| :--- | :--- | :--- |
| **Volatility** | **Higher** (tech-heavy) | **Lower** (broader index) |
| **Predictability** | Lower | **Higher** |
| **Best Model** | **LSTM** | **Random Forest** |
| **Strongest Features** | RSI, lag\_1 | MA14, volatility |

### Overall Finding
The S&P 500 is slightly easier to model because it is less volatile. The NASDAQ-100 requires more complex models (e.g., LSTM) but remains harder to forecast due to the higher inherent noise and concentration in volatile tech stocks.

---

## 🗓️ Project Timeline

| Date | NASDAQ-100 (Yug) | S&P 500 (Srihan) | Shared Milestones |
| :--- | :--- | :--- | :--- |
| **Nov 29** | Full preprocessing pipeline built | Basic preprocessing | Project structure defined |
| **Nov 30** | All feature engineering completed | Feature replication | Targets created |
| **Dec 1** | LR & RF trained and evaluated | LR & RF trained | Initial comparison |
| **Dec 2** | LSTM built, tuned, and visualized | Supporting plots created | Final insights |
| **Dec 3** | Cleaned notebooks + README + slides | Minor slide work | Final documentation |
