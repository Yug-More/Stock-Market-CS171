# Comparative Stock Market Forecasting: S&P 500 vs NASDAQ-100 Using Machine Learning

The main objective of this project is to explore the predictability of two major U.S. stock indices—the NASDAQ-100 (^NDX) and the S&P 500 (^GSPC) using various machine learning models. At the core of our project is one main question: **Can we predict how the NASDAQ-100 and S&P500 will move in the future using machine learning?**. In our notebooks we make use of several models ranging from Linear Regression, Random Forest to LSTM.

---

## Team Members
* **Yug More** — Lead Researcher & Developer (NASDAQ-100: ^NDX)
* **Srihan Cheemangunta** — Developer (S&P 500: ^GSPC)
  
---

## Project Overview

### Research Goals

* Make comparisons between the **next-day** and **next-week** return prediction accuracy.
* Evaluate performance of the different machine learning models such as **Linear Regression**, **Random Forest**, and **LSTM**.
* Analyze how predictability changes from next-day to next-week returns.

### Key Findings
* **Next-day returns** are significantly **more predictable** than next-week returns.
* Advanced models (**Random Forest, LSTM**) consistently outperform the baseline **Linear Regression**.
* The **S&P 500** is slightly **easier to predict** due to its lower volatility.
* The **LSTM** model generally yielded the best results for next-day forecasting on the high-volatility NASDAQ-100.

---

## Section A — NASDAQ-100 (^NDX)

**Lead Developer: Yug More**

### Main Research Question
**Can we predict how the NASDAQ-100 and S&P500 will move in the future using machine learning?**

### Sub-Research Questions to answer the main research question:
1. **RQ1:** Can simple classical models (Linear Regression and Random Forest) predict next-day returns?
2. **RQ2:** Does a deep learning model such as LSTM perform better than classical models?
3. **RQ3:** Are next-day returns easier to predict than next-week returns?

I kept the modeling process simple and focused on understanding rather than complexity. My goal here is to learn how different models behave on financial time-series data.

### 1. Data Preprocessing (Notebook 1)
* **Dataset:** 25 years of daily NASDAQ-100 data (2000–2024), downloaded via `yfinance`.
* **Engineered Features:**
    * `log_return`
    * Lagged returns: `lag_1`, `lag_2`, `lag_5`
    * Moving Averages: `ma5`, `ma20`
    * `20-day rolling volatility`
    * Momentum indicator: `RSI` (Relative Strength Index)
* **Regression Targets:** `target_next_day_return`, `target_next_week_return`
* **Classification Target:** `target_direction`
* **Visualization:** An exploratory visualization of the NASDAQ-100 sector distribution was added for context. 

### 2. Model Construction (Notebook 2)

| Model | Summary | Performance Notes |
| :--- | :--- | :--- |
| **Linear Regression** | Baseline model. | Poor performance, predictions remained near zero. |
| **Random Forest** | Captures nonlinear behavior. | Better than LR, but output was flat on next-week predictions. |
| **LSTM Neural Network** | Learns short-term sequential patterns. | **Best for next-day predictions**, but struggled with weekly forecasting due to noise. |

### Analysis & Findings (Notebook 2)
* **Research Question 1 (Next-day returns):** Predictions are noisy but limited predictability exists. **Linear Regression and Random Forest models perform poorly**, althought Random Forest has a slightly better result.
* **Research Question 2:** **Yes, the advanced LSTM/Deep LSTM models outperform the classical models**, but none can fully overcome the inherent randomness of financial returns.
* **Research Question 3 (Next-week returns):** **Yes, significantly harder.** Model error increases, Random Forest output flattens out and is closer to zero for the horizon, and LSTM becomes overly smoothed. Short-term indicators lose their predictive strength. So although some result is obtained the result for next-day return predictions are better.

### Why next-day returns are easier:
- Because of only one day of uncertainty, so the indicators still hold short-term meaning.
- Technical features like the MA5, MA20, RSI certainly relate more closely to next-day behavior
- Usually for the stock markets the daily noise is large, thus next-week noise is likely to be much larger.

### Why next-week returns are harder:
- Considering next-week is a longer-horizon it could involve many more uncertain and unpredictable events (news, sentiment shifts, interest rate changes, macro data)
- Short-term indicators lose most of their predictive power at this horizon
- Both models avoid making large predictions, which shows increased uncertainty

#### NASDAQ-100 Final Conclusion
Next-day returns are much more predictable than next-week returns. LSTM performs the best, but uncertainty increases rapidly over longer horizons.

### 4. Extra Notebook — Price Prediction of NASDAQ-100

* A simpler next-day **price prediction** extension was built for better interpretability.
* **Features:** Previous day's closing price, MA3, MA7, MA14.
* **Models Used:** Linear Regression, Gradient Boosting Regressor.
* **Result:** Produced a clean actual vs predicted price curve, enhancing project completeness and accessibility for non-technical readers.

---

## Section B — S&P 500 (^GSPC)

**Lead Developer: Srihan Cheemangunta**

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

## Project Timeline

| Date | NASDAQ-100 (Yug) | S&P 500 (Srihan) | Shared Milestones |
| :--- | :--- | :--- | :--- |
| **Nov 29** | Full preprocessing pipeline built | Basic preprocessing | Project structure defined |
| **Nov 30** | All feature engineering completed | Feature replication | Targets created |
| **Dec 1** | LR & RF trained and evaluated | LR & RF trained | Initial comparison |
| **Dec 2** | LSTM built, tuned, and visualized | Supporting plots created | Final insights |
| **Dec 3** | Cleaned notebooks + README + slides | Minor slide work | Final documentation |





