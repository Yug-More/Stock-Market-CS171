# Comparative Stock Market Forecasting: S&P 500 vs NASDAQ-100 Using Machine Learning
---

## Team Members
* **Yug More** — Lead Researcher & Developer (NASDAQ-100: ^NDX)
* **Srihan Cheemangunta** — Supporting Developer (S&P 500: ^GSPC)

---

## Description of the Question and Research Topic
This project explores the **predictability of major U.S. stock market indices** using machine learning. We focus on whether the **NASDAQ-100 (^NDX)** or the **S&P 500 (^GSPC)** is easier to forecast over **short-term horizons** using classical and deep learning models.

### Central Research Questions:
1. **Can we predict next-day and next-week (5-day) index returns** using ML models?
2. **Are next-week returns harder to predict** compared to next-day returns?
3. **Do advanced models** (Random Forest, LSTM) outperform **Linear Regression**?
4. Which **engineered features** (RSI, volatility, lagged returns, MACD) provide the most predictive value?

---

## Methodology and Design

### Datasets
We used **25 years of historical price data (2000–2024)** retrieved through `yfinance`:
* NASDAQ-100: `^NDX`
* S&P 500: `^GSPC`

### Feature Engineering
A consistent set of indicators was built for both indices:
* Lagged returns (`lag_1`, `lag_2`, `lag_5`)
* Moving averages (`MA_7`, `MA_14`, `MA_21`)
* Rolling volatility (7 & 14 days)
* Momentum (5-day % change)
* RSI, MACD

**Targets:**  
* `next_day_return` (1-day ahead)  
* `next_week_return` (5-day ahead)

---

## Machine Learning Models
We trained the following:
* **Linear Regression** (baseline)
* **Random Forest Regressor**
* **MLP Neural Network** (optional)
* **LSTM Neural Network** (optional)

---

## Evaluation Metrics
* MAE, MSE, RMSE  
* Directional Accuracy (correctly predicting up/down movement)

---

## Key Findings and Insights

### Cross-Index Insights
* **Next-week returns are slightly more predictable** than next-day returns due to reduced noise.
* **Random Forest consistently outperforms Linear Regression** on both indices.
* **Lagged returns, volatility, and RSI** were the strongest features overall.

### S&P 500 (^GSPC)
* More stable movements → **slightly easier to predict**
* Best features: **MA_14** and volatility

### NASDAQ-100 (^NDX)
* More volatile → **harder to model**
* Best features: **RSI** and `lag_1`
* Advanced models (RF, LSTM) help significantly more

---

## Project Outline / Division of Work

### **Yug More — Lead Work (NASDAQ-100: ^NDX & Full Pipeline Development)**
* Collected, cleaned, and normalized **all datasets**
* Engineered **every technical feature** (RSI, MACD, volatility, lagged returns, etc.)
* Created both prediction targets (next-day & next-week)
* Built **the complete modeling pipeline**
  - Linear Regression
  - Random Forest
  - LSTM neural network
* Designed and generated **all major plots & visualizations**  
  (prediction charts, training curves, feature analyses)
* Performed **full error evaluation**, comparisons, and interpretation
* Wrote the **cross-index comparative findings**
* Created the **presentation structure, visuals, and majority of the slides**
* Finalized and cleaned **both Jupyter notebooks**
* Wrote the majority of the README.md

### **Srihan Cheemangunta — Supporting Work (S&P 500: ^GSPC)**
* Reapplied Yug’s pipeline to S&P 500 dataset
* Ran Linear Regression and Random Forest models
* Generated supporting plots for S&P 500
* Assisted with presentation slides and minor documentation

---

## Project Timeline

| Date | NASDAQ-100 (Yug More) | S&P 500 (Srihan Cheemangunta) | Shared Work |
|------|------------------------|-------------------------------|-------------|
| **Nov 29** | Full preprocessing pipeline created | Basic preprocessing | Defined project structure |
| **Nov 30** | All feature engineering completed | Features replicated | Target creation |
| **Dec 1** | LR & RF fully trained + evaluated | LR & RF trained | Early comparison |
| **Dec 2** | LSTM built, tuned, and visualized + final plots | Generated supporting plots | Final joint insights |
| **Dec 3** | Cleaned notebooks + wrote README + built slides | Minor slide work | Final documentation |

---


