# Project Overview

This project aims to develop a stock price prediction app for the National Stock Exchange of India (NSE). The app utilizes historical stock data to predict future price movements using various machine learning techniques, primarily regression models.

## Data Collection and Preparation

* **Data Source:** Historical stock data is sourced from the NSE.
* **Feature Engineering:**
  * **Technical Indicators:**
    - Moving Averages (MA): Calculate simple moving averages (SMA) and exponential moving averages (EMA) over different time periods to identify trends.
    - Relative Strength Index (RSI): Measure the speed and change of price movements to identify overbought and oversold conditions.
    - Moving Average Convergence Divergence (MACD): Identify changes in the strength, direction, momentum, and duration of a trend.
    - Bollinger Bands: Determine volatility and potential price reversals.
  * **Time-Series Features:**
    - Lagged Features: Incorporate past values of the target variable and other features to capture historical patterns.
    - Rolling Statistics: Calculate rolling mean, standard deviation, and other statistical measures to identify trends and seasonality.

## Model Training and Evaluation

* **Regression Models:**
  - Linear Regression : A simple model to establish a linear relationship between features and the target variable.
  - Random Forest Regression : An ensemble method that combines multiple decision trees to improve accuracy.
  - Support Vector Regression (SVR) : A powerful model for non-linear regression tasks.
  - Gradient Boosting Regression : An ensemble method that iteratively builds models to improve predictions.
  - XGBoost : A scalable and efficient gradient boosting framework.

* **Model Evaluation:**
  - Root Mean Squared Error (RMSE) - Measures the average magnitude of the errors.
  - Mean Absolute Error (MAE) - Measures the average magnitude of the errors without considering their direction.
  - Mean Absolute Percentage Error (MAPE) -  Measures the average percentage error.
  - R-squared - Measures the proportion of the variance in the dependent variable explained by the independent variables.

## Future Work

* **Integration of LSTM Models:** Explore the use of Long Short-Term Memory (LSTM) networks to capture long-term dependencies in the time series data.
* **Enhanced Feature Engineering:** Experiment with additional technical indicators and time-series features to improve model performance.
* **Hyperparameter Tuning:** Optimize model hyperparameters using techniques like grid search and random search.
* **Ensemble Methods:** Combine multiple models to improve predictive accuracy.

## Disclaimer

**This app is intended for educational purposes only. It should not be used as a sole basis for investment decisions. Please consult with a qualified financial advisor before making any investment decisions.**

Note: While classification models were explored during development, they were ultimately excluded due to performance limitations probably due to unrealistic target variables.

*Note: While classification models were explored during development, they were ultimately excluded due to performance limitations and unrealistic target variables.*
