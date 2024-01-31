# Stock Market LSTM - Work in Progress

This project, currently under active development, is designed for comprehensive stock market analysis and prediction using technical indicators and a Long Short-Term Memory (LSTM) neural network model. The primary objective is to predict the percent change in stock prices based on historical data. This README outlines the main components of the project and future enhancements.

## Current Features

- **Data Fetching and Preprocessing**: Utilizes Yahoo Finance to fetch historical stock data. Preprocessing includes calculating various technical indicators (like ATR, moving averages, VWAP, RSI, MACD) and merging macroeconomic data. Data is scaled and cleaned using various techniques.

- **Technical Indicators Calculation**: Computes a range of indicators, focusing on volatility, moving averages, and price change percentages. Absolute price data in dollars is dropped to ensure the model's applicability to various stocks.

    ```py
    df = df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close',"VWAP","200DAY_ATR-","200DAY_ATR", 'Volume', 'atr', '200ma', '14ma'], axis=1)
    ```

- **LSTM Model for Prediction**: A Sequential LSTM model predicts the percent change in stock closing price. The model incorporates LSTM, dropout, and dense layers, optimizing using callbacks like Early Stopping and ReduceLROnPlateau.

- **Training and Validation**: The model is trained on a split dataset, with performance monitored using Mean Absolute Percentage Error (MAPE) for both training and validation sets.

- **Post-Prediction Analysis**: Performance visualization using Plotly, including prediction error histograms and scaling back data for interpretability.

- **Utility Functions**: Includes functions for data normalization, outlier handling, and error calculation.

- **Execution Flow**: The main function manages data fetching, indicator computation, scaling, prediction, and result plotting, saving predictions and error metrics to CSV files.

## Upcoming Enhancements

- **Modularization**: Transitioning the existing code in "LSTM_Example" into separate, modular files for better maintainability and scalability.

- **Automation**: Implementing scheduling capabilities to enable automatic execution of various components of the project.

- **Trading Interface Integration**: Integrating a trading interface to facilitate actual trading based on model predictions.

- **Backtesting Feature**: Incorporating a backtesting mechanism to evaluate the model's performance on historical data.

## Authors

- [@JonIsHere242](https://github.com/JonIsHere242)

