
# Stock market LSTM

This code is designed for stock market analysis and prediction using a range of technical indicators and a Long Short-Term Memory (LSTM) neural network model. The primary goal is to predict the percent change of stock prices based on historical data. Here's a summary of its main components:



## Features

Data Fetching and Preprocessing: It uses Yahoo Finance to fetch historical stock data. The data is then preprocessed by calculating various technical indicators such as Average True Range (ATR), moving averages, volume-weighted average price (VWAP), RSI, MACD, and more. It also includes functions to merge macroeconomic data from a specified directory and to scale and clean the data using various scalers.

Technical Indicators Calculation: A comprehensive set of indicators is computed, including volatility measures, moving averages, price change percentages, high/low percent ranges, and psychological levels. These indicators serve as features for the predictive model.

Important note, before any indicators are used for price prediction it will drop any columns containing data that is related to the absolute price in dollars instead using values that are precentage based so it should work with any stock that has some degree of change. Avoiding the common pitfall of simply using the last closing price as a prediction.

```py
    df = df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close',"VWAP","200DAY_ATR-","200DAY_ATR", 'Volume', 'atr', '200ma', '14ma'], axis=1)


```

LSTM Model for Prediction: A Sequential LSTM model is built and trained to predict the percent change in the stock's closing price. The model includes layers for LSTM, dropout, and dense operations, and uses Mean Absolute Percentage Error (MAPE) as the loss metric. Various callbacks like Early Stopping and ReduceLROnPlateau are used to optimize training.

Training and Validation: The dataset is split into training and test sets, and the model is trained using the training set. The training process involves monitoring MAPE for both training and validation datasets.

Post-Prediction Analysis: After predictions are made, the code plots various graphs to visualize the performance of the model against actual data using Plotly. It also includes a histogram of prediction errors and scales back the data to its original form for better interpretability.

Utility Functions: Several utility functions are included for data normalization, outlier handling, and error calculation.

Execution Flow: The main function orchestrates the execution by fetching data, computing indicators, scaling, predicting, and plotting the results. It concludes by saving the predictions and associated error metrics to a CSV file.

This script is a comprehensive tool for quantitative analysts, traders, or enthusiasts looking to understand and predict stock market movements using machine learning and technical analysis.
## Authors

- [@JonIsHere242](https://github.com/JonIsHere242)

