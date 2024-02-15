# Stock Market Analysis and Prediction Using LSTM and Random Forest

This project provides a comprehensive approach to stock market analysis and prediction. It utilizes a combination of machine learning models, including Random Forest and Long Short-Term Memory (LSTM) neural networks, to predict stock price movements based on historical data. The workflow encompasses data fetching, preprocessing, indicator calculations, machine learning model training, and trading signal generation.

## Project Components

### Data Collection and Preprocessing
- **SEC Ticker Download**: Fetches a list of tickers from the SEC.
- **Historical Data Retrieval**: Downloads historical daily stock prices back to 1990 using Yahoo Finance.
- **Data Preprocessing**: Includes scaling and cleaning of the data, calculation of various technical indicators, and data normalization.

### Indicator Calculations
- Extensive calculations of technical indicators such as ATR, moving averages, VWAP, RSI, MACD, and others.
- Utilizes Python libraries like Pandas and NumPy for efficient data manipulation and calculations.

### Machine Learning Models
- **Random Forest**: A Random Forest Regressor is used to predict stock price movements.
- **LSTM Neural Network**: Employs an LSTM model for predicting stock prices, incorporating features like Monte Carlo Dropout for better generalization.

### Trading Signal Generation
- Generates trading signals based on the LSTM's predictions of future stock prices.
- Includes validations to ensure the model's predictive power.

### Modular Codebase
- The code has been modularized into separate files for better maintainability and scalability.
- Includes utilities for logging, error handling, and file processing.

## Upcoming Enhancements

### Advanced Trading Interface
- Integration of a sophisticated trading interface for executing trades based on model predictions, including setting stop losses and take profits.

### Real-Time Market Adaptation
- Implementing a real-time news scanner to adapt trading strategies in response to market-affecting events and news.

### Model Optimization and Diversification
- Continuous optimization of individual stock predictors.
- Expansion of the LSTM model repository, feeding in the latest year's data for accurate next-day predictions.

### Automated Workflow
- Full automation of the data fetching, processing, and trading signal generation processes.
- Implementation of scheduling capabilities for automatic execution and updates.

### Backtesting and Performance Analysis
- Enhanced backtesting features to rigorously evaluate model performance over historical data.
- Detailed performance analysis and visualization tools.

## Authors
- [@JonIsHere242](https://github.com/JonIsHere242)
