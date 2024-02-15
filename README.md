# Stock Market Analysis and Prediction Using LSTM and Random Forest

This project offers a sophisticated approach to stock market analysis and prediction. It combines machine learning models, including Random Forest and Long Short-Term Memory (LSTM) neural networks, to predict stock price movements. This ensemble system is unique as the LSTM model is trained on data predicted by the Random Forest model, enhancing predictive accuracy. The workflow includes data fetching, preprocessing, indicator calculations, machine learning model training, and advanced trading signal generation.

## Project Components

### Data Collection and Preprocessing
- **SEC Ticker Download**: Fetches a list of tickers from the SEC.
- **Historical Data Retrieval**: Downloads historical daily stock prices back to 1990 using Yahoo Finance.
- **Data Preprocessing**: Involves scaling, cleaning, and normalizing the data, and calculating various technical indicators.

### Indicator Calculations
- In-depth calculations of technical indicators such as ATR, moving averages, VWAP, RSI, MACD, among others, using Python libraries like Pandas and NumPy.

### Machine Learning Models
- **Random Forest**: Utilizes a Random Forest Regressor to predict stock price movements. These predictions are subsequently used as training data for the LSTM model.
- **LSTM Neural Network**: Employs an LSTM model for predicting stock prices, featuring Monte Carlo Dropout for enhanced generalization.

### Trading Signal Generation
- The system generates trading signals based on predictions from the LSTM model, especially following its 'moments of clarity'—periods of calm news cycles and stable world events.
- Signals are only considered if the preceding prediction was accurate, leveraging the model's enhanced performance during these stable periods.
- Post-processing of signals includes capping prediction values positively or negatively based on historical maximum values over a given period to avoid unrealistic predictions.

### Modular Codebase
- Code has been organized into separate files for improved maintainability and scalability.
- Includes utility functions for logging, error handling, and file processing.

## Upcoming Enhancements

### Advanced Trading Interface
- Plans to integrate a sophisticated trading interface for executing model-based trades, complete with stop losses and take profits.

### Real-Time Market Adaptation
- Implementing a real-time news scanner to dynamically adjust trading strategies in response to significant market events.

### Model Optimization and Diversification
- Ongoing optimization of individual stock predictors.
- Expanding the repository of LSTM models to include the latest year's data for precise next-day predictions.

### Automated Workflow
- Fully automating the data fetching, processing, and trading signal generation processes.
- Adding scheduling capabilities for automatic execution and updates.

### Backtesting and Performance Analysis
- Implementing enhanced backtesting features for rigorous evaluation of model performance using historical data.
- Detailed performance analysis and visualization tools for better insight.

## Authors
- [@JonIsHere242](https://github.com/JonIsHere242)
