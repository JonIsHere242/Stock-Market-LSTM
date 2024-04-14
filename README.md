# Stock Market Analysis and Prediction Using Random Forest

This project employs sophisticated methods for stock market analysis and prediction, focusing primarily on the Random Forest model. This system predicts stock price movements by leveraging historical data processed through advanced data fetching, preprocessing, and technical indicator calculations. The workflow is designed for daily automated execution, making it an ideal solution for both novice and experienced traders seeking to automate their trading strategies.

## Project Components

### Data Collection and Preprocessing
- **SEC Ticker Download**: Automatically fetches a list of tickers from the SEC.
- **Historical Data Retrieval**: Downloads historical daily stock prices back to 1990 using Yahoo Finance.

### Indicator Calculations
- **Data Preprocessing**: Involves scaling, cleaning, and normalizing the data, and calculating various technical indicators such as ATR, moving averages, VWAP, RSI, and MACD.
- Comprehensive calculations of technical indicators using Python libraries like Pandas and NumPy.

### Machine Learning Models
- **Hybrid Approach**: The project uses a combination of Random Forest Classifier and Random Forest Regressor models to handle different aspects of market prediction. This dual approach leverages classification for directionality of the market movements and regression for quantifying the changes.

- **Trinary Classification**: The Random Forest Classifier discretizes the target variable into three classes (up, down, stable) based on a dynamic threshold that adapts to volatility, which is crucial for capturing market sentiment more accurately than binary classifications.

- **Dynamic Data Concatenation**: To manage training scalability and effectiveness, the system dynamically concatenates data frames based on a configurable percentage of available data files. This method is crucial for managing memory usage and computational efficiency, especially when dealing with vast datasets.

- **Automated Training Process**: The training pipeline is fully automated, allowing for models to be retrained with new data using scheduled jobs. This ensures that the models are always up-to-date and reflect the most recent market conditions.

### Trading Signal Generation
- Generates trading signals based on predictions from the Random Forest model. Signals are refined by historical performance to ensure reliability.
- Post-processing of signals includes capping prediction values to avoid unrealistic predictions based on historical extremes.

### Modular Codebase
- The codebase is structured into separate modules for each functionality to improve maintainability and scalability.
- Includes comprehensive logging, error handling, and file management utilities.
- Designed for easy scriptability: scripts can be easily set up to run automatically every night for up-to-date market analysis and predictions.

## Upcoming Enhancements

### Automated Daily Trade Execution
- Future versions will integrate with Interactive Brokers to execute trades automatically based on daily model predictions.
- This automation will include capabilities to adjust strategies based on pre-market conditions and historical data insights.

### Real-Time Market Adaptation
- Plans are in place to incorporate a real-time news scanner that adjusts trading strategies dynamically in response to significant market events.

### Model Optimization
- Continuous optimization of the Random Forest model to enhance prediction accuracy.
- Extension of the data repository to include the most recent market data for next-day trading strategies.

### Backtesting and Performance Analysis
- Advanced backtesting frameworks to rigorously evaluate model performance using extensive historical data.
- Development of sophisticated visualization tools for in-depth analysis of strategy effectiveness.

## Authors
- [@JonIsHere242](https://github.com/JonIsHere242)