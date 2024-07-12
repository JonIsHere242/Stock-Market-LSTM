# Advanced Stock Market Analysis and Trading System

This project is a comprehensive stock market analysis and trading system that combines data collection, preprocessing, machine learning prediction, backtesting, and live trading capabilities. It's designed to provide a complete pipeline from raw market data to executable trading strategies.

## System Components

### 1. Ticker Downloader
- Automatically fetches an up-to-date list of stock tickers from the SEC.
- Ensures the system always has the most current set of tradable securities.

### 2. Price Downloader
- Downloads historical daily stock prices dating back to 1990 using Yahoo Finance.
- Implements robust error handling and rate limiting to manage large-scale data retrieval.
- Stores data in an efficient format (Parquet) for quick access and analysis.

### 3. Asset Correlator
- Analyzes relationships between different stocks and market sectors.
- Generates a correlation matrix to identify potential portfolio diversification opportunities.
- Helps in risk management by highlighting strongly correlated assets.

### 4. AI Predictor Model
- Utilizes a Random Forest model for stock price movement prediction.
- Incorporates both traditional technical indicators and advanced synthetic indicators.
- Features an automated training pipeline that regularly updates the model with new market data.

#### Synthetic Genetic Indicators
- Employs genetic programming to evolve highly effective custom technical indicators.
- These indicators often outperform traditional ones, ranking high in feature importance.
- Adaptable system that can re-evolve indicators to match changing market conditions.

### 5. Backtesting Engine
- Rigorous backtesting framework to evaluate trading strategies using historical data.
- Simulates trading scenarios to assess strategy performance under various market conditions.
- Provides detailed performance metrics including Sharpe ratio, drawdown, and profit factor.

### 6. Live Trading System
- Integrates with Interactive Brokers for real-time market data and trade execution.
- Implements risk management rules and position sizing based on account equity.
- Features a real-time monitoring dashboard for tracking open positions and overall performance.

## Key Features

- **Modular Architecture**: Each component is designed as a separate module, allowing for easy maintenance and scalability.
- **Automated Workflow**: The entire process from data collection to trading can be automated for daily execution.
- **Advanced Data Processing**: Utilizes efficient data structures and processing techniques to handle large datasets.
- **Machine Learning Integration**: Leverages the power of Random Forest for predictive modeling.
- **Genetic Programming**: Creates evolved, high-performance technical indicators.
- **Comprehensive Backtesting**: Allows for thorough strategy validation before live deployment.
- **Real-time Trading**: Capable of executing trades automatically based on model predictions.

## Usage

- **Good old bash**: Currently the script is run as a series python files task sceduled and called with bash.

## Future Enhancements

- Integration of real-time news sentiment analysis for more responsive trading strategies.
- Expansion of the genetic programming module to evolve entire trading strategies.
- Development of a web-based interface for easier monitoring and control of the system.

## Disclaimer

This software is for educational and research purposes only. It is not intended to be used as financial advice or a recommendation to trade real money. Trading stocks carries a high level of risk, and may not be suitable for all investors. Please ensure you fully understand the risks involved before using this system with real funds.

## Author

- [@JonIsHere242](https://github.com/JonIsHere242)

## License

- **The Old Secret Mission Cia Edition**: The license is very fun 10/10 recommend usting it for projects
