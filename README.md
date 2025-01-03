# Advanced Stock Market Analysis and Trading System

A comprehensive end-to-end stock market analysis, prediction, and automated trading system combining machine learning, genetic programming, and real-time market execution. The system features a complete data pipeline from ticker acquisition to live trading, with sophisticated portfolio management and risk control.

## System Pipeline

### 1. Data Collection and Preprocessing

#### SEC Ticker Download (1__TickerDownloader.py)
- Automated fetching of ticker symbols from SEC database
- Exchange-specific filtering (NYSE, NASDAQ)
- Exclusion of OTC and CBOE markets
- Daily updates to maintain current market coverage
- Output: `TickerCIKs_{date}.parquet`

#### Market Data Collection (2__BulkPriceDownloader.py)
- Two operational modes:
  - ColdStart: Initial bulk download of all historical data
  - RefreshMode: Daily updates for existing symbols
- Configurable historical data depth (default: 2022-01-01)
- Robust error handling and rate limiting
- Data quality validation:
  - Minimum trading days requirement
  - Price consistency checks
  - Volume validation
- Output: Individual ticker parquet files in PriceData directory

#### Asset Derisking (AssetDerisker.ipynb)
- Not a daily requirement (can be done whenever as the values don't change drastically)
- K-means clustering for asset grouping based on multi timeframe weighted returns
- Implementation based on "Advances in Active Portfolio Management"
- Cross-asset correlation analysis
- Group identification for interchangeable assets
- Output: `Correlations.parquet` for portfolio diversification

### 2. Feature Engineering

#### Technical Indicator Generation (3__Indicators.py)
- Over 50 technical indicators including:
  - genetically discovered indicators
  - Volatility measures
  - Volume analysis
  - Price pattern recognition
- Kalman filter trend analysis
- Custom momentum indicators
- Data cleaning and outlier detection

(removing failing companies only staying listed through reverse stock splits)
- Output: Enhanced price data with indicators

#### Genetic Feature Discovery (Z_Genetic.py)
- Evolutionary algorithm for indicator creation
- Monte Carlo Tree Search optimization
- Custom fitness function combining:
  - Expected value
  - Information coefficient
  - Profit factor
- Feature importance analysis
- Output: Novel technical indicators

### 3. Prediction System

#### Machine Learning Pipeline (4__Predictor.py)
- Random Forest model with dual approach:
  - Classification for direction
  - Regression for magnitude
- Advanced hyperparameter optimization
- Cross-validation with time-series considerations
- Feature importance tracking
- Output: Trained model and daily predictions

### 4. Trading Systems

#### Backtesting Engine (5__NightlyBroker.py)
- Event-driven architecture
- Portfolio management with:
  - Dynamic position sizing
  - Risk-based allocation
  - Correlation-aware diversification
- Performance analytics
- Integration with asset grouping from `Correlations.parquet`

#### Live Trading System (6__DailyBroker.py)
- Interactive Brokers integration
- Real-time order execution
- Portfolio monitoring
- Risk management:
  - Position limits
  - Drawdown controls
  - Exposure management
- Automated trade reconciliation

### 5. Shared Infrastructure

#### Trading Functions (Trading_Functions.py)
- Common codebase for live and backtest environments
- Standardized:
  - Order management
  - Position sizing
  - Risk calculations
  - Market analysis
- Consistent behavior across environments

## Key Features
- Advanced Analytics
- Genetic Programming for feature discovery
- Asset clustering for risk management
- Custom technical indicators normal/genetic
- Complete Interactive Brokers integration (backtrader/ib_insync)
- Real-time market data processing (TWS datafeeds)
- Automated execution pipeline
- Portfolio synchronization (should really use a database oops)

## Technical Requirements

### Hardware
- Recommended: 32GB+ RAM for full dataset processing
- Multi-core CPU for parallel processing
(45mins at 32 cores 5.15GHz not including 1.5Hrs for ratelimited OHLCV download each night)
- SSD storage for data management (approx. 5 GB of data)

### Software
- Python 3.8+
- Interactive Brokers TWS or IB Gateway
- Required Python packages:
  ```
  pandas
  numpy
  scikit-learn
  gplearn    # or my custom genetic programming library
  backtrader
  ib_insync  #absolutely needed
  pyarrow
  ```

## Usage

### Pipeline Execution
1. Run `1__TickerDownloader.py` to get latest SEC data
2. Execute `2__BulkPriceDownloader.py` with appropriate mode
3. Run `AssetDerisker.ipynb` for correlation analysis
4. Process indicators with `3__Indicators.py`
5. Generate predictions using `4__Predictor.py`
6. Backtest strategies with `5__NightlyBroker.py`
7. Deploy live trading with `6__DailyBroker.py`

## Live Trading

The Live Trading component serves as the culmination of the entire trading system, executing real-time trades based on predictive analytics and ensuring robust portfolio management.

### Key Functionalities

#### Interactive Brokers Integration
- Connection Management
- Contract Handling
- Real-Time Data Processing
- Live Data Feeds
- Data Resampling

#### Strategy Execution (MyStrategy Class)
- Order Management
  - Order Tracking
  - Bracket Orders
  - Order Notifications
- Position Management
  - Dynamic Position Sizing
  - Risk Controls
  - Trade Reconciliation

#### Risk Management
- Stop-Loss and Take-Profit
- Trailing Stops
- Comprehensive Logging
- Heartbeat Mechanism

### Usage Instructions

1. Ensure TWS or IB Gateway is Running:
   - Start Interactive Brokers' TWS or Gateway
   - Configure API settings (default port: 7497)

2. Execute the Live Trading Script:
   ```bash
   python 6__DailyBroker.py
   ```

3. Monitor Logs and Outputs:
   - Real-time logs in console and `__BrokerLive.log`
   - Heartbeat messages for system status

## Future Enhancements
- Advanced Risk Metrics (VaR, CVaR)
- Strategy Diversification
- Automated Alerts
- Web Interface
- Docstrings and notes on why things have been done

## Acknowledgments
This project combines modern portfolio theory with machine learning and genetic programming techniques. A special thanks to the developers of the gplearn library, whose work laid the foundation for my own genetic programming library. Additionally, gratitude is extended to the pioneers of Monte Carlo Tree Search (MCTS) with backpropagation utilities. Integrating these two powerful concepts—tree-based algorithms from both gplearn and MCTS—has been instrumental in discovering new genetic indicators, contributing significantly to our market edge (around 70%). Lastly, heartfelt thanks to the Interactive Brokers API team and the open-source community for their invaluable support and resources such as ib_insync.