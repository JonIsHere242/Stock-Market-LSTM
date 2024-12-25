# Advanced Stock Market Analysis and Trading System

A comprehensive end-to-end stock market analysis, prediction, and automated trading system combining machine learning, genetic programming, and real-time market execution. The system features a complete data pipeline from ticker acquisition to live trading, with sophisticated portfolio management and risk control.

## System Pipeline

### 1. Data Collection and Preprocessing

#### SEC Ticker Download (1__TickerDownloader.py)
- Automated fetching of ticker symbols from SEC database
- Exchange-specific filtering (NYSE, NASDAQ)
- Exclusion of OTC and CBOE markets
- Daily updates to maintain current market coverage
- Output: TickerCIKs_{date}.parquet

#### Market Data Collection (2__BulkPriceDownloader.py)
- Two operational modes:
  - `ColdStart`: Initial bulk download of all historical data
  - `RefreshMode`: Daily updates for existing symbols
- Configurable historical data depth (default: 2022-01-01)
- Robust error handling and rate limiting
- Data quality validation:
  - Minimum trading days requirement
  - Price consistency checks
  - Volume validation
- Output: Individual ticker parquet files in PriceData directory

#### Asset Derisking (AssetDerisker.ipynb)
- K-means clustering for asset grouping
- Implementation based on "Advances in Active Portfolio Management"
- Cross-asset correlation analysis
- Group identification for interchangeable assets
- Output: Correlations.parquet for portfolio diversification

### 2. Feature Engineering

#### Technical Indicator Generation (3__Indicators.py)
- Over 50 technical indicators including:
  - Traditional indicators (MA, RSI, MACD)
  - Volatility measures
  - Volume analysis
  - Price pattern recognition
- Kalman filter trend analysis
- Custom momentum indicators
- Data cleaning and normalization
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
- Integration with asset grouping from Correlations.parquet

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

### Advanced Analytics
- Genetic Programming for feature discovery
- Asset clustering for risk management
- Custom technical indicators normal/genetic
- Complete Interactive Brokers integration (backtrader/ib_insync)
- Real-time market data processing (tws datafeeds)
- Automated execution pipeline
- Portfolio synchronization


## Technical Requirements

### Hardware
- Recommended: 32GB+ RAM for full dataset processing
- Multi-core CPU for parallel processing
- SSD storage for data management

### Software
- Python 3.8+
- Interactive Brokers TWS or IB Gateway
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - gplearn/or my custom genetric programming lib
  - backtrader
  - ib_insync
  - pyarrow


## Usage

### Pipeline Execution
1. Run `1__TickerDownloader.py` to get latest SEC data
2. Execute `2__BulkPriceDownloader.py` with appropriate mode
3. Run `AssetDerisker.ipynb` for correlation analysis
4. Process indicators with `3__Indicators.py`
5. Generate predictions using `4__Predictor.py`
6. Backtest strategies with `5__NightlyBroker.py`
7. Deploy live trading with `6__DailyBroker.py`


## Acknowledgments

This project combines modern portfolio theory with machine learning and genetic programming techniques. Special thanks to the Interactive Brokers API team and the open-source community.