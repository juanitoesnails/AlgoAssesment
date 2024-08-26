---

# Trading Algorithm Test README

## Overview

This Python project implements a trading algorithm that processes financial data from CSV files to generate and execute trading signals based on technical indicators. The algorithm calculates trading signals using moving averages and Bollinger Bands, places orders based on these signals, and manages an order book. It also tracks and reports on various performance metrics such as profit and loss (PnL), drawdown, and open positions.

The project is integrated with a dashboard, allowing users to visualize PnL and other key metrics in real-time. The dashboard is available in a Jupyter Notebook, providing an interactive interface for configuring the algorithm's parameters.

## Important Choices

The main goal was not to create an over-fitted algorithm that would simply generate PnL but rather to develop a tool that would be useful to a trader. Without access to test data, it was deemed more beneficial for a trader to visualize how their choice of parameters, execution costs, and order delays could affect their PnL.

The trading algorithm is designed to be easily expandable. Separate classes are used for the `OrderBook`, `Order Matching Algorithm`, and the signal generator. Currently, only limit and market orders are implemented, but adding a TWAP (Time-Weighted Average Price) order, for example, should be straightforward.

A Jupyter notebook was chosen to simplify installation. Typically, data would be stored in a database and sent to Grafana or Superset for visualization. Initial scripts were developed in Dash and Streamlit, but their refresh lag proved distracting. The current dashboard refreshes every 500 data points and stores 600 in memory, making it almost as responsive as a live system.

## Strategy

### Signal Generation Process

The algorithm uses two primary signals: Moving Averages and Bollinger Bands for stop-losses.

- **Moving Averages**: The trader selects three rolling windows for the moving averages. A strong downward trend, indicating a sell signal, is detected if the shortest-window moving average is lower than the medium-window moving average, and the medium-window moving average is lower than the long-window moving average. The reverse is true for a buy signal.
  
  Example:
  - `if sma_short < sma_medium < sma_long -> sell signal`
  - `if sma_short > sma_medium > sma_long -> buy signal`

- **Bollinger Bands**: Stop-losses are determined using Bollinger Bands. The user specifies the period for calculating these bands and the number of standard deviations from the mean to set them. If the price crosses the upper band while in a long position, a sell signal is generated. Conversely, if the price crosses the lower band while in a short position, a buy signal is triggered.

### Sizing and Orders

- **Precedence of Stop-Loss Signals**: A stop-loss signal, generated from Bollinger Bands, always takes precedence over a trade signal derived from moving averages.
  
- **Trade Execution**: The user specifies the percentage of the portfolio to expose at any given time. For a trade signal, the algorithm calculates the current exposure and determines how much to trade to achieve the target delta position. It then creates a limit order based on a user-defined percentage from the mid-price.

- **Order Flattening**: For a stop-loss signal, the algorithm immediately attempts to flatten the delta using a market order.

- **Execution Delay**: If a delay is specified, all orders will have an execution time set to the current time plus the specified millisecond delay.

- **Order Direction Consistency**: The algorithm strives to maintain a consistent direction in orders, meaning it will not have both buy and sell orders in the order book simultaneously. Once a sell order is added, all buy orders are canceled, and vice versa.

### Execution

Orders are filled based on price-time priority. If an order has an execution time later than the current time, it is assumed that the order has not yet been sent to the market.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/juanitoesnails/Baraktest.git
   cd trading-algo
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Download and Configure Data

1. **Download CSV Data**:
   - Obtain a CSV file containing trading data.

2. **Configure File Location**:
   - Place the downloaded CSV file in your desired directory.
   - Update the file location in the configuration file located at `etc/config.py`.
   - Refer to the sample config file for guidance.

### Step 2: Configure Algorithm Parameters

1. **Open the Dashboard**:
   - Navigate to the dashboard located at `src/dashboard/`.
   - Open the Jupyter Notebook (`dashboard.ipynb`) in this directory.

2. **Set Parameters**:
   - In the notebook, input the desired parameters for the trading algorithm, such as:
     - Moving averages (short, medium, long windows).
     - Bollinger Bands (rolling window, standard deviations).
     - Initial capital, risk levels, execution delays, etc.

3. **Run the Algorithm**:
   - Execute the notebook cells to run the trading algorithm with the configured parameters.
   - The algorithm will process the CSV data and generate trading signals.

### Step 3: Visualize Results

- The dashboard notebook will display real-time visualizations of PnL, open positions, drawdown, and other key metrics as the algorithm processes the data.

### Example Code Usage Directly from Python

If you prefer running the algorithm outside the notebook, you can use the following code snippet:

```python
from datetime import timedelta
from src.trading_algo import TradingAlgo, MovingAveragesParameter, BollingerBandParameters

# Example parameters
moving_averages_params = MovingAveragesParameter(10, 20, 50)
bollinger_bands_params = BollingerBandParameters(20, 2)

algo = TradingAlgo(
    file_location="path/to/data.csv",
    moving_averages_params=moving_averages_params,
    bollinger_bands_params=bollinger_bands_params,
    initial_capital=1000000,
    max_risk=15,
    limit_order_pct=5,
    millisec_execution_delay=timedelta(milliseconds=500),
    transaction_fees_per_contract=10,
)

while True:
    has_data, dashboard_data = algo.get_new_data()
    if not has_data:
        break
    print(dashboard_data)
```

## Classes and Methods

### Key Methods

- `calculate_moving_averages_signal`: Computes the trading signal using moving averages.
- `calculate_bollinger_bands_signal`: Computes the stop-loss signal using Bollinger Bands.
- `create_order`: Creates an order based on the current trading signal.
- `execute_orders`: Executes orders and updates trade history.
- `update_metrics`: Updates PnL, drawdown, and other metrics.

## Dependencies

- `numpy`: Numerical computations.
- `pandas`: Data manipulation and analysis.
- `logging`: Logging and debug information.
- `collections`: Efficient data structures such as `deque`.
- `Jupyter`: Interactive notebook environment for the dashboard.

---

Feel free to modify the parameters and experiment. Happy trading!

---
