
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import yfinance as yf

from typing import List

import streamlit as st

import sys

from datetime import datetime

import matplotlib.pyplot as plt



def download_ticker_data_from_yahoo(ticker, period: str) -> pd.DataFrame:

    stock_info = yf.Ticker(ticker)

    data = stock_info.history(period=period)

    data = data.reset_index()

    return data



def calculate_30_d_volatility(trading_data: pd.DataFrame) -> pd.DataFrame:

    trading_data["Daily_Return"] = trading_data["PNL"].pct_change()


    # Annualizing volatility (252 trading days)

    trading_data["Volatility"] = trading_data["Daily_Return"].rolling(window=30).std() * np.sqrt(

        252

    )

    return trading_data



def calculate_max_drawdown(trading_data: pd.DataFrame, initial_capital) -> pd.DataFrame:

    trading_data["Portfolio_value"] = initial_capital + trading_data["PNL"]


    # Fll NaNs with 0 for the first entry

    trading_data["daily_returns"] = trading_data["Portfolio_value"].pct_change().fillna(0)

    trading_data["cum_returns"] = (1 + trading_data["daily_returns"]).cumprod()

    cum_max = trading_data["cum_returns"].cummax()


    # Prevent division by zero for calculating drawdown

    trading_data["Max_Drawdown"] = (trading_data["cum_returns"] - cum_max) / cum_max.replace(

        0, np.nan

    )


    return trading_data



class TradingSignals:

    def __init__(self, data: pd.DataFrame, strategies: list):

        """

        Initialize the TradingSignals object with the historical stock data and a list of strategies.


        Args:

            data (pd.DataFrame): A DataFrame containing stock data (e.g., 'Close', 'High', 'Low', 'Volume').

            strategies (list): A list of strategies to use (e.g., ["SMA", "RSI", "MACD"]).

        """

        self.data = data.copy()

        self.strategies = strategies


        # Set signal columns to 0

        self.signal_columns = ["N_Buy", "N_Sell"]

        self.data[self.signal_columns] = 0


    def _generate_signals(self, buy_condition: pd.Series, sell_condition: pd.Series):

        buy_signals = buy_condition.astype(int)

        sell_signals = sell_condition.astype(int)


        self.data["N_Buy"] += buy_signals

        self.data["N_Sell"] += sell_signals


    def sma_crossover(self, short_window: int = 50, long_window: int = 200):

        sma_short = self.data["Close"].rolling(window=short_window).mean()

        sma_long = self.data["Close"].rolling(window=long_window).mean()


        buy_condition = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))

        sell_condition = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))


        self._generate_signals(buy_condition, sell_condition)


    def ema_crossover(self, short_period: int = 12, long_period: int = 26):

        ema_short = self.data["Close"].ewm(span=short_period, adjust=False).mean()

        ema_long = self.data["Close"].ewm(span=long_period, adjust=False).mean()


        buy_condition = (ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1))

        sell_condition = (ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1))


        self._generate_signals(buy_condition, sell_condition)


    def rsi(self, window: int = 14, overbought: int = 70, oversold: int = 30):

        delta = self.data["Close"].diff()

        gain = delta.where(delta > 0, 0).rolling(window=window).mean()

        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()


        rs = gain / loss

        rsi = 100 - (100 / (1 + rs))


        buy_condition = (rsi < oversold) & (rsi.shift(1) >= oversold)

        sell_condition = (rsi > overbought) & (rsi.shift(1) <= overbought)


        self._generate_signals(buy_condition, sell_condition)


    def macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):

        macd = (

            self.data["Close"].ewm(span=fast_period, adjust=False).mean()

            - self.data["Close"].ewm(span=slow_period, adjust=False).mean()

        )

        signal_line = macd.ewm(span=signal_period, adjust=False).mean()


        buy_condition = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))

        sell_condition = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))


        self._generate_signals(buy_condition, sell_condition)


    def bollinger_bands(self, window: int = 20, std_dev: int = 2):

        sma = self.data["Close"].rolling(window=window).mean()

        std = self.data["Close"].rolling(window=window).std()


        upper_band = sma + (std_dev * std)

        lower_band = sma - (std_dev * std)


        buy_condition = (self.data["Close"] < lower_band) & (

            self.data["Close"].shift(1) >= lower_band

        )

        sell_condition = (self.data["Close"] > upper_band) & (

            self.data["Close"].shift(1) <= upper_band

        )


        self._generate_signals(buy_condition, sell_condition)


    def stochastic_oscillator(self, window: int = 14, overbought: int = 80, oversold: int = 20):

        lowest_low = self.data["Low"].rolling(window=window).min()

        highest_high = self.data["High"].rolling(window=window).max()


        stoch = 100 * (self.data["Close"] - lowest_low) / (highest_high - lowest_low)


        buy_condition = (stoch < oversold) & (stoch.shift(1) >= oversold)

        sell_condition = (stoch > overbought) & (stoch.shift(1) <= overbought)


        self._generate_signals(buy_condition, sell_condition)


    def atr(self, window: int = 14):

        high_low = self.data["High"] - self.data["Low"]

        high_close = (self.data["High"] - self.data["Close"].shift(1)).abs()

        low_close = (self.data["Low"] - self.data["Close"].shift(1)).abs()


        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr = tr.rolling(window=window).mean()


        return atr


    def cci(self, window: int = 14, constant: int = 0.015):

        typical_price = (self.data["High"] + self.data["Low"] + self.data["Close"]) / 3

        sma = typical_price.rolling(window=window).mean()

        mad = typical_price.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean())


        cci = (typical_price - sma) / (constant * mad)


        buy_condition = (cci < -100) & (cci.shift(1) >= -100)

        sell_condition = (cci > 100) & (cci.shift(1) <= 100)


        self._generate_signals(buy_condition, sell_condition)


    def obv(self):

        obv = [0]

        for i in range(1, len(self.data)):

            if self.data["Close"].iloc[i] > self.data["Close"].iloc[i - 1]:

                obv.append(obv[-1] + self.data["Volume"].iloc[i])

            elif self.data["Close"].iloc[i] < self.data["Close"].iloc[i - 1]:

                obv.append(obv[-1] - self.data["Volume"].iloc[i])

            else:

                obv.append(obv[-1])

        self.data["OBV"] = obv


    def parabolic_sar(self, acceleration_factor: float = 0.02, max_acceleration: float = 0.2):

        # Initialize SAR, trend and acceleration factor

        self.data["SAR"] = self.data["Close"]

        uptrend = True

        af = acceleration_factor

        ep = self.data["High"].iloc[0]

        sar = self.data["Low"].iloc[0]


        for i in range(1, len(self.data)):

            if uptrend:

                sar = sar + af * (ep - sar)

                if self.data["Low"].iloc[i] < sar:

                    uptrend = False

                    sar = ep

                    ep = self.data["Low"].iloc[i]

                else:

                    ep = max(ep, self.data["High"].iloc[i])

            else:

                sar = sar + af * (ep - sar)

                if self.data["High"].iloc[i] > sar:

                    uptrend = True

                    sar = ep

                    ep = self.data["High"].iloc[i]

                else:

                    ep = min(ep, self.data["Low"].iloc[i])


            self.data["SAR"].iloc[i] = sar


        buy_condition = self.data["Close"] > self.data["SAR"]

        sell_condition = self.data["Close"] < self.data["SAR"]


        self._generate_signals(buy_condition, sell_condition)


    def calculate_signals(self, params: dict = None):

        for strategy in self.strategies:

            strategy_params = params.get(strategy, {}) if params else {}


            if strategy == "SMA":

                self.sma_crossover(**strategy_params)

            elif strategy == "EMA":

                self.ema_crossover(**strategy_params)

            elif strategy == "RSI":

                self.rsi(**strategy_params)

            elif strategy == "MACD":

                self.macd(**strategy_params)

            elif strategy == "BB":

                self.bollinger_bands(**strategy_params)

            elif strategy == "Stochastic":

                self.stochastic_oscillator(**strategy_params)

            elif strategy == "CCI":

                self.cci(**strategy_params)

            elif strategy == "OBV":

                self.obv()

            elif strategy == "SAR":

                self.parabolic_sar(**strategy_params)


        return self.data[["Date", "Open", "Close", "N_Buy", "N_Sell"]]



class Backtester:

    def __init__(

        self,

        signals_df: pd.DataFrame,

        initial_capital: float = 10_000,

        cash_reserve: float = 0.1,

        num_signals: int = 1,

    ):

        self.signals_df = signals_df

        self.cash_reserve = cash_reserve

        self.num_signals = num_signals


        self.initial_capital = initial_capital

        self.portfolio_value = initial_capital

        self.cash = initial_capital

        self.open_position = 0

        self.pnl = 0


        # Tracking performance and history

        self.pnls = []

        self.history = []


    def _calculate_signal(self, n_buy: int, n_sell: int) -> float:

        return (n_buy - n_sell) / self.num_signals


    def _calculate_trade_amt(self, avg_signal: float, close_price: float) -> int:

        capital_to_use = self.portfolio_value * abs(avg_signal)

        num_shares = int(capital_to_use / close_price)

        return num_shares


    def _buy(self, avg_signal: int, close_price: float):

        num_shares = self._calculate_trade_amt(avg_signal=avg_signal, close_price=close_price)

        self.open_position += num_shares

        self.cash -= num_shares * close_price


    def _sell(self, avg_signal: int, close_price: float):

        num_shares = self._calculate_trade_amt(avg_signal=avg_signal, close_price=close_price)

        self.open_position -= num_shares

        self.cash += num_shares * close_price


    def _calculate_pnl(self, close_price: float):

        self.portfolio_value = self.cash + self.open_position * close_price

        self.pnl = self.portfolio_value - self.initial_capital

        self.pnls.append(self.pnl)


    def simulate_trading(self) -> pd.DataFrame:

        for _, row in self.signals_df.iterrows():

            n_buy = row["N_Buy"]

            n_sell = row["N_Sell"]

            avg_signal = self._calculate_signal(n_buy, n_sell)


            close_price = row["Close"]


            if avg_signal > 0:

                self._buy(avg_signal, close_price)

            elif avg_signal < 0:

                self._sell(avg_signal, close_price)


            self._calculate_pnl(close_price)


            # Record the portfolio state

            self.history.append(

                {

                    "Date": row["Date"],

                    "Close": row["Close"],

                    "Cash": self.cash,

                    "Open Pos": self.open_position,

                    "PNL": self.pnl,

                }

            )


        return pd.DataFrame(self.history)



import streamlit as st

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np


# Streamlit Interface

st.title("Backtesting Different Trading Strategies")



sidebar_options = ["How to Use", "Backtester"]

selected_option = st.sidebar.radio("Select a Section", sidebar_options)


if selected_option == "How to Use":

    st.header("Welcome to the Trading Strategy Backtester!")

    st.write(

        """

        This app allows you to backtest various trading strategies on stock tickers, helping you evaluate the

        historical performance of different trading signals.


        ## How it Works:

        1. **Enter a Stock Ticker**: The app will download historical data for the ticker you provide (e.g., 'AAPL', 'GOOG').

        2. **Select a Time Period**: Choose the period over which you want to backtest the strategies (e.g., 1 year, 2 years, etc.).

        3. **Choose Trading Strategies**: You can select from a variety of popular trading strategies, including:

            - **SMA (Simple Moving Average)**

            - **EMA (Exponential Moving Average)**

            - **RSI (Relative Strength Index)**

            - **MACD (Moving Average Convergence Divergence)**

            - **Bollinger Bands (BB)**

            - **Stochastic Oscillator**

            - **CCI (Commodity Channel Index)**

            - **SAR (Parabolic SAR)**

        4. **Set Parameters**: Based on the strategies you select, you can adjust various parameters (e.g., window sizes, periods, thresholds) to customize the signals.

        5. **Run the Backtest**: Once all parameters are set, click "Run" to simulate the strategy over the selected time period.


        ## Results:

        - After running the backtest, the app will display performance metrics, including:

            - **Final Profit and Loss (PnL)**: How much profit or loss you would have made if you had followed the strategy.

            - **Volatility**: A measure of risk or price fluctuations during the backtesting period.

            - **Max Drawdown**: The largest peak-to-trough decline in value during the backtest.

        - The results are visualized in charts for easy analysis.


        ## Next Steps:

        1. Head over to the "Backtester" tab.

        2. Select your strategies and parameters.

        3. Hit "Run" to see the backtest results and have fun experimenting with different strategies!


        Enjoy exploring different trading strategies and see how they would have performed in the past!

        """

    )


elif selected_option == "Backtester":

    # Ticker Selection

    ticker = st.text_input("Enter Stock Ticker:", "AAPL")


    # Period Selection

    period = st.selectbox("Select Time Period to Test:", ["1y", "2y", "5y", "10y", "ytd"])


    # Strategy Selection

    strategies = st.multiselect(

        "Select Trading Strategies:", ["SMA", "EMA", "RSI", "MACD", "BB", "Stochastic", "CCI", "SAR"]

    )


    # Initial capital and cash reserve

    initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=10_000, step=100)

    cash_reserve = st.number_input("Cash Reserve (%)", min_value=0, max_value=100, value=10)



    # Function to display parameters based on selected strategy

    def display_strategy_params(strategy):

        params = {}

        if strategy == "SMA":

            params["short_window"] = st.number_input(

                "Short Window",

                min_value=5,

                max_value=50,

                value=10,

                help="Short window for the SMA calculation",

            )

            params["long_window"] = st.number_input(

                "Long Window",

                min_value=50,

                max_value=300,

                value=200,

                help="Long window for the SMA calculation",

            )


        elif strategy == "EMA":

            params["short_period"] = st.number_input(

                "Short Period",

                min_value=5,

                max_value=20,

                value=12,

                help="Short period for the EMA calculation",

            )

            params["long_period"] = st.number_input(

                "Long Period",

                min_value=20,

                max_value=200,

                value=26,

                help="Long period for the EMA calculation",

            )


        elif strategy == "RSI":

            params["window"] = st.number_input(

                "RSI Window", min_value=5, max_value=30, value=14, help="Window for the RSI calculation"

            )

            params["overbought"] = st.number_input(

                "Overbought Threshold",

                min_value=60,

                max_value=80,

                value=70,

                help="Threshold for overbought RSI level",

            )

            params["oversold"] = st.number_input(

                "Oversold Threshold",

                min_value=20,

                max_value=40,

                value=30,

                help="Threshold for oversold RSI level",

            )


        elif strategy == "MACD":

            params["fast_period"] = st.number_input(

                "Fast Period",

                min_value=5,

                max_value=20,

                value=12,

                help="Fast period for the MACD calculation",

            )

            params["slow_period"] = st.number_input(

                "Slow Period",

                min_value=20,

                max_value=50,

                value=26,

                help="Slow period for the MACD calculation",

            )

            params["signal_period"] = st.number_input(

                "Signal Period",

                min_value=5,

                max_value=15,

                value=9,

                help="Signal period for the MACD calculation",

            )


        elif strategy == "BB":

            params["window"] = st.number_input(

                "Bollinger Bands Window",

                min_value=10,

                max_value=50,

                value=20,

                help="Window for Bollinger Bands",

            )

            params["std_dev"] = st.number_input(

                "Standard Deviation",

                min_value=1,

                max_value=3,

                value=2,

                help="Standard deviation for Bollinger Bands",

            )


        elif strategy == "Stochastic":

            params["window"] = st.number_input(

                "Stochastic Window",

                min_value=5,

                max_value=20,

                value=14,

                help="Window for Stochastic Oscillator",

            )

            params["overbought"] = st.number_input(

                "Overbought Threshold",

                min_value=70,

                max_value=100,

                value=80,

                help="Overbought threshold for Stochastic",

            )

            params["oversold"] = st.number_input(

                "Oversold Threshold",

                min_value=0,

                max_value=30,

                value=20,

                help="Oversold threshold for Stochastic",

            )


        elif strategy == "CCI":

            params["window"] = st.number_input(

                "CCI Window",

                min_value=10,

                max_value=50,

                value=14,

                help="Window for the Commodity Channel Index",

            )

            params["constant"] = st.number_input(

                "Constant",

                min_value=0.01,

                max_value=0.1,

                value=0.015,

                help="Constant factor for CCI calculation",

            )


        elif strategy == "SAR":

            params["acceleration_factor"] = st.number_input(

                "Acceleration Factor",

                min_value=0.01,

                max_value=0.1,

                value=0.02,

                help="Acceleration factor for SAR",

            )

            params["max_acceleration"] = st.number_input(

                "Max Acceleration",

                min_value=0.1,

                max_value=0.3,

                value=0.2,

                help="Max acceleration for SAR",

            )


        return params



    

    if ticker and period:

        # Show config

        st.subheader("Configure Strategy Parameters")

        params_dict = {}

        for strategy in strategies:

            st.subheader(f"{strategy} Parameters:")

            params_dict[strategy] = display_strategy_params(strategy)


        # Button to run the backtest

        if st.button("Run"):

            

            data = download_ticker_data_from_yahoo(ticker, period)

            st.write(f"Showing data for {ticker} from {period}:", data.head())


            # Generate trading signals

            signal_calculator = TradingSignals(data, strategies)

            signals = signal_calculator.calculate_signals(params_dict)


            # Run the backtest with the parameters

            backtester = Backtester(

                signals_df=signals, initial_capital=initial_capital, cash_reserve=cash_reserve / 100

            )

            results = backtester.simulate_trading()


            results = calculate_30_d_volatility(results)

            results = calculate_max_drawdown(results, initial_capital)


            # Display metrics: PnL, Volatility, Max Drawdown

            st.subheader("Performance Metrics")

            final_pnl = results["PNL"].iloc[-1]

            st.write(f"Final PnL: ${final_pnl:,.2f}")


            st.write(f"Max Drawdown: {results['Max_Drawdown'].iloc[-1] * 100:.2f}%")


            # Plot PnL, Volatility, and Max Drawdown

            fig, ax = plt.subplots(3, 1, figsize=(10, 12))


            # Plot PnL

            ax[0].plot(results["Date"], results["PNL"], label="PnL", color="blue")

            ax[0].set_title("PnL over Time")

            ax[0].set_ylabel("PnL ($)")


            # Plot Volatility

            ax[1].plot(

                results["Date"], results["Volatility"], label="30 day Volatility", color="orange"

            )

            ax[1].set_title("20 Day Volatility")

            ax[1].set_ylabel("Volatility %")


            # Plot Max Drawdown

            ax[2].plot(results["Date"], results["Max_Drawdown"], label="Max Drawdown", color="red")

            ax[2].set_title("Max Drawdown over Time")

            ax[2].set_ylabel("Max Drawdown")


            st.pyplot(fig)
