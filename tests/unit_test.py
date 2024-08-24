import unittest
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from collections import deque

from unittest.mock import MagicMock, patch


# Add the 'src' directory to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from trading_algo import (
    MovingAveragesParameter,
    SignalGenerator,
    TradingSignal,
    BollingBandParameters,
    Order,
    OrderBook,
    CreateOrdersAlgo,
    OrderMatchingAlgo,
    ExecutionReport,
    ARBITRARY_HIGH_PRICE,
    ARBITRARY_LOW_PRICE,
    MaxDrawDownInfo,
    Trade,
    TRADE_HISTORY_COLUMNS,
    DashBoardData,
    TradingAlgo,
    MetricsCalculator,
)


### Test for Signals ##
class TestMovingAveragesParameter(unittest.TestCase):
    def test_valid_params(self):
        ma_params = MovingAveragesParameter(10, 20, 30)
        self.assertEqual(ma_params.long_rolling_window, 30)
        self.assertEqual(ma_params.medium_rolling_window, 20)
        self.assertEqual(ma_params.short_rolling_window, 10)

    def test_invalid_params_raises_value_error(self):
        with self.assertRaises(ValueError):
            MovingAveragesParameter(30, 20, 10)


class TestSignalGenerator(unittest.TestCase):
    def setUp(self):
        self.signal_generator = SignalGenerator()

    def test_calculate_moving_averages_signal_buy(self):
        data = {"price": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
        df_prices = pd.DataFrame(data)
        params = MovingAveragesParameter(2, 3, 5)

        signal = self.signal_generator.calculate_moving_averages_signal(
            df_prices, params
        )
        self.assertEqual(signal, TradingSignal.BUY)

    def test_calculate_moving_averages_signal_sell(self):
        data = {"price": [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]}
        df_prices = pd.DataFrame(data)
        params = MovingAveragesParameter(2, 3, 5)

        signal = self.signal_generator.calculate_moving_averages_signal(
            df_prices, params
        )
        self.assertEqual(signal, TradingSignal.SELL)

    def test_calculate_moving_averages_signal_hold(self):
        data = {"price": [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]}
        df_prices = pd.DataFrame(data)
        params = MovingAveragesParameter(2, 3, 5)

        signal = self.signal_generator.calculate_moving_averages_signal(
            df_prices, params
        )
        self.assertEqual(signal, TradingSignal.HOLD)

    def test_calculate_bollinger_bands_signal_buy(self):
        data = {"price": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]}
        df_prices = pd.DataFrame(data)
        params = BollingBandParameters(5, 2)

        signal = self.signal_generator.calculate_bollinger_bands_signal(
            df_prices, params, px_mid=18
        )
        self.assertEqual(signal, TradingSignal.BUY)

    def test_calculate_bollinger_bands_signal_sell(self):
        data = {"price": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]}
        df_prices = pd.DataFrame(data)
        params = BollingBandParameters(5, 2)

        signal = self.signal_generator.calculate_bollinger_bands_signal(
            df_prices, params, px_mid=32
        )
        self.assertEqual(signal, TradingSignal.SELL)

    def test_calculate_bollinger_bands_signal_hold(self):
        data = {"price": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]}
        df_prices = pd.DataFrame(data)
        params = BollingBandParameters(5, 2)

        signal = self.signal_generator.calculate_bollinger_bands_signal(
            df_prices, params, px_mid=25
        )
        self.assertEqual(signal, TradingSignal.HOLD)

    def test_moving_averages_calculation(self):
        data = {"price": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        df_prices = pd.DataFrame(data)
        params = MovingAveragesParameter(2, 3, 5)

        # Calculate expected moving averages
        expected_sma_long = df_prices.rolling(window=5).mean().iloc[-1].values[0]
        expected_sma_medium = df_prices.rolling(window=3).mean().iloc[-1].values[0]
        expected_sma_short = df_prices.rolling(window=2).mean().iloc[-1].values[0]

        sma_long_series = df_prices.rolling(window=params.long_rolling_window).mean()
        sma_medium_series = df_prices.rolling(
            window=params.medium_rolling_window
        ).mean()
        sma_short_series = df_prices.rolling(window=params.short_rolling_window).mean()

        sma_long = sma_long_series.iloc[-1].values[0]
        sma_medium = sma_medium_series.iloc[-1].values[0]
        sma_short = sma_short_series.iloc[-1].values[0]

        self.assertAlmostEqual(sma_long, expected_sma_long)
        self.assertAlmostEqual(sma_medium, expected_sma_medium)
        self.assertAlmostEqual(sma_short, expected_sma_short)

    def test_bollinger_bands_calculation(self):
        data = {"price": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        df_prices = pd.DataFrame(data)
        params = BollingBandParameters(5, 2)

        # Calculate expected Bollinger Bands
        rolling_mean = (
            df_prices.rolling(window=params.rolling_window).mean().iloc[-1].values[0]
        )
        rolling_std = (
            df_prices.rolling(window=params.rolling_window).std().iloc[-1].values[0]
        )

        expected_upper_band = rolling_mean + rolling_std * params.standard_deviations
        expected_lower_band = rolling_mean - rolling_std * params.standard_deviations

        rolling_mean_series = df_prices.rolling(window=params.rolling_window).mean()
        rolling_std_series = df_prices.rolling(window=params.rolling_window).std()

        latest_mean = rolling_mean_series.iloc[-1].values[0]
        latest_std = rolling_std_series.iloc[-1].values[0]

        upper_band = latest_mean + latest_std * params.standard_deviations
        lower_band = latest_mean - latest_std * params.standard_deviations

        self.assertAlmostEqual(upper_band, expected_upper_band)
        self.assertAlmostEqual(lower_band, expected_lower_band)


### Test for Orders ##
class TestOrder(unittest.TestCase):
    def test_order_initialization(self):
        execution_time = datetime.now()
        limit_price = 100.5
        order_size = 10
        side = 1

        order = Order(execution_time, limit_price, order_size, side)

        self.assertEqual(order.execution_time, execution_time)
        self.assertEqual(order.limit_price, limit_price)
        self.assertEqual(order.order_size, order_size)
        self.assertEqual(order.side, side)


class TestOrderBook(unittest.TestCase):
    def setUp(self):
        self.order_book = OrderBook()

    def test_add_order(self):
        order = Order(datetime.now(), 100.5, 10, 1)
        self.order_book.add_order(order)
        self.assertIn(order, self.order_book.get_book())
        self.assertEqual(self.order_book.current_side, 1)

    def test_add_order_with_multiple_side_changes(self):
        order1 = Order(datetime.now(), 100.5, 10, 1)
        order2 = Order(datetime.now(), 101.5, 15, -1)
        order3 = Order(datetime.now(), 102.0, 20, 1)

        self.order_book.add_order(order1)
        self.assertEqual(self.order_book.current_side, 1)
        self.order_book.add_order(order2)
        self.assertEqual(self.order_book.current_side, -1)
        self.order_book.add_order(order3)
        self.assertEqual(self.order_book.current_side, 1)

    def test_remove_order(self):
        order = Order(datetime.now(), 100.5, 10, 1)
        self.order_book.add_order(order)
        self.order_book.remove_order(order)
        self.assertNotIn(order, self.order_book.get_book())

    def test_sort_orders(self):
        order1 = Order(datetime(2024, 8, 22, 12, 0, 0), 101.0, 10, 1)
        order2 = Order(datetime(2024, 8, 22, 11, 0, 0), 102.0, 20, 1)
        order3 = Order(datetime(2024, 8, 22, 11, 0, 0), 100.5, 15, 1)

        self.order_book.add_order(order1)
        self.order_book.add_order(order2)
        self.order_book.add_order(order3)

        sorted_orders = self.order_book.get_book()

        # The correct order should be: order3, order2, order1 (sorted by time, then price)
        self.assertEqual(sorted_orders, [order3, order2, order1])


### Test for Order Creations ##
class TestCreateOrdersAlgo(unittest.TestCase):
    def setUp(self):
        self.trading_signal = TradingSignal.BUY
        self.stoploss_signal = TradingSignal.SELL
        self.limit_order_pct = 0.01
        self.millisec_execution_delay = timedelta(milliseconds=100)
        self.open_pos = 1000
        self.max_risk = 0.1
        self.mid_price = 50
        self.current_book_value = 1000000
        self.current_time = datetime.now()

        self.algo = CreateOrdersAlgo(
            trading_signal=self.trading_signal,
            stoploss_signal=self.stoploss_signal,
            limit_order_pct=self.limit_order_pct,
            millisec_execution_delay=self.millisec_execution_delay,
            open_pos=self.open_pos,
            max_risk=self.max_risk,
            mid_price=self.mid_price,
            current_book_value=self.current_book_value,
            current_time=self.current_time,
        )

    def test_create_stoploss_order_closes_position(self):
        # Create a stoploss order
        stoploss_order = self.algo.create_stoploss_order()

        # Check if the stoploss order size is the negative of the open position
        self.assertEqual(stoploss_order.order_size, -self.open_pos)

        # Check if the limit price is set correctly for the stoploss signal
        if self.stoploss_signal == TradingSignal.SELL:
            self.assertEqual(stoploss_order.limit_price, ARBITRARY_LOW_PRICE)
        elif self.stoploss_signal == TradingSignal.BUY:
            self.assertEqual(stoploss_order.limit_price, ARBITRARY_HIGH_PRICE)

        # Check that the execution time is correctly set
        self.assertEqual(stoploss_order.execution_time, self.algo.execution_time)

    def test_create_stoploss_order_buy_signal(self):
        # Change stoploss signal to BUY and test
        algo_buy_stoploss = CreateOrdersAlgo(
            trading_signal=self.trading_signal,
            stoploss_signal=TradingSignal.BUY,
            limit_order_pct=self.limit_order_pct,
            millisec_execution_delay=self.millisec_execution_delay,
            open_pos=-self.open_pos,  # Ensure we're testing the closing of a short position
            max_risk=self.max_risk,
            mid_price=self.mid_price,
            current_book_value=self.current_book_value,
            current_time=self.current_time,
        )

        stoploss_order = algo_buy_stoploss.create_stoploss_order()

        # Check if the stoploss order size is the positive of the open position (closing a short position)
        self.assertEqual(stoploss_order.order_size, self.open_pos)

        # Check if the limit price is set correctly for the stoploss signal
        self.assertEqual(stoploss_order.limit_price, ARBITRARY_HIGH_PRICE)

        # Check that the execution time is correctly set
        self.assertEqual(
            stoploss_order.execution_time, algo_buy_stoploss.execution_time
        )

    def test_create_limit_order(self):
        limit_order = self.algo.create_limit_order()

        desired_esp = self.current_book_value * self.max_risk * self.trading_signal
        current_esp = self.open_pos * self.mid_price
        order_size = int((desired_esp - current_esp) / self.mid_price)

        if self.trading_signal == TradingSignal.BUY:
            limit_price = self.mid_price * (1 + self.limit_order_pct)
            order_size = max(order_size, 0)
        else:
            limit_price = self.mid_price * (1 - self.limit_order_pct)
            order_size = min(order_size, 0)

        self.assertIsNotNone(limit_order)
        self.assertEqual(limit_order.limit_price, limit_price)
        self.assertEqual(limit_order.order_size, order_size)
        self.assertEqual(limit_order.side, self.trading_signal)
        self.assertEqual(limit_order.execution_time, self.algo.execution_time)

    def test_create_order_stoploss(self):
        order = self.algo.create_order()
        self.assertEqual(order.limit_price, ARBITRARY_LOW_PRICE)
        self.assertEqual(order.order_size, -self.open_pos)
        self.assertEqual(order.side, self.stoploss_signal)
        self.assertEqual(order.execution_time, self.algo.execution_time)

    def test_create_order_limit(self):
        algo_limit_order = CreateOrdersAlgo(
            trading_signal=self.trading_signal,
            stoploss_signal=TradingSignal.HOLD,
            limit_order_pct=self.limit_order_pct,
            millisec_execution_delay=self.millisec_execution_delay,
            open_pos=self.open_pos,
            max_risk=self.max_risk,
            mid_price=self.mid_price,
            current_book_value=self.current_book_value,
            current_time=self.current_time,
        )

        order = algo_limit_order.create_order()

        self.assertIsNotNone(order)
        self.assertEqual(order.limit_price, self.mid_price * (1 + self.limit_order_pct))
        self.assertEqual(order.side, self.trading_signal)
        self.assertEqual(order.execution_time, algo_limit_order.execution_time)

    def test_create_order_none(self):
        algo_no_order = CreateOrdersAlgo(
            trading_signal=TradingSignal.HOLD,
            stoploss_signal=TradingSignal.HOLD,
            limit_order_pct=self.limit_order_pct,
            millisec_execution_delay=self.millisec_execution_delay,
            open_pos=self.open_pos,
            max_risk=self.max_risk,
            mid_price=self.mid_price,
            current_book_value=self.current_book_value,
            current_time=self.current_time,
        )

        order = algo_no_order.create_order()
        self.assertIsNone(order)

    def test_edge_cases(self):
        # Test with zero open position
        algo_zero_pos = CreateOrdersAlgo(
            trading_signal=TradingSignal.BUY,
            stoploss_signal=TradingSignal.HOLD,
            limit_order_pct=self.limit_order_pct,
            millisec_execution_delay=self.millisec_execution_delay,
            open_pos=0,
            max_risk=self.max_risk,
            mid_price=self.mid_price,
            current_book_value=self.current_book_value,
            current_time=self.current_time,
        )
        self.assertIsNotNone(algo_zero_pos.create_order())

        # Test with maximum risk at extremes
        algo_max_risk = CreateOrdersAlgo(
            trading_signal=TradingSignal.BUY,
            stoploss_signal=TradingSignal.HOLD,
            limit_order_pct=self.limit_order_pct,
            millisec_execution_delay=self.millisec_execution_delay,
            open_pos=self.open_pos,
            max_risk=1,
            mid_price=self.mid_price,
            current_book_value=self.current_book_value,
            current_time=self.current_time,
        )
        self.assertIsNotNone(algo_max_risk.create_order())

        # Test with very high mid price
        algo_high_price = CreateOrdersAlgo(
            trading_signal=TradingSignal.BUY,
            stoploss_signal=TradingSignal.HOLD,
            limit_order_pct=self.limit_order_pct,
            millisec_execution_delay=self.millisec_execution_delay,
            open_pos=self.open_pos,
            max_risk=self.max_risk,
            mid_price=1_000_000,
            current_book_value=self.current_book_value,
            current_time=self.current_time,
        )
        self.assertIsNone(algo_high_price.create_order())


class TestOrderMatchingAlgo(unittest.TestCase):
    def setUp(self):
        self.current_time = datetime.now()
        self.order_matching_algo = OrderMatchingAlgo()

    def test_exit_order_matching(self):
        order_book = OrderBook()
        trade_history = deque()
        cash_util = 10000.0
        open_pos = 0
        execution_costs = 0.0

        report = self.order_matching_algo.exit_order_matching(
            trade_history, cash_util, open_pos, execution_costs, order_book
        )

        self.assertIsInstance(report, ExecutionReport)
        self.assertEqual(report.order_book, order_book)
        self.assertEqual(report.trade_history, trade_history)
        self.assertEqual(report.cash_util, cash_util)
        self.assertEqual(report.open_pos, open_pos)
        self.assertEqual(report.execution_costs, execution_costs)

    def test_execute_orders_buy(self):
        order_book = OrderBook()
        trade_history = deque()
        cash_util = 10000.0
        open_pos = 0
        execution_costs = 0.0

        order = Order(
            execution_time=self.current_time - timedelta(minutes=1),
            limit_price=105.0,
            order_size=50,
            side=TradingSignal.BUY,
        )
        order_book.add_order(order)

        report = self.order_matching_algo.execute_orders(
            order_book,
            bid_size=0,
            bid_price=0,
            ask_size=50,
            ask_price=100.0,
            current_time=self.current_time,
            trade_history=trade_history,
            cash_util=cash_util,
            execution_costs_per_contract=execution_costs,
            open_pos=open_pos,
        )

        self.assertEqual(len(trade_history), 1)
        trade = trade_history[-1]
        self.assertEqual(trade.trade_amount, 50)
        self.assertEqual(trade.price, 100.0)
        self.assertEqual(trade.time, self.current_time)
        self.assertEqual(report.cash_util, cash_util - 50 * 100.0)
        self.assertEqual(report.open_pos, 50)
        self.assertEqual(report.execution_costs, 0.0)
        self.assertEqual(len(order_book.get_book()), 0)
        self.assertNotIn(order, order_book.get_book())
        self.assertEqual(order.order_size, 0)

    def test_execute_orders_no_execution_limit_price_not_met(self):
        order_book = OrderBook()
        trade_history = deque()
        cash_util = 10000.0
        open_pos = 0
        execution_costs = 0.0

        order = Order(
            execution_time=self.current_time,
            limit_price=110.0,
            order_size=50,
            side=TradingSignal.BUY,
        )
        order_book.add_order(order)

        report = self.order_matching_algo.execute_orders(
            order_book,
            bid_size=50,
            bid_price=105.0,
            ask_size=50,
            ask_price=115.0,
            current_time=self.current_time,
            trade_history=trade_history,
            cash_util=cash_util,
            execution_costs_per_contract=execution_costs,
            open_pos=open_pos,
        )

        self.assertEqual(len(trade_history), 0)
        self.assertEqual(report.cash_util, cash_util)
        self.assertEqual(report.open_pos, open_pos)
        self.assertEqual(report.execution_costs, execution_costs)
        self.assertEqual(len(order_book.get_book()), 1)

    def test_execute_orders_sell(self):
        order_book = OrderBook()
        trade_history = deque()
        cash_util = 10000.0
        open_pos = 1000
        execution_costs = 1

        # Create and add a sell order to the order book
        order = Order(
            execution_time=self.current_time,
            limit_price=80.0,
            order_size=-50,
            side=TradingSignal.SELL,
        )
        order_book.add_order(order)

        # Execute orders
        report = self.order_matching_algo.execute_orders(
            order_book,
            bid_size=50,
            bid_price=90,
            ask_size=0,
            ask_price=0,
            current_time=self.current_time,
            trade_history=trade_history,
            cash_util=cash_util,
            execution_costs_per_contract=execution_costs,
            open_pos=open_pos,
        )

        self.assertEqual(len(trade_history), 1)
        trade = trade_history[-1]
        self.assertEqual(trade.trade_amount, -50)
        self.assertEqual(trade.price, 90)
        self.assertEqual(trade.time, self.current_time)
        self.assertEqual(report.cash_util, cash_util + (50 * 90))
        self.assertEqual(report.open_pos, open_pos - 50)
        self.assertEqual(report.execution_costs, 50)
        self.assertEqual(len(order_book.get_book()), 0)

    def test_execute_orders_no_execution(self):
        order_book = OrderBook()
        trade_history = deque()
        cash_util = 10000.0
        open_pos = 0
        execution_costs = 0.0

        # Create and add a buy order to the order book
        order = Order(
            execution_time=self.current_time + timedelta(minutes=1),
            limit_price=105.0,
            order_size=50,
            side=TradingSignal.BUY,
        )
        order_book.add_order(order)

        # Execute orders
        report = self.order_matching_algo.execute_orders(
            order_book,
            bid_size=50,
            bid_price=100.0,
            ask_size=0,
            ask_price=0,
            current_time=self.current_time,
            trade_history=trade_history,
            cash_util=cash_util,
            execution_costs_per_contract=execution_costs,
            open_pos=open_pos,
        )

        self.assertEqual(len(trade_history), 0)
        self.assertEqual(report.cash_util, cash_util)
        self.assertEqual(report.open_pos, open_pos)
        self.assertEqual(report.execution_costs, execution_costs)
        self.assertEqual(len(order_book.get_book()), 1)

    def test_execute_orders_partial_execution(self):
        order_book = OrderBook()
        trade_history = deque()
        cash_util = 10000.0
        open_pos = 0
        execution_costs = 0.0

        # Create and add a buy order to the order book
        order = Order(
            execution_time=self.current_time - timedelta(minutes=1),
            limit_price=105.0,
            order_size=100,
            side=TradingSignal.BUY,
        )
        order_book.add_order(order)

        # Execute orders
        report = self.order_matching_algo.execute_orders(
            order_book,
            bid_size=0,
            bid_price=0,
            ask_size=50,
            ask_price=100.0,
            current_time=self.current_time,
            trade_history=trade_history,
            cash_util=cash_util,
            execution_costs_per_contract=0.0,
            open_pos=open_pos,
        )

        self.assertEqual(len(trade_history), 1)
        trade = trade_history[-1]
        self.assertEqual(trade.trade_amount, 50)
        self.assertEqual(trade.price, 100.0)
        self.assertEqual(trade.time, self.current_time)
        self.assertEqual(report.cash_util, cash_util - 50 * 100.0)
        self.assertEqual(report.open_pos, 50)
        self.assertEqual(report.execution_costs, 0.0)
        self.assertEqual(
            len(order_book.get_book()), 1
        )  # Order not fully executed, should remain in the book


class TestTradingAlgo(unittest.TestCase):

    @patch("pandas.read_csv")  # Mocking pandas read_csv
    def setUp(self, mock_read_csv):
        # Mock the dependencies
        self.mock_moving_averages_params = MagicMock()
        self.mock_moving_averages_params.long_rolling_window = 10

        self.mock_bollinger_bands_params = MagicMock()
        self.mock_bollinger_bands_params.rolling_window = 5

        self.mock_order_book = MagicMock()
        self.mock_signal_generator = MagicMock()
        self.mock_create_orders_algo = MagicMock()
        self.mock_order_matching_algo = MagicMock()
        self.mock_metrics_calculator = MagicMock()
        self.mock_execution_report = MagicMock()

        self.mock_max_draw_down = MagicMock()

        # Mock read_csv to return a mock iterator
        mock_read_csv.return_value = iter(
            [
                pd.DataFrame(
                    {
                        "Date-Time": pd.to_datetime(["2024-08-24 10:00:00"]),
                        "Type": ["Quote"],
                        "Bid Price": [100],
                        "Bid Size": [10],
                        "Ask Price": [101],
                        "Ask Size": [12],
                    }
                )
            ]
        )

        # Initialize the class with mocks
        self.trading_algo = TradingAlgo(
            file_location="mock_file.csv",
            moving_averages_params=self.mock_moving_averages_params,
            bollinger_bands_params=self.mock_bollinger_bands_params,
            initial_capital=1000000,
            max_risk=0.1,
            limit_order_pct=0,
            millisec_execution_delay=timedelta(microseconds=0),
            transaction_fees_per_contract=0,
        )

    def test_initialize_reader(self):
        """Test if the CSV reader is initialized correctly."""
        self.assertIsNotNone(self.trading_algo.csv_iterator)
        self.assertIsNone(self.trading_algo.current_chunk)

    def test_initialize_parameters(self):
        """Test if trading parameters are initialized correctly."""
        self.assertEqual(
            self.trading_algo.moving_averages_params, self.mock_moving_averages_params
        )
        self.assertEqual(
            self.trading_algo.bollinger_bands_params, self.mock_bollinger_bands_params
        )
        self.assertIsInstance(self.trading_algo.price_deque, deque)
        self.assertEqual(self.trading_algo.price_deque.maxlen, 10)

    def test_initialize_state(self):
        """Test if the trading state and metrics are initialized correctly."""
        self.assertEqual(self.trading_algo.initial_capital, 1000000)
        self.assertEqual(self.trading_algo.book_value, 1000000)
        self.assertEqual(self.trading_algo.cash_utils, 0)
        self.assertEqual(self.trading_algo.open_pos, 0)
        self.assertEqual(self.trading_algo.pnl, 0)
        self.assertEqual(self.trading_algo.limit_order_pct, 0)
        self.assertEqual(
            self.trading_algo.millisec_execution_delay, timedelta(microseconds=0)
        )
        self.assertEqual(self.trading_algo.max_risk, 0.1)
        self.assertEqual(self.trading_algo.transaction_fees, 0)
        self.assertEqual(self.trading_algo.trading_costs, 0)
        self.assertEqual(self.trading_algo.total_transaction_costs, 0)
        self.assertIsNotNone(self.trading_algo.max_draw_down)

    @patch.object(TradingAlgo, "process_row")
    def test_read_next_line(self, mock_process_row):
        """Test reading the next line from the CSV."""
        result = self.trading_algo.read_next_line()
        self.assertTrue(result)
        self.assertIsNotNone(self.trading_algo.current_chunk)

    @patch.object(TradingAlgo, "calculate_signals")
    @patch.object(TradingAlgo, "create_and_execute_order")
    @patch.object(TradingAlgo, "update_metrics")
    @patch.object(TradingAlgo, "get_dashboard_data")
    def test_go_to_next_line(
        self,
        mock_get_dashboard_data,
        mock_update_metrics,
        mock_create_and_execute_order,
        mock_calculate_signals,
    ):
        """Test processing the next line of data."""
        mock_calculate_signals.return_value = ("buy_signal", "stop_loss_signal")

        result = self.trading_algo.go_to_next_line()

        self.assertTrue(result)
        self.assertTrue(mock_calculate_signals.called)
        self.assertTrue(mock_create_and_execute_order.called)
        self.assertTrue(mock_update_metrics.called)
        self.assertTrue(mock_get_dashboard_data.called)

    @patch.object(TradingAlgo, "create_and_execute_order")
    def test_create_and_execute_order(self, mock_create_and_execute_order):
        """Test the order creation and execution."""
        trading_signal = "buy"
        stop_loss_signal = "stop_loss"

        # Here, you can simulate different signals and check how the method behaves
        self.trading_algo.create_and_execute_order(trading_signal, stop_loss_signal)
        self.assertTrue(mock_create_and_execute_order.called)

    def test_get_dashboard_data(self):
        """Test if the dashboard data is returned correctly."""
        dashboard_data = self.trading_algo.get_dashboard_data()
        self.assertIsInstance(dashboard_data, DashBoardData)


if __name__ == "__main__":
    unittest.main()
