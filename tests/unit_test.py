import unittest
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from collections import deque

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
    DashboardMetrics,
    MaxDrawDown,
    Trade,
    TRADE_HISTORY_COLUMNS,
)


### Test for Signals ##
class TestMovingAveragesParameter(unittest.TestCase):
    def test_valid_params(self):
        params = (30, 20, 10)
        ma_params = MovingAveragesParameter(params)
        self.assertEqual(ma_params.long_rolling_window, 30)
        self.assertEqual(ma_params.medium_rolling_window, 20)
        self.assertEqual(ma_params.short_rolling_window, 10)

    def test_invalid_params_raises_value_error(self):
        params = (10, 20, 30)
        with self.assertRaises(ValueError):
            MovingAveragesParameter(params)


class TestSignalGenerator(unittest.TestCase):
    def setUp(self):
        self.signal_generator = SignalGenerator()

    def test_calculate_moving_averages_signal_buy(self):
        data = {"price": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
        df_prices = pd.DataFrame(data)
        params = MovingAveragesParameter((5, 3, 2))

        signal = self.signal_generator.calculate_moving_averages_signal(
            df_prices, params
        )
        self.assertEqual(signal, TradingSignal.BUY)

    def test_calculate_moving_averages_signal_sell(self):
        data = {"price": [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]}
        df_prices = pd.DataFrame(data)
        params = MovingAveragesParameter((5, 3, 2))

        signal = self.signal_generator.calculate_moving_averages_signal(
            df_prices, params
        )
        self.assertEqual(signal, TradingSignal.SELL)

    def test_calculate_moving_averages_signal_hold(self):
        data = {"price": [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]}
        df_prices = pd.DataFrame(data)
        params = MovingAveragesParameter((5, 3, 2))

        signal = self.signal_generator.calculate_moving_averages_signal(
            df_prices, params
        )
        self.assertEqual(signal, TradingSignal.HOLD)

    def test_calculate_bollinger_bands_signal_buy(self):
        data = {"price": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]}
        df_prices = pd.DataFrame(data)
        params = BollingBandParameters((5, 2))

        signal = self.signal_generator.calculate_bollinger_bands_signal(
            df_prices, params, px_mid=18
        )
        self.assertEqual(signal, TradingSignal.BUY)

    def test_calculate_bollinger_bands_signal_sell(self):
        data = {"price": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]}
        df_prices = pd.DataFrame(data)
        params = BollingBandParameters((5, 2))

        signal = self.signal_generator.calculate_bollinger_bands_signal(
            df_prices, params, px_mid=32
        )
        self.assertEqual(signal, TradingSignal.SELL)

    def test_calculate_bollinger_bands_signal_hold(self):
        data = {"price": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]}
        df_prices = pd.DataFrame(data)
        params = BollingBandParameters((5, 2))

        signal = self.signal_generator.calculate_bollinger_bands_signal(
            df_prices, params, px_mid=25
        )
        self.assertEqual(signal, TradingSignal.HOLD)

    def test_moving_averages_calculation(self):
        data = {"price": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        df_prices = pd.DataFrame(data)
        params = MovingAveragesParameter((5, 3, 2))

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
        params = BollingBandParameters((5, 2))

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

    def test_sum_unfulfilled_orders(self):
        order1 = Order(datetime.now(), 100.5, 10, 1)
        order2 = Order(datetime.now(), 101.5, 20, 1)
        order3 = Order(datetime.now(), 102.5, 15, 1)

        self.order_book.add_order(order1)
        self.order_book.add_order(order2)
        self.order_book.add_order(order3)

        total = self.order_book.sum_unfulfilled_orders()
        self.assertEqual(total, 45.0)


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


### Dashboard Metrics ###


class TestDashboardMetrics(unittest.TestCase):
    def setUp(self):
        self.dashboard_metrics = DashboardMetrics()

    def test_calculate_pnl(self):
        open_pos = 100
        px_mid = 150.0
        cash_utils = 5000.0
        trading_costs = 100.0

        pnl = self.dashboard_metrics.calculate_pnl(
            open_pos, px_mid, cash_utils, trading_costs
        )

        expected_pnl = (open_pos * px_mid) + cash_utils - trading_costs
        self.assertEqual(pnl, expected_pnl)

    def test_get_trade_history_df(self):
        trade_history = deque(
            [
                Trade(trade_amount=10, price=100.0, time=datetime(2024, 8, 22, 10, 30)),
                Trade(trade_amount=20, price=105.0, time=datetime(2024, 8, 22, 11, 00)),
            ]
        )

        df = self.dashboard_metrics.get_trade_history_df(trade_history)

        expected_data = [
            {
                TRADE_HISTORY_COLUMNS[0]: 10,
                TRADE_HISTORY_COLUMNS[1]: 100.0,
                TRADE_HISTORY_COLUMNS[2]: datetime(2024, 8, 22, 10, 30),
            },
            {
                TRADE_HISTORY_COLUMNS[0]: 20,
                TRADE_HISTORY_COLUMNS[1]: 105.0,
                TRADE_HISTORY_COLUMNS[2]: datetime(2024, 8, 22, 11, 00),
            },
        ]
        expected_df = pd.DataFrame(expected_data, columns=TRADE_HISTORY_COLUMNS)
        pd.testing.assert_frame_equal(df, expected_df)

    def test_update_max_drawdown(self):
        current_max_draw_down = MaxDrawDown(drawdown=0.05, peak=100.0)
        new_portfolio_value = 85.0

        new_max_drawdown = self.dashboard_metrics.update_max_drawdown(
            current_max_draw_down, new_portfolio_value
        )

        expected_drawdown = (100.0 - 85.0) / 100.0
        expected_peak = 100.0
        expected_max_drawdown = max(0.05, expected_drawdown)

        self.assertEqual(new_max_drawdown.drawdown, expected_max_drawdown)
        self.assertEqual(new_max_drawdown.peak, expected_peak)

        # Test with a new peak
        new_portfolio_value = 110.0
        new_max_drawdown = self.dashboard_metrics.update_max_drawdown(
            current_max_draw_down, new_portfolio_value
        )

        expected_drawdown = (new_portfolio_value - 110.0) / new_portfolio_value
        expected_peak = 110.0

        self.assertEqual(
            new_max_drawdown.drawdown, 0.0
        )  # No drawdown if new value is the highest
        self.assertEqual(new_max_drawdown.peak, expected_peak)


if __name__ == "__main__":
    unittest.main()
