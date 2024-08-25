import os
import sys
import unittest
from collections import deque
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

# Add the 'src' directory to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from trading_algo import (
    ARBITRARY_HIGH_PRICE,
    ARBITRARY_LOW_PRICE,
    BollingBandParameters,
    CreateOrdersAlgo,
    DashBoardData,
    ExecutionReport,
    MaxDrawDownInfo,
    MetricsCalculator,
    MovingAveragesParameter,
    Order,
    OrderBook,
    OrderMatchingAlgo,
    Side,
    SignalGenerator,
    TIME_FORMAT_STRING,
    TRADE_HISTORY_COLUMNS,
    Trade,
    TradingAlgo,
    TradingSignal,
)


# Unit tests for MovingAveragesParameter
class TestMovingAveragesParameter(unittest.TestCase):
    def test_validate_params_valid(self):
        valid_params = [(5, 10, 20), (3, 7, 14), (10, 20, 40), (1, 2, 3)]
        for short, medium, long in valid_params:
            try:
                MovingAveragesParameter(short, medium, long)
            except ValueError:
                self.fail(
                    f"MovingAveragesParameter({short}, {medium}, {long}) raised ValueError unexpectedly!"
                )

    def test_validate_params_invalid(self):
        with self.assertRaises(ValueError):
            MovingAveragesParameter(10, 5, 20)  # Invalid order


# Unit tests for SignalGenerator
class TestSignalGenerator(unittest.TestCase):
    def setUp(self):
        self.signal_generator = SignalGenerator()
        self.df_prices = pd.DataFrame(
            {"Price": [100, 102, 101, 103, 105, 104, 106, 107, 108, 110]}
        )

    def test_calculate_moving_averages_signal_hold_insufficient_data(self):
        params = MovingAveragesParameter(3, 5, 10)
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        df_prices = pd.DataFrame({"Price": prices})
        signal = self.signal_generator.calculate_moving_averages_signal(
            df_prices, params
        )
        self.assertEqual(signal, TradingSignal.BUY)

    def test_calculate_moving_averages_signal_buy(self):
        params = MovingAveragesParameter(3, 5, 10)
        signal = self.signal_generator.calculate_moving_averages_signal(
            self.df_prices, params
        )
        self.assertEqual(signal, TradingSignal.BUY)

    def test_calculate_bollinger_bands_signal_sell(self):
        params = BollingBandParameters(3, 2)
        self.df_prices = pd.DataFrame(
            {"Price": [110, 108, 107, 106, 105, 103, 102, 101, 100, 99]}
        )
        signal = self.signal_generator.calculate_bollinger_bands_signal(
            self.df_prices, params, 110
        )
        self.assertEqual(signal, TradingSignal.SELL)

    def test_calculate_bollinger_bands_signal_buy(self):
        params = BollingBandParameters(3, 2)
        self.df_prices = pd.DataFrame(
            {"Price": [90, 92, 93, 94, 95, 97, 98, 99, 100, 101]}
        )
        signal = self.signal_generator.calculate_bollinger_bands_signal(
            self.df_prices, params, 90
        )
        self.assertEqual(signal, TradingSignal.BUY)

    def test_calculate_bollinger_bands_signal_hold(self):
        params = BollingBandParameters(3, 2)
        self.df_prices = pd.DataFrame(
            {"Price": [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]}
        )
        signal = self.signal_generator.calculate_bollinger_bands_signal(
            self.df_prices, params, 100
        )
        self.assertEqual(signal, TradingSignal.HOLD)


# Unit tests for Order and OrderBook
class TestOrder(unittest.TestCase):
    def test_order_initialization(self):
        order = Order(datetime.now(), 100.0, 10, Side.BUY)
        self.assertEqual(order.limit_price, 100.0)
        self.assertEqual(order.order_size, 10)
        self.assertEqual(order.side, Side.BUY)


class TestOrderBook(unittest.TestCase):
    def setUp(self):
        self.order_book = OrderBook()

    def test_get_orderbook_dataframe(self):
        order1 = Order(datetime.now(), 100.0, 10, Side.BUY)
        order2 = Order(datetime.now(), 101.0, 20, Side.SELL)
        self.order_book.add_order(order1)
        self.order_book.add_order(order2)
        df = self.order_book.get_orderbook_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)

    def test_add_order(self):
        order = Order(datetime.now(), 100.0, 10, Side.BUY)
        self.order_book.add_order(order)
        self.assertEqual(len(self.order_book.get_book()), 1)
        self.assertEqual(self.order_book.get_book()[0], order)

    def test_sort_orders(self):
        time1 = datetime.now()
        time2 = time1 + timedelta(seconds=10)
        order1 = Order(time2, 101.0, 20, Side.BUY)
        order2 = Order(time1, 100.0, 10, Side.BUY)
        self.order_book.add_order(order1)
        self.order_book.add_order(order2)
        sorted_orders = self.order_book.get_book()
        self.assertEqual(sorted_orders[0], order2)
        self.assertEqual(sorted_orders[1], order1)


# Unit tests for CreateOrdersAlgo
class TestCreateOrdersAlgo(unittest.TestCase):
    def setUp(self):
        self.algo = CreateOrdersAlgo(
            trading_signal=TradingSignal.BUY,
            stoploss_signal=TradingSignal.HOLD,
            limit_order_pct=0.5,
            millisec_execution_delay=timedelta(milliseconds=100),
            open_pos=10,
            max_risk=0.02,
            mid_price=100.0,
            current_book_value=1_000_000,
            current_time=datetime.now(),
        )

    def test_create_limit_order(self):
        order = self.algo.create_limit_order()
        self.assertEqual(round(order.limit_price, 2), round(100.5, 2))
        self.assertEqual(order.order_size, 190)
        self.assertEqual(order.side, TradingSignal.BUY)

    def test_create_stoploss_order(self):
        self.algo.stoploss_signal = TradingSignal.SELL
        order = self.algo.create_stoploss_order()
        self.assertEqual(order.limit_price, ARBITRARY_LOW_PRICE)
        self.assertEqual(order.order_size, -10)
        self.assertEqual(order.side, TradingSignal.SELL)


# Unit tests for OrderMatchingAlgo
class TestOrderMatchingAlgo(unittest.TestCase):
    def setUp(self):
        self.order_book = OrderBook()
        self.trade_history = deque()
        self.cash_util = 100000.0
        self.open_pos = 0
        self.contracts_traded = 0
        self.algo = OrderMatchingAlgo()
        self.current_time = datetime.now()

    def test_successful_order_execution(self):
        buy_order = Order(self.current_time, 100.0, 10, 1)
        self.order_book.add_order(buy_order)

        report = self.algo.execute_orders(
            self.order_book,
            bid_size=10,
            bid_price=100.0,
            ask_size=10,
            ask_price=100.0,
            current_time=self.current_time,
            trade_history=self.trade_history,
            cash_util=self.cash_util,
            execution_costs_per_contract=1,
            open_pos=self.open_pos,
        )

        self.assertEqual(report.open_pos, 10)
        self.assertEqual(len(report.trade_history), 1)
        self.assertEqual(len(report.order_book.get_book()), 0)

    def test_no_orders_executed_if_bid_ask_zero(self):
        sell_order = Order(self.current_time, 100.0, -10, Side.SELL)
        self.order_book.add_order(sell_order)
        report = self.algo.execute_orders(
            self.order_book,
            bid_size=0,
            bid_price=100,
            ask_size=10,
            ask_price=100,
            current_time=self.current_time,
            trade_history=self.trade_history,
            cash_util=self.cash_util,
            execution_costs_per_contract=1,
            open_pos=self.open_pos,
        )
        self.assertEqual(report.cash_util, self.cash_util)
        self.assertEqual(report.open_pos, self.open_pos)
        self.assertEqual(len(report.trade_history), 0)
        self.assertEqual(len(report.order_book.get_book()), 1)


# Unit tests for MetricsCalculator
class TestMetricsCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = MetricsCalculator()
        self.current_time = datetime.now()

    def test_calculate_pnl(self):
        pnl = self.calculator.calculate_pnl(
            open_pos=10, px_mid=100.0, cash_utils=500.0, trading_costs=50.0
        )
        self.assertEqual(pnl, 1450.0)

    def test_update_max_drawdown_new_peak(self):
        current_max_drawdown = MaxDrawDownInfo(drawdown=5.0, peak=1000.0)
        new_portfolio_value = 1100.0
        updated_max_drawdown = self.calculator.update_max_drawdown(
            current_max_drawdown, new_portfolio_value
        )
        self.assertEqual(updated_max_drawdown.peak, 1100.0)
        self.assertEqual(updated_max_drawdown.drawdown, 5.0)

    def test_get_trade_history_df(self):
        trade_history = [
            Trade(10, 100, self.current_time),
            Trade(-5, 100, self.current_time),
        ]
        df = self.calculator.get_trade_history_df(trade_history)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()
