import unittest
import sys
import os
import pandas as pd
from datetime import datetime

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

    def test_add_order_with_side_change(self):
        order1 = Order(datetime.now(), 100.5, 10, 1)
        order2 = Order(datetime.now(), 101.5, 15, -1)

        self.order_book.add_order(order1)
        self.order_book.add_order(order2)

        self.assertNotIn(order1, self.order_book.get_book())  # Ensure order1 is removed
        self.assertIn(order2, self.order_book.get_book())  # Ensure order2 is added
        self.assertEqual(self.order_book.current_side, -1)

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


if __name__ == "__main__":
    unittest.main()
