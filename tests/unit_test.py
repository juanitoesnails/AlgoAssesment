import unittest
import sys
import os
import pandas as pd


# Add the 'src' directory to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from trading_algo import (
    MovingAveragesParameter,
    SignalGenerator,
    TradingSignal,
    BollingBandParameters,
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


if __name__ == "__main__":
    unittest.main()
