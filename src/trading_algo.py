import pandas as pd
from datetime import datetime, timedelta
from operator import attrgetter
from enum import Enum
from collections import deque

# We will use these two for limits prices when executing market
ARBITRARY_HIGH_PRICE = 100_000_000
ARBITRARY_LOW_PRICE = 0


class Side(Enum):
    BUY = 1
    SELL = 2


class Trade:
    def __init__(self, trade_amount: int, price: float, time: datetime):
        self.trade_amount = trade_amount
        self.price = price
        self.time = time


class TradingSignal:
    BUY = 1
    HOLD = 0
    SELL = -1


# Trading Signal Objects
class BollingBandParameters:
    def __init__(self, bollinger_bands_params: tuple[int, int]) -> None:
        self.rolling_window = bollinger_bands_params[0]
        self.standard_deviations = bollinger_bands_params[1]


class MovingAveragesParameter:
    def __init__(self, moving_averages_params: tuple[int, int, int]) -> None:
        self.validate_params(moving_averages_params)
        self.long_rolling_window = moving_averages_params[0]
        self.medium_rolling_window = moving_averages_params[1]
        self.short_rolling_window = moving_averages_params[2]

    def validate_params(self, params: tuple[int, int, int]) -> None:
        if not (params[0] > params[1] > params[2]):
            raise ValueError(
                "Invalid arguments: The values must be in descending order (long > medium > short)"
            )


class SignalGenerator:
    def calculate_moving_averages_signal(
        self, df_prices: pd.DataFrame, params: MovingAveragesParameter
    ) -> int:
        long_window = params.long_rolling_window
        medium_window = params.medium_rolling_window
        short_window = params.short_rolling_window

        # Ensure there is enough data
        if len(df_prices) < long_window:
            return TradingSignal.HOLD

        # Calculate rolling means
        sma_long_series = df_prices.rolling(window=long_window).mean()
        sma_medium_series = df_prices.rolling(window=medium_window).mean()
        sma_short_series = df_prices.rolling(window=short_window).mean()

        # Get the latest scalar values
        sma_long = sma_long_series.iloc[-1].values[0]
        sma_medium = sma_medium_series.iloc[-1].values[0]
        sma_short = sma_short_series.iloc[-1].values[0]

        # Determine the trading signal
        if sma_short > sma_medium > sma_long:
            return TradingSignal.BUY
        elif sma_short < sma_medium < sma_long:
            return TradingSignal.SELL
        else:
            return TradingSignal.HOLD

    def calculate_bollinger_bands_signal(
        self, df_prices: pd.DataFrame, params: BollingBandParameters, px_mid: float
    ) -> int:
        if len(df_prices) < params.rolling_window:
            return TradingSignal.HOLD

        # Calculate rolling mean and standard deviation
        rolling_mean = df_prices.rolling(window=params.rolling_window).mean()
        rolling_std = df_prices.rolling(window=params.rolling_window).std()

        # Get the latest values of the rolling mean and standard deviation
        latest_mean = rolling_mean.iloc[-1].values[0]
        latest_std = rolling_std.iloc[-1].values[0]

        # Calculate the upper and lower bands
        upper_band = latest_mean + latest_std * params.standard_deviations
        lower_band = latest_mean - latest_std * params.standard_deviations

        # Determine the trading signal
        if px_mid > upper_band:
            return TradingSignal.SELL
        elif px_mid < lower_band:
            return TradingSignal.BUY
        else:
            return TradingSignal.HOLD


# Order_Book_Objects
class Order:
    def __init__(
        self, execution_time: datetime, limit_price: float, order_size: int, side: Side
    ):
        self.execution_time = execution_time
        self.limit_price = limit_price
        self.order_size = order_size
        self.side = side


class OrderBook:
    def __init__(self):
        self.orders: list[Order] = []
        self.current_side: int = None

    def add_order(self, order: Order) -> None:
        # If we switch signals we should delete all previous orders
        if self.current_side and self.current_side != order.side:
            self.orders = []

        # Add the new order and update the current side
        self.orders.append(order)
        self.current_side = order.side
        self.sort_orders()

    def remove_order(self, order: Order) -> None:
        if order in self.orders:
            self.orders.remove(order)

    def sort_orders(self) -> None:
        self.orders.sort(key=attrgetter("execution_time", "limit_price"))

    def get_book(self) -> list[Order]:
        return self.orders

    def sum_unfulfilled_orders(self) -> float:
        return sum(order.order_size for order in self.orders)


# Takes Signals and Converts them into Objects
class CreateOrdersAlgo:
    def __init__(
        self,
        trading_signal: int,
        stoploss_signal: int,
        limit_order_pct: float,
        millisec_execution_delay: timedelta,
        open_pos: int,
        max_risk: float,
        mid_price: float,
        current_book_value: int,
        current_time: datetime,
    ) -> None:
        self.trading_signal = trading_signal
        self.stoploss_signal = stoploss_signal
        self.limit_order_pct = limit_order_pct
        self.open_pos = open_pos
        self.max_risk = max_risk
        self.mid_price = mid_price
        self.current_book_value = current_book_value
        self.execution_time = current_time + millisec_execution_delay

    def create_stoploss_order(self) -> Order:
        # Use a market order for stoplosses
        limit_price = (
            ARBITRARY_HIGH_PRICE
            if self.stoploss_signal == TradingSignal.BUY
            else ARBITRARY_LOW_PRICE
        )
        size = -self.open_pos

        return Order(
            execution_time=self.execution_time,
            limit_price=limit_price,
            order_size=size,
            side=self.stoploss_signal,
        )

    def create_limit_order(self) -> Order:  # Assuming Order is defined elsewhere
        desired_esp = self.current_book_value * self.max_risk * self.trading_signal
        current_esp = self.open_pos * self.mid_price
        target_order = int((desired_esp - current_esp) / self.mid_price)

        if self.trading_signal == TradingSignal.BUY:
            limit_price = self.mid_price * (1 + self.limit_order_pct)
            order_size = max(target_order, 0)
        else:
            limit_price = self.mid_price * (1 - self.limit_order_pct)
            order_size = min(target_order, 0)

        if order_size == 0:
            return None

        return Order(
            execution_time=self.execution_time,
            limit_price=limit_price,
            order_size=order_size,
            side=self.trading_signal,
        )

    def create_order(self) -> Order:
        # Check for stoplosses
        if (self.open_pos > 0 and self.stoploss_signal == TradingSignal.SELL) or (
            self.open_pos < 0 and self.stoploss_signal == TradingSignal.BUY
        ):
            return self.create_stoploss_order()

        elif self.stoploss_signal != TradingSignal.HOLD:
            return self.create_stoploss_order()

        elif self.trading_signal != TradingSignal.HOLD:
            return self.create_limit_order()
        else:
            return None


class ExecutionReport:
    def __init__(
        self,
        order_book: OrderBook,
        trade_history: deque[Trade],
        cash_util: float,
        open_pos: int,
        execution_costs: float,
    ):
        self.order_book = order_book
        self.trade_history = trade_history
        self.cash_util = cash_util
        self.open_pos = open_pos
        self.execution_costs = execution_costs


class OrderMatchingAlgo:
    def exit_order_matching(
        self,
        trade_history: deque[Trade],
        cash_util: float,
        open_pos: int,
        execution_costs: float,
        order_book=OrderBook,
    ) -> ExecutionReport:
        return ExecutionReport(
            order_book, trade_history, cash_util, open_pos, execution_costs
        )

    def execute_orders(
        self,
        order_book: OrderBook,
        bid_size: int,
        bid_price: int,
        ask_size: int,
        ask_price: int,
        current_time: datetime,
        trade_history: deque[Trade],
        cash_util: float,
        execution_costs_per_contract: float,
        open_pos: int,
    ) -> ExecutionReport:

        # Get orders from the order book
        orders = order_book.get_book()

        # Check if there are no available sizes for buying or selling
        if (
            bid_size == 0 and any(order.side == TradingSignal.SELL for order in orders)
        ) or (
            ask_size == 0 and any(order.side == TradingSignal.BUY for order in orders)
        ):
            return self.exit_order_matching(
                trade_history,
                cash_util,
                open_pos,
                execution_costs_per_contract,
                order_book,
            )

        # Iterate over orders and execute them
        for order in orders[:]:
            # Skip the order if the execution time has not been reached
            if order.execution_time > current_time:
                continue

            if order.side == TradingSignal.BUY and ask_price <= order.limit_price:
                trade_amount = min(order.order_size, ask_size)
                ask_size -= trade_amount

                traded_px = ask_price

            elif order.side == TradingSignal.SELL and bid_price >= order.limit_price:
                trade_amount = min(-order.order_size, bid_size)
                bid_size -= trade_amount
                trade_amount = trade_amount * -1
                traded_px = bid_price

            else:
                continue

            # Update our variables if we executed
            order.order_size -= trade_amount
            cash_util -= trade_amount * traded_px
            execution_costs_per_contract = (
                abs(trade_amount) * execution_costs_per_contract
            )
            open_pos += trade_amount
            trade_history.append(Trade(trade_amount, traded_px, current_time))

            # If the order is fully filled, remove it from the order book
            if order.order_size == 0:
                order_book.orders.remove(order)

        return self.exit_order_matching(
            trade_history, cash_util, open_pos, execution_costs_per_contract, order_book
        )
