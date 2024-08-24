import pandas as pd
from datetime import datetime, timedelta
from operator import attrgetter
from enum import Enum
from collections import deque


# We will use these two for limits prices when executing market
ARBITRARY_HIGH_PRICE = 100_000_000
ARBITRARY_LOW_PRICE = 0

# Columns for Dataframe
TRADE_HISTORY_COLUMNS = ["Trade Amount", "Price", "Time"]
MAX_LENGTH_TRADE_HISTORY_DEQUE = 10


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
    def __init__(self, rolling_window: int, sd_deviation: int) -> None:
        self.rolling_window = rolling_window
        self.standard_deviations = sd_deviation


class MovingAveragesParameter:
    def __init__(self, small_window: int, medium_window: int, long_window: int) -> None:
        self.validate_params(small_window, medium_window, long_window)
        self.long_rolling_window = long_window
        self.medium_rolling_window = medium_window
        self.short_rolling_window = small_window

    def validate_params(self, small_window, medium_window, long_window) -> None:
        if not (long_window > medium_window > small_window):
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

    def get_orderbook_dataframe(self) -> pd.DataFrame:
        # Convert the list of orders to a list of dictionaries
        orders_data = [
            {
                "Execution Time": order.execution_time,
                "Limit Price": order.limit_price,
                "Order Size": order.order_size,
                "Side": order.side,
            }
            for order in self.orders
        ]
        return pd.DataFrame(orders_data)


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


class MaxDrawDownInfo:
    def __init__(self, drawdown: float, peak: float):
        self.drawdown = drawdown
        self.peak = peak


# Calculate_Data_for_Dashboard
class MetricsCalculator:
    def calculate_pnl(
        self, open_pos: int, px_mid: float, cash_utils: float, trading_costs: float
    ):
        return (open_pos * px_mid) + cash_utils - trading_costs

    def get_trade_history_df(self, trade_history: deque[Trade]) -> pd.DataFrame:
        trade_data_list = []

        for trade in trade_history:
            trade_data = {
                TRADE_HISTORY_COLUMNS[0]: trade.trade_amount,
                TRADE_HISTORY_COLUMNS[1]: trade.price,
                TRADE_HISTORY_COLUMNS[2]: trade.time,
            }
            trade_data_list.append(trade_data)

        return pd.DataFrame(trade_data_list, columns=TRADE_HISTORY_COLUMNS)

    def update_max_drawdown(
        self, current_max_draw_down: MaxDrawDownInfo, new_portfolio_value: float
    ) -> MaxDrawDownInfo:
        # Update the peak if the new portfolio value is greater
        if new_portfolio_value > current_max_draw_down.peak:
            new_peak = new_portfolio_value
            new_max_drawdown = 0.0  # Reset drawdown since we have a new peak
        else:
            new_peak = current_max_draw_down.peak
            if new_peak == 0:  # Prevent division by zero
                drawdown = 0.0
            else:
                drawdown = (new_peak - new_portfolio_value) / new_peak

            # Calculate the new max drawdown
            new_max_drawdown = max(current_max_draw_down.drawdown, drawdown)

        return MaxDrawDownInfo(drawdown=new_max_drawdown, peak=new_peak)


class DashBoardData:
    def __init__(
        self,
        current_price: float,
        current_pnl: float,
        open_pos: int,
        current_trading_costs: float,
        current_max_drawdown: float,
        current_unfilled_trades: pd.DataFrame,
        df_last_trades: pd.DataFrame,
    ) -> None:
        self.current_pnl = current_pnl
        self.current_price = current_price
        self.open_pos = open_pos
        self.current_trading_costs = current_trading_costs
        self.pnl_exc_trading_costs = current_pnl - current_trading_costs
        self.current_max_drawdown = current_max_drawdown
        self.current_unfilled_trades = current_unfilled_trades
        self.df_last_trades = df_last_trades


class TradingAlgo:
    def __init__(
        self,
        file_location: str,
        moving_averages_params: MovingAveragesParameter,
        bollinger_bands_params: BollingBandParameters,
        initial_capital: int = 1000000,
        max_risk: int = 0.1,
        limit_order_pct: float = 0,
        millisec_execution_delay: timedelta = timedelta(microseconds=0),
        transaction_fees_per_contract: int = 0,
    ):
        self.initialize_reader(file_location)
        self.initialize_parameters(moving_averages_params, bollinger_bands_params)
        self.initialize_state(
            initial_capital,
            max_risk,
            limit_order_pct,
            millisec_execution_delay,
            transaction_fees_per_contract,
        )

    def initialize_reader(self, file_location: str):
        """Initialize the CSV reader."""
        self.csv_iterator = pd.read_csv(
            file_location,
            usecols=[
                "Date-Time",
                "Type",
                "Bid Price",
                "Bid Size",
                "Ask Price",
                "Ask Size",
            ],
            chunksize=1,
            parse_dates=["Date-Time"],
            date_format="%Y-%m-%dT%H:%M:%S.%f%z",
        )

        self.current_chunk = None

    def initialize_parameters(
        self,
        moving_averages_params: MovingAveragesParameter,
        bollinger_bands_params: BollingBandParameters,
    ):
        """Initialize the trading parameters."""
        self.moving_averages_params = moving_averages_params
        self.bollinger_bands_params = bollinger_bands_params
        self.price_deque = deque(
            maxlen=max(
                self.moving_averages_params.long_rolling_window,
                self.bollinger_bands_params.rolling_window,
            )
        )

    def initialize_state(
        self,
        initial_capital: int,
        max_risk: float,
        limit_order_pct: float,
        millisec_execution_delay: timedelta,
        transaction_fees_per_contract: int,
    ):
        """Initialize the trading state and metrics."""
        # Market data
        self.date_time = None
        self.bid_price = None
        self.bid_size = None
        self.ask_price = None
        self.ask_size = None
        self.signal = None
        self.mid_price = None

        # Financial metrics
        self.initial_capital = initial_capital
        self.book_value = initial_capital
        self.cash_utils = 0
        self.open_pos = 0
        self.pnl = None

        # Order and trade management
        self.order_book = OrderBook()
        self.trade_history = deque(maxlen=MAX_LENGTH_TRADE_HISTORY_DEQUE)
        self.limit_order_pct = limit_order_pct
        self.millisec_execution_delay = millisec_execution_delay
        self.max_risk = max_risk

        # Transaction costs
        self.transaction_fees = transaction_fees_per_contract
        self.trading_costs = 0
        self.total_transaction_costs = 0

        # Risk metrics
        self.max_draw_down = MaxDrawDownInfo(0, 0)

    def read_next_line(self):
        """Read the next line from the CSV file."""
        if self.current_chunk is None or len(self.current_chunk) == 0:
            try:
                self.current_chunk = next(self.csv_iterator)
            except StopIteration:
                return False

        return True

    def process_row(self, row: pd.Series):
        """Process a single row of data."""
        self.date_time = row["Date-Time"]
        self.bid_price = row["Bid Price"]
        self.bid_size = row["Bid Size"]
        self.ask_price = row["Ask Price"]
        self.ask_size = row["Ask Size"]
        self.mid_price = (self.bid_price + self.ask_price) / 2

        self.price_deque.append(self.mid_price)

    def calculate_signals(self):
        """Calculate trading signals."""
        prices_df = pd.DataFrame(list(self.price_deque))
        signal_generator = SignalGenerator()

        stop_loss_signal = signal_generator.calculate_bollinger_bands_signal(
            prices_df, self.bollinger_bands_params, self.mid_price
        )
        trading_signal = signal_generator.calculate_moving_averages_signal(
            prices_df, self.moving_averages_params
        )

        return trading_signal, stop_loss_signal

    def create_and_execute_order(self, trading_signal, stop_loss_signal):
        """Create and execute orders based on the signals."""
        if new_order := CreateOrdersAlgo(
            trading_signal,
            stop_loss_signal,
            self.limit_order_pct,
            self.millisec_execution_delay,
            self.open_pos,
            self.max_risk,
            self.mid_price,
            self.book_value,
            self.date_time,
        ).create_order():
            self.order_book.add_order(new_order)

        execution_report = OrderMatchingAlgo().execute_orders(
            self.order_book,
            self.bid_size,
            self.bid_price,
            self.ask_size,
            self.ask_price,
            self.date_time,
            self.trade_history,
            self.cash_utils,
            self.transaction_fees,
            self.open_pos,
        )

        self.update_state(execution_report)

    def update_state(self, execution_report: ExecutionReport):
        """Update the state based on the execution report."""
        self.order_book = execution_report.order_book
        self.trade_history = execution_report.trade_history

        self.cash_utils = execution_report.cash_util
        self.open_pos = execution_report.open_pos
        self.execution_costs = execution_report.execution_costs

    def update_metrics(self):
        """Update PnL and other dashboard metrics."""
        self.pnl = MetricsCalculator().calculate_pnl(
            open_pos=self.open_pos,
            px_mid=self.mid_price,
            cash_utils=self.cash_utils,
            trading_costs=self.execution_costs,
        )
        self.book_value = self.initial_capital + self.pnl

        self.max_draw_down = MetricsCalculator().update_max_drawdown(
            self.max_draw_down, self.book_value
        )

    def get_dashboard_data(self) -> DashBoardData:

        return DashBoardData(
            current_pnl=self.pnl,
            current_price=self.mid_price,
            open_pos=self.open_pos,
            current_trading_costs=self.trading_costs,
            current_max_drawdown=self.max_draw_down.drawdown,
            current_unfilled_trades=self.order_book.get_orderbook_dataframe(),
            df_last_trades=MetricsCalculator().get_trade_history_df(self.trade_history),
        )

    def go_to_next_line(self):
        """Main loop to go to the next line and process it."""
        if not self.read_next_line():
            return False

        while len(self.current_chunk) > 0:
            row = self.current_chunk.iloc[0]

            if row["Type"] != "Quote":
                self.current_chunk = self.current_chunk.iloc[1:]
                continue

            self.process_row(row)

            trading_signal, stop_loss_signal = self.calculate_signals()

            self.create_and_execute_order(trading_signal, stop_loss_signal)

            self.update_metrics()

            self.get_dashboard_data()
            self.current_chunk = self.current_chunk.iloc[1:]
            return True

        return self.go_t
