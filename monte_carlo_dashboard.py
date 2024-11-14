import streamlit as st

import math

import numpy as np

from scipy.stats import norm

import io

import matplotlib.pyplot as plt

import logging

import os

from enum import Enum


# Constants

TRADING_DAYS_PER_YEAR = 252


# Set up logging

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


# Suppress matplotlib debug messages

logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


class VanillaOptionType(Enum):

    European = "European"

    American = "American"


class OptionType(Enum):

    Call = "Call"

    Put = "Put"


class LookBackType(Enum):

    Min = "Min"

    Max = "Max"


class ExoticOptionTypes(Enum):

    KnockOut = "Knock-Out"

    Cliquet = "Cliquet"

    Lookback = "Lookback"


class StreamLitTabs(Enum):

    MonteCarloDescription = "Monte Carlo Basics"

    VanillaOptions = "Vanilla Options"

    ExoticOptions = "Exotic Options"


class VanillaOptionPricer:

    def __init__(
        self,
        expiry: float,
        strike: float,
        spot: float,
        volatility: float,
        risk_free_rate: float,
        num_paths: int,
    ) -> None:

        self.expiry = expiry

        self.strike = strike

        self.spot = spot

        self.volatility = volatility

        self.risk_free_rate = risk_free_rate

        self.num_paths = num_paths

        # Precompute values to avoid recalculating them

        self.variance = volatility**2 * expiry

        self.root_variance = math.sqrt(self.variance)

        self.ito_corr = -0.5 * self.variance

        self.moved_spot = spot * math.exp(risk_free_rate * expiry + self.ito_corr)

    def _simulate_spot_prices(self, num_paths: int) -> np.ndarray:

        logger.debug(f"Simulating {num_paths} spot prices.")

        # Generate all random normal values (one per path)

        gauss = np.random.normal(size=num_paths)

        spot_prices = self.moved_spot * np.exp(self.root_variance * gauss)

        return spot_prices

    def simulate_asset_paths(self) -> np.ndarray:

        logger.debug("Simulating asset paths.")

        # Simulate random normal values for all paths and time steps

        random_normals = np.random.normal(
            size=(self.num_paths, int(self.expiry * TRADING_DAYS_PER_YEAR))
        )

        # Simulate the asset paths

        paths = np.zeros_like(random_normals)

        paths[:, 0] = self.spot

        # Apply the SDE to simulate each time step for all paths

        for t in range(1, paths.shape[1]):

            paths[:, t] = paths[:, t - 1] * np.exp(
                (self.risk_free_rate - 0.5 * self.volatility**2) / TRADING_DAYS_PER_YEAR
                + self.volatility
                * np.sqrt(1 / TRADING_DAYS_PER_YEAR)
                * random_normals[:, t]
            )

        return paths

    def price_european_option(self, option_type: str) -> float:

        logger.debug(f"Pricing European {option_type} option.")

        payoffs = self._simulate_spot_prices(self.num_paths)

        if option_type == OptionType.Call.value:

            mean_payoff = np.mean(np.maximum(payoffs - self.strike, 0))

        else:

            mean_payoff = np.mean(np.maximum(self.strike - payoffs, 0))

        discounted_price = mean_payoff * np.exp(-self.risk_free_rate * self.expiry)

        return round(discounted_price, 2)

    def black_scholes_price(self, option_type: str) -> float:

        # Handle the case where spot, strike, or expiry is invalid

        if self.spot <= 0 or self.strike <= 0 or self.expiry <= 0:

            raise ValueError("Spot, strike, and expiry must be greater than zero")

        # Calculate d1 and d2

        d1 = (
            math.log(self.spot / self.strike)
            + (self.risk_free_rate + 0.5 * self.volatility**2) * self.expiry
        ) / (self.volatility * math.sqrt(self.expiry))

        d2 = d1 - self.volatility * math.sqrt(self.expiry)

        if option_type.lower() == "call":

            # Call option pricing

            price = self.spot * norm.cdf(d1) - self.strike * np.exp(
                -self.risk_free_rate * self.expiry
            ) * norm.cdf(d2)

        elif option_type.lower() == "put":

            # Put option pricing

            price = self.strike * np.exp(-self.risk_free_rate * self.expiry) * norm.cdf(
                -d2
            ) - self.spot * norm.cdf(-d1)

        else:

            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        return round(price, 2)

    def visualize_paths(self, paths_to_show: int = 100) -> io.BytesIO:

        paths = self.simulate_asset_paths()

        # Select a random subset of paths for visualization

        if self.num_paths > paths_to_show:

            selected_paths = np.random.choice(
                self.num_paths, paths_to_show, replace=False
            )

            paths = paths[selected_paths]

        logger.debug(f"Visualizing {paths_to_show} asset paths.")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the selected paths

        for path in paths:

            ax.plot(path, linewidth=1)

        ax.set_title("Simulated Asset Paths", fontsize=14)

        ax.set_xlabel("Time Steps (Daily)", fontsize=12)

        ax.set_ylabel("Spot Price", fontsize=12)

        ax.grid(True)

        # Save the plot to a BytesIO buffer

        img_buf = io.BytesIO()

        plt.savefig(img_buf, format="png")

        plt.close(fig)

        # Rewind the buffer to the beginning

        img_buf.seek(0)

        return img_buf


class ExoticOptionPricer(VanillaOptionPricer):

    def __init__(
        self,
        expiry: float,
        strike: float,
        spot: float,
        volatility: float,
        risk_free_rate: float,
        num_paths: int,
        option_type: VanillaOptionType = VanillaOptionType.European,
        barrier: float = None,  # For Knock-Out options
        cliquet_strikes: list = None,  # For Cliquet options
        lookback_type: str = LookBackType.Min.value,
    ) -> None:

        # Common parameters with the base class

        super().__init__(expiry, strike, spot, volatility, risk_free_rate, num_paths)

        # Additional parameters for exotic options

        self.option_type = option_type

        self.barrier = barrier

        self.cliquet_strikes = cliquet_strikes

        self.lookback_type = lookback_type

    def _get_payoff(self, option_type: str, prices: np.ndarray) -> np.ndarray:
        """Helper function to calculate the payoff for a call or put option."""

        if option_type == OptionType.Call.value:

            return np.maximum(prices - self.strike, 0)

        elif option_type == OptionType.Put.value:

            return np.maximum(self.strike - prices, 0)

        else:

            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    def price_knock_out(self) -> float:

        logger.debug("Pricing Knock-Out barrier option.")

        paths = self.simulate_asset_paths()

        # Get paths that hit the barrier and set their payoff to 0

        knock_outs = np.any(paths < self.barrier, axis=1)

        payoffs = self._get_payoff(self.option_type, paths[:, -1])

        payoffs[knock_outs] = 0

        # Discount and return the mean payoff

        mean_payoff = np.mean(payoffs)

        discounted_price = mean_payoff * np.exp(-self.risk_free_rate * self.expiry)

        return round(discounted_price, 2)

    def price_cliquet(self) -> float:

        logger.debug("Pricing Cliquet option.")

        paths = self.simulate_asset_paths()

        total_payoff = 0

        for i in range(len(self.cliquet_strikes)):

            discrete_payoff = self._get_payoff(self.option_type, paths[:, i])

            total_payoff += np.mean(discrete_payoff)

        discounted_price = total_payoff * np.exp(-self.risk_free_rate * self.expiry)

        return round(discounted_price, 2)

    def price_lookback(self) -> float:

        logger.debug(f"Pricing Lookback option ({self.lookback_type}).")

        paths = self.simulate_asset_paths()

        if self.lookback_type == LookBackType.Min.value:

            # Min lookback: Exercise at the lowest price during the life of the option

            lookback_prices = np.min(paths, axis=1)

        else:

            # Max lookback: Exercise at the highest price during the life of the option

            lookback_prices = np.max(paths, axis=1)

        payoffs = self._get_payoff(self.option_type, lookback_prices)

        discounted_price = np.mean(payoffs) * np.exp(-self.risk_free_rate * self.expiry)

        return round(discounted_price, 2)

    def visualize_paths(self, paths_to_show: int = 100) -> io.BytesIO:

        return super().visualize_paths(paths_to_show)


def main():

    st.title("Monte Carlo Option Pricing Tool")

    tabs = [
        StreamLitTabs.MonteCarloDescription.value,
        StreamLitTabs.VanillaOptions.value,
        StreamLitTabs.ExoticOptions.value,
    ]

    tab = st.selectbox("Select a Tab", tabs)

    if tab == StreamLitTabs.MonteCarloDescription.value:

        st.header("Monte Carlo Method Description")

        st.write(
            """

            The Monte Carlo method is a powerful tool used for pricing options by simulating random asset price paths.

            We assume the asset price follows a geometric Brownian motion and generate a large number of paths to

            estimate the option's payoff at maturity. The final option price is the discounted average of these payoffs.

        """
        )

        st.write(
            """

            **Assumptions**:

            - Asset price follows geometric Brownian motion.

            - The risk-free rate is constant.

            - Volatility is constant.

            - The number of simulations should be large enough to capture the randomness.

        """
        )

        st.write(
            "The method works by simulating random price paths for the underlying asset, calculating the payoff for each path, and then averaging them to estimate the option price."
        )

    elif tab == StreamLitTabs.VanillaOptions.value:

        st.subheader("European Vanilla Options")

        option_type = st.radio(
            "Select Option Type", [OptionType.Call.value, OptionType.Put.value]
        )

        # Inputs for the vanilla options

        expiry = st.number_input(
            "Option Expiry (Years)", min_value=0.01, max_value=10.0, value=1.0
        )

        strike = st.number_input("Strike Price", min_value=1.0, value=100.0)

        spot = st.number_input("Spot Price", min_value=1.0, value=100.0)

        volatility = st.number_input(
            "Volatility (%)", min_value=0.1, max_value=100.0, value=20.0
        )

        risk_free_rate = (
            st.number_input(
                "Risk-Free Rate (%)", min_value=0.0, max_value=100.0, value=5.0
            )
            / 100
        )

        num_paths = st.number_input(
            "Number of Paths", min_value=100, max_value=500000, value=10000
        )

        pricer = VanillaOptionPricer(
            expiry, strike, spot, volatility / 100, risk_free_rate, num_paths
        )

        # Button to trigger the simulation

        if st.button("Run Simulation"):

            mc_price = pricer.price_european_option(option_type)

            # Black-Scholes price for comparison

            bs_price = pricer.black_scholes_price(option_type)

            st.subheader("Results")

            st.write(f"Monte Carlo Price: {mc_price}")

            st.write(f"Black-Scholes Price: {bs_price}")

            st.write(f"Difference: {round((mc_price - bs_price), 3)}")

            # Visualize paths

            st.image(pricer.visualize_paths(), use_column_width=True)

    elif tab == StreamLitTabs.ExoticOptions.value:

        st.subheader("Exotic Option Pricing")

        option_type = st.selectbox(
            "Select Option Type",
            [
                OptionType.Call.value,
                OptionType.Put.value,
            ],
        )

        exotic_option_type = st.selectbox(
            "Select Exotic Option Type",
            [
                ExoticOptionTypes.KnockOut.value,
                ExoticOptionTypes.Cliquet.value,
                ExoticOptionTypes.Lookback.value,
            ],
        )

        # Knock-Out Option

        if exotic_option_type == ExoticOptionTypes.KnockOut.value:

            st.write("**Knock-Out Option**")

            expiry = st.number_input("Time to Expiry (years)", min_value=0.1, value=1.0)

            strike = st.number_input("Strike Price", min_value=1.0, value=100.0)

            spot = st.number_input("Spot Price", min_value=1.0, value=100.0)

            volatility = (
                st.number_input(
                    "Volatility (%)", min_value=0.0, max_value=100.0, value=20.0
                )
                / 100
            )

            risk_free_rate = (
                st.number_input(
                    "Risk-Free Rate (%)", min_value=0.0, max_value=100.0, value=5.0
                )
                / 100
            )

            num_paths = st.number_input(
                "Number of Paths", min_value=100, max_value=500000, value=10000
            )

            barrier = st.number_input("Barrier Level", min_value=1.0, value=95.0)

            if st.button("Run Simulation Knock-Out"):

                exotic_pricer = ExoticOptionPricer(
                    expiry=expiry,
                    strike=strike,
                    spot=spot,
                    volatility=volatility,
                    risk_free_rate=risk_free_rate,
                    num_paths=num_paths,
                    option_type=option_type,  # Pass the selected option type (Call/Put)
                    barrier=barrier,
                )

                price = exotic_pricer.price_knock_out()

                st.write(f"**Knock-Out Option Price**: {price:.2f}")

                st.subheader("Simulated Asset Paths")

                st.image(exotic_pricer.visualize_paths(), use_column_width=True)

        # Cliquet Option

        elif exotic_option_type == ExoticOptionTypes.Cliquet.value:

            st.write("**Cliquet Option**")

            expiry = st.number_input("Time to Expiry (years)", min_value=0.1, value=1.0)

            strike = st.number_input("Strike Price", min_value=1.0, value=100.0)

            spot = st.number_input("Spot Price", min_value=1.0, value=100.0)

            volatility = (
                st.number_input(
                    "Volatility (%)", min_value=0.0, max_value=100.0, value=20.0
                )
                / 100
            )

            risk_free_rate = (
                st.number_input(
                    "Risk-Free Rate (%)", min_value=0.0, max_value=100.0, value=5.0
                )
                / 100
            )

            num_paths = st.number_input(
                "Number of Paths", min_value=100, max_value=500000, value=10000
            )

            num_intervals = st.number_input("Number of Intervals", min_value=1, value=5)

            cliquet_strikes = []

            for i in range(num_intervals):

                cliquet_strikes.append(
                    st.number_input(
                        f"Strike Price for Interval {i+1}", min_value=1.0, value=100.0
                    )
                )

            if st.button("Run Simulation Cliquet"):

                exotic_pricer = ExoticOptionPricer(
                    expiry=expiry,
                    strike=strike,
                    spot=spot,
                    volatility=volatility,
                    risk_free_rate=risk_free_rate,
                    num_paths=num_paths,
                    option_type=option_type,
                    cliquet_strikes=cliquet_strikes,
                )

                price = exotic_pricer.price_cliquet()

                st.write(f"**Cliquet Option Price**: {price:.2f}")

                st.subheader("Simulated Asset Paths")

                st.image(exotic_pricer.visualize_paths(), use_column_width=True)

        # Lookback Option

        elif exotic_option_type == ExoticOptionTypes.Lookback.value:

            st.write("**Lookback Option**")

            expiry = st.number_input("Time to Expiry (years)", min_value=0.1, value=1.0)

            strike = st.number_input("Strike Price", min_value=1.0, value=100.0)

            spot = st.number_input("Spot Price", min_value=1.0, value=100.0)

            volatility = (
                st.number_input(
                    "Volatility (%)", min_value=0.0, max_value=100.0, value=20.0
                )
                / 100
            )

            risk_free_rate = (
                st.number_input(
                    "Risk-Free Rate (%)", min_value=0.0, max_value=100.0, value=5.0
                )
                / 100
            )

            num_paths = st.number_input(
                "Number of Paths", min_value=100, max_value=500000, value=10000
            )

            lookback_type = st.selectbox(
                "Lookback Type", [LookBackType.Max.value, LookBackType.Min.value]
            )

            if st.button("Run Simulation Lookback"):

                exotic_pricer = ExoticOptionPricer(
                    expiry=expiry,
                    strike=strike,
                    spot=spot,
                    volatility=volatility,
                    risk_free_rate=risk_free_rate,
                    num_paths=num_paths,
                    option_type=option_type,
                    lookback_type=lookback_type,
                )

                price = exotic_pricer.price_lookback()

                st.write(f"**Lookback Option Price**: {price:.2f}")

                st.subheader("Simulated Asset Paths")

                st.image(exotic_pricer.visualize_paths(), use_column_width=True)


if __name__ == "__main__":

    main()
