import pandas as pd
import numpy as np
import pickle
import sys

sys.path.append("C:/Users/WilliamFetzner/Documents/Trading/")
from gym_mtsim_forked.gym_mtsim.data import FOREX_DATA_PATH, FOREX_DATA_PATH_15MIN

with open(FOREX_DATA_PATH_15MIN, "rb") as f:
    symbols_1hr = pickle.load(f)
# convert symbols_1hr to a pd.dataframe
symbols_1hr[1]["EURUSD"].index = pd.to_datetime(symbols_1hr[1]["EURUSD"].index)
full_data = symbols_1hr[1]["EURUSD"]
# full_data schema = Index: 'Time' dtype: datetime64[ns, UTC], Columns: ['Open', 'High', 'Low', 'Close']


class TemplateStrategy:
    def __init__(
        self,
        stop_loss_mult: float = 1.5,
        take_profit_mult: float = 1.5,
        # other parameters as needed
    ):
        self.stop_loss_mult = stop_loss_mult
        self.take_profit_mult = take_profit_mult
        # other parameters as needed

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on strategy rules"""
        # Initialize signal column
        df["signal"] = 0
        # Long signal conditions
        long_conditions = (
            # input long conditions for the trade here
        )
        # Short signal conditions
        short_conditions = (
            # input short conditions for the trade here
        )

        # Set signals
        df.loc[long_conditions, "signal"] = 1
        df.loc[short_conditions, "signal"] = -1

        return df

    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the complete strategy to the dataframe"""

        # strategy here

        df["stop_distance"] = (
            df["atr"] * self.stop_loss_mult
        )  # Use 1.5 * ATR for stop distance - 0.0025
        df["profit_distance"] = df["atr"] * self.take_profit_mult
