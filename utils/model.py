from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List
from tradestation_python.types.responses.bars import Bar


class Consolidation(BaseModel):
    start_date: datetime
    end_date: datetime
    range_high: float
    range_low: float
    high_slope: float
    low_slope: float
    next_bars: List[Bar]

    @property
    def duration(self) -> timedelta:
        return self.end_date - self.start_date

    @property
    def range_percent(self) -> float:
        return (self.range_high - self.range_low) / self.range_low

    @property
    def breakout_risk_reward(self):
        # if price breaks out above range_high, potential reward above range_high is unlimited
        highest_price, lowest_price = None, None
        entry = None
        for bar in self.next_bars:
            if bar.high > self.range_high:
                entry = "LONG"
            elif bar.low < self.range_low:
                entry = "SHORT"
            if entry is not None:
                if highest_price is None or bar.high > highest_price:
                    highest_price = bar.high
                if lowest_price is None or bar.low < lowest_price:
                    lowest_price = bar.low

        if entry == "LONG" and highest_price is not None:
            potential_reward = highest_price - self.range_high
            # risk is how much it could fall until the last bar
            potential_risk = (
                self.range_high - lowest_price if lowest_price is not None else 0
            )
            return (
                potential_reward / potential_risk
                if potential_risk != 0
                else potential_reward / (self.range_high - self.range_low)
            )
        elif entry == "SHORT" and lowest_price is not None:
            potential_reward = self.range_low - lowest_price
            # risk is how much it could rise until the last bar
            potential_risk = self.next_bars[-1].close - self.range_low
            return (
                potential_reward / potential_risk
                if potential_risk != 0
                else potential_reward / (self.range_low - self.range_high)
            )
        return 0
