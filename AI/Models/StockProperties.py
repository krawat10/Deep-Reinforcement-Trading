import json


class StockProperties:
    end_date: str
    start_date: str
    ticker: str
    trading_cost_bps = 1e-3
    time_cost_bps = 1e-4
    leverage = 1

    def __init__(self, ticker: str, start_date: str, end_date: str, leverage: int):
        self.end_date = end_date
        self.start_date = start_date
        self.ticker = ticker
        self.leverage = leverage

    def toJSON(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)
