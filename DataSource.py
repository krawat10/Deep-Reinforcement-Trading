from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from ta import add_all_ta_features

from FinancialmodelingprepProvider import FinancialmodelingprepProvider
from YahooProvider import YahooProvider


class DataSource:
    def __init__(self, start_date: str, end_date, ticker, trading_days=252, normalize=True):
        self.start_date = datetime.fromisoformat(start_date)
        self.end_date = datetime.fromisoformat(end_date)
        self.ticker = ticker
        self.trading_days = trading_days
        self.normalize = normalize
        self.provider = YahooProvider()
        self.ohlc = self.load_data()
        self.data: pd.DataFrame = self.preprocess_data()
        self.min_values = self.data.min()
        self.max_values = self.data.max()
        self.step = 0
        self.offset = None

    def load_data(self) -> pd.DataFrame:
        print('loading data for {}...'.format(self.ticker))
        df = self.provider.get_full_price_in_range(self.ticker, '2000-01-01', self.end_date.date())
        df = df[['open', 'high', 'low', 'close', 'volume']]
        print('got data for {}...'.format(self.ticker))
        return df

    def find_closest_trading_day(self, target_date):
        # Implement logic to find the closest previous trading day here
        # For example, you can use a loop to iterate back in time until you find a trading day.
        # Replace this with your logic.
        closest_date = target_date
        while closest_date not in self.ohlc.index:
            closest_date -= pd.Timedelta(days=1)
        return closest_date

    def preprocess_data(self):
        """calculate returns and percentiles, then removes missing values"""
        extended_data = add_all_ta_features(self.ohlc.copy(), open="open", high="high", low="low", close="close",
                                            volume="volume")
        data = pd.DataFrame()

        data['returns'] = self.ohlc.close.pct_change()
        data['ret_2'] = self.ohlc.close.pct_change(2)
        data['ret_5'] = self.ohlc.close.pct_change(5)
        data['ret_10'] = self.ohlc.close.pct_change(10)
        data['ret_21'] = self.ohlc.close.pct_change(21)
        data['stc'] = extended_data['trend_stc']
        data['aroon_up'] = extended_data['trend_aroon_up']
        data['aroon_down'] = extended_data['trend_aroon_down']
        data['psar_down_indicator'] = extended_data['trend_psar_down_indicator']
        # self.data['vol_ret_5'] = self.data.volume.pct_change(5)
        # self.data['vol_ret_15'] = self.data.volume.pct_change(15)

        data['rsi'] = talib.STOCHRSI(self.ohlc.close)[1]
        data['macd'] = talib.MACD(self.ohlc.close)[1]
        data['atr'] = talib.ATR(self.ohlc.high, self.ohlc.low, self.ohlc.close)
        # data['obv'] = talib.OBV(self.ohlc.close, self.ohlc.volume)
        slowk, slowd = talib.STOCH(self.ohlc.high, self.ohlc.low, self.ohlc.close)
        data['stoch'] = slowd - slowk
        data['atr'] = talib.ATR(self.ohlc.high, self.ohlc.low, self.ohlc.close)
        data['ultosc'] = talib.ULTOSC(self.ohlc.high, self.ohlc.low, self.ohlc.close)
        data = (data.replace((np.inf, -np.inf), np.nan).dropna())

        if self.start_date == self.end_date:
            closest_start_date = self.find_closest_trading_day(self.start_date)
            data = data.loc[closest_start_date: closest_start_date]
        else:
            data = data.loc[(data.index >= self.start_date) & (data.index <= self.end_date)]

        r = data.returns.copy()
        if self.normalize:
            data = pd.DataFrame(scale(data),
                                columns=data.columns,
                                index=data.index)
        features = data.columns.drop('returns')
        data['returns'] = r  # don't scale returns

        data = data.loc[:, ['returns'] + list(features)]
        print(data.info())

        return data

    def reset(self):
        """Provides starting index for time series and resets step"""
        high = len(self.data.index) - self.trading_days  # maxDate - trading days
        self.offset = np.random.randint(low=0, high=high)  # select start between 0 and (maxDate - trading days)
        self.step = 0

    def take_step(self):
        """Returns data for current trading day and done signal"""

        if self.trading_days == 1:
            data_row = self.data.iloc[-1]
        else:
            data_row = self.data.iloc[self.offset + self.step]
        obs = data_row.values
        date = data_row.name
        self.step += 1
        done = self.step > self.trading_days
        return obs, date, done
