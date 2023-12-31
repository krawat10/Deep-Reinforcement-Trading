import datetime
import pandas as pd
import requests
from dotenv import dotenv_values
import yfinance as yf

class YahooProvider:
    def get_full_price_in_range(self, ticker: str, start_date, end_date) -> pd.DataFrame:
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)

        # Drop rows with missing data
        data.dropna(inplace=True)

        # Sort DataFrame by date
        data.sort_index(ascending=False, inplace=True)

        # Rename columns to lowercase
        data.columns = data.columns.str.lower()

        # Select only the columns we're interested in
        data = data[['open', 'high', 'low', 'close', 'volume']]

        return data
