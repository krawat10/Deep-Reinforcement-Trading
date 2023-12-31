import datetime
import pandas as pd
import requests
from dotenv import dotenv_values


class FinancialmodelingprepProvider:
    def __init__(
            self,
            url='https://financialmodelingprep.com/api/v3/historical-price-full'):
        self.api_key = dotenv_values('../.env')['API_KEY']
        self.url = url

    def get_adj_price_in_range(self, ticker: str, start_date, end_date) -> pd.DataFrame:
        url = f'https://financialmodelingprep.com/api/v4/historical-price-adjusted/{ticker}/1/day/{start_date}/{end_date}?apikey={self.api_key}'

        if 'USD' in ticker:
            url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={self.api_key}'
            json = requests.get(url).json()['historical']
            df = pd.DataFrame(json).dropna().sort_index(ascending=False)
            df.index = pd.to_datetime(df['date'])
            return df

        print(f'GET: {url}')

        r = requests.get(url)
        json = r.json()['results']
        df = pd.DataFrame(json).dropna().sort_index(ascending=False)
        df = df.rename({
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'},
            axis='columns')
        df.index = pd.to_datetime(df['formated'])
        return df

    def get_full_price_in_range(self, ticker: str, start_date, end_date) -> pd.DataFrame:
        url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={self.api_key}'
        json = requests.get(url).json()['historical']
        df = pd.DataFrame(json).dropna().sort_index(ascending=False)
        df.index = pd.to_datetime(df['date'])
        return df

    def get_historical_price(self, ticker: str, years: int):
        start_date = (datetime.datetime.now() - datetime.timedelta(days=years * 365)).strftime("%Y-%m-%d")
        sdend_date = datetime.datetime.now().strftime("%Y-%m-%d")

        return self.get_full_price_in_range(ticker, start_date, end_date)
