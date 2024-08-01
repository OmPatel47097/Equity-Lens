import pandas as pd
from utils.DatabaseManager import DatabaseManager
import yfinance as yf
from utils.Constants import intervals
from pathlib import Path
from datetime import datetime, timedelta, timezone
pd.set_option('future.no_silent_downcasting', True)

class StockDataManager:
    def __init__(self):
        self.dbManager = DatabaseManager()

    def get_symbols(self):
        try:
            db_path = Path(__file__).resolve().parent.parent / "data" / "sp500.csv"
            symbols = pd.read_csv(db_path)['Symbol'].to_list()
            symbols = self._format_tickers(symbols)
            return symbols
        except FileNotFoundError:
            print("The file sp500.csv was not found.")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def adj_close(self, tickers, interval):
        tickers_list = ', '.join([f"'{ticker}'" for ticker in tickers])
        data = self.dbManager.select_tickers(interval, tickers_list)
        if data.empty:
            return pd.DataFrame()
        df = data.pivot(index='datetime', columns='symbol', values='adj_close').reset_index()
        df.set_index('datetime', inplace=True)
        return df

    def history(self, ticker, period, interval):
        data = self.dbManager.select_history(ticker=ticker, interval=interval, period=period)
        # print(data.head())
        data['dividends'] = data['dividends'].fillna(0)
        data['splits'] = data['splits'].fillna(0)

        if len(data) == 0:
            print("No data found.")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['datetime', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividends',
                                         'splits', 'symbol'])
        df.set_index('datetime', inplace=True)
        return df

    def get_last_date(self, interval, table_name, symbol):
        return self.dbManager.get_last_date(interval, table_name, symbol)

    def download_adj_close(self):
        tickers = self.get_symbols()
        for interval in intervals:
            period = self._get_period_for_interval(interval)
            for ticker in tickers:
                last_date = self.get_last_date(interval, interval, ticker)
                start_date = self._adjust_start_date(last_date, interval)
                if interval == "1m":
                    self._download_in_chunks(ticker, interval, start_date, period)
                else:
                    df = self._download_data(ticker, start_date, interval, period)
                    if not df.empty:
                        self._add_data_to_db(df, ticker, interval)

    def download_histories(self):
        tickers = self.get_symbols()
        for interval in intervals:
            period = self._get_period_for_interval(interval)
            for ticker in tickers:
                last_date = self.get_last_date(interval, f'history_{interval}', ticker)
                start_date = self._adjust_start_date(last_date, interval)
                if interval == "1m":
                    self._download_in_chunks(ticker, interval, start_date, period)
                else:
                    df = self._download_data(ticker, start_date, interval, period, include_all_columns=True)
                    if not df.empty:
                        self._add_data_to_db(df, ticker, f'history_{interval}')

    def _download_in_chunks(self, ticker, interval, start_date, period):
        end_date = datetime.now(timezone.utc)
        while start_date < end_date:
            chunk_end_date = min(start_date + timedelta(days=7), end_date)
            df = self._download_data(ticker, start_date, interval, period, chunk_end_date, include_all_columns=True)
            if not df.empty:
                self._add_data_to_db(df, ticker, interval)
            start_date = chunk_end_date

    def _download_data(self, ticker, start_date, interval, period, end_date=None, include_all_columns=False):
        try:
            if end_date:
                df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            else:
                df = yf.download(ticker, start=start_date, interval=interval, period=period)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.reset_index()
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]  # Standardize column names
                df.rename(columns={'date': 'datetime', 'adj_close': 'adj_close'}, inplace=True)
                df['symbol'] = ticker
            return df
        except Exception as e:
            print(f"Error downloading data for {ticker} at interval {interval}: {e}")
            return pd.DataFrame()

    def _add_data_to_db(self, df, ticker, table_name):
        self.dbManager.add_table(df, table_name)

    def _format_tickers(self, tickers):
        return [ticker.replace('.', '-') for ticker in tickers]

    def _get_period_for_interval(self, interval):
        if interval in ["2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
            return "1mo"
        return "max"

    def _adjust_start_date(self, last_date, interval):
        if last_date is None:
            # Set default start date if no data exists in the table
            last_date = '1900-01-01'

        try:
            last_date = datetime.strptime(last_date, "%Y-%m-%d")
        except ValueError:
            last_date = datetime.fromisoformat(last_date)

        # Ensure the last_date is timezone-aware
        if last_date.tzinfo is None:
            last_date = last_date.replace(tzinfo=timezone.utc)

        if interval == "1m":
            last_date = max(datetime.now(timezone.utc) - timedelta(days=6), last_date)
        elif interval in ["2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
            last_date = max(datetime.now(timezone.utc) - timedelta(days=59), last_date)
        return last_date
