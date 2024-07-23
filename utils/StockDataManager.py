import pandas as pd
from utils.DatabaseManager import DatabaseManager
import yfinance as yf
from utils.Constants import intervals

class StockDataManager:
    def __init__(self):
        self.dbManager = DatabaseManager()
        pass

    def get_symbols(self):
        symbols = pd.read_csv('../data/sp500.csv')['Symbol'].to_list()
        symbols = [ticker.replace('.', '-') for ticker in symbols]
        return symbols

    def adj_close(self, tickers, interval):
        data = self.dbManager.select_tickers(interval, ', '.join([f"{ticker}" for ticker in tickers]))
        df = pd.DataFrame(data, columns=['date'] + tickers)
        df.set_index('date', inplace=True)
        return df

    def history(self, ticker, period, interval):
        data = self.dbManager.select_history(ticker=ticker, interval=interval, period=period)
        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'])
        df.set_index('date', inplace=True)
        return df

    def download_adj_close(self):
        tickers = self.tickers()
        for interval in intervals:
            period = "max"
            if interval in ["2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
                period = "1mo"
            df = yf.download(tickers, period=period, interval=interval)['Adj Close']
            df['date'] = df.index
            self.dbManager.add_table(df, interval)
            print(df.shape)

    def download_histories(self):
        tickers = self.tickers()
        for interval in intervals:
            period = "max"
            if interval in ["2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
                period = "1mo"
            for ticker in tickers:
                df = yf.download(ticker, period=period, interval=interval)
                df['date'] = df.index
                df['symbol'] = ticker
                self.dbManager.add_table(df, 'history_' + interval)

    def tickers(self):
        df_tickers = pd.read_csv('../data/sp500.csv')
        tickers = df_tickers['Symbol'].to_list()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
