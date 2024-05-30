import sqlite3
import pandas as pd
import yfinance as yf
pd.options.mode.chained_assignment = None


class StockDataFetcher:
    def __init__(self, db_name='D:\\Projects\\portfolio optimization\\data\\stocks_data.db'):
        self.db_name = db_name
        self.conn = None

    def connect_to_db(self):
        self.conn = sqlite3.connect(self.db_name)
        self.create_table_if_not_exists()

    def create_table_if_not_exists(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                ticker TEXT,
                date TEXT,
                UNIQUE(ticker, date)
            )
        ''')

    def fetch_and_store_data(self, ticker):
        df = yf.download(ticker)
        if df.empty:
            print(f"No data found for {ticker}")
            return

        df.reset_index(inplace=True)
        df.sort_index(inplace=True, ascending=False)
        df = df[:int(len(df) * 0.65)]
        df['ticker'] = ticker
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df.columns = df.columns.str.lower()

        for row in df.itertuples(index=False, name=None):
            self.conn.execute('''
                INSERT INTO stock_data (date, open, high, low, close, adj_close, volume, ticker)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO NOTHING;
            ''', row)

        self.conn.commit()
        print(f"Data stored successfully for {ticker}")

    def query_data(self):
        self.connect_to_db()
        query = 'SELECT * FROM stock_data'
        result = pd.read_sql_query(query, self.conn)
        return result

    def close_connection(self):
        if self.conn:
            self.conn.close()

    def insert_stock_data(self, ticker, date, open_price, high, low, close, adj_close, volume):
        try:
            self.conn.execute('''
                INSERT INTO stock_data (date, open, high, low, close, adj_close, volume, ticker)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO NOTHING;
            ''', (date, open_price, high, low, close, adj_close, volume,ticker))
            self.conn.commit()
            print(f"Data inserted successfully for {ticker} on {date}")
        except sqlite3.IntegrityError:
            print(f"Data for {ticker} on {date} already exists, skipping insertion.")


# Example usage
if __name__ == "__main__":
    fetcher = StockDataFetcher()
    fetcher.connect_to_db()

    df = pd.read_csv('../data/sp500.csv')

    tickers = df['Symbol'].tolist()

    for ticker in tickers:
        fetcher.fetch_and_store_data(ticker)

    data = fetcher.query_data()
    print(data)

    fetcher.close_connection()

