import sqlite3


class DatabaseManager:
    tables = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1mo", "3mo"]

    def __init__(self):
        self.connection = None
        self.cursor = None

    def connect(self):
        self.cursor = self.connection.cursor()
        return self.connection, self.cursor

    def close(self):
        self.connection.close()

    def add_table(self, df, name):
        self.connect()
        df.to_sql(name, self.connection, if_exists='append', index=False)
        self.close()

    def select_tickers(self, table, tickers):
        self.connect()
        data = self.cursor.fetchall()
        self.close()
        return data

    def select_history(self, interval, ticker, period):
        period_mapping = {
            "3mo": "-3 months",
            "6mo": "-6 months",
            "1y": "-1 year",
            "2y": "-2 years",
            "5y": "-5 years",
            "max": "1900-01-01"  # This represents a date far in the past
        }

        self.connect()
        if period == "max":
            query = """
                SELECT date, Open, High, Low, Close, `Adj Close`, Volume
                FROM `history_{interval}`
                WHERE symbol = ?
            """.format(interval=interval)
            self.cursor.execute(query, (ticker,))
        else:
            query = """
                SELECT date, Open, High, Low, Close, `Adj Close`, Volume
                FROM `history_{interval}`
                WHERE symbol = ? AND date >= DATE('now', ?)
            """.format(interval=interval)
            self.cursor.execute(query, (ticker, period_mapping[period]))

        data = self.cursor.fetchall()
        self.close()
        return data
