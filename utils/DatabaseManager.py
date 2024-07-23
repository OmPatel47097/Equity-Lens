import os.path
import sqlite3


class DatabaseManager:
    tables = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1mo", "3mo"]

    def __init__(self):
        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))    # Get the current directory
            # construct path to the database
            db_path = os.path.join(current_dir, "..", "data", "db", "equity_lens_db.db")
            db_path = os.path.normpath(db_path)
            # self.connection = sqlite3.connect("../data/db/equity_lens_db.db")
            self.connection = sqlite3.connect(db_path)
            self.cursor = self.connection.cursor()

            print("Connected to the database successfully!")
            return self.connection, self.cursor

        except sqlite3.Error as error:
            print(f"SQLite operational error: {error}")
            print(f"Attempted to open database at: {db_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Directory contents of the parent of 'data': {os.listdir(os.path.dirname(os.path.dirname(db_path)))}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    def close(self):
        # self.connection.close()
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def add_table(self, df, name):
        self.connect()
        df.to_sql(name, self.connection, if_exists='append', index=False)
        self.close()

    def select_tickers(self, table, tickers):
        self.connect()
        self.cursor.execute(f"SELECT `date`, {tickers} FROM `{table}`")
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
