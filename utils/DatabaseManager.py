import os
import sqlite3
import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta
from datetime import datetime, timezone

class DatabaseManager:
    intervals = {
        "1m": "history_1m", "2m": "history_2m", "5m": "history_5m",
        "15m": "history_15m", "30m": "history_30m", "60m": "history_60m",
        "90m": "history_90m", "1h": "history_1h", "1d": "history_1d",
        "5d": "history_5d", "1mo": "history_1mo", "3mo": "history_3mo"
    }

    def __init__(self):
        self.connection = None

    def connect(self):
        try:
            db_path = Path(__file__).resolve().parent.parent / "data" / "db" / "equity_lens_db.db"
            self.connection = sqlite3.connect(db_path)
            self.cursor = self.connection.cursor()
            print("Connected to the database successfully!")
        except sqlite3.Error as error:
            print(f"SQLite operational error: {error}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def add_table(self, df, name):
        table_name = self.intervals.get(name, name)
        try:
            with self._get_connection() as conn:
                if not self.table_exists(conn, table_name):
                    self.create_table(conn, table_name)
                df.to_sql(table_name, conn, if_exists='append', index=False)
        except Exception as e:
            print(f"An error occurred while adding the table {table_name}: {e}")

    def create_table(self, conn, table_name):
        try:
            create_table_query = f'''
                CREATE TABLE IF NOT EXISTS "{table_name}" (
                    "datetime" TEXT NOT NULL,
                    "open" REAL,
                    "high" REAL,
                    "low" REAL,
                    "close" REAL,
                    "adj_close" REAL,
                    "volume" REAL,
                    "dividends" REAL,
                    "splits" REAL,
                    "symbol" TEXT NOT NULL,
                    PRIMARY KEY("datetime", "symbol")
                )
            '''
            conn.execute(create_table_query)
            print(f"Table {table_name} created successfully.")
        except Exception as e:
            print(f"An error occurred while creating table {table_name}: {e}")

    def table_exists(self, conn, table_name):
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        table_exists = cursor.fetchone() is not None
        cursor.close()
        return table_exists

    def get_last_date(self, interval, table_name, symbol):
        table_name = self.intervals.get(table_name, table_name)
        try:
            with self._get_connection() as conn:
                if not self.table_exists(conn, table_name):
                    print(f"Table {table_name} does not exist.")
                    return None
                query = f"SELECT MAX(datetime) as last_date FROM `{table_name}` WHERE symbol = ?"
                cursor = conn.cursor()
                cursor.execute(query, (symbol,))
                last_date = cursor.fetchone()
                cursor.close()
                return last_date[0] if last_date else None
        except Exception as e:
            print(f"An error occurred while fetching last date from table {table_name}: {e}")
            return None

    def select_tickers(self, interval, tickers):
        table_name = self.intervals.get(interval, interval)
        try:
            with self._get_connection() as conn:
                query = f"SELECT datetime, adj_close, symbol FROM `{table_name}` WHERE symbol IN ({tickers})"
                df = pd.read_sql(query, conn)
                return df
        except Exception as e:
            print(f"An error occurred while selecting tickers from table {table_name}: {e}")
            return pd.DataFrame()

    def select_history(self, ticker, interval, period):
        table_name = self.intervals.get(interval, interval)
        try:
            with self._get_connection() as conn:
                end_date = datetime.now(timezone.utc)
                if period.endswith('y'):
                    years = int(period[:-1])
                    start_date = end_date - relativedelta(years=years)
                elif period.endswith('mo'):
                    months = int(period[:-2])
                    start_date = end_date - relativedelta(months=months)
                elif period.endswith('d'):
                    days = int(period[:-1])
                    start_date = end_date - relativedelta(days=days)
                else:
                    raise ValueError(f"Invalid period format: {period}")

                query = f"SELECT * FROM `{table_name}` WHERE symbol = ? AND datetime >= ? ORDER BY datetime DESC"
                df = pd.read_sql(query, conn, params=(ticker, start_date.isoformat()))
                return df
        except Exception as e:
            print(f"An error occurred while selecting history from table {table_name}: {e}")
            return pd.DataFrame()

    def _get_connection(self):
        db_path = Path(__file__).resolve().parent.parent / "data" / "db" / "equity_lens_db.db"
        return sqlite3.connect(db_path)
