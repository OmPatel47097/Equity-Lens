import sys
import os
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# Add the parent directory of the current script to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from utils.StockDataManager import StockDataManager

def _fetch_top_performing_stocks():
    symbols = StockDataManager().get_symbols()[:10]
    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        obj = {}
        obj['symbol'] = symbol
        obj['name'] = ticker.info['shortName']
        obj['current_price'] = ticker.info['currentPrice']
        obj['previousClose'] = ticker.info['previousClose']
        obj['change'] = ((ticker.info['currentPrice'] * 100) / ticker.info['previousClose']) - 100
        obj['marketCap'] = ticker.info['marketCap']
        data[symbol] = obj

    # top_stocks = sorted(data.items(), key=lambda x: x[1]['change'].iloc[-1], reverse=True)[:3]

    return data


def fetch_news(symbol):
    ticker = yf.Ticker(symbol)
    return ticker.news[:2]

def format_timestamp(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')


def show():
    st.title("Home")

    stock_data_manager = StockDataManager()

    # S&P 500 Performance
    st.subheader("S&P 500 Performance")
    sp500_data = yf.download("^GSPC", period="1y")
    sp500_data.columns = [col.lower() for col in sp500_data.columns]  # Ensure column names are in lowercase
    if not sp500_data.empty:
        st.line_chart(sp500_data["close"])

    st.divider()

    # Top Performing Stocks
    st.subheader("Stock Screener")
    top_stocks = _fetch_top_performing_stocks()

    if top_stocks:

        top_stocks_df = pd.DataFrame(top_stocks).T

        st.table(top_stocks_df)

        st.divider()

        # Combined news for top-performing stocks
        st.subheader("News for Top Performing Stocks")
        combined_news = []
        for symbol in top_stocks_df['symbol']:
            news_items = fetch_news(symbol)
            for item in news_items:
                combined_news.append({
                    'title': item['title'],
                    'publisher': item['publisher'],
                    'link': item['link'],
                    'date': format_timestamp(item['providerPublishTime'])
                })

        # Sort news by date
        combined_news.sort(key=lambda x: x['date'], reverse=True)

        # Display news in a grid
        news_cols = st.columns(3)
        for i, news in enumerate(combined_news):
            with news_cols[i % 3]:
                st.markdown(f"[{news['title']}]({news['link']})")
                st.markdown(f"by _{news['publisher']}_ | {news['date']}")
                st.markdown(f"---")
