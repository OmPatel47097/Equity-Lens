import streamlit as st
import requests


def fetch_news(ticker):
    # Placeholder for actual news API integration
    return [{"title": "News headline 1", "sentiment": "Positive"},
            {"title": "News headline 2", "sentiment": "Neutral"}]


def show():
    st.title("Market News and Sentiment")
    ticker = st.text_input("Enter Stock Ticker for News", value="AAPL")

    if ticker:
        news_items = fetch_news(ticker)
        for item in news_items:
            st.write(f"{item['title']} - Sentiment: {item['sentiment']}")
