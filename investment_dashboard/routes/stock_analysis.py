import sys
import os
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import ta

# Add the parent directory of the current script to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from services.FinancialSentimentAnalysis import FinancialSentimentAnalyzer

def get_technical_indicators(df):
    # Add technical indicators using the `ta` library
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
    df['ROC'] = ta.momentum.roc(df['Close'], window=12)
    return df

def get_fundamental_ratios(ticker):
    ticker_info = yf.Ticker(ticker).info
    pe_ratio = ticker_info.get('forwardPE', None)
    pb_ratio = ticker_info.get('priceToBook', None)
    roe = ticker_info.get('returnOnEquity', None)
    return pe_ratio, pb_ratio, roe

def calculate_beta(df, market_df):
    cov_matrix = df['Returns'].cov(market_df['Returns'])
    market_var = market_df['Returns'].var()
    beta = cov_matrix / market_var
    return beta

def calculate_var(df, confidence_level=0.95):
    return df['Returns'].quantile(1 - confidence_level)

def fetch_news(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news
    return news

def analyze_sentiment(news_articles):
    analyzer = FinancialSentimentAnalyzer()
    sentiment_scores = []
    for article in news_articles:
        title = article['title']
        sentiment, score = analyzer.analyze_sentiment(title)
        sentiment_scores.append({'title': title, 'sentiment': sentiment, 'score': score, 'link': article['link']})
    return sentiment_scores

def generate_insights(pe_ratio, pb_ratio, roe, beta, var):
    insights = {}

    if pe_ratio is not None:
        insights['pe_ratio'] = f"P/E Ratio: {pe_ratio} \nCompares a company's current share price to its per-share earnings. A P/E ratio of {pe_ratio} means investors are willing to pay ${pe_ratio:.2f} for every 1 dollar of earnings. Higher P/E can mean high growth expectations."
    else:
        insights['pe_ratio'] = "P/E Ratio: Not available."

    if pb_ratio is not None:
        insights['pb_ratio'] = f"P/B Ratio: {pb_ratio} \nCompares a company's market value to its book value. A P/B ratio of {pb_ratio} means the stock is trading at {pb_ratio:.2f} times its book value. Lower P/B could indicate that the stock is undervalued."
    else:
        insights['pb_ratio'] = "P/B Ratio: Not available."

    if roe is not None:
        insights['roe'] = f"ROE: {roe} \nReturn on Equity measures a corporation's profitability in relation to stockholders’ equity. A ROE of {roe:.2f}% means that for every dollar of equity, the company generates {roe:.2f} cents in profit. Higher ROE is generally better."
    else:
        insights['roe'] = "ROE: Not available."

    if beta is not None:
        insights['beta'] = f"Beta: {beta} \nBeta measures a stock's volatility in relation to the overall market. A beta of {beta} means the stock is {abs(beta - 1) * 100:.1f}% {'more' if beta > 1 else 'less'} volatile than the market. Higher beta indicates higher risk and potential reward."
    else:
        insights['beta'] = "Beta: Not available."

    if var is not None:
        insights['var'] = f"Value at Risk (VaR): {var:.2f} \nEstimates the maximum potential loss over a specific period at a given confidence level. For example, a 5% VaR of ${var:.2f} means there's a 5% chance of losing more than this amount in the given period."
    else:
        insights['var'] = "VaR: Not available."

    return insights

def summarize_analysis(ticker, pe_ratio, pb_ratio, roe, beta, var):
    summary = {
        'Ticker': ticker,
        'P/E Ratio': pe_ratio,
        'P/B Ratio': pb_ratio,
        'ROE (%)': roe,
        'Beta': beta,
        'VaR': var
    }
    return summary

def format_value(value, is_good):
    color = "green" if is_good else "red"
    return f"<span style='color:{color}'>{value}</span>"

def show():
    st.title("Stock Analysis")

    # User inputs for tickers and period
    tickers = st.text_input("Enter Tickers (comma separated)", value="AAPL,MSFT,AMZN")
    period = st.selectbox("Select Period for Analysis", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

    tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

    summary_data = []
    if st.button("Analyze"):
        with st.spinner("Fetching data and analyzing..."):
            try:
                for ticker in tickers:
                    st.divider()
                    st.markdown(f"## Analysis for {ticker}")
                    st.divider()

                    # Fetch historical market data
                    df = yf.download(ticker, period=period)
                    df['Returns'] = df['Close'].pct_change()

                    # Fetch market data for benchmarking (e.g., S&P 500)
                    market_df = yf.download('^GSPC', period=period)
                    market_df['Returns'] = market_df['Close'].pct_change()

                    # Calculate technical indicators
                    df = get_technical_indicators(df)

                    # Calculate fundamental ratios
                    pe_ratio, pb_ratio, roe = get_fundamental_ratios(ticker)

                    # Calculate beta and VaR
                    beta = calculate_beta(df, market_df)
                    var = calculate_var(df)

                    # Fetch and analyze news sentiment
                    news_articles = fetch_news(ticker)
                    sentiment_scores = analyze_sentiment(news_articles)

                    summary_data.append(summarize_analysis(ticker, pe_ratio, pb_ratio, roe, beta, var))

                    col1, col2, col3 = st.columns([3, 3, 2])

                    with col1:
                        # Plotting the closing price with SMA
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'))
                        fig.update_layout(title=f'Price and SMAs for {ticker}', xaxis_title='Date', yaxis_title='Price')
                        st.plotly_chart(fig)
                        st.caption("SMA (Simple Moving Average) indicates the average price over a specific period. It's used to smooth out price data and identify trends.")

                        # Plotting RSI
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
                        fig.update_layout(title=f'RSI for {ticker}', xaxis_title='Date', yaxis_title='RSI')
                        st.plotly_chart(fig)
                        st.caption("RSI (Relative Strength Index) measures the speed and change of price movements. Values above 70 indicate overbought conditions, while values below 30 indicate oversold conditions.")

                        # Plotting MACD
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
                        fig.update_layout(title=f'MACD for {ticker}', xaxis_title='Date', yaxis_title='MACD')
                        st.plotly_chart(fig)
                        st.caption("MACD (Moving Average Convergence Divergence) indicates the relationship between two moving averages. It's used to identify changes in the strength, direction, momentum, and duration of a trend.")

                    with col2:
                        # Plotting EMA
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], mode='lines', name='EMA'))
                        fig.update_layout(title=f'Price and EMA for {ticker}', xaxis_title='Date', yaxis_title='Price')
                        st.plotly_chart(fig)
                        st.caption("EMA (Exponential Moving Average) gives more weight to recent prices. It's used to identify trends and potential reversals.")

                        # Plotting CCI
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['CCI'], mode='lines', name='CCI'))
                        fig.update_layout(title=f'CCI for {ticker}', xaxis_title='Date', yaxis_title='CCI')
                        st.plotly_chart(fig)
                        st.caption("CCI (Commodity Channel Index) measures the current price level relative to an average price level over a given period. Values above 100 indicate overbought conditions, while values below -100 indicate oversold conditions.")

                        # Plotting ROC
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['ROC'], mode='lines', name='ROC'))
                        fig.update_layout(title=f'ROC for {ticker}', xaxis_title='Date', yaxis_title='ROC')
                        st.plotly_chart(fig)
                        st.caption("ROC (Rate of Change) measures the percentage change in price between the current price and the price a certain number of periods ago. It's used to identify momentum.")

                    with col3:
                        st.subheader("Sentiment Analysis")
                        for sentiment in sentiment_scores[:4]:
                            st.write(f"{sentiment['title']}")
                            st.caption(f"{sentiment['sentiment']} ({sentiment['score']})")
                            st.write(f"[Read article.]({sentiment['link']})")
                            st.divider()
                        st.caption("""
                        **Sentiment Analysis:**
                        - This analysis determines the overall sentiment of recent news articles about the stock. 
                        - Positive sentiment indicates favorable news, while negative sentiment indicates unfavorable news.
                        """)

                    st.divider()

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Fundamental Analysis")
                        insights = generate_insights(pe_ratio, pb_ratio, roe, beta, var)
                        st.write("P/E Ratio:", pe_ratio)
                        st.caption(insights['pe_ratio'].split('\n')[1])
                        st.write("P/B Ratio:", pb_ratio)
                        st.caption(insights['pb_ratio'].split('\n')[1])
                        st.write("ROE Ratio:", roe)
                        st.caption(insights['roe'].split('\n')[1])

                    with col2:
                        st.subheader("Risk Analysis")
                        st.write("Beta", beta)
                        st.caption(insights['beta'].split('\n')[1])
                        st.write("VaR", var)
                        st.caption(insights['var'].split('\n')[1])

                if summary_data:
                    st.subheader("Summary Table")
                    summary_df = pd.DataFrame(summary_data)
                    st.table(summary_df)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    show()
