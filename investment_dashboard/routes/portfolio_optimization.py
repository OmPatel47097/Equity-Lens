import sys
import os
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# Add the parent directory of the current script to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from services.BlackLittermanOptimization import BlackLittermanOptimization

def show():
    st.title("Portfolio Optimization")

    st.header("Portfolio Settings")


    # Input for initial capital
    capital = st.number_input("Initial Capital ($)", min_value=1000, value=100000, step=1000)

    # Input for tickers
    tickers = st.text_input("Tickers (comma separated)", value="AAPL,MSFT,AMZN,GOOGL,META,MMM")
    tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

    if st.button("Optimize Portfolio"):
        with st.spinner("Optimizing..."):
            try:
                bl = BlackLittermanOptimization()
                result = bl.calculate_portfolio_value(capital, tickers)

                st.divider()
                if result:



                    col1, col2 = st.columns(2)

                    # Create a DataFrame for display
                    df = pd.DataFrame(result)

                    with col1:
                        st.subheader("Optimized Portfolio")
                        st.dataframe(df)
                        st.write("Total investment value: $", round(sum(df['investment']), 2))

                    with col2:
                        # Asset Allocation Pie Chart using Plotly
                        fig = px.pie(df, values='investment', names='ticker', title='Asset Allocation')
                        st.plotly_chart(fig)

                    # Fetch historical data for cumulative return comparison
                    st.subheader("Portfolio vs Benchmark(S&P 500)")
                    portfolio_hist = yf.download(tickers, period="1y", interval="1d")['Adj Close']
                    sp500_hist = yf.download("^GSPC", period="1y", interval="1d")['Adj Close']

                    # Calculate cumulative returns
                    portfolio_cum_returns = (portfolio_hist / portfolio_hist.iloc[0]).mean(axis=1)
                    sp500_cum_returns = sp500_hist / sp500_hist.iloc[0]

                    # Create a DataFrame for cumulative returns
                    cum_returns_df = pd.DataFrame({
                        'Date': portfolio_cum_returns.index,
                        'Portfolio': portfolio_cum_returns.values,
                        'S&P 500': sp500_cum_returns.values
                    })

                    # Plot cumulative returns using Plotly
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=cum_returns_df['Date'], y=cum_returns_df['Portfolio'], mode='lines', name='Portfolio'))
                    fig.add_trace(go.Scatter(x=cum_returns_df['Date'], y=cum_returns_df['S&P 500'], mode='lines', name='S&P 500'))
                    fig.update_layout(title='Cumulative Portfolio Return vs S&P 500', xaxis_title='Date', yaxis_title='Cumulative Return')
                    st.plotly_chart(fig)

                else:
                    st.error("Could not optimize portfolio. Please check the tickers and try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    show()
