import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# Function to download stock data
def get_data(symbols, start, end):
    data = yf.download(symbols, start=start, end=end)['Adj Close']
    returns = data.pct_change().dropna()
    return data, returns


# Function to optimize portfolio
def optimize_portfolio(returns, mean_returns, cov_matrix, risk_tolerance):
    mean_returns = np.array(mean_returns)
    n_assets = len(mean_returns)
    weights = cp.Variable(n_assets)

    portfolio_return = mean_returns @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix)

    # Calculate risk aversion factor based on risk tolerance
    # A lower risk tolerance should lead to a higher risk aversion factor
    risk_aversion = 1 / risk_tolerance

    # Define the objective as a trade-off between return and risk
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return weights.value


# Streamlit app
def main():
    st.title("Portfolio Optimization App")

    # User inputs
    st.sidebar.header("User Inputs")
    symbols = st.sidebar.text_input("Enter stock symbols (comma separated)", "AAPL, MSFT, GOOGL, AMZN, META")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    risk_tolerance = st.sidebar.slider("Risk Tolerance (Lower is more risk-averse)", 0.01, 0.10, step=0.01, value=0.02)

    # Process symbols
    symbols = [symbol.strip() for symbol in symbols.split(',')]

    # Get data
    data, returns = get_data(symbols, start_date, end_date)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Optimize portfolio
    optimal_weights = optimize_portfolio(returns, mean_returns, cov_matrix, risk_tolerance)

    # Display results
    st.subheader("Optimal Weights")
    for symbol, weight in zip(symbols, optimal_weights):
        st.write(f"{symbol}: {weight:.4f}")

    # Plot cumulative returns
    st.subheader("Cumulative Returns")
    portfolio_returns = returns.dot(optimal_weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    fig, ax = plt.subplots()
    cumulative_returns.plot(ax=ax, title='Cumulative Returns')
    st.pyplot(fig)


if __name__ == "__main__":
    main()
