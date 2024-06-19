import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class StockClassifier:
    def __init__(self, stock_df, sp500_df, tickers):
        self.stock_df = stock_df
        self.sp500_df = sp500_df
        self.tickers = tickers

    def risk_based_clustering(self):
        stock_data = self.stock_df

        # Step 3: Calculate Risk Metrics
        daily_returns = stock_data.pct_change().dropna()
        volatility = daily_returns.std()

        sp500 = self.sp500_df
        sp500_returns = sp500.pct_change().dropna()

        betas = {}
        for ticker in self.tickers:
            covariance = np.cov(daily_returns[ticker], sp500_returns)[0, 1]
            variance = np.var(sp500_returns)
            beta = covariance / variance
            betas[ticker] = beta

        betas = pd.Series(betas)

        # Step 4: Normalize the Risk Metrics
        risk_metrics = pd.DataFrame({'Volatility': volatility, 'Beta': betas})
        scaler = StandardScaler()
        risk_metrics_scaled = scaler.fit_transform(risk_metrics)

        # Step 5: Classify Stocks Using K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(risk_metrics_scaled)
        risk_metrics['Cluster'] = kmeans.labels_
        return risk_metrics

    def performance_clustering(self):
        daily_returns = self.stock_df.pct_change().dropna()
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        annualized_return = daily_returns.mean() * 252
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        risk_free_rate = 0.01  # Assume a risk-free rate of 1%
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

        # Step 4: Normalize the Performance Metrics
        performance_metrics = pd.DataFrame({
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio
        })
        scaler = StandardScaler()
        performance_metrics_scaled = scaler.fit_transform(performance_metrics)

        # Step 5: Cluster Stocks Using K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(performance_metrics_scaled)
        performance_metrics['Cluster'] = kmeans.labels_
        return performance_metrics
