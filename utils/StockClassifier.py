<<<<<<< HEAD
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
=======
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

'''
Data:
symbol, volatility, beta, annualized return, sharpe ratio
'''

class StockClassifier:
    def __init__(self, stock_df, volume_df,sp500_df, tickers):
        self.stock_df = stock_df
        self.sp500_df = sp500_df
        self.volume_df = volume_df
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

        # Step 5: Classify Stocks Using Various Clustering Methods
        risk_metrics['KMeans_Cluster'] = KMeans(n_clusters=3, random_state=0).fit_predict(risk_metrics_scaled)
        risk_metrics['DBSCAN_Cluster'] = DBSCAN(eps=0.5, min_samples=2).fit_predict(risk_metrics_scaled)
        risk_metrics['Agg_Cluster'] = AgglomerativeClustering(n_clusters=3).fit_predict(risk_metrics_scaled)
        risk_metrics['GMM_Cluster'] = GaussianMixture(n_components=3, random_state=0).fit_predict(risk_metrics_scaled)

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

        # Step 5: Classify Stocks Using Various Clustering Methods
        performance_metrics['KMeans_Cluster'] = KMeans(n_clusters=3, random_state=0).fit_predict(performance_metrics_scaled)
        performance_metrics['DBSCAN_Cluster'] = DBSCAN(eps=0.5, min_samples=2).fit_predict(performance_metrics_scaled)
        performance_metrics['Agg_Cluster'] = AgglomerativeClustering(n_clusters=3).fit_predict(performance_metrics_scaled)
        performance_metrics['GMM_Cluster'] = GaussianMixture(n_components=3, random_state=0).fit_predict(performance_metrics_scaled)

        return performance_metrics

    def volume_based_clustering(self):

        volume_data = self.volume_df['Volume']

        # Step 3: Normalize the Volume Data
        scaler = StandardScaler()
        volume_scaled = scaler.fit_transform(volume_data.values.reshape(-1, 1))

        # Step 4: Classify Stocks Using Various Clustering Methods
        volume_data['KMeans_Cluster'] = KMeans(n_clusters=3, random_state=0).fit_predict(volume_scaled)
        volume_data['DBSCAN_Cluster'] = DBSCAN(eps=0.5, min_samples=2).fit_predict(volume_scaled)
        volume_data['Agg_Cluster'] = AgglomerativeClustering(n_clusters=3).fit_predict(volume_scaled)
        volume_data['GMM_Cluster'] = GaussianMixture(n_components=3, random_state=0).fit_predict(volume_scaled)

        return volume_data

    def price_based_clustering(self):
        price_data = self.stock_df['Close']

        # Step 3: Normalize the Price Data
        scaler = StandardScaler()
        price_scaled = scaler.fit_transform(price_data.values.reshape(-1, 1))

        # Step 4: Classify Stocks Using Various Clustering Methods
        price_data['KMeans_Cluster'] = KMeans(n_clusters=3, random_state=0).fit_predict(price_scaled)
        price_data['DBSCAN_Cluster'] = DBSCAN(eps=0.5, min_samples=2).fit_predict(price_scaled)
        price_data['Agg_Cluster'] = AgglomerativeClustering(n_clusters=3).fit_predict(price_scaled)
        price_data['GMM_Cluster'] = GaussianMixture(n_components=3, random_state=0).fit_predict(price_scaled)

        return price_data

    def fundamental_based_clustering(self, fundamentals_df):
        # Step 3: Normalize the Fundamental Data
        scaler = StandardScaler()
        fundamentals_scaled = scaler.fit_transform(fundamentals_df)

        # Step 4: Classify Stocks Using Various Clustering Methods
        fundamentals_df['KMeans_Cluster'] = KMeans(n_clusters=3, random_state=0).fit_predict(fundamentals_scaled)
        fundamentals_df['DBSCAN_Cluster'] = DBSCAN(eps=0.5, min_samples=2).fit_predict(fundamentals_scaled)
        fundamentals_df['Agg_Cluster'] = AgglomerativeClustering(n_clusters=3).fit_predict(fundamentals_scaled)
        fundamentals_df['GMM_Cluster'] = GaussianMixture(n_components=3, random_state=0).fit_predict(fundamentals_scaled)

        return fundamentals_df


>>>>>>> S_2_UC2_Retail_Investor_Enhancement
