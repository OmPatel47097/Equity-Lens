import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize

from utils.StockDataManager import StockDataManager


# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        return x


# GAN models
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class BlackLittermanOptimization:
    def __init__(self):
        self.stockDataManger = StockDataManager()

    def collect_data(self, tickers, period):
        # TODO: collect data from local datasource
        '''
        Collect historical data for the given tickers and periods
        :param tickers: list of tickers to optimize the portfolio
        :param period: period to collect the data
        :return: the dataframe
        '''

        data = self.stockDataManger.adj_close(tickers, interval='1d')
        data.dropna(inplace=True)
        if len(data) > 5 * 252:
            data = data.tail(5 * 252)
        return data

    def normalize_data(self, data, tickers):
        '''
        Normalize the data using MinMaxScaler
        :param data: the dataframe to normalize
        :return: the normalized dataframe
        '''
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data)
        print(tickers)
        data_normalized = pd.DataFrame(data_normalized, columns=tickers, index=data.index)
        return data_normalized

    def train_generator(self, n_assets, data):
        # TODO: Hypertune model parameters
        '''
        Train the generator model
        :param data: the normalized historical data
        :param n_assets: number of assets
        :return: the trained generator model
        '''
        input_dim = n_assets
        output_dim = n_assets
        model_dim = 64
        num_heads = 4
        num_layers = 2

        # Initialize models
        transformer = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)
        generator = Generator(input_dim, output_dim)
        discriminator = Discriminator(output_dim)

        # Loss and optimizers
        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

        # Training loop for GAN
        num_epochs = 200
        for epoch in range(num_epochs):
            for real_data in DataLoader(TensorDataset(torch.tensor(data.values, dtype=torch.float32)),batch_size=128, shuffle=True):
                real_data = real_data[0]
                batch_size = real_data.size(0)

                # Train Discriminator
                noise = torch.randn(batch_size, input_dim)
                fake_data = generator(noise)
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                optimizer_d.zero_grad()
                outputs = discriminator(real_data)
                real_loss = criterion(outputs, real_labels)
                outputs = discriminator(fake_data.detach())
                fake_loss = criterion(outputs, fake_labels)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()
                outputs = discriminator(fake_data)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                optimizer_g.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
        return generator

    def optimize_weights(self, data, n_assets, generator, delta=2.5, tau=0.05, risk_free_rate=0.02):
        '''
        Optimize the weights using the Black-Litterman model
        :param data: the dataframe with the normalized historical data
        :param n_assets: number of assets
        :param generator: the trained generator model
        :param delta: risk aversion parameter
        :param tau: uncertainty parameter
        :param risk_free_rate: risk free rate
        :return: the optimized weights
        '''

        # Calculate the implied equilibrium returns (Π)
        cov_matrix = data.cov()
        market_weights = np.ones(n_assets) / n_assets
        pi = delta * np.dot(cov_matrix, market_weights)

        views = generator(torch.randn(n_assets, n_assets)).detach().numpy()

        # Calculate the Omega matrix (uncertainty in views)
        P = np.eye(n_assets)
        Q = views.mean(axis=0)
        omega = np.diag(np.diag(np.dot(np.dot(P, tau * cov_matrix), P.T)))

        # Adjusted covariance matrix (Σ̂)
        adjusted_cov_matrix = cov_matrix + np.linalg.inv(
            np.linalg.inv(tau * cov_matrix) + np.dot(np.dot(P.T, np.linalg.inv(omega)), P))

        # Black-Litterman expected returns (µBL)
        mu_bl = np.dot(np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(np.dot(P.T, np.linalg.inv(omega)), P)),
                       np.dot(np.linalg.inv(tau * cov_matrix), pi) + np.dot(np.dot(P.T, np.linalg.inv(omega)), Q))

        # Portfolio optimization (maximizing Sharpe Ratio)
        def objective(weights):
            return - (np.dot(weights, mu_bl) - risk_free_rate) / np.sqrt(
                np.dot(np.dot(weights, adjusted_cov_matrix), weights.T))

        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = [(0, 1) for _ in range(n_assets)]
        initial_guess = market_weights

        result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
        optimal_weights = result.x

        print('Optimal Portfolio Weights:', optimal_weights)
        return optimal_weights

    def calulate_portfolio_value(self, capital, tickers):
        '''
        Calculate the portfolio value using the optimized weights
        :param capital: the initial capital
        :param tickers: the list of tickers
        :return: the portfolio value
        '''

        data = self.collect_data(tickers, '3y')
        data_normalized = self.normalize_data(data, tickers)
        generator = self.train_generator(len(tickers), data_normalized)
        optimal_weights = self.optimize_weights(data_normalized, len(tickers), generator)
        output = []
        for i in range(len(optimal_weights)):
            investment = capital * optimal_weights[i]
            print(f'Investment in {tickers[i]}: ${investment:.2f}')
            # obj = {}
            # obj['ticker'] =
            obj = {'ticker': tickers[i], 'investment': investment, 'weight': optimal_weights[i]}
            output.append(obj)
        return output

# if __name__ == '__main__':
#     # TODO: Calculate the no of shares to buy
#     capital = 100000
#     tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
#     bl = BlackLittermanOptimization()
#     bl.calulate_portfolio_value(capital, tickers)
