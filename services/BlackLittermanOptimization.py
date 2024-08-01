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
from utils.LoggerManager import LoggerManager

# Constants
MODEL_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
OUTPUT_DIM = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 500
BATCH_SIZE = 128
RISK_FREE_RATE = 0.02
DELTA = 2.5
TAU = 0.05

logger = LoggerManager.get_logger(__name__)

class Generator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class BlackLittermanOptimization:
    def __init__(self):
        self.stockDataManager = StockDataManager()

    def collect_data(self, tickers: list[str], period: str) -> pd.DataFrame:
        """
        Collect historical data for the given tickers and periods.
        :param tickers: List of tickers to optimize the portfolio.
        :param period: Period to collect the data.
        :return: The dataframe with historical data.
        """
        data = self.stockDataManager.adj_close(tickers, interval='1d')
        data.dropna(inplace=True)
        if len(data) > 5 * 252:
            data = data.tail(5 * 252)
        return data

    def normalize_data(self, data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """
        Normalize the data using MinMaxScaler.
        :param data: The dataframe to normalize.
        :param tickers: List of tickers.
        :return: The normalized dataframe.
        """
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data)
        data_normalized = pd.DataFrame(data_normalized, columns=tickers, index=data.index)
        return data_normalized

    def train_generator(self, n_assets: int, data: pd.DataFrame) -> Generator:
        """
        Train the generator model.
        :param data: The normalized historical data.
        :param n_assets: Number of assets.
        :return: The trained generator model.
        """
        input_dim = n_assets
        output_dim = n_assets

        generator = Generator(input_dim, output_dim)
        discriminator = Discriminator(output_dim)

        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            for real_data in DataLoader(TensorDataset(torch.tensor(data.values, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True):
                real_data = real_data[0]
                batch_size = real_data.size(0)

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

                optimizer_g.zero_grad()
                outputs = discriminator(fake_data)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                optimizer_g.step()

            logger.info(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
        return generator

    def optimize_weights(self, data: pd.DataFrame, n_assets: int, generator: Generator, delta: float = DELTA, tau: float = TAU, risk_free_rate: float = RISK_FREE_RATE) -> np.ndarray:
        """
        Optimize the weights using the Black-Litterman model.
        :param data: The dataframe with the normalized historical data.
        :param n_assets: Number of assets.
        :param generator: The trained generator model.
        :param delta: Risk aversion parameter.
        :param tau: Uncertainty parameter.
        :param risk_free_rate: Risk-free rate.
        :return: The optimized weights.
        """
        cov_matrix = data.cov()
        market_weights = np.ones(n_assets) / n_assets
        pi = delta * np.dot(cov_matrix, market_weights)

        views = generator(torch.randn(n_assets, n_assets)).detach().numpy()

        P = np.eye(n_assets)
        Q = views.mean(axis=0)
        omega = np.diag(np.diag(np.dot(np.dot(P, tau * cov_matrix), P.T)))

        adjusted_cov_matrix = cov_matrix + np.linalg.inv(
            np.linalg.inv(tau * cov_matrix) + np.dot(np.dot(P.T, np.linalg.inv(omega)), P))

        mu_bl = np.dot(np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(np.dot(P.T, np.linalg.inv(omega)), P)),
                       np.dot(np.linalg.inv(tau * cov_matrix), pi) + np.dot(np.dot(P.T, np.linalg.inv(omega)), Q))

        def objective(weights: np.ndarray) -> float:
            return - (np.dot(weights, mu_bl) - risk_free_rate) / np.sqrt(
                np.dot(np.dot(weights, adjusted_cov_matrix), weights.T))

        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = [(0, 1) for _ in range(n_assets)]
        initial_guess = market_weights

        result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
        optimal_weights = result.x

        logger.info(f'Optimal Portfolio Weights: {optimal_weights}')
        return optimal_weights

    def calculate_portfolio_value(self, capital: float, tickers: list[str]) -> list[dict]:
        """
        Calculate the portfolio value using the optimized weights.
        :param capital: The initial capital.
        :param tickers: The list of tickers.
        :return: The portfolio value.
        """
        try:
            data = self.collect_data(tickers, '3y')
            data_normalized = self.normalize_data(data, tickers)
            generator = self.train_generator(len(tickers), data_normalized)
            optimal_weights = self.optimize_weights(data_normalized, len(tickers), generator)

            output = []

            for i, weight in enumerate(optimal_weights):
                investment = capital * weight
                ticker = tickers[i]

                stock = yf.Ticker(ticker)
                latest_price = stock.history(period="1d")['Close'].iloc[-1]

                num_shares = round(investment / latest_price)  # Round the number of shares to an integer

                output.append({
                    'ticker': ticker,
                    'investment': investment,
                    'weight': weight,
                    'latest_price': latest_price,
                    'num_shares': num_shares
                })

            logger.info(output)
            return output
        except Exception as e:
            logger.error(e)
            return []

if __name__ == '__main__':
    capital = 100000
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    bl = BlackLittermanOptimization()
    result = bl.calculate_portfolio_value(capital, tickers)
    print(result)
