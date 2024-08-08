import mlflow
import mlflow.pytorch
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from typing import List, Tuple
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from BlackLittermanOptimization import (
    Generator,
    BlackLittermanOptimization,
)


@task
def collect_data(tickers: List[str], period: str) -> pd.DataFrame:
    bl_optimizer = BlackLittermanOptimization()
    return bl_optimizer.collect_data(tickers, period)


@task
def preprocess_data(
    data: pd.DataFrame, tickers: List[str]
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    data_normalized = pd.DataFrame(data_normalized, columns=tickers, index=data.index)
    return data_normalized, scaler


@task
def train_model(data: pd.DataFrame, n_assets: int) -> Generator:
    bl_optimizer = BlackLittermanOptimization()
    return bl_optimizer.train_generator(n_assets, data)


@task
def optimize_weights(
    data: pd.DataFrame, n_assets: int, generator: Generator
) -> List[float]:
    bl_optimizer = BlackLittermanOptimization()
    return bl_optimizer.optimize_weights(data, n_assets, generator)


@task
def calculate_portfolio(
    capital: float, tickers: List[str], optimal_weights: List[float]
) -> List[dict]:
    bl_optimizer = BlackLittermanOptimization()
    return bl_optimizer.calculate_portfolio_value(capital, tickers)


@flow(task_runner=SequentialTaskRunner())
def portfolio_optimization_pipeline(
    capital: float, tickers: List[str], period: str = "3y"
):
    with mlflow.start_run():
        # Data Collection
        data = collect_data(tickers, period)
        mlflow.log_param("tickers", tickers)
        mlflow.log_param("period", period)

        # Preprocessing
        data_normalized, scaler = preprocess_data(data, tickers)
        mlflow.sklearn.log_model(scaler, "data_scaler")

        # Model Training
        n_assets = len(tickers)
        generator = train_model(data_normalized, n_assets)
        mlflow.pytorch.log_model(generator, "generator_model")

        # Optimization
        optimal_weights = optimize_weights(data_normalized, n_assets, generator)
        mlflow.log_param("optimal_weights", optimal_weights)

        # Portfolio Calculation
        portfolio = calculate_portfolio(capital, tickers, optimal_weights)
        mlflow.log_param("portfolio", portfolio)

        return portfolio


if __name__ == "__main__":
    capital = 100000
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    result = portfolio_optimization_pipeline(capital, tickers)
    print(result)
