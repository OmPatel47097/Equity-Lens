import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from utils.IndicatorGenerator import IndicatorGenerator
from utils.StockDataManager import StockDataManager
from utils.LoggerManager import LoggerManager
from BlackLittermanOptimization import BlackLittermanOptimization
from dotenv import load_dotenv
import os
load_dotenv()

# Constants - Made them Secrets
SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH', 60))
BATCH_SIZE = int(os.getenv('BATCH_SIZE_SMP', 64))
EPOCHS = int(os.getenv('EPOCHS', 50))
LEARNING_RATE = float(os.getenv('LEARNING_RATE_SMP', 0.001))

logger = LoggerManager.get_logger(__name__)

class StockPricePrediction:
    def __init__(self, prediction_days):
        self.prediction_days = prediction_days
        self.stockDataManager = StockDataManager()
        self.scaler = StandardScaler()

    def load_data(self, ticker, period='5y'):
        data = self.stockDataManager.history(ticker, period, interval='1d')
        if data is None or data.empty:
            logger.error(f"No data found for {ticker} with period {period}")
            return None
        data.dropna(inplace=True)
        logger.debug(f"Loaded data for {ticker}: {data.head()}")
        return data

    def create_features(self, data):
        if data is None or data.empty:
            logger.error("No data available for feature creation")
            return None
        indicator_generator = IndicatorGenerator(data)
        data['ma50'] = indicator_generator.moving_average(50)
        data['ema50'] = indicator_generator.exponential_moving_average(50)
        data['macd'], data['macd_signal'] = indicator_generator.macd()
        data['rsi'] = indicator_generator.rsi()
        data['stochastic'] = indicator_generator.stochastic_oscillator()
        data['cci'] = indicator_generator.cci()
        data['roc'] = indicator_generator.roc()
        # data['bb_middle'], data['bb_upper'], data['bb_lower'] = indicator_generator.bollinger_bands()
        # data['atr'] = indicator_generator.atr()
        # data['obv'] = indicator_generator.obv()
        # data['ad_line'] = indicator_generator.ad_line()
        # data['tenkan_sen'], data['kijun_sen'], data['senkou_span_a'], data['senkou_span_b'], data['chikou_span'] = indicator_generator.ichimoku_cloud()
        # data['parabolic_sar'] = indicator_generator.parabolic_sar()
        # data['williams_r'] = indicator_generator.williams_r()
        data.dropna(inplace=True)
        logger.debug(f"Features created: {data.head()}")
        return data

    def normalize_data(self, data):
        if data is None or data.empty:
            logger.error("Data is empty, cannot normalize")
            return None
        if 'adj_close' not in data.columns:
            logger.error(f"'adj_close' column missing in data: {data.columns}")
            return None
        data_numeric = data.drop(columns=['symbol'])  # Exclude the 'symbol' column
        scaled_data = self.scaler.fit_transform(data_numeric)
        return pd.DataFrame(scaled_data, columns=data_numeric.columns, index=data.index)

    def create_sequences(self, data):
        if data is None or data.empty:
            logger.error("No data available to create sequences")
            return None, None
        X, y = [], []
        for i in range(len(data) - SEQUENCE_LENGTH - self.prediction_days + 1):
            X.append(data.iloc[i:i+SEQUENCE_LENGTH].values)
            y.append(data.iloc[i+SEQUENCE_LENGTH+self.prediction_days-1]['adj_close'])
        if len(X) == 0 or len(y) == 0:
            logger.error("Insufficient data to create sequences")
            return None, None
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
        return model

    def train_model(self, model, X_train, y_train):
        if X_train is None or y_train is None or len(X_train) == 0 or len(y_train) == 0:
            logger.error("No training data available")
            return None
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[early_stopping])
        return history

    def predict(self, model, X_test):
        if X_test is None or len(X_test) == 0:
            logger.error("No test data available for prediction")
            return None
        predictions = model.predict(X_test)
        return predictions

    def predict_single_stock_price(self, ticker):
        try:
            data = self.load_data(ticker)
            print(data.shape)
            if data is None:
                return None
            data = self.create_features(data)
            print(data.shape)
            if data is None:
                return None
            data_normalized = self.normalize_data(data)
            if data_normalized is None:
                return None
            X, y = self.create_sequences(data_normalized)
            if X is None or y is None:
                return None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = self.build_model((SEQUENCE_LENGTH, X.shape[2]))
            self.train_model(model, X_train, y_train)

            predicted_price = self.predict(model, X_test[-1].reshape(1, SEQUENCE_LENGTH, X.shape[2]))[0][0]
            if predicted_price is None:
                return None
            predicted_price = self.scaler.inverse_transform([[predicted_price] + [0]*(X.shape[2]-1)])[0][0]

            logger.info(f"Predicted price for {ticker} after {self.prediction_days} days: {predicted_price}")
            return predicted_price
        except Exception as e:
            logger.error(e)
            return None

    def calculate_portfolio_value(self, capital, tickers):
        try:
            results = []
            for ticker in tickers:
                data = self.load_data(ticker)
                if data is None:
                    continue
                data = self.create_features(data)
                if data is None:
                    continue
                data_normalized = self.normalize_data(data)
                if data_normalized is None:
                    continue
                X, y = self.create_sequences(data_normalized)
                if X is None or y is None:
                    continue
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = self.build_model((SEQUENCE_LENGTH, X.shape[2]))
                self.train_model(model, X_train, y_train)

                predicted_price = self.predict(model, X_test[-1].reshape(1, SEQUENCE_LENGTH, X.shape[2]))[0][0]
                if predicted_price is None:
                    continue
                predicted_price = self.scaler.inverse_transform([[predicted_price] + [0]*(X.shape[2]-1)])[0][0]

                latest_price = data['adj_close'].iloc[-1]
                num_shares = round(capital / latest_price)
                predicted_value = num_shares * predicted_price

                results.append({
                    'ticker': ticker,
                    'latest_price': latest_price,
                    'predicted_price': predicted_price,
                    'num_shares': num_shares,
                    'predicted_value': predicted_value
                })

            logger.info(results)
            return results
        except Exception as e:
            logger.error(e)
            return []

if __name__ == '__main__':
    prediction_days = 30  # Predict the stock price 30 days into the future
    spp = StockPricePrediction(prediction_days)

    # Predict the price of a single stock
    ticker = "AAPL"
    predicted_price = spp.predict_single_stock_price(ticker)
    print(f"Predicted price for {ticker} after {prediction_days} days: {predicted_price}")

    # Calculate the portfolio value with a given capital
    capital = 100000
    tickers = ["AAPL", "MSFT", "GOOGL"]
    portfolio_value = spp.calculate_portfolio_value(capital, tickers)
    print(f"Portfolio value after {prediction_days} days: {portfolio_value}")
