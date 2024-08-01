import numpy as np
import pandas as pd
from keras import Input, Model
from keras.src.layers import concatenate
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
import os
import tensorflow as tf


pd.options.mode.chained_assignment = None

from utils.IndicatorGenerator import IndicatorGenerator
from utils.StockDataManager import StockDataManager


class StockTrendPrediction:

    def __init__(self, sequence_length=60):
        self.model = None
        self.df_test = None
        self.df_train = None
        self.scaler = None
        self.data = None
        self.sequence_length = sequence_length
        self.stockDataManager = StockDataManager()

    def load_data(self, ticker, period='2y', interval = '1d'):
        self.data = self.stockDataManager.history(ticker, period=period, interval=interval)
        self.data = self.data[:int(len(self.data)*60)]
        self.data.drop(columns=['symbol'], inplace=True)

    def create_features(self):
        indicatorGen = IndicatorGenerator(self.data)
        rsi = indicatorGen.rsi()
        macd, signal = indicatorGen.macd()
        ema_5 = indicatorGen.exponential_moving_average(5)
        ema_20 = indicatorGen.exponential_moving_average(20)
        ema_50 = indicatorGen.exponential_moving_average(50)
        stochastic_oscillator = indicatorGen.stochastic_oscillator()
        cci = indicatorGen.cci()
        roc = indicatorGen.roc()
        self.data['rsi'] = rsi
        self.data['macd'] = macd
        self.data['macd_signal'] = signal
        self.data['ema_5'] = ema_5
        self.data['ema_20'] = ema_20
        self.data['ema_50'] = ema_50
        self.data['stochastic_oscillator'] = stochastic_oscillator
        self.data['cci'] = cci
        self.data['roc'] = roc

        self.data.bfill(inplace=True)

    def split_data(self):
        data_len = len(self.data)
        self.df_train = self.data[:int(data_len * 0.8)]
        self.df_test = self.data[int(data_len * 0.8):]

    def normalize_data(self):
        columns = self.data.columns
        self.scaler = StandardScaler()
        self.scaler.fit(self.df_train[columns])
        self.df_train[columns] = self.scaler.transform(self.df_train[columns])
        self.df_test[columns] = self.scaler.transform(self.df_test[columns])

    def create_sequences(self, data):
        features = self.data.columns
        target = 'adj_close'
        X = []
        y = []
        x_raw = data[features].values
        y_raw = data[target].values
        for i in range(x_raw.shape[0] - self.sequence_length):
            X.append(x_raw[i:i+self.sequence_length])
            y.append(y_raw[i+self.sequence_length ])
        return np.array(X), np.array(y)

    def build_model(self):
        input_layer = Input(shape=(self.sequence_length, self.data.shape[1]))

        # LSTM
        lstm1 = LSTM(units=50, return_sequences=True)(input_layer)
        lstm2 = LSTM(units=50, return_sequences=False)(lstm1)

        # CNN
        cnn1 = Conv1D(filters=64, kernel_size=2, activation='relu')(input_layer)
        max_pool = MaxPooling1D(pool_size=2)(cnn1)
        flatten = Flatten()(max_pool)

        combined = concatenate([lstm2, flatten])
        dense1 = Dense(units=64, activation='relu')(combined)
        dropout = Dropout(0.5)(dense1)
        output = Dense(1)(dropout)

        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(loss='mse', optimizer='adam')

    def train_model(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=256)

    def cross_validation_train(self, X_train, y_train, X_test, y_test):
        tscv = TimeSeriesSplit(n_splits=5)
        history = []

        for train_index, val_index in tscv.split(X_train):
            X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
            Y_train_cv, Y_val_cv = y_train[train_index], y_train[val_index]

            history.append(self.model.fit(X_train_cv, Y_train_cv, epochs=50, batch_size=32, validation_data=(X_val_cv, Y_val_cv)))

        # Evaluate on the test set
        test_loss = self.model.evaluate(X_test, y_test)
        print(f'Test Loss: {test_loss}')


    def _build_hypertune_model(self,hp):
        input_layer = Input(shape=(self.sequence_length, self.data.shape[1]))

        lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=16)
        lstm_1 = LSTM(lstm_units, return_sequences=True)(input_layer)
        lstm_2 = LSTM(lstm_units, return_sequences=False)(lstm_1)

        cnn_filters = hp.Int('cnn_filters', min_value=32, max_value=128, step=16)
        cnn_1 = Conv1D(filters=cnn_filters, kernel_size=2, activation='relu')(input_layer)
        max_pooling = MaxPooling1D(pool_size=2)(cnn_1)
        flatten = Flatten()(max_pooling)

        dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=16)
        combined = concatenate([lstm_2, flatten])
        dense_1 = Dense(dense_units, activation='relu')(combined)
        dropout_rate = hp.Float(
            "dropout_rate",
            min_value=0,
            max_value=0.5)
        dropout = Dropout(dropout_rate)(dense_1)
        output_layer = Dense(1)(dropout)

        self.model = Model(inputs=input_layer, outputs=output_layer )
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model

    def hypertune_model(self, X_train, y_train, X_test, y_test):
        tuner = kt.RandomSearch(self._build_hypertune_model, objective='val_loss', max_trials=10, executions_per_trial=1, directory='hypertune', project_name='stock_prediction')
        tuner.search(X_train, y_train, epochs=50,batch_size=32, validation_split=0.2)
        best_model = tuner.get_best_models(num_models=1)[0]

        best_model_loss = best_model.evaluate(X_test, y_test)
        print(f'Best model loss: {best_model_loss}')
        best_model.save(f'../models/stock_prediction_model_{best_model_loss}.h5')
        return best_model

    def predict(self, data):
        model_path = os.path.join('models','stock_prediction_model_1.h5')
        model = tf.keras.models.load_model(model_path)
        return model.predict(data)

    def train_and_predict(self, ticker, period='2y', interval='1d'):
        stockTrendPrediction = StockTrendPrediction()
        stockTrendPrediction.load_data(ticker, period=period, interval=interval)
        stockTrendPrediction.create_features()
        stockTrendPrediction.split_data()
        stockTrendPrediction.normalize_data()
        X_train, y_train = stockTrendPrediction.create_sequences(stockTrendPrediction.df_train)
        X_test, y_test = stockTrendPrediction.create_sequences(stockTrendPrediction.df_test)
        model = stockTrendPrediction.hypertune_model(X_train, y_train, X_test, y_test)
        test_prediction = model.predict(X_test)
        return test_prediction, y_test

# D:\College\Sem 2\Projects In ML\EquityLens\models\stock_prediction_model_1.h5