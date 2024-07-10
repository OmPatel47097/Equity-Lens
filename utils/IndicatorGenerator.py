import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


class IndicatorGenerator:
    def __init__(self, data):
        self.data = data

    def moving_average(self, window=50):
        return self.data['adj_close'].rolling(window=window).mean()

    def exponential_moving_average(self, span=50):
        return self.data['adj_close'].ewm(span=span, adjust=False).mean()

    def macd(self, short_span=12, long_span=26, signal_span=9):

        adj_close = self.data['adj_close']
        short_ema = adj_close.ewm(span=short_span, adjust=False).mean()
        long_ema = adj_close.ewm(span=long_span, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_span, adjust=False).mean()
        return macd, signal

    def rsi(self, window=14):
        delta = self.data['adj_close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def stochastic_oscillator(self, window=14):
        low_min = self.data['low'].rolling(window=window).min()
        high_max = self.data['high'].rolling(window=window).max()
        return 100 * (self.data['adj_close'] - low_min) / (high_max - low_min)

    def cci(self, window=20):
        tp = (self.data['high'] + self.data['low'] + self.data['adj_close']) / 3
        tp_sma = tp.rolling(window=window).mean()
        mean_deviation = (tp - tp_sma).abs().rolling(window=window).mean()
        return (tp - tp_sma) / (0.015 * mean_deviation)

    def roc(self, window=12):
        return ((self.data['adj_close'] - self.data['adj_close'].shift(window)) / self.data['adj_close'].shift(window)) * 100

    def bollinger_bands(self, window=20):
        sma = self.data['adj_close'].rolling(window=window).mean()
        std = self.data['adj_close'].rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return sma, upper_band, lower_band

    def atr(self, window=14):
        high_low = self.data['high'] - self.data['low']
        high_adj_close = np.abs(self.data['high'] - self.data['adj_close'].shift())
        low_adj_close = np.abs(self.data['low'] - self.data['adj_close'].shift())
        tr = high_low.combine(high_adj_close, max).combine(low_adj_close, max)
        return tr.rolling(window=window).mean()

    def obv(self):
        obv = np.where(self.data['adj_close'] > self.data['adj_close'].shift(), self.data['volume'], -self.data['volume'])
        return pd.Series(obv).cumsum()

    def ad_line(self):
        mfm = ((self.data['adj_close'] - self.data['low']) - (self.data['high'] - self.data['adj_close'])) / (
                    self.data['high'] - self.data['low'])
        mfv = mfm * self.data['volume']
        return mfv.cumsum()

    def ichimoku_cloud(self):
        high_9 = self.data['high'].rolling(window=9).max()
        low_9 = self.data['low'].rolling(window=9).min()
        high_26 = self.data['high'].rolling(window=26).max()
        low_26 = self.data['low'].rolling(window=26).min()
        high_52 = self.data['high'].rolling(window=52).max()
        low_52 = self.data['low'].rolling(window=52).min()

        tenkan_sen = (high_9 + low_9) / 2
        kijun_sen = (high_26 + low_26) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)
        chikou_span = self.data['adj_close'].shift(-26)

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

    def parabolic_sar(self):
        # Simplified version
        psar = self.data['adj_close'][0]
        psar_series = [psar]
        for i in range(1, len(self.data)):
            psar = psar_series[-1] + 0.02 * (self.data['high'][i] - psar_series[-1]) if self.data['adj_close'][i] > \
                                                                                        psar_series[-1] else \
            psar_series[-1] - 0.02 * (psar_series[-1] - self.data['low'][i])
            psar_series.append(psar)
        return pd.Series(psar_series, index=self.data.index)

    def williams_r(self, window=14):
        high_max = self.data['high'].rolling(window=window).max()
        low_min = self.data['low'].rolling(window=window).min()
        return -100 * ((high_max - self.data['adj_close']) / (high_max - low_min))


# Sample usage:
if __name__ == "__main__":
    data = pd.read_csv('../data/processed/GOOGL.csv', parse_dates=['Date'], index_col='Date')
    indicators = IndicatorGenerator(data)

    print(indicators.moving_average())
    print(indicators.rsi())
    print(indicators.bollinger_bands())
