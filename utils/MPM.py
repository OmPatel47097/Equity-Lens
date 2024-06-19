# import pandas as pd
# import numpy as np
import yfinance as yf
# from arch import arch_model
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# def fetch_data(tickers, start_date, end_date):
#     data = yf.download(tickers, start=start_date, end=end_date)
#     return data['Adj Close']
#
# def compute_cov_matrix(returns):
#     dcc = FitDCC(returns)
#     dcc_fitted = dcc.fit()
#     cov_matrices = dcc_fitted.cov_matrices
#     return cov_matrices
#
# def compute_nrp_weights(cov_matrix):
#     inv_var = 1 / np.diag(cov_matrix)
#     weights = inv_var / np.sum(inv_var)
#     return weights
#
# def compute_hrp_weights(cov_matrix):
#     weights = np.ones(len(cov_matrix)) / len(cov_matrix)
#     return weights
#
# def compute_features(returns, cov_matrix, weights_nrp, weights_hrp):
#     features = {}
#     features['mean_return'] = np.mean(returns, axis=0)
#     features['std_return'] = np.std(returns, axis=0)
#     features['nrp_volatility'] = np.sqrt(np.dot(weights_nrp.T, np.dot(cov_matrix, weights_nrp)))
#     features['hrp_volatility'] = np.sqrt(np.dot(weights_hrp.T, np.dot(cov_matrix, weights_hrp)))
#     return pd.Series(features)
#
# def select_strategy(features, model):
#     prediction = model.predict(features)
#     return 'HRP' if prediction >= 0 else 'NRP'
#
# tickers = ['AGG', 'VNQ', 'EEM', 'XLV', 'IWD']
# start_date = '2010-01-01'
# end_date = '2023-01-01'
#
# data = fetch_data(tickers, start_date, end_date)
# returns = data.pct_change().dropna()
#
# lookback_period = 252
# X = []
# y = []
#
# for t in range(len(returns) - lookback_period):
#     historical_returns = returns.iloc[t:t+lookback_period]
#     cov_matrix = compute_cov_matrix(historical_returns)[-1]
#     weights_nrp = compute_nrp_weights(cov_matrix)
#     weights_hrp = compute_hrp_weights(cov_matrix)
#     features = compute_features(historical_returns, cov_matrix, weights_nrp, weights_hrp)
#     X.append(features)
#     sharpe_nrp = np.mean(historical_returns @ weights_nrp) / np.std(historical_returns @ weights_nrp)
#     sharpe_hrp = np.mean(historical_returns @ weights_hrp) / np.std(historical_returns @ weights_hrp)
#     y.append(sharpe_hrp - sharpe_nrp)
#
# X = pd.DataFrame(X)
# y = np.array(y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# model = XGBRegressor(objective='reg:squarederror')
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
# print('Test MSE:', mean_squared_error(y_test, y_pred))
#
# results = []
# for t in range(len(returns) - lookback_period):
#     historical_returns = returns.iloc[t:t+lookback_period]
#     cov_matrix = compute_cov_matrix(historical_returns)[-1]
#     weights_nrp = compute_nrp_weights(cov_matrix)
#     weights_hrp = compute_hrp_weights(cov_matrix)
#     features = compute_features(historical_returns, cov_matrix, weights_nrp, weights_hrp)
#     selected_strategy = select_strategy(features, model)
#     if selected_strategy == 'HRP':
#         returns_next_period = returns.iloc[t+lookback_period] @ weights_hrp
#     else:
#         returns_next_period = returns.iloc[t+lookback_period] @ weights_nrp
#     results.append(returns_next_period)
#
# cumulative_returns = np.cumsum(results)
# sharpe_ratio = np.mean(results) / np.std(results)
# print('Cumulative Returns:', cumulative_returns[-1])
# print('Sharpe Ratio:', sharpe_ratio)



















import pandas as pd
import numpy as np
from rmgarch import dcc_garch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class MetaPortfolioMethod:
    def __init__(self, returns, var_cov_matrix):
        self.returns = returns
        self.var_cov_matrix = var_cov_matrix

    def compute_features(self):
        features = {}
        features['avg_return'] = self.returns.mean()
        features['realized_volatility'] = self.returns.std()
        features['max_drawdown'] = self.returns.min()
        features['downside_deviation'] = self.returns[self.returns < 0].std()
        features['mean_corr'] = self.var_cov_matrix.mean().mean()
        features['std_corr'] = self.var_cov_matrix.std().std()
        return features

    def train_xgb_model(self, features, target):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'MSE: {mse}')
        return xgb_model

    def select_strategy(self, features, model):
        prediction = model.predict(pd.DataFrame(features, index=[0]))
        if prediction > 0:
            return 'HRP'
        else:
            return 'NRP'

    def meta_portfolio_method(self, model):
        features = self.compute_features()
        strategy = self.select_strategy(features, model)
        if strategy == 'HRP':
            # Implement HRP strategy
            portfolio_returns = ...
        else:
            # Implement NRP strategy
            portfolio_returns = ...
        return portfolio_returns

# Example usage
tickers = ['SPY']  # List of tickers
data = yf.download(tickers, start='2010-01-01', end='2023-05-31')['Adj Close']
returns = data.pct_change().dropna()

# Compute variance-covariance matrix using DCC GARCH
dcc_model = dcc_garch(returns)
dcc_fit = dcc_model.fit()
var_cov_matrix = dcc_fit.covariance

# Initialize MPM
mpm = MetaPortfolioMethod(returns, var_cov_matrix)

# Compute features
features = mpm.compute_features()

# Compute target variable (Sharpe ratio spread)
sharpe_ratio_hrp = ...
sharpe_ratio_nrp = ...
target = sharpe_ratio_hrp - sharpe_ratio_nrp

# Train XGBoost model
model = mpm.train_xgb_model(features, target)

# Run MPM
portfolio_returns = mpm.meta_portfolio_method(model)
print(f'Portfolio Returns: {portfolio_returns}')

