# import yfinance as yf
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Input, concatenate
#
# stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
#
# # Define date range
# start_date = '2015-01-01'
# end_date = '2022-12-31'
#
# # Load data from Yahoo Finance
# data = yf.download(stock_symbols, start=start_date, end=end_date)['Adj Close']
#
# # Convert data to a pandas DataFrame
# data_df = pd.DataFrame(data)
# # print(data_df)
#
# # Calculate daily returns
# # Calculate daily returns
# data_df['AAPL_returns'] = data_df['AAPL'].pct_change()
# data_df['MSFT_returns'] = data_df['MSFT'].pct_change()
# data_df['GOOGL_returns'] = data_df['GOOGL'].pct_change()
# data_df['AMZN_returns'] = data_df['AMZN'].pct_change()
#
# # Calculate daily returns
# for stock in stock_symbols:
#     data_df[f'{stock}_returns'] = data_df[stock].pct_change()
#
# # Calculate cumulative returns
# data_df['AAPL_cumulative_returns'] = (1 + data_df['AAPL_returns']).cumprod()
# data_df['MSFT_cumulative_returns'] = (1 + data_df['MSFT_returns']).cumprod()
# data_df['GOOGL_cumulative_returns'] = (1 + data_df['GOOGL_returns']).cumprod()
# data_df['AMZN_cumulative_returns'] = (1 + data_df['AMZN_returns']).cumprod()
#
# # Calculate cumulative returns
# for stock in stock_symbols:
#     data_df[f'{stock}_cumulative_returns'] = (1 + data_df[f'{stock}_returns']).cumprod()
#
#
# # Normalize data
# # Normalize data
# for stock in stock_symbols:
#     std = data_df[f'{stock}_returns'].std()
#     if std != 0:
#         data_df[f'{stock}_normalized_returns'] = data_df[f'{stock}_returns'] / std
#     else:
#         data_df[f'{stock}_normalized_returns'] = 0
#
# # Create state and action spaces
# state_dim = data_df.shape[1]
# action_dim = data_df.shape[1]
#
#
# # # Create state and action spaces
# # state_dim = data_df.shape[1]
# # action_dim = data_df.shape[1]
#
# class DRLAgent:
#     def __init__(self, state_dim, action_dim):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.model = self.create_model()
#
#     def create_model(self):
#         state_input = Input(shape=(1,))  # Adjust the input shape to (1,)
#         x = Dense(64, activation='relu')(state_input)
#         x = Dense(64, activation='relu')(x)
#         output = Dense(self.action_dim, activation='softmax')(x)
#         model = Model(inputs=state_input, outputs=output)
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         return model
#
#     def act(self, state):
#         return self.model.predict(state)
#
#
# class EIIE:
#     def __init__(self, num_agents, state_dim, action_dim):
#         self.num_agents = num_agents
#         self.agents = [DRLAgent(state_dim, action_dim) for _ in range(num_agents)]
#
#     def act(self, state):
#         actions = [agent.act(state) for agent in self.agents]
#         return np.mean(actions, axis=0)
#
# class PVM:
#     def __init__(self, max_size):
#         self.max_size = max_size
#         self.data = []
#
#     def store(self, state, action, reward, next_state):
#         self.data.append((state, action, reward, next_state))
#         if len(self.data) > self.max_size:
#             self.data.pop(0)
#
#     def sample(self, batch_size):
#         return np.random.sample(self.data, batch_size)
#
# class OSBL:
#     def __init__(self, pvm, eiie, batch_size):
#         self.pvm = pvm
#         self.eiie = eiie
#         self.batch_size = batch_size
#
#     def update(self):
#         batch = self.pvm.sample(self.batch_size)
#         states, actions, rewards, next_states = zip(*batch)
#         states = np.array(states)
#         actions = np.array(actions)
#         rewards = np.array(rewards)
#         next_states = np.array(next_states)
#
#         # Compute target values
#         target_values = rewards + 0.99 * np.max(self.eiie.act(next_states), axis=1)
#
#         # Compute loss
#         loss = np.mean((target_values - self.eiie.act(states)) ** 2)
#
#         # Update weights
#         self.eiie.model.fit(states, target_values, epochs=1, verbose=0)
#
# # Initialize EIIE, PVM, and OSBL
# eiie = EIIE(num_agents=5, state_dim=data_df.shape[1], action_dim=data_df.shape[1])
# pvm = PVM(max_size=10000)
# osbl = OSBL(pvm, eiie, batch_size=32)
#
# # Train and Test
# for episode in range(1000):
#     state = data_df.iloc[0]
#     done = False
#     rewards = 0
#     while not done:
#         action = eiie.act(state)
#         next_state, reward, done, _, _ = data_df.iloc[1]
#         pvm.store(state, action, reward, next_state)
#         osbl.update()
#         state = next_state
#         rewards += reward
#     print(f'Episode {episode+1}, Reward: {rewards}')
#
# # Back-testing
# backtest_data = pd.read_csv('backtest_data.csv')
# backtest_rewards = []
# for episode in range(100):
#     state = backtest_data.iloc[0]
#     done = False
#     rewards = 0
#     while not done:
#         action = eiie.act(state)
#         next_state, reward, done, _, _ = backtest_data.iloc[1]
#         rewards += reward
#         state = next_state
#     backtest_rewards.append(rewards)
# print(f'Backtest Reward: {np.mean(backtest_rewards)}')









import yfinance as yf
import pandas as pd
import numpy as np

class PortfolioEnv:
    def __init__(self, tickers, start_date, end_date, initial_cash=10000):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.current_step = 0
        self.num_assets = len(tickers)
        self.data = self._load_data()

    def _load_data(self):
        # Load historical data using yfinance
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        return data

    def reset(self):
        self.cash = self.initial_cash
        self.asset_holdings = np.zeros(self.num_assets)
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        action = action / np.sum(action)  # Normalize action to sum to 1
        asset_prices = self.data.iloc[self.current_step].values
        portfolio_value = self.cash + np.sum(self.asset_holdings * asset_prices)
        self.cash = portfolio_value * action[0]
        self.asset_holdings = (portfolio_value * action[1:]) / asset_prices
        self.current_step += 1
        reward = portfolio_value - self.initial_cash
        done = self.current_step >= len(self.data) - 1
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        asset_prices = self.data.iloc[self.current_step].values
        return np.concatenate(([self.cash], asset_prices))

    def get_total_portfolio_value(self):
        asset_prices = self.data.iloc[self.current_step].values
        portfolio_value = self.cash + np.sum(self.asset_holdings * asset_prices)
        return portfolio_value


def run_portfolio_strategy(env, rebalance_period='M'):
    # Reset the environment
    obs = env.reset()
    done = False
    while not done:
        # Rebalance the portfolio monthly
        if pd.to_datetime(env.data.index[env.current_step]).is_month_end:
            weights = np.ones(env.num_assets + 1) / (env.num_assets + 1)  # Equal weights including cash
            obs, reward, done, _ = env.step(weights)
        else:
            # Ensure weights are defined even in non-month-end steps
            weights = np.ones(env.num_assets + 1) / (env.num_assets + 1)  # Default equal weights
            obs, reward, done, _ = env.step(weights)

    # Get final portfolio value
    final_value = env.get_total_portfolio_value()
    return final_value


if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Example tickers
    start_date = '2010-01-01'
    end_date = '2023-01-01'

    env = PortfolioEnv(tickers, start_date, end_date)
    final_portfolio_value = run_portfolio_strategy(env)
    print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")



