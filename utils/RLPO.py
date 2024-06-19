# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Input, concatenate
#
# # Step 1: Load Data
# data = pd.read_csv('data.csv')
#
# # Step 2: Preprocess Data
# data['date'] = pd.to_datetime(data['date'])
# data.set_index('date', inplace=True)
# data.sort_index(inplace=True)
#
# # Step 3: Create State and Action Spaces
# state_dim = data.shape[1]
# action_dim = data.shape[1]
#
# # Step 4: Define Reward Function
# def reward_function(portfolio_weights, market_data):
#     portfolio_returns = np.sum(portfolio_weights * market_data, axis=1)
#     sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
#     return sharpe_ratio
#
# # Step 5: Define Deep Reinforcement Learning Framework
# class DRLAgent:
#     def __init__(self, state_dim, action_dim):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.model = self.create_model()
#
#     def create_model(self):
#         state_input = Input(shape=(self.state_dim,))
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
# # Step 6: Define Ensemble of Identical Independent Evaluators (EIIE)
# class EIIE:
#     def __init__(self, num_agents, state_dim, action_dim):
#         self.num_agents = num_agents
#         self.agents = [DRLAgent(state_dim, action_dim) for _ in range(num_agents)]
#
#     def act(self, state):
#         actions = [agent.act(state) for agent in self.agents]
#         return np.mean(actions, axis=0)
#
# # Step 7: Define Portfolio-Vector Memory (PVM)
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
#         return random.sample(self.data, batch_size)
#
# # Step 8: Define Online Stochastic Batch Learning (OSBL)
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
# # Step 9: Train and Test
# eiie = EIIE(num_agents=5, state_dim=state_dim, action_dim=action_dim)
# pvm = PVM(max_size=10000)
# osbl = OSBL(pvm, eiie, batch_size=32)
#
# for episode in range(1000):
#     state = data.iloc[0]
#     done = False
#     rewards = 0
#     while not done:
#         action = eiie.act(state)
#         next_state, reward, done, _ = data.iloc[1]
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
#         next_state, reward, done, _ = backtest_data.iloc[1]
#         rewards += reward
#         state = next_state
#     backtest_rewards.append(rewards)
# print(f'Backtest Reward: {np.mean(backtest_rewards)}')



# ---------------------------------------------------------------------------------------------------------------------------

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
        self.state_dim = self.num_assets + 1  # Number of dimensions in state (cash + asset prices)
        self.action_dim = self.num_assets + 1  # Number of dimensions in action space (weights for each asset + cash)

    def _load_data(self):
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        return data

    def reset(self):
        self.cash = self.initial_cash
        self.asset_holdings = np.zeros(self.num_assets)
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        if np.sum(action) == 0:
            # Handle the case where action sums to 0 (optional)
            return self._get_observation(), 0, True, {}

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


class QLearningAgent:
    def __init__(self, state_dim, action_dim, epsilon=0.1, lr=0.1, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon  # Exploration-exploitation trade-off parameter
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor

        # Initialize Q-table with zeros
        self.q_table = np.zeros((self.state_dim, self.action_dim))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random action
            return np.random.randint(self.action_dim)
        else:
            state_idx = self._state_to_index(state)
            return np.argmax(self.q_table[state_idx])

    def learn(self, state, action, reward, next_state, done):
        # Update Q-table using Q-learning equation
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)

        if not done:
            td_target = reward + self.gamma * np.max(self.q_table[next_state_idx])
        else:
            td_target = reward

        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.lr * td_error

    def _state_to_index(self, state):
        # Example: Discretize cash into 10 bins, and use asset holdings as is
        cash_index = int(state[0] / 1000)  # Example: Bin cash into 10K increments
        # Ensure cash_index is within bounds
        cash_index = np.clip(cash_index, 0, self.state_dim - 1)
        return cash_index


# Example of setting up and running an RL episode
# Example of setting up and running an RL episode
if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Example tickers
    start_date = '2010-01-01'
    end_date = '2023-01-01'

    env = PortfolioEnv(tickers, start_date, end_date)
    agent = QLearningAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    num_episodes = 1000  # Number of episodes for training

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            # Ensure action is numpy array or list with correct dimensions
            if isinstance(action, np.ndarray):
                action = action.tolist()  # Convert numpy array to list if necessary
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state

            # Debugging: Check action before passing to env.step
            print(f"Episode: {episode}, Action: {action}")

        final_portfolio_value = env.get_total_portfolio_value()
        print(f"Episode: {episode}, Final Portfolio Value: ${final_portfolio_value:.2f}")

    print("Training completed.")






