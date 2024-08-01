from flask import Blueprint, jsonify, request
from flask_cors import CORS, cross_origin
import pandas as pd

from services.BlackLittermanOptimization import BlackLittermanOptimization
from utils.StockDataManager import StockDataManager
import yfinance as yf

bp_portfolio_optimizer = Blueprint('portfolio_optimizer', __name__)
CORS(bp_portfolio_optimizer)


@bp_portfolio_optimizer.route('/optimize_portfolio', methods=['POST'])
def optimizePortfolio():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No data provided'}), 400
        if data.get('capital') is None:
            return jsonify({'error': 'No capital provided'}), 400
        if data.get('symbols') is None:
            return jsonify({'error': 'No symbols provided'}), 400

        capital = int(data['capital'])
        if capital == 0:
            return jsonify({'error': 'Capital must me larger than zero.'}), 400

        symbols = data['symbols']
        symbols = [symbol.strip() for symbol in symbols]

        if len(symbols) == 0:
            return jsonify({'error': 'No symbols provided'}), 400

        print(symbols)

        stockDataManager = StockDataManager()
        valid_symbols = stockDataManager.get_symbols()

        non_valid_symbols = [symbol for symbol in symbols if symbol not in valid_symbols]
        if len(non_valid_symbols) > 0:
            return jsonify({'error': 'Invalid symbols provided', 'invalid_symbols': non_valid_symbols}), 400

        bl_optimizer = BlackLittermanOptimization()
        optimized_result = bl_optimizer.calculate_portfolio_value(capital, symbols)

        return jsonify({'data': optimized_result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp_portfolio_optimizer.route('/asset_allocation', methods=['POST'])
def asset_allocation_chart():
    """
    returns data for asset allocation chart
    """
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No data provided'}), 400
        if data.get('assets') is None:
            return jsonify({'error': 'No assets provided'}), 400
        assets = data['assets']
        if type(assets) is not list:
            return jsonify({'error': 'Provide assets in list format'}), 400
        if len(assets) == 0:
            return jsonify({'error': 'No assets provided'}), 400

        data = []
        total_value = 0

        for asset in assets:
            n_shares = float(asset['shares'])
            avg_value = float(asset['avg_value'])
            total_value += avg_value * n_shares

        for asset in assets:
            symbol = str(asset['symbol'])
            n_shares = float(asset['shares'])
            avg_value = float(asset['avg_value'])
            asset_value = avg_value * n_shares
            asset_alloc = asset_value / total_value
            data.append({
                'asset_alloc': asset_alloc,
                'total_shares': n_shares,
                'asset_value': asset_value,
                'avg_value': avg_value,
                'symbol': symbol
            })

        response = jsonify(data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


@bp_portfolio_optimizer.route('/asset_info', methods=['POST'])
def asset_info():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No data provided'}), 400
        if data.get('assets') is None:
            return jsonify({'error': 'No assets provided'}), 400
        assets = data['assets']
        if type(assets) is not list:
            return jsonify({'error': 'Provide assets in list format'}), 400
        if len(assets) == 0:
            return jsonify({'error': 'No assets provided'}), 400

        data = []

        for asset in assets:
            symbol = str(asset['symbol'])
            ticker = yf.Ticker(symbol)
            data.append({
                'ticker': symbol,
                'current_price': ticker.info['currentPrice'],
                'name': ticker.info['shortName'],
                'previousClose': ticker.info['previousClose'],
                'marketCap': ticker.info['marketCap'],
                'change': ((ticker.info['currentPrice'] * 100) / ticker.info['previousClose']) - 100,
            })

        return jsonify(data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp_portfolio_optimizer.route('/cumulative_return', methods=['POST'])
def cumulative_return():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No data provided'}), 400
        if data.get('assets') is None:
            return jsonify({'error': 'No assets provided'}), 400
        if data.get('period') is None:
            return jsonify({'error': 'No period provided'}), 400

        assets = data['assets']
        period = data['period']

        if type(assets) is not list:
            return jsonify({'error': 'Provide assets in list format'}), 400
        if len(assets) == 0:
            return jsonify({'error': 'No assets provided'}), 400

        symbols = [asset['symbol'] for asset in assets]

        stockDataManager = StockDataManager()

        # Fetch historical data for the portfolio
        portfolio_data_list = [stockDataManager.history(ticker=symbol, period=period, interval='1d')[['adj_close']] for
                               symbol in symbols]
        portfolio_data = pd.concat(portfolio_data_list, axis=1, keys=[symbol for symbol in symbols])
        portfolio_data.columns = portfolio_data.columns.droplevel(1)
        portfolio_data = portfolio_data.fillna(method='ffill').fillna(method='bfill')
        portfolio_data = portfolio_data.dropna()
        mean_portfolio_data = portfolio_data.mean(axis=1)

        # Fetch historical data for S&P 500, matching the dates with portfolio
        sp500_data = yf.download('^GSPC', period=period, interval='1d')['Adj Close']
        sp500_data = sp500_data.reindex(mean_portfolio_data.index).fillna(method='ffill').fillna(method='bfill')

        # Calculate cumulative returns
        portfolio_cumulative_returns = (mean_portfolio_data / mean_portfolio_data.iloc[0] - 1) * 100
        sp500_cumulative_returns = (sp500_data / sp500_data.iloc[0] - 1) * 100

        # Convert dates to strings
        portfolio_cumulative_returns.index = portfolio_cumulative_returns.index.astype(str)
        sp500_cumulative_returns.index = sp500_cumulative_returns.index.astype(str)

        result = {
            'portfolio_cumulative_returns': portfolio_cumulative_returns.to_dict(),
            'sp500_cumulative_returns': sp500_cumulative_returns.to_dict()
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
