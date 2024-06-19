from flask import Flask, jsonify, request
import yfinance as yf
import json
import pandas as pd
from backend.AnalysisController import AnalysisController

app = Flask(__name__)


@app.route('/api/stock/<symbol>/<period>', methods=['GET'])
def get_stock_history(symbol, period):
    try:
        interval = '1d'
        if period == '1d':
            interval = '5m'
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty:
            return jsonify({'message': 'No data found for symbol'}), 404
        response_data = jsonify(json.loads(data.to_json(orient='table')))  # Convert pandas dataframe to dictionary
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
    except Exception as e:
        return jsonify({'message': f'Error retrieving data: {str(e)}'}), 500


@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    try:
        df = pd.read_csv('../data/sp500.csv')
        response_data = jsonify(json.loads(df['Symbol'].to_json(orient='table')))
        response_data.headers.add('Access-Control-Allow-Origin', '*')
        return response_data
    except Exception as e:
        return jsonify({'message': f'Error retrieving data: {str(e)}'}), 500


@app.route('/api/optimize_portfolio', methods=['POST'])
def get_optimized_weights():
    try:
        # request_json = request.json
        stock_weights = [
            {"stock": "AAPL", "weight": "0.3"},
            {"stock": "MSFT", "weight": "0.2"},
            {"stock": "GOOG", "weight": "0.5"}
        ]
        json_data = json.dumps(stock_weights, indent=4)

        return jsonify(stock_weights), 200
    except Exception as e:
        return jsonify({'message': f'Error retrieving data: {str(e)}'}), 500


@app.route('/api/risk_based_clustering', methods=['GET'])
def get_risk_based_clustering():
    try:
        clusters = {
        "AAPL": {
        "Volatility": 0.22,
        "Beta": 1.15,
        "VaR": -0.025,
        "Cluster": 2
        },
        "MSFT": {
        "Volatility": 0.21,
        "Beta": 1.10,
        "VaR": -0.023,
        "Cluster": 2
        },
        "GOOGL": {
        "Volatility": 0.20,
        "Beta": 1.05,
        "VaR": -0.022,
        "Cluster": 1
        },

        }

        return jsonify(clusters), 200
    except Exception as e:
        return jsonify({'message': f'Error retrieving data: {str(e)}'}), 500

@app.route('/api/performance_based_clustering', methods=['GET'])
def get_performance_based_clustering():
    try:
        clusters = {
            "AAPL": {
                "Annualized Return": 0.324,
                "Annualized Volatility": 0.223,
                "Sharpe Ratio": 1.4,
                "Cluster": 2
            },
            "MSFT": {
                "Annualized Return": 0.298,
                "Annualized Volatility": 0.211,
                "Sharpe Ratio": 1.35,
                "Cluster": 2
            },
            "GOOGL": {
                "Annualized Return": 0.275,
                "Annualized Volatility": 0.195,
                "Sharpe Ratio": 1.3,
                "Cluster": 1
            },

        }

        return jsonify(clusters), 200
    except Exception as e:
        return jsonify({'message': f'Error retrieving data: {str(e)}'}), 500

@app.route('/api/analysis/get_indicators', methods=['POST'])
def get_indicators():
    try:
        request_json = request.json
        data = AnalysisController.get_indicators(request_json)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({'message': f'Error retrieving data: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
