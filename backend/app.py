from flask import Flask, jsonify, request
import yfinance as yf
import json
import pandas as pd

app = Flask(__name__)


@app.route('/api/stock/<symbol>/<period>', methods=['GET'])
def get_stock_history(symbol, period):
    try:
        data = yf.download(symbol, period=period)
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


if __name__ == '__main__':
    app.run(debug=True)
