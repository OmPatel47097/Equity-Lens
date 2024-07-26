from flask import Blueprint, request, jsonify
from flask_cors import CORS
from utils.StockDataManager import StockDataManager
import yfinance as yf
import pandas as pd

stock_controller = Blueprint('stock_controller', __name__)
CORS(stock_controller)

@stock_controller.route('/symbols', methods=['GET'])
def get_symbols():
    try:
        stockDataManager = StockDataManager()
        symbols = stockDataManager.get_symbols()
        return jsonify(symbols)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_controller.route('/stock_info/<symbol>', methods=['GET'])
def get_stock_info(symbol):
    try:
        stockDataManager = StockDataManager()
        # Fetch the latest price and other details from yfinance
        stock_data = yf.Ticker(symbol)
        stock_info = stock_data.info
        latest_price = stock_info['regularMarketPrice']
        previous_close = stock_info['regularMarketPreviousClose']
        price_change = latest_price - previous_close
        price_change_percent = (price_change / previous_close) * 100

        # Calculate additional metrics
        history = stockDataManager.history(ticker=symbol, period='1y', interval='1d')
        avg_return = history['adj_close'].pct_change().mean() * 252  # Annualized average return
        risk = history['adj_close'].pct_change().std() * (252**0.5)  # Annualized standard deviation (risk)

        stock_details = {
            'symbol': symbol,
            'name': stock_info['shortName'],
            'latest_price': latest_price,
            'price_change': price_change,
            'price_change_percent': price_change_percent,
            'avg_return': avg_return,
            'risk': risk,
            'market_cap': stock_info['marketCap'],
            'pe_ratio': stock_info['trailingPE'],
            'dividend_yield': stock_info.get('dividendYield', 0),
            '52_week_high': stock_info['fiftyTwoWeekHigh'],
            '52_week_low': stock_info['fiftyTwoWeekLow']
        }
        return jsonify(stock_details)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_controller.route('/history/<symbol>/<period>/<interval>', methods=['GET'])
def get_history(symbol, period, interval):
    try:
        stockDataManager = StockDataManager()
        history = stockDataManager.history(ticker=symbol, period=period, interval=interval)
        history.reset_index(inplace=True)
        history_dict = history.to_dict(orient='records')
        return jsonify(history_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_controller.route('/all_stock_info/<n_records>', methods=['GET'])
def get_all_stock_info(n_records):
    try:
        stockDataManager = StockDataManager()
        symbols = stockDataManager.get_symbols()
        symbols = symbols[:int(n_records)]
        all_stock_info = []

        for symbol in symbols:
            stock_data = yf.Ticker(symbol)
            stock_info = stock_data.info
            latest_price = stock_info['currentPrice']
            previous_close = stock_info['regularMarketPreviousClose']
            price_change = latest_price - previous_close
            price_change_percent = (price_change / previous_close) * 100

            # Calculate additional metrics
            history = stockDataManager.history(ticker=symbol, period='1y', interval='1d')
            avg_return = history['adj_close'].pct_change().mean() * 252  # Annualized average return
            risk = history['adj_close'].pct_change().std() * (252**0.5)  # Annualized standard deviation (risk)

            stock_details = {
                'symbol': symbol,
                'name': stock_info['shortName'],
                'latest_price': latest_price,
                'price_change': price_change,
                'price_change_percent': price_change_percent,
                'avg_return': avg_return,
                'risk': risk,
                'market_cap': stock_info['marketCap'],
                'pe_ratio': stock_info['trailingPE'],
                'dividend_yield': stock_info.get('dividendYield', 0),
                '52_week_high': stock_info['fiftyTwoWeekHigh'],
                '52_week_low': stock_info['fiftyTwoWeekLow']
            }
            all_stock_info.append(stock_details)

        return jsonify(all_stock_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
