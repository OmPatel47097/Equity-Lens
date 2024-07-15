from flask import Blueprint, Flask, request, jsonify
from flask_cors import CORS

from utils.StockDataManager import StockDataManager

stock_controller = Blueprint('stock_controller', __name__)
CORS(stock_controller)


@stock_controller.route('/symbols', methods=['GET'])
def get_symbols():
    try:
        stockDataManager = StockDataManager()
        symbols = stockDataManager.get_symbols()
        return jsonify(symbols)
    except Exception as e:
        return jsonify({'error': str(e)})


@stock_controller.route('/history/<symbol>/<interval>/<period>', methods=['GET'])
def get_history():
    pass