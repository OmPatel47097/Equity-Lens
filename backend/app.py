from flask import Flask, jsonify, request
import yfinance as yf
import json
import pandas as pd


from backend.PortfolioOptimizeController import bp_portfolio_optimizer
from backend.StockController import stock_controller

app = Flask(__name__)
app.register_blueprint(bp_portfolio_optimizer, url_prefix='/portfolio')
app.register_blueprint(stock_controller, url_prefix='/stocks')


if __name__ == '__main__':
    app.run(debug=True)
