from flask import Flask, jsonify, request
import yfinance as yf
import json
import pandas as pd
import logging


from backend.PortfolioOptimizeController import bp_portfolio_optimizer
from backend.StockController import stock_controller

app = Flask(__name__)
app.register_blueprint(bp_portfolio_optimizer, url_prefix='/portfolio')
app.register_blueprint(stock_controller, url_prefix='/stocks')

import logging


def configure_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler('application.log')

    # Create a formatter and set it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
    print("logger configured")

if __name__ == '__main__':
    configure_logging()
    logging.info("Backend app starting...")
    app.run(debug=True)
