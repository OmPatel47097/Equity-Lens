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
