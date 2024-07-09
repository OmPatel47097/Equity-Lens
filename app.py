import json
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from backend.FinancialSentimentAnalysis import FinancialSentimentAnalyzer

app = Flask(__name__)


@app.route("/api/stock/<symbol>/<period>", methods=["GET"])
def get_stock_history(symbol, period):
    try:
        data = yf.download(symbol, period=period)
        if data.empty:
            return jsonify({"message": "No data found for symbol"}), 404
        response_data = jsonify(
            json.loads(data.to_json(orient="table"))
        )  # Convert pandas dataframe to dictionary
        response_data.headers.add("Access-Control-Allow-Origin", "*")
        return response_data
    except Exception as e:
        return jsonify({"message": f"Error retrieving data: {str(e)}"}), 500


@app.route("/api/symbols", methods=["GET"])
def get_symbols():
    try:
        df = pd.read_csv("../data/sp500.csv")
        response_data = jsonify(json.loads(df["Symbol"].to_json(orient="table")))
        response_data.headers.add("Access-Control-Allow-Origin", "*")
        return response_data
    except Exception as e:
        return jsonify({"message": f"Error retrieving data: {str(e)}"}), 500


analyzer = FinancialSentimentAnalyzer()


@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    try:
        data = request.json
        if "text" not in data:
            return jsonify({"error": "No text provided"}), 400
        text = data["text"]
        sentiment, score = analyzer.analyze_sentiment(text)
        return jsonify({"text": text, "sentiment": sentiment, "score": score})
    except Exception as e:
        return jsonify({"message": f"Error analyzing sentiment: {str(e)}"}), 500


@app.route("/")
def home():
    return {"message": "Welcome to Equity Lens"}


if __name__ == "__main__":
    app.run(debug=True)
