import json
from bs4 import BeautifulSoup
import pandas as pd
import requests
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


# Function to get article text
def get_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    article_text = " ".join([para.get_text() for para in paragraphs])
    return article_text


@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    try:
        data = request.json
        if "ticker_symbol" not in data:
            return jsonify({"error": "No ticker provided"}), 400
        ticker_symbol = data.get("ticker_symbol", None)
        news = yf.Ticker(ticker_symbol).news
        for i, article in enumerate(news[:1], 1):
            title = f"Title: {article['title']}"
            publisher = f"Publisher: {article['publisher']}"
            snippet = f"Snippet: {article.get('snipplet', 'N/A')}"
            text = title + "\n" + publisher + "\n" + snippet
        sentiment, score = analyzer.analyze_sentiment(text)
        return jsonify({"text": text, "sentiment": sentiment, "score": score})
    except Exception as e:
        return jsonify({"message": f"Error analyzing sentiment: {str(e)}"}), 500


@app.route("/")
def home():
    return {"message": "Welcome to Equity Lens"}


if __name__ == "__main__":
    app.run(debug=True)
