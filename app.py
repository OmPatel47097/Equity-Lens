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


# Function to get article text using requests and BeautifulSoup
def get_article_text(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx, 5xx)

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted elements
        for script in soup(["script", "style", "aside", "header", "footer", "nav"]):
            script.decompose()

        # Try to find the main content
        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_="main-content")
        )
        if main_content:
            paragraphs = main_content.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        # Join paragraphs with double newlines for better readability
        article_text = "\n\n".join([para.get_text() for para in paragraphs])

        # Check if the extracted text is empty or too short
        if len(article_text.strip()) < 100:
            raise ValueError("Extracted article text is too short or empty.")

        return article_text
    except requests.RequestException as e:
        raise f"Network error fetching article text: {str(e)}"
    except Exception as e:
        raise f"Error parsing article text: {str(e)}"


@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    try:
        data = request.json
        if "ticker_symbol" not in data:
            return jsonify({"error": "No ticker provided"}), 400
        ticker_symbol = data.get("ticker_symbol", None)
        news = yf.Ticker(ticker_symbol).news
        results = []
        for article in news[:5]:  # Limiting to the first 5 articles for demonstration
            url = article["link"]
            title = article["title"]
            article_text = get_article_text(url)  # Get the full article text
            sentiment, score = analyzer.analyze_sentiment(article_text)
            results.append(
                {
                    "url": url,
                    "title": title,
                    "article_text": article_text,
                    "sentiment": sentiment,
                    "score": score,
                }
            )
        return jsonify(results)
    except Exception as e:
        return jsonify({"message": f"Error analyzing sentiment: {str(e)}"}), 500


@app.route("/")
def home():
    return {"message": "Welcome to Equity Lens"}


if __name__ == "__main__":
    app.run(debug=True)