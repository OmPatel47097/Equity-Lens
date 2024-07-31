import streamlit as st
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from backend.FinancialSentimentAnalysis import FinancialSentimentAnalyzer

# Initialize sentiment analyzer
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
        st.error(f"Network error fetching article text: {str(e)}")
        return ""
    except Exception as e:
        st.error(f"Error parsing article text: {str(e)}")
        return ""


# Function to get article text using Selenium
def get_article_text_selenium(url):
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        paragraphs = soup.find_all("p")
        article_text = "\n\n".join([para.get_text() for para in paragraphs])
        driver.quit()
        return article_text
    except Exception as e:
        st.error(f"Error fetching article text with Selenium: {str(e)}")
        return ""


def analyze_sentiment(ticker_symbol):
    try:
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
        return results
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return []


# Streamlit app layout
def main():
    st.title("Welcome to Equity Lens")

    ticker_symbol = st.text_input("Enter the ticker symbol:")
    if st.button("Analyze Sentiment"):
        if ticker_symbol:
            with st.spinner("Analyzing sentiment..."):
                results = analyze_sentiment(ticker_symbol)
                if results:
                    for result in results:
                        st.write(f"**Title:** {result['title']}")
                        st.write(f"**Text:**\n{result['article_text']}")
                        st.write(f"**Sentiment:** {result['sentiment']}")
                        st.write(f"**Score:** {result['score']}")
                        st.write("---")
        else:
            st.error("Please enter a ticker symbol.")


if __name__ == "__main__":
    main()
