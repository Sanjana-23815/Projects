import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import yfinance as yf
from datetime import datetime

# Load FinBERT sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Streamlit page configuration
st.set_page_config(page_title="📊 Financial News Sentiment Analysis", layout="wide")
st.title("📊 Financial News Sentiment Analysis with AI")

# Define news sources
news_sources = {
    "Yahoo Finance": "https://finance.yahoo.com/news/",
    "Google News": "https://news.google.com/search?q=stock+market",
    "Bloomberg": "https://www.bloomberg.com/markets",
    "CNBC": "https://www.cnbc.com/finance/",
    "Reuters": "https://www.reuters.com/finance/",
    "Investing.com": "https://www.investing.com/news/stock-market-news",
    "MarketWatch": "https://www.marketwatch.com/latest-news",
    "Business Insider": "https://www.businessinsider.com/stock-market",
}

# Function to analyze sentiment using FinBERT
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result["label"], result["score"]

# Function to scrape financial news headlines
def fetch_financial_news():
    news_data = []
    for source, url in news_sources.items():
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.find_all("h3", limit=10)

            for article in articles:
                headline = article.text
                link = article.a["href"] if article.a else None
                if link and not link.startswith("http"):
                    link = url + link
                
                sentiment, confidence = analyze_sentiment(headline)
                news_data.append({
                    "Source": source,
                    "Headline": headline,
                    "Sentiment": sentiment,
                    "Confidence": confidence,
                    "Link": link,
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                })
        except Exception as e:
            st.warning(f"⚠️ Error fetching news from {source}: {e}")
    
    return pd.DataFrame(news_data)

# Sidebar to fetch news
if st.sidebar.button("Fetch Latest News"):
    st.sidebar.write("📡 Fetching news... Please wait.")
    df_news = fetch_financial_news()
    
    if not df_news.empty:
        st.sidebar.success("✅ News fetched successfully!")
        st.session_state["news_data"] = df_news  # Save in session state

# Display fetched news
if "news_data" in st.session_state:
    df_news = st.session_state["news_data"]
    
    # Display table
    st.write("### 📰 Latest Financial News & Sentiment Analysis")
    st.dataframe(df_news)

    # Sentiment distribution visualization
    st.write("### 📊 Sentiment Distribution")
    sentiment_counts = df_news["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    
    fig_sentiment = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment", title="Sentiment Distribution", text_auto=True)
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Historical Sentiment Trend
    st.write("### 📈 Sentiment Trend Over Time")
    df_news["Date"] = pd.to_datetime(df_news["Date"])
    sentiment_trend = df_news.groupby(["Date", "Sentiment"]).size().reset_index(name="Count")
    fig_trend = px.line(sentiment_trend, x="Date", y="Count", color="Sentiment", title="Historical Sentiment Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Stock Price Correlation
    stock_ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, AMZN)")
    if stock_ticker:
        try:
            stock_data = yf.download(stock_ticker, period="6mo")
            
            if "Close" not in stock_data.columns:
                st.warning("⚠️ No adjusted close price available for this stock. Using 'Close' instead.")
                stock_data["Close"] = stock_data["Close"]  # Fallback to Close price

            stock_data["Returns"] = stock_data["Close"].pct_change()
            st.line_chart(stock_data["Close"], use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error fetching stock data: {e}")

st.write("---")
st.write("🚀 Developed using AI-powered FinBERT & Streamlit")
