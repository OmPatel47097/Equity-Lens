import streamlit as st
from routes import home, portfolio_analysis, stock_analysis, market_news, portfolio_optimization

st.set_page_config(page_title="Equity Lens", layout="wide")


# Sidebar navigation
st.sidebar.title("Equity Lens")
page = st.sidebar.radio("Go to", ["Home", "Portfolio Optimization", "Stock Analysis",])

# Load the selected page
if page == "Home":
    home.show()
elif page == "Portfolio Optimization":
    portfolio_optimization.show()
elif page == "Stock Analysis":
    stock_analysis.show()

