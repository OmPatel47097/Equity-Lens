import streamlit as st
from services.BlackLittermanOptimization import BlackLittermanOptimization

def main():
    st.title("Equity Lens")
    st.header("Portfolio Optimization using Black Litterman")
    blpo = BlackLittermanOptimization()


    # Input for capital
    capital = st.number_input("Enter the amount of capital to invest:", min_value=0.0, step=1.0)

    # Input for stocks
    stocks = st.text_input("Enter the stocks you want to invest in (comma-separated):")

    if st.button("Submit"):
        if capital > 0 and stocks:
            # Process the input
            stock_list = [stock.strip() for stock in stocks.split(",")]
            output = blpo.calulate_portfolio_value(capital, stock_list)

            # Display the inputs
            st.write(f"Capital to invest: ${capital}")
            st.write("Stocks to invest in:", output)
        else:
            st.error("Please enter a valid amount of capital and at least one stock.")

if __name__ == "__main__":
    main()
