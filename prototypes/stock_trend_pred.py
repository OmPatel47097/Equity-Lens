import streamlit as st
import numpy as np
import pandas as pd


from services.StockTrendPrediction import StockTrendPrediction

# Define the main function
def stock_trend_prediction():
    st.title("Stock Trend Prediction")

    # Sidebar for user input
    st.sidebar.header("User Input")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    sequence_length = 60

    # Create an instance of StockTrendPrediction
    stp = StockTrendPrediction(sequence_length)

    # Load data
    st.write("Loading data...")
    stp.load_data(ticker)
    st.write(f"Data for {ticker} loaded successfully!")

    # Create features
    st.write("Creating features...")
    stp.create_features()
    st.write("Features created successfully!")

    # Split data
    st.write("Splitting data into training and testing sets...")
    stp.split_data()
    st.write("Data split successfully!")

    # Normalize data
    st.write("Normalizing data...")
    stp.normalize_data()
    st.write("Data normalized successfully!")

    # Create sequences for prediction
    st.write("Creating sequences for prediction...")
    X_test, y_test = stp.create_sequences(stp.df_test)
    st.write("Sequences created successfully!")

    # Load model and make predictions
    st.write("Loading model and making predictions...")
    predictions = stp.predict(X_test)
    st.write("Predictions made successfully!")

    # # Display predictions
    # st.write("Predictions:")
    # st.write(predictions)

    # Create a DataFrame for the actual and predicted values
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions.flatten()
    })

    # Plot the results
    st.write("Plotting results...")
    st.line_chart(results)

# Run the app
if __name__ == "__main__":
    main()
