import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

# Function to plot predictions
def plot_predictions(ticker):
    # Load the dataset
    df = pd.read_csv(f'data\{ticker}_data.csv').dropna()
    
    # Convert 'Date' column to datetime
    if df['Date'].dtype == 'object':
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Drop 'Date' column from features, but keep it for plotting
    dates = df['Date']
    X = df.drop(columns=['Close', 'Date'])
    y = df['Close']
    
    # Split into training and testing datasets
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, shuffle=False)
    
    # Load the pre-trained model
    with open(f'Models\{ticker}_xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions using the model
    y_pred = model.predict(X_test)
    
    # Plotting the actual vs predicted prices
    plt.figure(figsize=(10, 5))
    plt.plot(dates_test, y_test.values, label='Actual', linewidth=2, color='blue')
    plt.plot(dates_test, y_pred, label='Predicted', linewidth=2, color='red', linestyle='dashed')
    plt.title(f'{ticker} - XGBoost Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate dates for better visibility
    plt.tight_layout()  # Prevent label cut-off
    st.pyplot(plt)

# Streamlit UI
def main():
    # Set the title of the app
    st.title('Stock Price Prediction and Forecasting')
    
    # Dropdown for selecting a company ticker
    ticker = st.selectbox(
        'Select the company to forecast:',
        ['AAPL', 'MSFT', 'GOOG', 'TCS.NS', 'TATAMOTORS.NS', 'NBCC.NS']
    )
    
    # Button to trigger prediction
    if st.button(f'Generate Forecast for {ticker}'):
        st.write(f"Generating forecast for {ticker}...")
        plot_predictions(ticker)

# Run the app
if __name__ == '__main__':
    main()
