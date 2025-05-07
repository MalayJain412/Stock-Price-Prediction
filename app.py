# app3.py
import streamlit as st
import yfinance as yf
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

# Configure page
st.set_page_config(page_title="Stock Predictor", layout="wide")
plt.style.use('ggplot')

# Session state initialization
if 'preserve_data' not in st.session_state:
    st.session_state.preserve_data = {
        'live_data': None,
        'model': None,
        'prediction': None,
        'inr_amount': None,
        'currency_symbol': '$',
        'show_results': False
    }

# --- UI styling ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1578662996442-48f60103fc96?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .sidebar-notice {
        position: fixed;
        left: 0;
        top: 15%;
        transform: translateY(-50%);
        background: rgba(255, 255, 255, 0.20);
        border: 2px solid #4CAF50;
        border-left: none;
        padding: 10px 10px 10px 10px;
        border-radius: 0 15px 15px 0;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        z-index: 999;
        backdrop-filter: blur(2px);
    }
    .sidebar-notice::before {
        content: '👆';
        # position: absolute;
        top: 20px;
        font-size: 1.2em;
    }
    .sidebar-notice:hover {
        background: rgba(100, 100, 100, 0.95);
    }
    @media (max-width: 768px) {
        .sidebar-notice {
            # top: auto;
            top: 15%;
            font-size: 0.9em;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_currency_symbol(ticker):
    """Determine currency symbol based on ticker"""
    return '₹' if ticker.endswith('.NS') else '$'

def get_exchange_rate():
    """Safely get current USD to INR exchange rate"""
    try:
        exchange_data = yf.download('USDINR=X', period='1d')
        if not exchange_data.empty:
            rate = exchange_data['Close'].iloc[0]
            return float(rate)
        return 83.0
    except Exception as e:
        st.error(f"Exchange rate error: {str(e)}")
        return 83.0

def main():
    st.markdown(
        """<div class="sidebar-notice"><strong> Currency Converter</strong><br>Available in sidebar</div>""", 
        unsafe_allow_html=True
    )
    
    st.title("💹 Stock Price Prediction and Forecasting")
    st.sidebar.caption("💡 Use this converter for predicted price conversions!")

    # Company selection
    TICKERS = ['AAPL', 'MSFT', 'GOOG', 'TCS.NS', 'TATAMOTORS.NS', 'NBCC.NS']
    selected_ticker = st.selectbox("Select Company:", TICKERS)
    is_indian = selected_ticker.endswith('.NS')

    if st.button("Generate Forecast"):
        st.session_state.preserve_data['show_results'] = True
        with st.spinner("Processing..."):
            try:
                # Always fetch fresh data
                live_data = fetch_live_data(selected_ticker)
                model = load_model(selected_ticker)
                prediction = None

                if model and live_data is not None:
                    features = prepare_features(live_data)
                    prediction = float(model.predict(features)[0])

                # Update session state
                st.session_state.preserve_data.update({
                    'live_data': live_data,
                    'model': model,
                    'prediction': prediction,
                    'currency_symbol': '₹' if is_indian else '$',
                    'inr_amount': None
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.preserve_data['show_results'] = False

    # Display results from session state
    if st.session_state.preserve_data['show_results']:
        if st.session_state.preserve_data['live_data'] is not None:
            st.header(f"Latest Market Data - {selected_ticker}")
            st.dataframe(st.session_state.preserve_data['live_data'].style.format(precision=2))

        if st.session_state.preserve_data['prediction'] is not None:
            st.header("Price Prediction")
            display_results(
                st.session_state.preserve_data['prediction'],
                st.session_state.preserve_data['currency_symbol']
            )
            
            if not is_indian:
                st.markdown("""
                <div style="background: rgba(100, 100, 100, 0.95);padding:10px; border-radius:5px; margin:15px 0;">
                💱 Want to convert USD to INR? Use the currency converter in the sidebar!
                </div>
                """, unsafe_allow_html=True)

            st.header("Model Performance Analysis")
            plot_historical(selected_ticker, st.session_state.preserve_data['currency_symbol'])

    # Currency Converter
    with st.sidebar:
        st.markdown("---")
        st.header("Currency Converter")
        exchange_rate = get_exchange_rate()
        
        # Use form submit handler to prevent full refresh
        with st.form(key='converter_form'):
            default_value = st.session_state.preserve_data['prediction'] if (
                not is_indian and 
                st.session_state.preserve_data['prediction'] is not None
            ) else 0.0
            
            usd_amount = st.number_input(
                "USD Amount:", 
                min_value=0.0, 
                value=default_value,
                key='usd_input',
                format="%.2f",
                step=0.01
            )
            
            if st.form_submit_button("Convert"):
                st.session_state.preserve_data['inr_amount'] = usd_amount * exchange_rate

        if st.session_state.preserve_data['inr_amount'] is not None:
            st.markdown(f"""
            **Current Rate:** 1 USD = ₹{exchange_rate:.2f}  
            **Converted Amount:** ₹{st.session_state.preserve_data['inr_amount']:.2f}
            """)
        st.markdown("---")

def fetch_live_data(ticker):
    """Fetch and format live market data"""
    try:
        data = yf.download(ticker, period="2d", interval="1d")
        if not data.empty:
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
            data = data[numeric_cols].reset_index()
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d %H:%M')
            return data.dropna()
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return pd.DataFrame()

def load_model(ticker):
    """Load model from Models directory"""
    try:
        model_path = f"Models/{ticker}_xgb_model.pkl"
        if not os.path.exists(model_path):
            st.error(f"Model not found: {model_path}")
            return None
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model load error: {str(e)}")
        return None

def prepare_features(data):
    """Prepare and validate features"""
    try:
        latest = data.iloc[-1]
        return pd.DataFrame({
            'High': [float(latest['High'])],
            'Low': [float(latest['Low'])],
            'Open': [float(latest['Open'])],
            'Volume': [float(latest['Volume'])]
        })
    except Exception as e:
        raise ValueError(f"Feature preparation failed: {str(e)}")

def display_results(prediction, symbol):
    """Display prediction results"""
    st.subheader("Predicted Closing Price")
    st.markdown(f"## {symbol}{prediction:.2f}")

def plot_historical(ticker, currency_symbol):
    """Plot historical predictions"""
    try:
        hist_path = f"data/{ticker}_data.csv"
        df = pd.read_csv(hist_path)
        df['Date'] = pd.to_datetime(df['Date'])
        numeric_cols = ['High', 'Low', 'Open', 'Volume', 'Close']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        test_size = int(len(df) * 0.2)
        X_test = df[['High', 'Low', 'Open', 'Volume']].iloc[-test_size:]
        y_test = df['Close'].iloc[-test_size:]
        dates_test = df['Date'].iloc[-test_size:]
        
        model = load_model(ticker)
        if model:
            y_pred = model.predict(X_test)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(dates_test, y_test.values, label='Actual', linewidth=2, color='blue')
            ax.plot(dates_test, y_pred, label='Predicted', linewidth=2, 
                    color='red', linestyle='dashed')
            
            ax.set_title(f'{ticker} - Past Predictions vs Actual')
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Price ({currency_symbol})')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Historical plot failed: {str(e)}")

if __name__ == "__main__":
    main()