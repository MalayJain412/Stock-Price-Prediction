import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import os

st.set_page_config(page_title="Next-Day Stock Forecast", layout="wide")

# ─── Helper: fetch yesterday & today ─────────────────────────────
def fetch_two_days(ticker):
    # 1. Download a 5-day window (to capture two trading days)
    df = yf.download(ticker, period="5d", interval="1d")
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    # 2. Flatten multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 3. Rename “Adj Close” → “Close” so it matches your CSV
    if 'Adj Close' in df.columns:
        df.rename(columns={'Adj Close': 'Close'}, inplace=True)

    # 4. Reset index so Date becomes a column
    df = df.reset_index()

    # 5. Keep only the exact columns your model was trained on
    keep = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = df.loc[:, [c for c in keep if c in df.columns]]

    # 6. Parse the 'Date' column (capital “D”) to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    # 7. Return just the last two rows (yesterday + today)
    return df.tail(2)



# ─── Main forecasting ────────────────────────────────────────────
def run_forecast(ticker):
    data = fetch_two_days(ticker)
    if data.empty:
        st.error("Failed to fetch at least two days of data.")
        return

    # Display fetched data
    st.subheader(f"Fetched Data for {ticker}")
    # Data already indexed by date
    st.dataframe(data)

    # Prepare features from today (last row)
    today = data.iloc[-1]
    X = today[['open', 'high', 'low', 'volume']].to_frame().T

    # Load model
    model_path = f"Models/{ticker}_xgb_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
        return
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Predict next-day close
    pred = model.predict(X)[0]

    st.markdown(f"### Predicted Next-Day Close for **{ticker}**:  **${pred:,.2f}**")
    st.write(
        f"• Using today's features: Open=${today['open']:.2f}, "
        f"High=${today['high']:.2f}, "
        f"Low=${today['low']:.2f}, "
        f"Volume={int(today['volume']):,}"
    )

# ─── Streamlit UI ────────────────────────────────────────────────
st.title("📈 Next-Day Stock Price Forecast")
ticker = st.selectbox(
    "Choose a ticker to forecast:",
    ["AAPL", "MSFT", "GOOG", "TCS.NS", "TATAMOTORS.NS", "NBCC.NS"]
)

if st.button("🔮 Forecast Next-Day Close"):
    run_forecast(ticker)
