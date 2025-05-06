def fetch_latest_data(ticker):
    data = yf.download(ticker, period="2d", interval="1d")

    if data.empty:
        return pd.DataFrame()

    # If index is datetime, reset and rename
    data.reset_index(inplace=True)

    # Handle case where Date column may not be named 'Date'
    if 'Date' not in data.columns:
        if 'index' in data.columns:
            data.rename(columns={'index': 'Date'}, inplace=True)
        else:
            # assume first column is the date
            data.rename(columns={data.columns[0]: 'Date'}, inplace=True)

    # Ensure proper datetime format
    try:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    except Exception as e:
        print(f"Date conversion error: {e}")
        return pd.DataFrame()

    # Drop rows where date couldn't be parsed
    if 'Date' in data.columns:
        data.dropna(subset=['Date'], inplace=True)
    else:
        return pd.DataFrame()

    return data

print("Fetched columns:", data.columns)
print("First few rows:\n", data.head())
