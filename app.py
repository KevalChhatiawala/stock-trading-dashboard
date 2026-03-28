import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Trading Dashboard", layout="wide")

st.markdown("""
<style>
.card {
    background: linear-gradient(145deg, #1c1f26, #111);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.6);
}
.metric-title {
    color: #aaa;
    font-size: 14px;
}
.metric-value {
    font-size: 26px;
    font-weight: bold;
}
.buy { color: #00ff9c; font-weight: bold; }
.sell { color: #ff4d4d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Smart Stock Trading Dashboard")

stocks = [
    "AAPL","MSFT","TSLA","GOOGL","AMZN","META","NVDA",
    "NFLX","IBM","ORCL","INTC","AMD","BA","JPM",
    "WMT","DIS","NKE","PFE","KO","PEP"
]

col1, col2, col3 = st.columns(3)

with col1:
    selected_stocks = st.multiselect("Select Stocks", stocks, default=["AAPL","MSFT"])

with col2:
    start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))

with col3:
    end_date = st.date_input("End Date", pd.to_datetime("2020-01-01"))

if not selected_stocks:
    st.warning("Select at least one stock")
    st.stop()

refresh = st.sidebar.slider("Refresh (sec)", 5, 60, 10)

@st.cache_data(ttl=60)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

st.subheader("📈 Growth Comparison")

fig = go.Figure()

for stock in selected_stocks:
    df = load_data(stock, start_date, end_date)
    if df.empty:
        continue

    df["Growth"] = (df["Close"] / df["Close"].iloc[0]) * 100

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Growth"],
        mode="lines",
        name=stock
    ))

fig.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📌 Market Insights")

for stock in selected_stocks:

    st.markdown("---")
    st.header(f"🔎 {stock}")

    df = load_data(stock, start_date, end_date)

    if df.empty:
        st.error("No data available")
        continue

    live_price = None
    try:
        live = yf.download(stock, period="1d", interval="1m")
        if not live.empty:
            live_price = float(live["Close"].iloc[-1])
    except:
        pass

    live_display = f"${live_price:.2f}" if live_price else "N/A"

    ml_df = df.copy()
    ml_df["Prediction"] = ml_df["Close"].shift(-1)
    ml_df.dropna(inplace=True)

    if len(ml_df) < 50:
        st.warning("Not enough data for prediction")
        continue

    X = ml_df[["Open","High","Low","Close","Volume"]]
    y = ml_df["Prediction"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    next_price = float(model.predict(X.tail(1))[0])
    last_close = float(ml_df["Close"].iloc[-1])

    change = ((next_price - last_close) / last_close) * 100

    last_date = ml_df["Date"].iloc[-1]
    pred_date = last_date + pd.Timedelta(days=1)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="card">
        <div class="metric-title">Live Price</div>
        <div class="metric-value">{live_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
        <div class="metric-title">MAE</div>
        <div class="metric-value">{mae:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
        <div class="metric-title">Change %</div>
        <div class="metric-value">{change:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="card">
        <div class="metric-title">Prediction Date</div>
        <div class="metric-value">{pred_date.date()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.success(f"📌 Predicted Price: ${next_price:.2f}")

    if change > 0:
        st.markdown(f"<p class='buy'>📈 BUY SIGNAL (+{change:.2f}%)</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='sell'>📉 SELL SIGNAL ({change:.2f}%)</p>", unsafe_allow_html=True)

time.sleep(refresh)
st.rerun()