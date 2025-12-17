import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="PM2.5 Forecast Comparison Dashboard",
    layout="wide"
)

st.title("🌫️ Dashboard Perbandingan Forecast PM2.5")
st.markdown("""
Perbandingan **ARIMAX** dan **LSTM Multivariate**  
Resolusi data: **30 menit** (Tahun 2024)
""")

# =====================================================
# LOAD MODEL & SCALER
# =====================================================
@st.cache_resource
def load_assets():
    lstm_model = load_model("model/lstm_model.h5")
    scaler_X = joblib.load("model/scaler_X.pkl")
    scaler_y = joblib.load("model/scaler_y.pkl")
    with open("model/arimax_model.pkl", "rb") as f:
        arimax_model = pickle.load(f)
    return lstm_model, arimax_model, scaler_X, scaler_y

lstm_model, arimax_model, scaler_X, scaler_y = load_assets()

# =====================================================
# LOAD & PREPROCESS DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/Data Stasiun Surabaya Tandes Tahun 2024- Surabaya Tandes - Data Stasiun Surabaya Tandes Tahun 2024.xlsx - Surabaya Tandes.csv"
    )

    df = df.rename(columns={
        'Waktu': 'waktu',
        'PM2.5': 'pm25',
        'Kec.Angin': 'kec_angin',
        'Arah Angin': 'arah_angin',
        'Kelembaban': 'kelembaban',
        'Suhu': 'suhu',
        'Tek.Udara': 'tek_udara',
        'Sol.Rad': 'sol_rad',
        'Curah Hujan': 'curah_hujan'
    })

    df['waktu'] = pd.to_datetime(df['waktu'])
    df = df.sort_values('waktu')
    df = df.set_index('waktu')

    return df

df = load_data()

# =====================================================
# FEATURE LIST (HARUS SAMA DENGAN TRAINING)
# =====================================================
features = [
    'kec_angin',
    'arah_angin',
    'kelembaban',
    'suhu',
    'tek_udara',
    'sol_rad',
    'curah_hujan'
]

WINDOW = 48

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Pengaturan Forecast")

horizon = st.sidebar.slider(
    "Forecast Horizon (30 menit)",
    min_value=1,
    max_value=96,
    value=48
)

# =====================================================
# LSTM FORECAST
# =====================================================
def forecast_lstm(model, last_seq, steps):
    preds = []
    seq = last_seq.copy()

    for _ in range(steps):
        p = model.predict(seq.reshape(1, *seq.shape), verbose=0)[0, 0]
        preds.append(p)

        seq = np.vstack([
            seq[1:],
            np.hstack([p, seq[-1, 1:]])
        ])

    return np.array(preds)

X_scaled = scaler_X.transform(df[features])
last_seq = X_scaled[-WINDOW:]

lstm_scaled = forecast_lstm(lstm_model, last_seq, horizon)
lstm_forecast = scaler_y.inverse_transform(lstm_scaled.reshape(-1, 1)).flatten()

# =====================================================
# ARIMAX FORECAST
# =====================================================
exog_future = df[features].iloc[-horizon:]
arimax_forecast = arimax_model.forecast(
    steps=horizon,
    exog=exog_future
).values

# =====================================================
# FUTURE TIME INDEX
# =====================================================
future_index = pd.date_range(
    start=df.index[-1] + pd.Timedelta(minutes=30),
    periods=horizon,
    freq='30T'
)

# =====================================================
# MAIN VISUALIZATION
# =====================================================
st.subheader("📈 Perbandingan Forecast")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    df.index[-200:],
    df['pm25'][-200:],
    label="Actual",
    color="black"
)

ax.plot(
    future_index,
    lstm_forecast,
    '--',
    label="LSTM Forecast",
    linewidth=2
)

ax.plot(
    future_index,
    arimax_forecast,
    '--',
    label="ARIMAX Forecast",
    linewidth=2
)

ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_xlabel("Waktu")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# =====================================================
# ERROR METRICS
# =====================================================
y_true = df['pm25'].iloc[-horizon:]

mae_lstm = mean_absolute_error(y_true, lstm_forecast)
rmse_lstm = np.sqrt(mean_squared_error(y_true, lstm_forecast))

mae_arimax = mean_absolute_error(y_true, arimax_forecast)
rmse_arimax = np.sqrt(mean_squared_error(y_true, arimax_forecast))

# =====================================================
# METRIC CARDS
# =====================================================
st.subheader("📊 Evaluasi Kinerja Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ARIMAX")
    st.metric("MAE", f"{mae_arimax:.2f}")
    st.metric("RMSE", f"{rmse_arimax:.2f}")

with col2:
    st.markdown("### LSTM")
    st.metric("MAE", f"{mae_lstm:.2f}")
    st.metric("RMSE", f"{rmse_lstm:.2f}")

# =====================================================
# ERROR BAR CHART
# =====================================================
st.subheader("📉 Perbandingan Error")

error_df = pd.DataFrame({
    "Model": ["ARIMAX", "LSTM"],
    "MAE": [mae_arimax, mae_lstm],
    "RMSE": [rmse_arimax, rmse_lstm]
})

fig2, ax2 = plt.subplots(figsize=(6,4))
error_df.set_index("Model").plot(kind="bar", ax=ax2)
ax2.set_ylabel("Error")
ax2.grid(axis="y")

st.pyplot(fig2)
