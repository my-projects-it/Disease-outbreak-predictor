import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Disease Outbreak Predictor")
st.title("🦠 Disease Outbreak Predictor")

model = joblib.load('outbreak_model.pkl')

temp = st.slider("🌡️ Temperature (°C)", 20, 45, 30)
humidity = st.slider("💧 Humidity (%)", 30, 100, 70)
prev_cases = st.number_input("📊 Previous Week Cases", 0, 100, 5)
region = st.selectbox("📍 Region", ["north", "south", "east", "west"])

input_data = {
    'temperature': temp,
    'humidity': humidity,
    'previous_cases': prev_cases,
    'region_east': 0,
    'region_north': 0,
    'region_south': 0,
    'region_west': 0
}
input_data[f'region_{region}'] = 1

df_input = pd.DataFrame([input_data])

if st.button("🔍 Predict Outbreak"):
    result = model.predict(df_input)[0]
    if result == 1:
        st.error("⚠️ Warning: Outbreak Likely!")
    else:
        st.success("✅ No Outbreak Expected.")
