!npm install -g cloudflared

import os
os.makedirs("app", exist_ok=True)
os.makedirs("model", exist_ok=True)

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = {
    'temperature': [28, 32, 30, 35, 25, 29, 31, 27],
    'humidity': [75, 60, 80, 55, 85, 70, 65, 90],
    'previous_cases': [5, 15, 10, 20, 3, 7, 13, 2],
    'region': ['north', 'south', 'east', 'west', 'north', 'south', 'east', 'west'],
    'outbreak': [1, 1, 0, 1, 0, 0, 1, 0]
}
df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=['region'])
X = df.drop('outbreak', axis=1)
y = df['outbreak']
X_train, _, y_train, _ = train_test_split(X, y, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'model/outbreak_model.pkl')

with open("app/streamlit_app.py", "w") as f:
    f.write("""
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Disease Outbreak Predictor")
st.title("🦠 Disease Outbreak Predictor")

model = joblib.load('model/outbreak_model.pkl')

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
""")

!streamlit run app/streamlit_app.py &>/content/logs.txt &

!cloudflared tunnel --url http://localhost:8501
