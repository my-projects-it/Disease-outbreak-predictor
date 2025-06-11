import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 📦 Model file path
model_path = "outbreak_model.pkl"

# 🤖 Train the model automatically if not found
if not os.path.exists(model_path):
    st.info("🔄 Training model for the first time...")

    # 📊 Dummy data (replace with real data for actual use)
    data = {
        "temperature": [98.6, 99.1, 101.2, 97.0, 100.4],
        "humidity": [70, 80, 60, 90, 75],
        "population_density": [1000, 2000, 1500, 1800, 2200],
        "outbreak": [0, 1, 1, 0, 1]
    }

    df = pd.DataFrame(data)

    # 🧠 Training
    X = df.drop("outbreak", axis=1)
    y = df["outbreak"]

    model = RandomForestClassifier()
    model.fit(X, y)

    # 💾 Save model
    joblib.dump(model, model_path)
    st.success("✅ Model trained and saved!")

else:
    # 🔓 Load model
    model = joblib.load(model_path)

# 💡 Prediction UI
st.title("🦠 Disease Outbreak Prediction")

temp = st.number_input("🌡️ Temperature (F)", value=98.6)
humid = st.number_input("💧 Humidity (%)", value=70)
pop_density = st.number_input("👥 Population Density", value=1500)

if st.button("🔍 Predict"):
    input_data = pd.DataFrame([[temp, humid, pop_density]],
                              columns=["temperature", "humidity", "population_density"])
    result = model.predict(input_data)[0]

    if result == 1:
        st.error("🚨 High Risk of Outbreak Detected!")
    else:
        st.success("🛡️ Low Risk of Outbreak")
