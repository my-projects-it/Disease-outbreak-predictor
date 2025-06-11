import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample dummy data (replace with your real dataset)
data = pd.DataFrame({
    'temperature': [30, 35, 20, 40, 28],
    'humidity': [60, 65, 55, 80, 70],
    'rainfall': [100, 120, 80, 200, 90],
    'disease_outbreak': [1, 1, 0, 1, 0]
})

X = data[['temperature', 'humidity', 'rainfall']]
y = data['disease_outbreak']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model in compatible format
joblib.dump(model, 'outbreak_model.pkl')
