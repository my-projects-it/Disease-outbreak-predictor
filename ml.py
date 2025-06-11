from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib

# Sample training data (mock, simple)
data = {
    'temperature': [30, 45, 25, 35, 40, 28],
    'humidity': [70, 40, 60, 50, 20, 65],
    'previous_cases': [5, 25, 3, 10, 40, 2],
    'region_east': [0, 0, 1, 0, 0, 1],
    'region_north': [1, 0, 0, 0, 1, 0],
    'region_south': [0, 1, 0, 1, 0, 0],
    'region_west': [0, 0, 0, 0, 0, 0]
}
X = pd.DataFrame(data)
y = [0, 1, 0, 1, 1, 0]  # Labels (0 = No Outbreak, 1 = Outbreak)

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model
model_path = "/mnt/data/outbreak_model.pkl"
joblib.dump(model, model_path)

model_path
