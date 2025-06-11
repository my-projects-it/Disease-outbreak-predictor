!pip install scikit-learn==1.4.2 joblib pandas

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Sample data
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

joblib.dump(model, 'outbreak_model.pkl')
print("✅ Model saved successfully!")
