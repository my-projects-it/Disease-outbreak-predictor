from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import pandas as pd

import joblib

import os



# Sample dataset

data = {

    'temperature': [28, 32, 30, 35, 25, 29, 31, 27],

    'humidity': [75, 60, 80, 55, 85, 70, 65, 90],

    'previous_cases': [5, 15, 10, 20, 3, 7, 13, 2],

    'region': ['north', 'south', 'east', 'west', 'north', 'south', 'east', 'west'],

    'outbreak': [1, 1, 0, 1, 0, 0, 1, 0]

}

df = pd.DataFrame(data)



# One-hot encoding for 'region'

df = pd.get_dummies(df, columns=['region'])



# Split features and target

X = df.drop('outbreak', axis=1)

y = df['outbreak']



# Split data

X_train, _, y_train, _ = train_test_split(X, y, random_state=42)



# Train model

model = RandomForestClassifier()

model.fit(X_train, y_train)



# Save model to 'model' directory

os.makedirs("/mnt/data/model", exist_ok=True)

joblib.dump(model, '/mnt/data/model/outbreak_model.pkl')
