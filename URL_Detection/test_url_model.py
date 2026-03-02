import pandas as pd
import joblib

# Load trained model
model = joblib.load("url_model.pkl")

# Load dataset to get column names
df = pd.read_csv("PhishingData.csv")
feature_columns = df.drop("Result", axis=1).columns

print("Enter feature values (-1 or 1)")

user_data = []

for col in feature_columns:
    value = int(input(f"{col}: "))
    user_data.append(value)

# Convert to DataFrame
input_df = pd.DataFrame([user_data], columns=feature_columns)

# Predict
prediction = model.predict(input_df)

if prediction[0] == 1:
    print("\nThis URL is Legitimate ✅")
else:
    print("\nThis URL is Phishing ❌")
