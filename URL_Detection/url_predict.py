import pandas as pd
import joblib
import re
from urllib.parse import urlparse

# Load trained model
model = joblib.load("url_model.pkl")
feature_names = joblib.load("url_feature_names.pkl")

# -------- FEATURE EXTRACTION FUNCTION --------
def extract_features(url):

    features = []

    # 1. Having IP Address
    if re.search(r'\d+\.\d+\.\d+\.\d+', url):
        features.append(1)
    else:
        features.append(-1)

    # 2. URL Length
    if len(url) < 54:
        features.append(-1)
    elif 54 <= len(url) <= 75:
        features.append(0)
    else:
        features.append(1)

    # 3. Having @ Symbol
    if "@" in url:
        features.append(1)
    else:
        features.append(-1)

    # 4. Double Slash Redirecting
    if url.count("//") > 1:
        features.append(1)
    else:
        features.append(-1)

    # 5. Having Hyphen
    if "-" in url:
        features.append(1)
    else:
        features.append(-1)

    # 6. Subdomain count
    parsed = urlparse(url)
    if parsed.hostname:
        if parsed.hostname.count(".") > 2:
            features.append(1)
        else:
            features.append(-1)
    else:
        features.append(1)

    # Fill remaining features with -1 (temporary simple version)
    while len(features) < len(feature_names):
        features.append(-1)

    return features

# -------- PREDICTION --------
url = input("Enter URL: ")

features = extract_features(url)

input_df = pd.DataFrame([features], columns=feature_names)

prediction = model.predict(input_df)

print("\n===== DETECTION RESULT =====")

if prediction[0] == -1:
    print("⚠ Phishing URL Detected!")
else:
    print("✅ Legitimate URL")