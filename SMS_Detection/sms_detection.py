import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("sms_data.csv")

# If dataset has more columns, keep only first two
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Clean label column
df["label"] = df["label"].astype(str).str.strip().str.lower()

# Keep only ham and spam
df = df[df["label"].isin(["ham", "spam"])]

# Convert to numbers
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' link ', text)
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return text

df["message"] = df["message"].apply(clean_text)

# Check class balance
print("Ham count:", (df["label"] == 0).sum())
print("Spam count:", (df["label"] == 1).sum())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.3,
    random_state=42,
    stratify=df["label"]
)

# TF-IDF Vectorizer (strong version)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    max_features=5000
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Logistic Regression (stronger than Naive Bayes)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("SMS Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "sms_model.pkl")
joblib.dump(vectorizer, "sms_vectorizer.pkl")

print("SMS model saved successfully!")
