import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset (CREATE simple dataset first)
df = pd.read_csv("url_text_dataset.csv")

# Columns must be: url , label
df.columns = ["url", "label"]

# Convert label to number
df["label"] = df["label"].map({"legitimate": 0, "phishing": 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["url"], df["label"], test_size=0.3, random_state=42
)

# TFIDF
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("URL Model Accuracy:", accuracy)

# Save
joblib.dump(model, "url_model.pkl")
joblib.dump(vectorizer, "url_vectorizer.pkl")

print("New URL model saved successfully!")