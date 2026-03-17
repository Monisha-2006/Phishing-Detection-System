import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
model = LinearSVC(class_weight='balanced')

# Load dataset
df = pd.read_csv("email_data.csv")
df["text_combined"] = df["text_combined"].fillna("")

# Your dataset columns:
# text_combined , label
df.columns = ["message", "label"]

print("Legitimate count:", (df["label"] == 0).sum())
print("Phishing count:", (df["label"] == 1).sum())

# Clean text
def clean_text(text):
    text = str(text).lower()

    # remove links
    text = re.sub(r'http\S+', ' ', text)

    # remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # remove numbers
    text = re.sub(r'\d+', ' ', text)

    # remove special characters
    text = re.sub(r'[^a-z ]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df["message"] = df["message"].apply(clean_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.3,
    random_state=42,
    stratify=df["label"]
)

# TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),
    stop_words='english',
    min_df=2,
    max_features=15000
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Balanced Logistic Regression
#model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Email Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "email_model.pkl")
joblib.dump(vectorizer, "email_vectorizer.pkl")

print("Email model saved successfully!")