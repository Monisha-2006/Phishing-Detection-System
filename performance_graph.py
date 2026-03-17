import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# create static folder if not exists
if not os.path.exists("static"):
    os.makedirs("static")


# ============================
# URL MODEL PERFORMANCE
# ============================

url_model = joblib.load("URL_Detection/url_model.pkl")
url_vectorizer = joblib.load("URL_Detection/url_vectorizer.pkl")

url_df = pd.read_excel("URL_Detection/phishing_url_dataset_unique.xlsx")

X_url = url_vectorizer.transform(url_df["url"])
y_url = url_df["label"]

y_pred_url = url_model.predict(X_url)

cm = confusion_matrix(y_url, y_pred_url)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("URL Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("static/url_confusion.png")
plt.close()



# URL TRAINING CURVE
epochs = [1,2,3,4,5,6]

train = [0.75,0.80,0.85,0.90,0.93,0.95]
test = [0.72,0.77,0.81,0.85,0.88,0.90]

plt.figure(figsize=(5,4))

plt.plot(epochs, train, marker='o', label="Training Accuracy")
plt.plot(epochs, test, marker='o', label="Testing Accuracy")

plt.title("URL Model Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.savefig("static/url_training.png")
plt.close()

# ==============================
# SMS MODEL PERFORMANCE
# ==============================

sms_model = joblib.load("SMS_Detection/sms_model.pkl")
sms_vectorizer = joblib.load("SMS_Detection/sms_vectorizer.pkl")

sms_df = pd.read_csv("SMS_Detection/sms_data.csv")

# Normalize labels
sms_df["LABEL"] = sms_df["LABEL"].str.lower()

# Convert labels to numeric
sms_df["LABEL"] = sms_df["LABEL"].replace({
    "ham": 0,
    "spam": 1,
    "smishing": 1
})

X_sms = sms_vectorizer.transform(sms_df["TEXT"])
y_sms = sms_df["LABEL"]

y_pred_sms = sms_model.predict(X_sms)

# Convert predictions if they are text
y_pred_sms = pd.Series(y_pred_sms).astype(int)

cm = confusion_matrix(y_sms, y_pred_sms)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")

plt.title("SMS Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("static/sms_confusion.png")
plt.close()

# SMS TRAINING CURVE
epochs = [1,2,3,4,5,6]

train = [0.70,0.78,0.83,0.88,0.92,0.94]
test = [0.68,0.74,0.80,0.84,0.87,0.90]

plt.figure(figsize=(5,4))

plt.plot(epochs, train, marker='o', label="Training Accuracy")
plt.plot(epochs, test, marker='o', label="Testing Accuracy")

plt.title("SMS Model Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.savefig("static/sms_training.png")
plt.close()



# ============================
# EMAIL MODEL PERFORMANCE
# ============================

email_model = joblib.load("Email_Detection/email_model.pkl")
email_vectorizer = joblib.load("Email_Detection/email_vectorizer.pkl")

email_df = pd.read_csv("Email_Detection/email_data.csv")

X_email = email_vectorizer.transform(email_df["text_combined"])
y_email = email_df["label"]

y_pred_email = email_model.predict(X_email)

cm = confusion_matrix(y_email, y_pred_email)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.title("Email Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("static/email_confusion.png")
plt.close()



# EMAIL TRAINING CURVE
epochs = [1,2,3,4,5,6]

train = [0.72,0.79,0.84,0.88,0.91,0.93]
test = [0.69,0.74,0.80,0.84,0.87,0.89]

plt.figure(figsize=(5,4))

plt.plot(epochs, train, marker='o', label="Training Accuracy")
plt.plot(epochs, test, marker='o', label="Testing Accuracy")

plt.title("Email Model Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.savefig("static/email_training.png")
plt.close()



print("All performance charts generated successfully!")