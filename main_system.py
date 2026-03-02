import joblib

print("==== PHISHING DETECTION SYSTEM ====")
print("1. URL Detection")
print("2. SMS Detection")
print("3. Email Detection")

choice = input("Enter your choice (1/2/3): ")

if choice == "1":
    print("URL Detection Model Loaded")
    model = joblib.load("URL_Detection/url_model.pkl")
    vectorizer = joblib.load("URL_Detection/url_vectorizer.pkl")

    url = input("Enter URL: ")
    url_vector = vectorizer.transform([url])
    prediction = model.predict(url_vector)

    print("\n===== DETECTION RESULT =====")

    prob = model.predict_proba(url_vector)[0][1]
    risk_score = round(prob * 100, 2)
    print("Risk Score:", risk_score, "%")

    if prediction[0] == 1:
        print("⚠ Phishing URL Detected!")
    else:
        print("✅ Legitimate URL")
elif choice == "2":
    print("SMS Detection Model Loaded")
    model = joblib.load("SMS_Detection/sms_model.pkl")
    vectorizer = joblib.load("SMS_Detection/sms_vectorizer.pkl")

    text = input("Enter SMS message: ")
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    print("\n===== DETECTION RESULT =====")
    prob = model.predict_proba(text_vector)[0][1]
    risk_score = round(prob * 100, 2)
    print("Risk Score:", risk_score, "%")

    if prediction[0] == 1:
        print("⚠ SPAM SMS Detected!")
    else:
        print("✅ Legitimate SMS")

elif choice == "3":
    print("Email Detection Model Loaded")
    model = joblib.load("Email_Detection/email_model.pkl")
    vectorizer = joblib.load("Email_Detection/email_vectorizer.pkl")

    text = input("Enter Email text: ")
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    print("\n===== DETECTION RESULT =====")
    prob = model.predict_proba(text_vector)[0][1]
    risk_score = round(prob * 100, 2)
    print("Risk Score:", risk_score, "%")


    if prediction[0] == 1:
        print("⚠ Phishing Email Detected!")
    else:
        print("✅ Legitimate Email")

else:
    print("Invalid choice")
