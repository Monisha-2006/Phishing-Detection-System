import streamlit as st
import joblib

# st.markdown("---")
# st.info("Model Accuracy: 77% (URL Detection)")

st.set_page_config(page_title="Phishing Detection System")

st.title("🔐 Phishing Detection System")
st.markdown("### 📊 Model Performance")

st.info("""
URL Detection Accuracy: ~80%  
SMS Detection Accuracy: 97.9%  
Email Detection Accuracy: 98.3%
""")

st.markdown("---")
st.write("Detect Phishing URLs, SMS, and Emails using Machine Learning")


option = st.selectbox(
    "Select Detection Type",
    ("URL Detection", "SMS Detection", "Email Detection")
)

# ---------------- URL DETECTION ----------------
if option == "URL Detection":

    url = st.text_input("Enter URL")

    if st.button("Check URL"):

        model = joblib.load("URL_Detection/url_model.pkl")
        vectorizer = joblib.load("URL_Detection/url_vectorizer.pkl")

        url_vector = vectorizer.transform([url])
        prediction = model.predict(url_vector)
        prob = model.predict_proba(url_vector)[0][1]
        risk_score = round(prob * 100, 2)

        st.write(f"Risk Score: {risk_score}%")

        if prediction[0] == 1:
            st.error("⚠ Phishing URL Detected!")
        else:
            st.success("✅ Legitimate URL")


# ---------------- SMS DETECTION ----------------
elif option == "SMS Detection":

    text = st.text_area("Enter SMS Message")

    if st.button("Check SMS"):

        model = joblib.load("SMS_Detection/sms_model.pkl")
        vectorizer = joblib.load("SMS_Detection/sms_vectorizer.pkl")

        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)
        prob = model.predict_proba(text_vector)[0][1]
        risk_score = round(prob * 100, 2)

        st.write(f"Risk Score: {risk_score}%")

        if prediction[0] == 1:
            st.error("⚠ Spam SMS Detected!")
        else:
            st.success("✅ Legitimate SMS")


# ---------------- EMAIL DETECTION ----------------
elif option == "Email Detection":

    text = st.text_area("Enter Email Content")

    if st.button("Check Email"):

        model = joblib.load("Email_Detection/email_model.pkl")
        vectorizer = joblib.load("Email_Detection/email_vectorizer.pkl")

        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)
        prob = model.predict_proba(text_vector)[0][1]
        risk_score = round(prob * 100, 2)

        st.write(f"Risk Score: {risk_score}%")

        if prediction[0] == 1:
            st.error("⚠ Phishing Email Detected!")
        else:
            st.success("✅ Legitimate Email")