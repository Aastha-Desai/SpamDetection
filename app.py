import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to predict if an email is spam or not
def predict_spam(email_text):
    email_vec = vectorizer.transform([email_text])  # Transform the input text
    prediction = model.predict(email_vec)  # Get the prediction (0 = not spam, 1 = spam)
    
    if prediction[0] == 1:
        return "Spam"
    else:
        return "Not Spam"

# Streamlit UI to get user input
st.title("Email Spam Detector")

# User input for email text
email_text = st.text_area("Enter email text here:")

# Predict button to classify the email
if st.button("Predict"):
    if email_text:
        result = predict_spam(email_text)
        st.write(f"Prediction: {result}")
    else:
        st.write("Please enter the email text for prediction.")
