import streamlit as st
import pickle
import string

# Load saved model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Clean input text
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    return text

# Streamlit UI
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")
st.title("📩 SMS Spam Classifier")
st.write("Type an SMS message and check whether it's spam or not.")

user_input = st.text_area("✉️ Enter your SMS here:")

if st.button("🔍 Predict"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        st.error("🚨 This message is **SPAM**!")
    else:
        st.success("✅ This message is **HAM** (Not Spam).")
