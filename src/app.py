import streamlit as st
import joblib as job
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

#Load Model and TF-IDF
model = job.load("model/spam_model.pkl")
tfidf = job.load("model/tfidf.pkl")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(w) for w in words if w.isalpha() and w not in stop_words]
    return " ".join(words)

#UI
st.title("Spam SMS Detector")
st.write("Enter a Message to check if it is spam or not!")

message = st.text_area("Enter your Message here: ")

if st.button("Check"):
    if message.strip() == "":
        st.warning("Please enter a messsage!")
    else:
        cleaned = clean_text(message)
        result = model.predict(tfidf.transform([cleaned]))

        if result[0] == 1:
            st.error("Spam Message Detected!")
        else:
            st.success("Not Spam Safe Message!")
