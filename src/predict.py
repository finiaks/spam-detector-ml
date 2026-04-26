import joblib as job
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Load Model and TF_IDF
model = job.load("../model/spam_model.pkl")
tfidf = job.load("../model/tfidf.pkl")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(w) for w in words if w.isalpha() and w not in stop_words]
    return " ".join(words)

#Text with your own message
message = input("Enter your message: ")
cleaned = clean_text(message)
result = model.predict(tfidf.transform([cleaned]))

if result[0] == 1:
    print("Spam")
else:
    print("Not Spam")