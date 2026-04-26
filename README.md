# 📩 Spam SMS Detector

A Machine Learning project that detects whether an SMS message
is spam or not spam using Natural Language Processing (NLP)
and Naive Bayes Classification.

## 🔗 Live Demo

[Click here to try the app](https://spam-detectorapp.streamlit.app)

## 📌 Project Overview

This project uses the SMS Spam Collection dataset containing
5,572 real SMS messages. The model learns patterns from spam
and ham messages and predicts whether a new message is spam
or not with 98% accuracy.

## 🧠 How It Works

1. User enters an SMS message
2. Text is cleaned (lowercase, stopwords removed, stemming)
3. TF-IDF converts text to numbers
4. Naive Bayes model predicts Spam or Ham
5. Result is displayed with color indicator

## 🛠️ Tech Stack

- Python
- NLTK (Natural Language Processing)
- Scikit-learn (Machine Learning)
- Pandas (Data handling)
- Streamlit (Web UI)
- Joblib (Model saving)

## 📊 Model Performance

- Accuracy: 98%
- Precision: 99%
- Algorithm: Multinomial Naive Bayes

## 📁 Project Structure

```
spam_detector_ml/
├── dataset/
│ └── spam.csv
├── model/
│ ├── spam_model.pkl
│ └── tfidf.pkl
├── src/
│ ├── main.py
│ └── predict.py
│ └── app.py
└── requirements.txt
```

## ⚙️ How to Run Locally

1. Clone the repo
2. Install requirements
   pip install -r requirements.txt
3. Run the app
   streamlit run frontend_ui/app.py

## 📦 Dataset

SMS Spam Collection Dataset from UCI ML Repository

- 5,572 messages
- 4,825 Ham messages
- 747 Spam messages

## Author

Akshay Prakash
