import pandas as pd
import nltk as nk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#Load Dataset
df = pd.read_csv("../dataset/spam.csv", encoding = 'latin-1')
df = df[['Category','Message']]
df.columns = ['label', 'message']

#Clean Text
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    words = nk.word_tokenize(text)
    words = [stemmer.stem(w) for w in words if w.isalpha() and w not in stop_words]
    return " ".join(words)

df['cleaned'] = df['message'].apply(clean_text)
df['label'] = df['label'].map({'ham':0,'spam':1})

#TF-IDF
tfidf = TfidfVectorizer(max_features = 3000)
x = tfidf.fit_transform(df['cleaned'])
y = df['label'].values

#Split Data
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state = 55)

#Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(x_train,y_train)
nb_pred = nb_model.predict(x_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test,nb_pred))

#SVM
svm_model = SVC(kernel = 'linear')
svm_model.fit(x_train,y_train)
svm_pred = svm_model.predict(x_test)
print("SVM Accuracy:", accuracy_score(y_test,svm_pred))

print("\nNaive Bayes Report:", classification_report(y_test,nb_pred))
print("\nSVM Report:", classification_report(y_test,svm_pred))
