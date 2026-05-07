import pandas as pd
import nltk as nk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

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
svm_model = SVC(kernel = 'linear', probability = True)
svm_model.fit(x_train,y_train)
svm_pred = svm_model.predict(x_test)
print("SVM Accuracy:", accuracy_score(y_test,svm_pred))

print("\nNaive Bayes Report:", classification_report(y_test,nb_pred))
print("\nSVM Report:", classification_report(y_test,svm_pred))

#Cross Validation
nb_cv = cross_val_score(nb_model, x, y, cv = 5)
print("NB Average CV Score:", nb_cv.mean())

svm_cv = cross_val_score(svm_model, x, y, cv = 5)
print("SVM Average CV Score:", svm_cv.mean())

#AUC Score
y_prob = nb_model.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, y_prob)
print("AUC Score:", auc)

#Confusion Matrix
cm = confusion_matrix(y_test, svm_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"Spam Caught: {cm[1][1]}")
print(f"Spam Missed: {cm[1][0]}")
print(f"Ham as Spam: {cm[0][1]}")
print(f"Ham Correct: {cm[0][0]}")
