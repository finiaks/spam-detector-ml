import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib as job

#Load Dataset
df = pd.read_csv("../dataset/spam.csv", encoding = 'latin-1')

#Keep useful Columns
df = df[["Category","Message"]]
df.columns = ['label','message']

#Clean text function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    #Lowercase
    text = text.lower()
    #Tokenize
    words = nltk.word_tokenize(text)
    #Remove stopwords and stem
    words = [stemmer.stem(w) for w in words if w.isalpha() and w not in stop_words]
    return " ".join(words)

#Apply cleaning
df['cleaned'] = df['message'].apply(clean_text)

#Convert label to numbers
df['label'] = df['label'].map({'ham':0,'spam':1})

#Features and Target
x = df['cleaned']
y = df['label']

#TF-IDF
tfidf = TfidfVectorizer(max_features = 3000)
X = tfidf.fit_transform(x)

#Split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

#Train Model
model = MultinomialNB()
model.fit(X_train,y_train)

#Accuracy
y_pred = model.predict(X_test)
print("Accuracy :",accuracy_score(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))

#Save Model
job.dump(model,"../model/spam_model.pkl")
job.dump(tfidf,"../model/tfidf.pkl")
