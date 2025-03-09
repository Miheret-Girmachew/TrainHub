import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'text']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def clean_text(text):
    try:
     
        text = re.sub(r'<[^>]+>', '', text)
       
        text = re.sub(r'[^a-zA-Z]', ' ', text)
       
        words = text.lower().split()
   
        stop_words = set(stopwords.words("english"))
        words = [w for w in words if not w in stop_words]
 
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
      
        return ' '.join(words)
    except LookupError as e:
        print(f"Error: Required NLTK resource missing: {e}.  Please download it using nltk.download().  Make sure to run this command in a Python interpreter *before* running this script.")
        return "" 

def split_data(df, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def vectorize_data(X_train, X_test, max_features=5000):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf

def preprocess(file_path, test_size=0.2, random_state=42, max_features=5000):
    df = load_data(file_path)
    if df is None:
        return None, None, None, None


    df['text'] = df['text'].apply(clean_text)

    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)
    X_train_tfidf, X_test_tfidf = vectorize_data(X_train, X_test, max_features)
    return X_train_tfidf, X_test_tfidf, y_train, y_test