import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ----------- Preprocessing Functions -----------

def handle_missing_values(df):
    if 'text' in df.columns:
        df['text'] = df['text'].fillna('')
    if 'title' in df.columns:
        df['title'] = df['title'].fillna('')
    df = df.dropna(subset=['text', 'title'], how='all')
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.split())

def preprocess_text(text):
    tokens = word_tokenize(str(text))
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def preprocess_dataset(df):
    if 'title' in df.columns and 'text' in df.columns:
        df['combined_text'] = df['title'] + ' ' + df['text']
    elif 'text' in df.columns:
        df['combined_text'] = df['text']
    elif 'title' in df.columns:
        df['combined_text'] = df['title']
    else:
        raise ValueError("No 'text' or 'title' column found in dataset.")

    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    df['processed_text'] = df['cleaned_text'].apply(preprocess_text)
    return df

def vectorize_text(df, max_features=3000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['processed_text']).toarray()
    y = df['label'].values
    return X, y, vectorizer

# ----------- Main Preprocessing Function -----------

def preprocess_train_csv(train_csv_path, output_path=None):
    if not os.path.exists(train_csv_path):
        print(f"Error: File not found at {train_csv_path}")
        return None, None, None, None

    print(f"Loading train.csv from: {train_csv_path}")
    df = pd.read_csv(train_csv_path)

    df = handle_missing_values(df)
    df = preprocess_dataset(df)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Processed train.csv saved to: {output_path}")

    X, y, vectorizer = vectorize_text(df)
    print("✅ Preprocessing complete. Data is ready for modeling.")
    return X, y, vectorizer, df

# ----------- Run Preprocessing -----------

if __name__ == "__main__":
    train_csv_path = 'D:/projects/ongoing-projs/fake_news_detection/data/train.csv'
    output_csv_path = 'D:/projects/ongoing-projs/fake_news_detection/data/train_processed.csv'

    X, y, vectorizer, processed_df = preprocess_train_csv(train_csv_path, output_csv_path)

    if X is not None:
        print(f"Vectorized shape: {X.shape}")
        print(f"Sample processed text: {processed_df['processed_text'].iloc[0][:100]}...")
