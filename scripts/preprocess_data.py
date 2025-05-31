import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('punkt_tab')  # Added for newer NLTK versions
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Load the datasets
def load_datasets(true_path='D:\fake_news_detection\data\True.csv', false_path=
'D:\fake_news_detection\data\False.csv'):
    """
    Load true and false CSV files and assign labels.
    True news: label = 1, False news: label = 0
    """
    print(f"Checking if True.csv exists: {os.path.exists(true_path)}")
    print(f"Checking if False.csv exists: {os.path.exists(false_path)}")
    try:
        true_df = pd.read_csv(true_path)
        false_df = pd.read_csv(false_path)

        # Add label column
        true_df['label'] = 1
        false_df['label'] = 0

        # Combine datasets
        combined_df = pd.concat([true_df, false_df], ignore_index=True)
        print(f"Combined data shape: {combined_df.shape}")
        return combined_df
    except FileNotFoundError:
        print("Error: One or both CSV files not found. Please check the file paths.")
        return None
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None

# Step 2: Handle missing values
def handle_missing_values(df):
    """
    Check for missing values and handle them.
    Fill missing text with empty string, drop rows with critical missing data.
    """
    print("Missing values before cleaning:")
    print(df.isnull().sum())

    # Fill missing 'text' or 'title' with empty string
    if 'text' in df.columns:
        df['text'] = df['text'].fillna('')
    if 'title' in df.columns:
        df['title'] = df['title'].fillna('')

    # Drop rows where both text and title are empty
    df = df.dropna(subset=['text', 'title'], how='all')

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    return df

# Step 3: Clean text data
def clean_text(text):
    """
    Clean text by removing punctuation, converting to lowercase, removing numbers,
    and removing extra whitespace.
    """
    try:
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ''

# Step 4: Tokenize, remove stopwords, and lemmatize
def preprocess_text(text):
    """
    Tokenize text, remove stopwords, and lemmatize words.
    """
    try:
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Join tokens back into text
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ''

# Step 5: Combine preprocessing steps
def preprocess_dataset(df):
    """
    Apply text cleaning and preprocessing to the data.
    Combine title and text if both exist, else use available column.
    """
    try:
        # Combine title and text if both exist
        if 'title' in df.columns and 'text' in df.columns:
            df['combined_text'] = df['title'] + ' ' + df['text']
        elif 'text' in df.columns:
            df['combined_text'] = df['text']
        elif 'title' in df.columns:
            df['combined_text'] = df['title']
        else:
            print("Error: No 'title' or 'text' column found in data.")
            return None

        # Apply cleaning and preprocessing
        print("Starting text cleaning...")
        df['cleaned_text'] = df['combined_text'].apply(clean_text)
        print("Starting text preprocessing...")
        df['processed_text'] = df['cleaned_text'].apply(preprocess_text)
        print("Text preprocessing completed.")
        return df
    except Exception as e:
        print(f"Error in preprocess_dataset: {e}")
        return None

# Step 6: Vectorize text for machine learning
def vectorize_text(df, max_features=3000):
    """
    Convert processed text to TF-IDF features.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(df['processed_text']).toarray()
        y = df['label'].values
        print("Text vectorization completed.")
        return X, y, vectorizer
    except Exception as e:
        print(f"Error during vectorization: {e}")
        return None, None, None

# Main function to run preprocessing
def main(true_path=r'D:\fake_news_detection\data\True.csv', false_path=r'D:\fake_news_detection\data\False.csv', output_path=r'D:\fake_news_detection\processed_data.csv'):
    """
    Main function to preprocess the fake news data.
    """
    try:
        # Load datasets
        print("Loading datasets...")
        df = load_datasets(true_path, false_path)
        if df is None:
            print("Failed to load datasets.")
            return None, None, None, None

        # Handle missing values
        print("Handling missing values...")
        df = handle_missing_values(df)
        if df is None:
            print("Failed to handle missing values.")
            return None, None, None, None

        # Preprocess text
        print("Preprocessing data...")
        df = preprocess_dataset(df)
        if df is None:
            print("Failed to preprocess data.")
            return None, None, None, None

        # Save processed data
        print("Saving processed data...")
        try:
            df.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
        except Exception as e:
            print(f"Error saving processed data: {e}")

        # Vectorize text
        print("Vectorizing text...")
        X, y, vectorizer = vectorize_text(df)
        if X is not None:
            print(f"Vectorized data shape: {X.shape}")
            return X, y, vectorizer, df
        print("Failed to vectorize text.")
        return None, None, None, df
    except Exception as e:
        print(f"Error in main function: {e}")
        return None, None, None, None

# Run the preprocessing
if __name__ == "__main__":
    # Use the provided CSV file paths
    true_csv_path = r'D:\fake_news_detection\data\True.csv'
    false_csv_path = r'D:\fake_news_detection\data\False.csv'
    output_csv_path = r'D:\fake_news_detection\processed_data.csv'

    # Check if files exist before running
    print(f"Checking paths: True.csv={os.path.exists(true_csv_path)}, False.csv={os.path.exists(false_csv_path)}")
    X, y, vectorizer, processed_df = main(true_csv_path, false_csv_path, output_path=output_csv_path)
    if X is not None:
        print("Preprocessing completed successfully!")
        print(f"Sample processed text: {processed_df['processed_text'].iloc[0][:100]}...")