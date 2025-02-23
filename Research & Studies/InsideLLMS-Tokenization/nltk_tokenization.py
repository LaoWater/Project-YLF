import re
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Remove special characters and convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

    # Tokenize the cleaned text
    tokens = word_tokenize(cleaned_text)

    return tokens

def save_preprocessed_text(tokens, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(token + "\n")

# Load the diary_text.txt file with the correct encoding
with open('diary_text.txt', 'r', encoding='utf-8') as f:
    text_main = f.read()

# Preprocess the text
tokens_main = preprocess_text(text_main)

# Save the preprocessed text
save_preprocessed_text(tokens_main, 'Simple_Tokenization.txt')
