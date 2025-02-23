import re
from transformers import AutoTokenizer
import nltk
import json

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


def split_into_paragraphs(text):
    # Split text into paragraphs based on double newlines
    paragraphs = text.split('\n\n')
    return paragraphs


def save_as_json(paragraphs, filename):
    data = [{"text": paragraph} for paragraph in paragraphs]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def preprocess_text(text):
    # Remove special characters and convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return cleaned_text


def save_preprocessed_text(tokens, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(token + "\n")


def split_text(text, max_length):
    # Split the text into chunks of max_length characters
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


# Load the diary_text.txt file with the correct encoding
with open('diary_text.txt', 'r', encoding='utf-8') as f:
    text_main = f.read()

# Preprocess the text to remove special characters and convert to lowercase
cleaned_text = preprocess_text(text_main)

# Split the cleaned text into chunks before tokenization
max_chunk_length = 1000  # Adjust the chunk size as needed
text_chunks = split_text(cleaned_text, max_chunk_length)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

all_tokens = []

# Tokenize each chunk and collect all tokens
for chunk in text_chunks:
    tokens = tokenizer.tokenize(chunk)
    all_tokens.extend(tokens)

# Save the collected tokens into a single file
save_preprocessed_text(all_tokens, 'preprocessed_diary_text.txt')

# Assuming 'cleaned_text' contains your preprocessed text
paragraphs = split_into_paragraphs(cleaned_text)
save_as_json(paragraphs, 'tokenized_preprocessed_diary_text.json')

print(f"Total tokens: {len(all_tokens)}")
