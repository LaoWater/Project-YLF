import json
from langdetect import detect_langs, LangDetectException


# Function to check if the text is in English and return the probability
def is_english_with_probability(text):
    try:
        # Get probabilities for all detected languages
        lang_probabilities = detect_langs(text)
        for lang in lang_probabilities:
            if lang.lang == 'en':  # Check if English is detected
                return True, lang.prob  # Return True and the probability
        return False, 0.0  # If English is not detected, return False and 0 probability
    except LangDetectException:
        print(f"Could not detect language for the text: {text[:50]}...")  # Print a portion of the text for context
        return False, 0.0


# File path
file_path = "../Data/v4_step2_cleaned_diary.json"


# Read and process the diary file
def process_diary(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            diary_entries = json.load(f)  # Assuming the file contains a JSON array of entries

        # Process each entry
        for idx, entry in enumerate(diary_entries, start=1):
            text = entry.get("chunk", "")  # Assuming each entry has a 'chunk' key with the text
            if not text:
                continue

            is_english, probability = is_english_with_probability(text)
            # print(f"Is English? {is_english} with probability {probability}")
            if is_english and probability < 0.11:
                print(f"Entry {idx}:")
                print(f"  Text: {text[:100]}...")  # Print a portion of the text for context
                print(f"  Detected as English with probability: {probability:.2f}")
                print("-" * 40)

            if not is_english:
                print(f"Entry {idx}:")
                print(f"  Text: {text[:100]}...")  # Print a portion of the text for context
                print(f"  Detected as Non-English with probability: {probability:.2f}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from the file: {file_path}")


# Run the script
if __name__ == "__main__":
    process_diary(file_path)
