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


# Test texts
texts = [
    "This is a clear English sentence.",
    "Acesta este un text în engleză și română.",  # Mixed English and Romanian
    "This is partially în română.",  # English with little Romanian
    "Acesta este un text complet în română.",  # Fully Romanian
    "Acesta este română with some English mixed in."  # Romanian with little English
]

# Process each text
for idx, text in enumerate(texts, start=1):
    print(f"Text {idx}: {text}")
    is_english, probability = is_english_with_probability(text)
    if is_english:
        print(f"  Detected as English with probability: {probability:.2f}")
    else:
        print("  Not detected as English or has low confidence.")
    print("-" * 40)
