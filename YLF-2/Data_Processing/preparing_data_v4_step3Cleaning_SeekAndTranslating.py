import json
import openai
from langdetect import detect, LangDetectException
from openai import OpenAI


# Function to check if the text is in English
def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        print(f"Could not detect language for the text: {text[:50]}...")  # Print a portion of the text for context
        return False


# Function to translate text to English using GPT-3.5
def translate_to_english(text, model="gpt-4o-mini", temperature=0.7):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """You are a helpful assistant that translates text to English.
            Consider that language translation are incomplete and may often be lacking context understanding.
            You are free to read the text and interpret it with medium temperature and make the necessary 
            changes/adjustments so that the broader understanding and art of writing is kept.
            
            - Return specifically the new translated message and only the message, no additional stuff.
            - In case text is a combination of romanian + english, keep english as it is and integrate translated romanian chunk
            """},





            {"role": "user", "content": f"Translate the following text from Romanian to English:\n\n{text}"}
        ],
        temperature=temperature
    )

    response_text = response.choices[0].message.content
    return response_text


# Function to process the JSON data and store translations
def process_json_data(json_data, translation_log_file):
    translated_entries = []
    translations_only = []

    for entry in json_data:
        text = entry['chunk']

        # Check if the text is in English
        if is_english(text):
            print("Detected English text. Adding to the final entries.")
            print(text)
            print("\n\n")
            translated_entries.append(entry)  # Keep the original if it's in English
        else:
            print("Detected other language! Calling Translation LLM for text:")
            print(f"ORIGINAL TEXT:\n{text}\n")

            # Translate the text to English
            translated_text = translate_to_english(text)

            # Debugging notification after translation
            print("Translated text (after API call):")
            print(f"{translated_text}\n")

            # Append the translated text to the list
            translated_entries.append({'chunk': translated_text})

            # Save strictly the translations
            translations_only.append(f"Original: {text}\nTranslated: {translated_text}\n")

    # Write all translations to the log file
    with open(translation_log_file, 'w', encoding='utf-8') as log_file:
        log_file.write("\n".join(translations_only))

    return translated_entries


# Function to format the translations nicely in a text file for review
def write_translations_to_txt(json_file, output_txt_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_txt_file, 'w', encoding='utf-8') as txt_file:
        for i, entry in enumerate(data):
            txt_file.write(f"--- Entry {i+1} ---\n")
            txt_file.write(f"Original Text:\n{entry['chunk']}\n")
            txt_file.write(f"Translated Text:\n{entry.get('text', 'No translation needed.')}\n")
            txt_file.write("\n")

    print(f"Translations written to: {output_txt_file}")


# Paths and filenames
input_file_path = 'Data/v4_step2_cleaned_diary.json'
output_file_path = 'Data/v4_step3_translated_diary.json'
translation_log_file = 'Data/training_data_checking_LLM_mass_translations.txt'

# Read the JSON data from the file
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    diary_json = json.load(input_file)

# Process the JSON data to translate non-English text and save the translations to a log file
translated_json_data = process_json_data(diary_json, translation_log_file)

# Save the translated JSON data to a new file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(translated_json_data, output_file, indent=2)

print("Translated JSON data has been saved to", output_file_path)

# Write the translations to a .txt file for easy review
# write_translations_to_txt(output_file_path, translation_log_file)
