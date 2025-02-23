import re
import json


def process_text(txt_file_path):
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split the text into paragraphs using two or more newlines as separators
    paragraphs = re.split(r'\n\s*\n', text.strip())

    # List of months to check for in exclusion rule
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December',
              'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept',
              'Oct', 'Nov', 'Dec']

    # Set of strings for the second exclusion rule
    exclusion_strings = {'~', 'Recovery'}

    processed_paragraphs = []

    count_single_lines = 0

    exclude = False

    for paragraph in paragraphs:
        # Strip leading and trailing whitespace
        paragraph = paragraph.strip()

        # Skip empty paragraphs
        if not paragraph:
            continue

        # Check if paragraph is a single line
        is_single_line = '\n' not in paragraph
        # Count single lines to test previous step efficiency
        if is_single_line:
            count_single_lines += 1

        exclude = False


        # Rule 1: Exclude if single line, standalone, and contains any of the exclusion strings
        if all(excl_str in paragraph for excl_str in exclusion_strings):
            exclude = True

        if not exclude:
            # Append the paragraph as a dictionary with key 'chunk'
            processed_paragraphs.append({'chunk': str(paragraph)})

    # Convert the list of dictionaries to a JSON object
    json_output = json.dumps(processed_paragraphs, ensure_ascii=False, indent=4)

    return json_output


# Specify the path to your text file
file_path = r'Data\v4_step1_cleaned_diary.txt'

# Process the text and get the JSON output
json_result = process_text(file_path)

# Optionally, write the JSON output to a file
with open(r'Data\v4_step2_cleaned_diary.json', 'w', encoding='utf-8') as f:
    f.write(json_result)

# Print the JSON output
print(json_result)
