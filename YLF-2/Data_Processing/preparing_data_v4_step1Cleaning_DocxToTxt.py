import docx
import re

def clean_ppi_information(text):
    """Replace sensitive PPI information in all word forms (case-insensitive, keeps original casing)."""
    replacements = {
        "yardi": "work",
        "ammy": "leia",
        "ariana": "ramona",
        "fuck": "duck",
        "sex": "dance",
        "octavu": "bosman",
        "masturbate": "m*",
        "masturbating": "m*",
        "pelicanu": "jon",
        "alex": "jon",
        "tommy": "tomtom",
        "lavinia": "country girl"
    }

    # Extend replacements to handle different word variations
    word_variants = {}
    for word, replacement in replacements.items():
        # Standard word replacement
        word_variants[word] = replacement
        word_variants[word + "s"] = replacement + "s"
        word_variants[word + "ed"] = replacement + "ed"
        word_variants[word + "ing"] = replacement + "ing"
        word_variants[word + "i"] = replacement + "i"

    # Special cases: Replace "sex" but NOT words like "sexual" - this is because the "Sexual Energy",
    # chakra, etc are a big part of these writings and replacing would pollute the teachings
    word_variants.pop("sex", None)  # Remove default plural extensions for "sex"
    word_variants["sex"] = "dance"  # Ensure "sex" is replaced

    # Ensure all keys in dictionary are lowercase for case-insensitive matching
    word_variants = {k.lower(): v for k, v in word_variants.items()}

    # Use regex to match and replace words inside text (but only standalone "sex", not "sexual")
    def replace_match(match):
        original_word = match.group(0)
        replacement_word = word_variants.get(original_word.lower(), original_word)  # Safe lookup

        # Preserve original case
        if original_word.isupper():
            return replacement_word.upper()
        elif original_word[0].isupper():
            return replacement_word.capitalize()
        else:
            return replacement_word.lower()

    # Regex pattern to match full words but exclude words like "sexual"
    pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in word_variants.keys()) + r')\b', re.IGNORECASE)
    return pattern.sub(replace_match, text)


def clean_diary(input_path, output_path):
    # Load the .docx file
    empty_line_count = 0
    doc = docx.Document(input_path)
    entries = []  # Store the final entries (groups of paragraphs)
    current_entry = []  # Temporarily store paragraphs for the current entry

    # List of months to check for in exclusion rule
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December',
              'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept',
              'Oct', 'Nov', 'Dec']

    numbers = set(map(str, range(1, 32)))  # Set of strings '1' to '31'
    paragraph_count = 0
    # Iterate through all paragraphs in the document
    for para in doc.paragraphs:
        text = para.text.strip()  # Strip leading/trailing whitespace
        is_date_line = False
        print(f'{text}, count = {paragraph_count}, Empty lines count: {empty_line_count}')
        if text:
            # Exclude Date lines for Time Dimension is not relevant
            if (empty_line_count >= 1 and any(month in text for month in months)
                    and any(num in text for num in numbers)):
                # skip appending
                print("Date line encountered!")
                is_date_line = True
                continue

            # Reset empty line counter if a non-empty line is encountered
            if not is_date_line:
                empty_line_count = 0
                current_entry.append(text)  # Add non-empty paragraph to the current entry
        else:
            # Increment empty line counter for an empty line
            empty_line_count += 1

            if empty_line_count == 2:  # Check for two consecutive empty lines
                if current_entry:
                    # Finalize the current entry when two empty lines are encountered
                    entries.append('\n'.join(current_entry))
                    current_entry = []  # Reset for the next entry



        # paragraph_count += 1
        #
        # if paragraph_count > 333:
        #     break

    # Add the last entry if it exists and no double empty lines followed
    if current_entry:
        entries.append('\n'.join(current_entry))

    # Join all entries with a newline separator (indicating a new entry)
    full_text = '\n\n'.join(entries)

    # Apply PPI cleaning
    full_text = clean_ppi_information(full_text)

    # Save the cleaned text to a .txt file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(full_text)

    return full_text  # Optionally return the cleaned text for further use


#####################################
#### Cleaning & Processing Diary ####
#####################################

print("Cleaning Docx and transcribing to txt...")
clean_diary(r'Data\Lao_All Writings.docx', r'Data\v4_step1_cleaned_diary.txt')


# docx_to_json(r'Data\Extra Organic Dataset\Discord Conversation until 26 Dec 2024.docx.docx', r'Data\Extra Organic Dataset\formatted_discord_conversations.json')
