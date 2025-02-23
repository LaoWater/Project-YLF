import docx


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

    # Save the cleaned text to a .txt file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(full_text)

    return full_text  # Optionally return the cleaned text for further use


#####################################
#### Cleaning & Processing Diary ####
#####################################

print("Cleaning Docx and transcribing to txt...")
clean_diary(r'Data\Lao_All Writings 1-Dec.docx', r'Data\v4_step1_cleaned_diary.txt')


# docx_to_json(r'Data\Extra Organic Dataset\Discord Conversation until 26 Dec 2024.docx.docx', r'Data\Extra Organic Dataset\formatted_discord_conversations.json')
