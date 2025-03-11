import concurrent.futures
import os
import json
import re
import logging
import openai
import time
from openai import OpenAI
import sys


######################################
## Parallel Processing of API calls ##
## Use Carefully as it can become very taxing ##
## Currently implemented to use the CPU with concurrent.futures Library ##
################################################


# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Read the translated diary data
input_file_path = 'Data/v4_step3_translated_diary.json'
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    translated_entries = json.load(input_file)


def categorize_chunk(chunk, model="gpt-4o-mini", temperature=0.8):
    client = OpenAI()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that categorizes text into predefined main subjects."
        },
        {
            "role": "user",
            "content": f"""
Please read the following text and categorize it into one of the main subjects listed below. 
Allow broader integration - not just by specific words but view it broader.
If you can link it to multiple categories - link it to the highest chance one.
If All categories chances per prompt are under ~10% - 
{{"category": "Uncertain", "chunk": "The original text chunk"}}

-----------------------
Main Subjects:
1. Food & Digestive System: The body's core and highest Life Quality Conditioner ~ Life-Sustaining Force.
2. Movement - Biomechanical Pains, Performance, Recovery
3. Into the Mind: The Miracle CPU of Nature—thoughts, pains, reading, processing, autopilot.
4. Into the Soul: Philosophies, Religions, God, Purpose, Meaning of Life and Death, Ikigai.
5. The Art of Life: Life's hidden skills and revelations emerging from deep suffering, studies, and travel.
6. Love: The binding of two souls ~ Life-Creating Force.
7. The Alchemy of the Body, Mind, and Soul: Sufferings, joys, contentment of the mind, the interconnected and 
interchanging forces.
8. Science & Math: Into God's World - The laws of this universe, experienced firsthand.
9. The Natural Truths: Air, Food, Movement ~ The inner natural truths: Emotions, Feelings, Conditionings.

These are the main subjects for you to categorize on.
When outputting - use the summary of category names, with no numbers preceding them, only text:
1. Food & Digestive System
2. Movement
3. Into the Mind
4. Into the Soul
5. The Art of Life
6. Love
7. The Alchemy of the Body
8. Science & Math & Computer World
9. The Natural Truths

------------------------

Text:
\"\"\"{chunk}\"\"\"

Return the result in JSON format:
{{"category": "Selected Category", "chunk": "The original text chunk"}}

**Important Instructions:**
- **Return only the JSON object and nothing else.**
- **Ensure the JSON is properly formatted:**
  - Strings must be enclosed in double quotes (`"`).
  - Do not use triple quotes or single quotes.
  - Escape any double quotes within the text with a backslash (`\\`).
- **Do not include any additional text or explanations.**
"""
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.8,
        max_tokens=7777,
        top_p=0.85
    )

    response_text = response.choices[0].message.content
    # Fixing Model failing to complete json object with "}"
    if not response_text.strip().endswith("}"):
        response_text += "}"

    # Escape invalid backslashes
    # response_text = response_text.replace("\\", "\\\\")

    # Handle backslash used as "And" connector which might mess up final JSON
    response_text = replace_word_slash_word(response_text)

    # Parse the JSON output
    try:
        result_chunks = json.loads(response_text)
    except json.JSONDecodeError:
        # Check if the last character of the response is not '}' - most probably Model's Reponse was not finished due to token limitation
        if not response_text.strip().endswith('}'):
            print("The token limit is too low for this message; the LLM did not finish the response.")
        else:
            print(f"An error occurred: {json.JSONDecodeError}")
        print(f"While parsing:\n{chunk}\n")
        print(f"Chat GPT Response:\n{response_text}")
        result_chunks = None
    return result_chunks


# Function to process each chunk
def process_chunk(chunk_idx, entry_text):
    chunk_text = entry_text.get('text') or entry_text.get('chunk')
    if not chunk_text:
        return None

    print(f"Categorizing chunk {chunk_idx + 1}/{len(translated_entries)}")
    processed_result = categorize_chunk(chunk_text)
    return processed_result


def process_entries_with_cost_estimate(entries):
    # Define the cost per 100 API calls (adjust as per GPT-4 API pricing)
    cost_per_100_calls = 0.01  # $0.01 per 100 calls

    # Step 1: Count total entries and estimate costs
    total_entries = len(entries)
    total_cost_estimation = (total_entries / 100) * cost_per_100_calls

    # Output the number of entries and estimated cost
    print(f"Number of entries: {total_entries}")
    print(f"Estimated cost for API calls: ${total_cost_estimation:.4f}")

    # Step 2: Await user confirmation to proceed
    user_input = 'y'
    # (input("Press 'Y' to confirm and begin API calling, or any other key to exit: ").strip().lower())

    # Step 3: Check user input and proceed or exit
    if user_input == 'y':
        print("Proceeding with API calls...")

    else:
        print("Exiting script...")
        sys.exit(0)  # Exit the script


def replace_word_slash_word(text):
    """
    Replace patterns like 'purpose/environment' with 'purpose and environment'.
    """

    def replacement(match):
        parts = match.group(0).split("/")
        return f"{parts[0]} and {parts[1]}"

    pattern = r'\b\w+/\w+\b'
    return re.sub(pattern, replacement, text)


###################
## Script Starts ##
###################

start_time = time.time()
process_entries_with_cost_estimate(translated_entries)

categorized_entries = []
# Limit number of processed entries with index
index_limit = 11111
count_index = 0

# Set up parallel processing using ThreadPoolExecutor
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = []

    # Track submitted tasks
    logging.debug(f"Starting ThreadPoolExecutor with max_workers=10.")
    for idx, entry in enumerate(translated_entries):
        if count_index >= index_limit:
            logging.debug(f"Reached index limit: {index_limit}. Exiting loop.")
            break

        count_index += 1
        logging.debug(f"Submitting task {idx} to process_chunk.")
        # Submit the processing of each chunk to the thread pool
        futures.append(executor.submit(process_chunk, idx, entry))

    logging.debug(f"All tasks have been submitted. Total: {len(futures)} tasks.")

    # Collect results as they complete
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        try:
            result = future.result()
            if result:
                categorized_entries.append(result)
                logging.debug(f"Task {i+1}/{len(futures)} completed successfully.")
            else:
                logging.warning(f"Task {i+1}/{len(futures)} returned None.")
        except Exception as e:
            logging.error(f"Error occurred while processing task {i+1}/{len(futures)}: {e}")
        time.sleep(0.2)  # Adjust the sleep time if needed between API calls

logging.info("Finished processing all tasks. Proceeding to save data.")

post_llm_processing_compute = time.time() - start_time
print(f"LLM processing Compute time (Before entering JSON write: {post_llm_processing_compute}")
print("Finished LLM processing, moving on to saving the Data in JSON")

# Save the categorized data
output_file_path = 'Data/v4_step4_categorized_diary_ParallelProcessing.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(categorized_entries, output_file, indent=2)

print(f"Categorized data has been saved to {output_file_path}")

final_compute_time = time.time() - start_time
print(f"Total Compute Time: {final_compute_time}")
