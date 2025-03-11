import concurrent.futures
import datetime
import os
import re
import json
import time
import openai
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def is_prompt_similar(new_prompt, existing_prompts, threshold=0.8):
    if not existing_prompts:
        return False  # No prompts to compare with
    vectorizer = TfidfVectorizer().fit_transform([new_prompt] + existing_prompts)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    # Compare the new prompt with each existing prompt
    similarities = cosine_matrix[0, 1:]
    return any(sim >= threshold for sim in similarities)


def clean_text(text):
    return ' '.join(text.strip().split())


import re


def sanitize_response_text(raw_response_text):
    """
    Sanitize the response text by handling triple quotes, matching all quotes,
    and ensuring the JSON structure is intact.

    Args:
        raw_response_text (str): The raw response text to be sanitized.

    Returns:
        str: The sanitized response text.
    """
    # Replace triple quotes or double double-quotes with a single double-quote
    response_text = re.sub(r'"""|""', '"', raw_response_text)

    # Remove the last character (assumed to be '}') for processing
    if response_text.endswith('}'):
        response_text = response_text[:-1]

    # Ensure all escaped quotes are matched
    if response_text.count('\"') % 2 != 0:
        response_text += '\"'  # Add a closing quote if unmatched

    # Ensure all quotes are matched
    if response_text.count('"') % 2 != 0:
        response_text += '"'  # Add a closing quote if unmatched

    # Add back the closing '}'
    response_text += '}'

    return response_text


def generate_prompt_content(chunk_category, chunk, model="gpt-4o-mini", temperature=0.8):
    client = OpenAI()
    messages = [
        {
            "role": "system",
            "content": "You are a creative assistant that generates prompts based on provided text & text category."
                       "Take the time to properly set the category before reading and before generating prompt."
                       "The prompts should be clear and precise, they are going to be used in a PromptGenerated Dataset"
                       "Used for LLM training."
        },
        {
            "role": "user",
            "content": f"""
Using the following category and text, create an imagination-inviting prompt of 3 to 20 words that reflects the 
sentiment of the category and is relevant to the text. The prompt should not be too specific and should 
encourage creative thinking.
Don't use big, sophisticated, uncommon words - keep in mind this is to be later used in a LLM Training Dataset.

**Important Instructions:**
- Return only the JSON object and nothing else.
- Do **not** include any code block markers (e.g., triple backticks) or language identifiers.
- Ensure the JSON is properly formatted, using double quotes for strings.
- Strings must be enclosed in double quotes (`"`).
- Do NOT use triple quotes or single quotes. DO NOT USE TRIPLE QUOTES.
- Escape any special characters as needed.
- Do not include any explanations or additional text.

Category: {chunk_category}

Text:
\"\"\"{chunk}\"\"\"

Return the result in JSON format:
{{"prompt": "Generated prompt", "content": "Original text chunk"}}
"""
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2222
    )

    raw_response_text = response.choices[0].message.content

    # Fix Model fails regarding respecting the JSON object.
    response_text = sanitize_response_text(raw_response_text)

    # Parse the JSON output
    try:
        generated_result = json.loads(response_text)
    except json.JSONDecodeError:
        # If parsing fails, print the error and assistant's response for debugging
        print(f"JSON decoding error: {json.JSONDecodeError}")
        print(f"While processing chunk with category '{chunk_category}'")
        print(f"Assistant's response:\n{response_text}")
        generated_result = None

    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred: {e}")
        generated_result = None

    return generated_result


# Function to process each entry and generate prompt-content pairs
def process_entry(idx, entry):
    category = entry.get('category')
    chunk_text = entry.get('chunk')

    if not chunk_text or not category:
        return None  # Skip if missing data

    if category != "Uncertain":
        print(f"Processing entry {idx + 1}/{len(categorized_entries)}")

        result = generate_prompt_content(category, chunk_text)
        if result:
            return {
                "result": result,
                "prompt": result['prompt']
            }
        else:
            print(f"Failed to generate prompt-content pair for entry {idx + 1}")
            return None
    else:
        print(f"Skipping entry {idx + 1}/{len(categorized_entries)} (Uncertain Category Detected)")
        return None


################
# Main Starts ##
################

start_time = time.time()

# Read the categorized diary data
input_file_path = 'Data/v4_step4_categorized_diary_ParallelProcessing.json'
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    categorized_entries = json.load(input_file)

prompt_content_pairs = []
all_generated_prompts = []

# Use parallel processing to handle the entries
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = []
    for idx, entry in enumerate(categorized_entries):
        futures.append(executor.submit(process_entry, idx, entry))
        time.sleep(0.1)  # Adjust as needed to respect rate limits

    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            prompt_content_pairs.append(result["result"])
            all_generated_prompts.append(result["prompt"])

# Save the output to a file
output_file_path = 'Data/v4_step5_creating_prompt_content_pairs.jsonl'
with open(output_file_path, 'w', encoding='utf-8') as file:
    for pair in prompt_content_pairs:
        prompt = clean_text(pair['prompt'])
        content = clean_text(pair['content'])
        data = {
            "prompt": prompt,
            "content": content
        }
        file.write(json.dumps(data) + '\n')

# Print time taken and results
print(f"Prompt-content pairs have been saved to {output_file_path}")
elapsed_time = time.time() - start_time
print(f"Total time taken: {elapsed_time}")

print("All generated prompts for review before creating Final DataSet:")
for pair in prompt_content_pairs:
    prompt = clean_text(pair['prompt'])
    # print(f"Prompt: {prompt}")

print("All generated prompts list:")
# print(all_generated_prompts)
