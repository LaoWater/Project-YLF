import os
import json
import openai
import time
from openai import OpenAI
import sys

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Read the translated diary data
input_file_path = 'Data/v4_step3_translated_diary.json'
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    translated_entries = json.load(input_file)


meta_prompt = """Read the provided text and categorize it into one of the main subjects listed below. Consider broader implications and not just specific words. If the text links to multiple categories, select the one with the highest likelihood. If the chances are below 10% for all categories, classify it as "Uncertain."

# Main Subjects:
- Food & Digestive System: The body's core and highest Body, Mind, and Life Quality Conditioner ~ Life-Sustaining Force.
- Movement - Biomechanical Pains, Performance, Recovery.
- Into the Mind: The Miracle CPU of Nature—thoughts, pains, reading, processing, autopilot.
- Into the Soul: Philosophies, Religions, God, Purpose, Meaning of Life and Death, Ikigai.
- The Art of Life: Life's hidden skills and revelations emerging from deep suffering, studies, and travel.
- Love: The binding of two souls ~ Life-Creating Force.
- The Alchemy of the Body, Mind, and Soul: Sufferings, joys, contentment of the mind, the interconnected and interchanging forces.
- Science & Math: Into God's World - The laws of this universe, experienced firsthand.
- The Natural Truths: Air, Food, Movement ~ The inner natural truths: Emotions, Feelings, Conditionings.

# Steps
1. Analyze the text to understand its main theme and broader context.
2. Match the theme to the most relevant category among the main subjects.
3. Determine if the confidence for any category is less than 10% and if so, classify it as "Uncertain."

# Output Format
Produce the output in JSON format, strictly as: 
```json
{"category": "Selected Category", "chunk": "The original text chunk"}
```
- Ensure the JSON is properly formatted with double quotes.
- The "category" should be the exact category name from the provided list.
- If uncertain, use "Uncertain" as the category.

# Examples

**Example 1:**
- **Text:** "The importance of balanced diet and digestion in maintaining health is unmatched."
- **Result:** 
```json
{"category": "Food & Digestive System", "chunk": "The importance of balanced diet and digestion in maintaining health is unmatched."}
```

**Example 2:**
- **Text:** "Exploring different cultures through travel opens up new perspectives."
- **Result:**
```json
{"category": "The Art of Life", "chunk": "Exploring different cultures through travel opens up new perspectives."}
```

# Notes
- Pay careful attention to nuanced expressions that may indicate broader subject linkages.
- Validate the confidence of the categorization before finalizing it in the output JSON."""




def categorize_chunk(chunk, model="gpt-4o-mini", temperature=0.8):
    client = OpenAI()
    messages = [
        {
            "role": "system",
            "content": meta_prompt
        },
        {
            "role": "user",
            "content": f"""
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
- do not include "```json ``` brackets" - output strictly the json
"""
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1111
    )

    response_text = response.choices[0].message.content
    print(response_text)

    # Parse the JSON output
    try:
        result_chunks = json.loads(response_text)
    except json.JSONDecodeError:
        # Check if the last character of the response is not '}'
        if not response_text.strip().endswith('}'):
            print("The token limit is too low for this message; the LLM did not finish the response.")
        else:
            print(f"An error occurred: {json.JSONDecodeError}")
        print(f"While parsing:\n{chunk}\n")
        print(f"Chat GPT Response:\n{response_text}")
        result_chunks = None
    return result_chunks


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
    user_input = input("Press 'Y' to confirm and begin API calling, or any other key to exit: ").strip().lower()

    # Step 3: Check user input and proceed or exit
    if user_input == 'y':
        print("Proceeding with API calls...")

    else:
        print("Exiting script...")
        sys.exit(0)  # Exit the script


###################
## Script Starts ##
###################

start_time = time.time()

process_entries_with_cost_estimate(translated_entries)

categorized_entries = []
count_index = 0
for idx, entry in enumerate(translated_entries):
    chunk_text = entry.get('text') or entry.get('chunk')
    if not chunk_text:
        continue
    count_index += 1
    if count_index >= 20:
        break

    print(f"Categorizing chunk {idx + 1}/{len(translated_entries)}")
    result = categorize_chunk(chunk_text)
    categorized_entries.append(result)
    time.sleep(1)  # Adjust the sleep time as needed

# Save the categorized data
output_file_path = 'Data/v4_step4_categorized_diary.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(categorized_entries, output_file, indent=2)

print(f"Categorized data has been saved to {output_file_path}")

final_compute_time = time.time() - start_time
print(f"Total Compute Time: {final_compute_time}")
