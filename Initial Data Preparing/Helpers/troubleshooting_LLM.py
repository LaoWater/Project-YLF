import re
import sys
import os

from openai import OpenAI
import json

# Modify the Python path to include the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def replace_word_slash_word(text):
    """
    Replace patterns like 'purpose/environment' with 'purpose and environment'.
    """

    def replacement(match):
        parts = match.group(0).split("/")
        return f"{parts[0]} and {parts[1]}"

    pattern = r'\b\w+/\w+\b'
    return re.sub(pattern, replacement, text)


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
        temperature=0.9,
        max_tokens=7777,
        top_p=0.8
    )

    response_text = response.choices[0].message.content
    # Fixing Model failing to complete json object with "}"

    if not response_text.strip().endswith("\"") and not response_text.strip().endswith("}"):
        response_text += "\""

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


prompt = """
The Alchemist talking to the wind and desert.
"Well, what good would it be to you if you had to die?" the alchemist answered. "Your money saved us for three days. It's not often that money saves a person's life."
But the boy was too frightened to listen to words of wisdom. He had no idea how he was going to transform himself into the wind. He wasn’t an alchemist!
The alchemist asked one of the soldiers for some tea, and poured some on the boy's wrists. A wave of relief washed over him, and the alchemist muttered some words that the boy didn’t understand.
"Don't give in to your fears," said the alchemist, in a strangely gentle voice. "If you do, you won’t be able to talk to your heart."
"But I have no idea how to turn myself into the wind."
"If a person is living out his Personal Legend, he knows everything he needs to know. There is only one thing that makes a dream impossible to achieve: the fear of failure."
"I'm not afraid of failing. It's just that I don’t know how to turn myself into the wind."
"Well, you’ll have to learn; your life depends on it."
"But what if I can’t?"
"Then you’ll die in the midst of trying to realize your Personal Legend. That’s a lot better than dying like millions of other people, who never even knew what their Personal Legends were."
"""

processed_text = categorize_chunk(prompt)
print(processed_text)

#
# # Parse JSON
# json_data = json.loads(processed_text)
# print(json_data)
