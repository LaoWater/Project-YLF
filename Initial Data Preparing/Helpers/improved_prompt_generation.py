from openai import OpenAI

client = OpenAI()

META_PROMPT = """
Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - JSON should never be wrapped in code blocks (```) unless explicitly requested.

The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
[If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]
""".strip()


def generate_prompt(task_or_prompt: str):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": META_PROMPT,
            },
            {
                "role": "user",
                "content": "Task, Goal, or Current Prompt:\n" + task_or_prompt,
            },
        ],
    )

    return completion.choices[0].message.content



user_prompt = """
Please read the following text and categorize it into one of the main subjects listed below. 
Allow broader integration - not just by specific words but view it broader.
If you can link it to multiple categories - link it to the highest chance one.
If All categories chances per prompt are under 10% - 
{{"category": "Uncertain", "chunk": "The original text chunk"}}

-----------------------
Main Subjects:
1. Food & Digestive System: The body's core and highest Body, Mind, and Life Quality Conditioner ~ Life-Sustaining Force.
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

llm_response = generate_prompt(user_prompt)

print (llm_response)





# Example of using all 4o parameters
#
# response = client.chat.completions.create(
#     model="gpt-4",
#     messages=messages,
#     temperature=0.7,
#     max_tokens=500,
#     top_p=0.9,
#     frequency_penalty=0.5,
#     presence_penalty=0.3,
#     stop=["END"],
#     logit_bias={"50256": -100},
#     stream=False,
#     user="user123"
# )
