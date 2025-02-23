###########################################################################
###########################################################################
# YLF Alpha was trained on medium to bigger - size Dataset of improved Quality -
# Compared to first training rounds, the tokens have increased more than 10x, from 100k to around 2 millions.
# This is a checkpoint #1 at 1/3 of training - thus we'll refer to this as Teenager.


from util import split_into_paragraphs
from openai import OpenAI

client = OpenAI()

prompt = "Who is PonPon?"
completion = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:personal::AioG35Tv",
    messages=[
        {"role": "system", "content": "You are Lao, a Healer and philosopher - but most of all,"
                                      "A humble student of life, sharing his experiences and lessons."
                                      "Structure and format your response beautifully when outputting."
                                      "Give complete full-hearted answer when it's time and hold back little bit when it's time - "
                                      "as in when user asks you too much personal questions which might imply PPIs or too intimacy responses"},
        {"role": "user", "content": prompt},

    ],

    temperature=0.8,  # Controls creativity; 0 is deterministic, 2 is maximum creativity
    top_p=0.9,  # Controls diversity; higher means more varied completions
    max_tokens=2222,  # Limits the length of the response
    frequency_penalty=0.1,  # Penalizes frequent words
    presence_penalty=0.0  # Encourages topic diversity
)

print(f"Q: {prompt} \n")
print("YLF Model:")
model_response = completion.choices[0].message.content

# Generate paragraphs
paragraphs = split_into_paragraphs(model_response)

# Print each paragraph
for paragraph in paragraphs:
    print(paragraph)

####################################
## Comparing to Normal Base Model ##
####################################

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt}
    ]
)

print("\n\n\nBase Model:")
print(completion.choices[0].message.content)
