###########################################################################
###########################################################################
# Lao Alpha Was trained on a very minimal dataset for training purposes #


from openai import OpenAI
from util import split_into_paragraphs

client = OpenAI()

prompt = """What is the fine line between the mine interfering with the body
                                and guiding it towards healthful, mindful, blessed eating?"""

completion = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:personal::A8phTRw2",
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt}
    ]
)

print(f"Q: {prompt} \n")
print("Trained Model:")
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
