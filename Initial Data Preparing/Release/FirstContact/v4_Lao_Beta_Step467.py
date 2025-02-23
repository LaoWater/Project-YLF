###########################################################################
###########################################################################
# Lao Beta #1 Was trained on medium-size Dataset of medium Quality - #
# This is a checkpoint #1 at 1/3 of training, where the pre-training basline LLM architecture is clearly being felt more than "Lao's Touch"


from util import split_into_paragraphs
from openai import OpenAI
client = OpenAI()


prompt = "What a beautiful day to"
completion = client.chat.completions.create(
  model="ft:gpt-4o-mini-2024-07-18:personal::AAh0aubN:ckpt-step-467",
  messages=[
    {"role": "system", "content": ""},
    {"role": "user", "content": prompt}
  ]
)


print(f"Q: {prompt} \n")
print("Trained Model:")
print(completion.choices[0].message.content)


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

