###########################################################################
###########################################################################
# YLF Alpha was trained on medium to bigger - size Dataset of improved Quality -
# Compared to first training rounds, the tokens have increased more than 10x, from 100k to around 2 millions.
# This is a checkpoint #1 at 1/3 of training - thus we'll refer to this as Teenager.


from util import split_into_paragraphs
from openai import OpenAI
client = OpenAI()


prompt = "Hello my friend"
completion = client.chat.completions.create(
  model="ft:gpt-4o-mini-2024-07-18:personal::AioG37fT:ckpt-step-329",
  messages=[
    {"role": "system", "content": ""},
    {"role": "user", "content": prompt}
  ]
)


print(f"Q: {prompt} \n")
print("YLF Model(teenager):")
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

