from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="ft:gpt-4o-mini-2024-07-18:personal::A8phTRw2",
  messages=[
    {"role": "system", "content": ""},
    {"role": "user", "content": "What is the fine line between the mind interfering with the body"
                                "and guiding it towards healthful, mindful, blessed eating?"}
  ]
)

print("Trained Model:")
print(completion.choices[0].message.content)


####################################
## Comparing to Normal Base Model ##
####################################

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": ""},
    {"role": "user", "content": "What is the fine line between the mind interfering with the body "
                                "and guiding it towards healthful, mindful, blessed eating?"}
  ]
)

print("\n\n\nBase Model:")
print(completion.choices[0].message.content)

