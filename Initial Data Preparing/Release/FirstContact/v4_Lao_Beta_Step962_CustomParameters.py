###########################################################################
###########################################################################
# Lao Beta #2 Was trained on medium-size Dataset of medium Quality - #
# This is a checkpoint #1 at 2/3 of training,
# Where we have touched a sweetspot between Pre-training data and training data
# Lao's touch is being felt clearly, yet not to a point where the binding of the pre-train with training becomes stainde and overwhelming for the LLM.
# Most stable Model yet and producer of Marvelling asnwers worthy of labeling them as "Complete Transcendence"
# That is when the Model produces something similar or better than the writer itself, although using different words, expressions and styles of "painting", but springing from a similar "Neural Network".


import re
from openai import OpenAI
import random

client = OpenAI()

prompt = """
“Blessed are you, Lord our God, King of the Universe, for giving me the strength to Fight and Wisdom to Learn.”
"""







def comparing_to_base_model(prompt_input):
    completion_p = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt_input}
        ]
    )

    print("\n\n\nBase Model:")
    print(completion_p.choices[0].message.content)




prompt_2 = ("how to reprogram 7 years of night eating, causing now a beast(body) secreting lots of Acid as evening comes?"
            "I want to heal.. balance the body and digestive system")

completion = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:personal::AAh0b7Uy:ckpt-step-934",
    messages=[
        {"role": "system",
         "content": "You are Lao, a student of Life who has searched for understanding of the Body, Mind "
                    "And Soul - all his Life. "
                    "Has traveled the world and oceans, deeply immersed in cultures in both study"
                    "and practice, love, habits of the body, Mind and Soul."
                    "He returns home to spread his teachings with his fellow brothers, in "
                    "Truth Discerning awareness, wise but yet not speaking as if he is better than others."},
        {"role": "user", "content": prompt_2}
    ],
    temperature=0.7,  # Controls creativity; 0 is deterministic
    top_p=0.9,  # Controls diversity; higher means more varied completions
    max_tokens=500,  # Limits the length of the response
    frequency_penalty=0.0,  # Penalizes frequent words
    presence_penalty=0.6  # Encourages topic diversity
)

completion_content = completion.choices[0].message.content

# Generate paragraphs
paragraphs = split_into_paragraphs(completion_content)

# Print each paragraph
print("Lao Beta 2:")
for paragraph in paragraphs:
    print(paragraph)
    print()  # Adds a blank line between paragraphs

####################################
## Comparing to Normal Base Model ##
####################################

comparing_to_base_model(prompt_2)
