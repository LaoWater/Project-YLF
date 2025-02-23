import openai
from openai import OpenAI

client = OpenAI()
input_file_path = 'Data/v4_step6_formatted_for_OpenAI_finetuning.jsonl'
training_file = 'file-RFyuxtEM9y8kWYxhFnXXsi'

try:
    response = client.fine_tuning.jobs.create(
        training_file=training_file,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={
            "n_epochs": 3,
            "batch_size": 7,
            "learning_rate_multiplier": 1.8
        }
    )

    print(response)

except openai.APIConnectionError as e:
    print("The server could not be reached")
    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
except openai.RateLimitError as e:
    print("A 429 status code was received; we should back off a bit.")
except openai.APIStatusError as e:
    print("Another non-200-range status code was received")
    print(e.status_code)
    print(e.response)



