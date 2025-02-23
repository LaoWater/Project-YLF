import openai
from openai import OpenAI

# Initialize the client
client = OpenAI()

fine_tuning_job_id = 'ftjob-mYPRY7sPQoDsB5tr0XiizPGK'
client.fine_tune.list_events(fine_tuning_job_id)


