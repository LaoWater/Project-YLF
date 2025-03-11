from datetime import datetime, timezone
from openai import OpenAI
import pprint

# Initialize the client
client = OpenAI()

# List 10 fine-tuning jobs
jobs_list = client.fine_tuning.jobs.list(limit=10)

# Pretty print the list of fine-tuning jobs
print("List 10 fine-tuning jobs:")
pprint.pprint(jobs_list.data, indent=4)

# Print the JOB IDs and store them in a list
finetuning_jobs = []
for job in jobs_list.data:
    print(f"JOB ID: {job.id}")
    finetuning_jobs.append(job.id)

current_job = finetuning_jobs[0]

# Retrieve the state of a specific fine-tune
job_state = client.fine_tuning.jobs.retrieve(current_job)

# Pretty print the job state
print("\nJob State:")
pprint.pprint(job_state, indent=10)

# List up to 10 events from a fine-tuning job
jobs_events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=current_job, limit=1000)

# Pretty print the job events
print(f"\nJob {current_job} events:")
pprint.pprint(jobs_events.data, indent=4)
print("\n\n\n Available Models: \n")


# List all models, including fine-tuned models
models_list = client.models.list()

# Print the models with formatted dates
for model in models_list.data:
    # Filtering out OpenAI System models, leaving only personal
    if "personal" in model.id:
        # Convert the Unix timestamp to a readable date format (dd/mm/yyyy)
        creation_date = datetime.fromtimestamp(model.created, tz=timezone.utc).strftime('%d/%m/%Y')

        print(f"Model ID: {model.id}, Owned by: {model.owned_by}, Created on: {creation_date}")

# Cancel a job
# client.fine_tuning.jobs.cancel("ftjob-abc123")

# Delete a fine-tuned model (must be an owner of the org the model was created in)
# client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")




