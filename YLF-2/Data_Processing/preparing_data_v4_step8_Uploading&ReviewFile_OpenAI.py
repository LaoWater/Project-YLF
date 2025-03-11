from datetime import datetime, timezone
import json
import os

import openai
import requests
from openai import OpenAI
client = OpenAI()
input_file_path = 'Data/v4_step6_formatted_for_OpenAI_finetuning.jsonl'
upload_flag = False


# Uploading File

if upload_flag:
    openai_upload_response = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="fine-tune"
    )
    print(openai_upload_response)

# Viewing Uploaded Files
# CMD/PS: curl https://api.openai.com/v1/files -H "Authorization: Bearer YOUR_OPENAI_API_KEY"
# Or Python:

files = client.files.list()
print(files)
# Extract file IDs
file_ids = [file.id for file in files.data]


# Function to convert timestamp to a human-readable date
def format_timestamp(timestamp):
    # Use timezone-aware datetime for UTC
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


# Iterating through file IDs and retrieving file details
for file_id in file_ids:
    print(f"Processing file ID: {file_id}")

    # Retrieving File
    retrieved_file = client.files.retrieve(file_id)

    # Extracting file attributes
    file_details = {
        "id": retrieved_file.id,
        "bytes": retrieved_file.bytes,
        "created_at": format_timestamp(retrieved_file.created_at),
        "filename": retrieved_file.filename,
        "object": retrieved_file.object,
        "purpose": retrieved_file.purpose,
        "status": retrieved_file.status,
        "status_details": retrieved_file.status_details
    }

    # Formatting file details for human readability
    print("File Details:")
    print(json.dumps(file_details, indent=4))


# Retrieving File
retrieved_file = client.files.retrieve(file_ids[0])
# print(retrieved_file)

# Retrieve Files Content

content = client.files.content(file_ids[0])
# print(f"Content of Retrieved File: \n {content}")


################################################
## Retrieve File Content Through HTTP Request ##
################################################

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Get your file ID (e.g., file_ids[0])
file_id = file_ids[0]

# Define the endpoint URL
url = f'https://api.openai.com/v1/files/{file_id}/content'

# Set the headers for authentication
headers = {
    'Authorization': f'Bearer {openai.api_key}',
}

# Make the GET request to download the file content
response = requests.get(url, headers=headers)

if response.status_code == 200:
    # The content is in response.content
    file_content = response.content.decode('utf-8')
    # print(f"Content of Retrieved File:\n{file_content}")
else:
    print(f"Failed to retrieve file content. Status code: {response.status_code}, Response: {response.text}")


# Deleting Files

# client.files.delete(file_id)

exit()

file_ids = ['file-CqdWfXbHU3JudzwYLh9P4G', 'file-1wbDNyXwjCQoRFKdd2bVCu', 'file-5NHzSBjt5sKoiW7p4XpY6J']

for file_id in file_ids:
    print(f" Deleting {file_id}...")
    client.files.delete(file_id)
