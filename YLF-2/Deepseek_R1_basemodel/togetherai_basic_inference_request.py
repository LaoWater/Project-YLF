import requests
import os
from dotenv import load_dotenv


api_key=os.getenv("TOGETHER_API_KEY")
url = "https://api.together.xyz/v1/completions"

payload = {
    "model": "meta-llama/Llama-2-70b-hf",
    "prompt": "What is to go for a diba"
}
# Request headers
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {api_key}"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)