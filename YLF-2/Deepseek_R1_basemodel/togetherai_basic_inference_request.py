import requests

url = "https://api.together.xyz/v1/completions"

payload = {
    "model": "meta-llama/Llama-2-70b-hf",
    "prompt": "What is to go for a diba"
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer f0fcd534ca738e751fe69bda3d22f05b6eee410de1f155bbbbb27753b59ac703"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)