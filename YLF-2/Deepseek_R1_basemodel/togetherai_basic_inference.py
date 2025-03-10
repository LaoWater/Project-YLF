import os
from dotenv import load_dotenv
from together import Together  # New client-based approach

# Load environment variables
load_dotenv()

# Initialize client with API key from environment
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

def test_base_model(prompt: str):
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    test_prompt = "Write a short poem about AI in healthcare:"
    print("Base Model Response:\n", test_base_model(test_prompt))