import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Load the fine-tuned model and tokenizer
model_name = "./fine-tuned-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the prompt for text generation
prompt = "the meaning of life is"

# Check if CUDA (GPU) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move the model to the appropriate device
model.to(device)

# Encode the prompt into a format suitable for the model
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Start timing the generation process
start_time = time.time()

# Generate text using the model with adjusted parameters
output = model.generate(
    **inputs,
    max_length=100,        # Maximum length of the generated text
    num_return_sequences=3, # Number of generated sequences
    no_repeat_ngram_size=2, # Prevents repeating the same n-grams
    early_stopping=True,    # Stops the generation early if an end token is generated
    temperature=0.7,        # Adjusting temperature for more focused text
    top_k=50,               # Using top-k sampling
    top_p=0.9,              # Using top-p (nucleus) sampling
    pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
    do_sample=True          # Enable sampling for diverse output
)

# End timing the generation process
end_time = time.time()

# Decode the generated text to a human-readable format
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)

# Print the time taken for generation
print(f"Time taken for text generation: {end_time - start_time:.2f} seconds")

# Print GPU utilization
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
