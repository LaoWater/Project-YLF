from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

start_time = time.time()

# Check CUDA availability and cuDNN version
print("CUDA available:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("cuDNN version:", torch.backends.cudnn.version())

# Load the pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # or any other model you want to use
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the prompt for text generation
prompt = "Who are You?"

# Check if CUDA (GPU) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move the model to the appropriate device
model.to(device)

# Encode the prompt into a format suitable for the model
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text using the model with adjusted parameters
output = model.generate(
    **inputs,
    max_length=1000,  # Maximum length of the generated text
    num_return_sequences=1,  # Number of generated sequences
    no_repeat_ngram_size=2,  # Prevents repeating the same n-grams
    # early_stopping=True,  # Stops the generation early if an end token is generated
    #    temperature=0.7,        # Adjusting temperature for more focused text
    top_k=50,  # Using top-k sampling
    # top_p=0.9,  # Using top-p (nucleus) sampling
    pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id
)

# Decode the generated text to a human-readable format
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)

# End timing the generation process
end_time = time.time()

# Print the time taken for generation
print(f"Time taken for text generation: {end_time - start_time:.2f} seconds")

# Print GPU utilization
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")
