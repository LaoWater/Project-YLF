import torch
from torch.profiler import profile, record_function, ProfilerActivity

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the prompt for text generation
prompt = "Once upon a time"

# Check if CUDA (GPU) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move the model to the appropriate device
model.to(device)

# Encode the prompt into a format suitable for the model
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Verify tensors are on the correct device
print(f"Inputs device: {inputs['input_ids'].device}")

# Use PyTorch Profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        output = model.generate(
            **inputs,
            max_length=100,        # Maximum length of the generated text
            num_return_sequences=1, # Number of generated sequences
            no_repeat_ngram_size=2, # Prevents repeating the same n-grams
            early_stopping=True,    # Stops the generation early if an end token is generated
            temperature=0.7,        # Adjusting temperature for more focused text
            top_k=50,               # Using top-k sampling
            top_p=0.9,              # Using top-p (nucleus) sampling
            pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id
        )

# Decode the generated text to a human-readable format
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)

# Print profiler results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
