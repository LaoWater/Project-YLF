import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Path to LLaMA 3.1 model files
model_path = "C:\\Users\\baciu\\Desktop\\Neo Training\\Transcendence\\LLama3-1\\llama-models\\models\\llama3_1\\Meta-Llama-3.1-8B"

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Load the model
model = LlamaForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,  # To save memory
    torch_dtype=torch.float16,  # Use float16 for memory efficiency, remove this line for CPU
    device_map="auto"  # Automatically assign to available devices
)

# Make sure the model is in evaluation mode
model.eval()

# Test input prompt
prompt = "Once upon a time, in a land far, far away,"

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text using the model
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,  # Adjust the max_length as needed
        num_return_sequences=1,
        do_sample=True,  # Sampling for more diverse outputs
    )

# Decode and print the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
