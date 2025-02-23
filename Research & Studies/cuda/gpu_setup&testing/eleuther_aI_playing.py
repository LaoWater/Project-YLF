from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"  # Using a smaller model for faster generation
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the prompt for text generation
prompt = "hello how are you my friend"

# Encode the prompt into a format suitable for the model
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text using the model with adjusted parameters
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
