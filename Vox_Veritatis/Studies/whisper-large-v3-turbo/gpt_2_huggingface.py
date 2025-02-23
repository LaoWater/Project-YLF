from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Step 1: Load the model and tokenizer for text generation
model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Step 2: Setup the pipeline
text_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Step 3: Pass input data to the pipeline
text_input = "The future of AI is"
result = text_pipe(text_input, max_length=1000, truncation=True)

# Step 4: Print the output
print(result[0]["generated_text"])
