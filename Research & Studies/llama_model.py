# Hugging Face API Key

import transformers
import torch

# Define the model ID
model_id = "meta-llama/Meta-Llama-3.1-8B-Instant"

# Initialize the pipeline for text generation with CUDA support
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Print out the device the model is using
print(f"Model is using device: {pipeline.model.device}")

# Combine messages into a single prompt
prompt = """System: You are a ideea specialist. You give insights of new ideeas!\n
User: Give me 5 apps ideeas?\nAssistent: """

# Generate text based on the combined prompt
outputs = pipeline(
    prompt,
    max_new_tokens=256,
)

# Print the generated response
print(outputs[0]["generated_text"])