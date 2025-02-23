import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B"

pipeline = transformers.pipeline(
  "text-generation",
  model="Meta-Llama-3.1-8B",
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
)

result = pipeline("Today is a beautiful day and", max_length=50)
print(result[0]['generated_text'])