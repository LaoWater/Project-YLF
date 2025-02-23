#############################################################################
#### Project Transcendence: A look into the intimacy of an Awakened Mind ####
#### Author: Lao Water, 7-June-2024 ####
#############################################################################
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import json
import os
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)

# Debugging: Print the current working directory
logging.info(f"Current working directory: {os.getcwd()}")

# Check if the file exists
file_path = 'training_data_top30_forGPTtest.jsonl'
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")


# Load and prepare the JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Load the training data
training_data = load_jsonl(file_path)

# Save the data in a format that the datasets library can load
prepared_jsonl_path = 'training_data_prepared.jsonl'
with open(prepared_jsonl_path, 'w', encoding='utf-8') as f:
    for entry in training_data:
        f.write(json.dumps({"text": entry["text"]}) + '\n')

# Debugging: Check if the prepared file was created
if not os.path.isfile(prepared_jsonl_path):
    raise FileNotFoundError(f"Prepared file not found: {prepared_jsonl_path}")

# Load dataset
dataset = load_dataset('json', data_files={'train': prepared_jsonl_path})

# Load pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add padding token if it's not already present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))


# Tokenize the dataset and add labels
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
)


# Custom Trainer class to compute the loss manually
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").clone().detach()
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Shift the logits and labels to align them
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute the loss using CrossEntropyLoss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# Initialize the Trainer with the custom compute_loss function
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
output_dir = './fine-tuned-model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

logging.info("Model fine-tuning and saving completed successfully.")

