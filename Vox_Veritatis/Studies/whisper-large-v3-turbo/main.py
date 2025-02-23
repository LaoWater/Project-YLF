import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import os

# Set the environment variable to disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]
dataset_size = len(dataset)


# View the available audio files (loop only over the available number of samples)
for i in range(min(5, dataset_size)):  # Loop through up to 5 samples, but respect the dataset size
    audio_info = dataset[i]["audio"]
    print(f"Sample {i+1}:")
    print(f"  Sampling rate: {audio_info['sampling_rate']} Hz")
    print(f"  Path to file: {audio_info['path']}")
    print("-" * 50)

# result = pipe(sample)
# print(result["text"])