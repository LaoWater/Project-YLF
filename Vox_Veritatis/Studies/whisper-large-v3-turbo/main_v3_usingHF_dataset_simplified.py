import os
# Disable unnecessary warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from warnings import filterwarnings


filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)

# 1. Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# 2. Load Whisper model and processor
def load_model(model_id, language="en"):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                                                      use_safetensors=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    return model, processor


# 3. Setup ASR pipeline
def setup_pipeline(model, processor):
    return pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor, torch_dtype=torch_dtype,
                    device=device, return_timestamps=True)


# 4. Load and list dataset
def load_dataset_samples(dataset_name, split="validation"):
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} audio samples from the dataset")
    return dataset


# 5. Transcribe audio sample
def transcribe_audio_sample(pipe, dataset, index=0):
    audio_sample = dataset[index]["audio"]["array"]
    result = pipe(audio_sample)
    print(f"Transcription: {result['text']}")


# Main function
def main():
    model_id = "openai/whisper-large-v3-turbo"
    dataset_name = "distil-whisper/librispeech_long"

    # Load model and processor
    model, processor = load_model(model_id)

    # Set up ASR pipeline
    pipe = setup_pipeline(model, processor)

    # Load dataset
    dataset = load_dataset_samples(dataset_name)

    # Transcribe the first audio sample
    transcribe_audio_sample(pipe, dataset)


if __name__ == "__main__":
    main()
