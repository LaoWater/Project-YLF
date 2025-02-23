import os
# Set the environment variable to disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from IPython.display import Audio
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The input name `inputs` is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Torch was not compiled with flash attention")
warnings.filterwarnings("ignore", category=FutureWarning, message="Passing a tuple of `past_key_values` is deprecated")
warnings.filterwarnings("ignore")



# 1. Device and dtype configuration
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, dtype


# 2. Load the Whisper model and set language
def load_whisper_model(model_id, device, torch_dtype, language="en"):
    print(f"Loading model {model_id}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Obtain forced_decoder_ids for the specified language and task
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    # Set forced_decoder_ids in the model's config
    model.config.forced_decoder_ids = forced_decoder_ids

    # Return both the model and processor
    return model, processor


# 3. Setup ASR pipeline
def setup_pipeline(model, processor, device, torch_dtype):
    print("Setting up the ASR pipeline...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
    )
    return pipe


# 4. Load and list the dataset
def load_and_list_dataset(dataset_name, split="validation"):
    print(f"Loading dataset {dataset_name} ({split} split)...")
    dataset = load_dataset(dataset_name, split=split)

    # List the available audio samples
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size} samples\n")

    for i in range(min(5, dataset_size)):  # Show up to 5 audio samples
        audio_info = dataset[i]["audio"]
        print(f"Sample {i + 1}:")
        print(f"  Sampling rate: {audio_info['sampling_rate']} Hz")
        print(f"  Path to file: {audio_info['path']}")
        print("-" * 50)

    return dataset


# 5. Play an audio sample, requires Jupyter
def play_audio_sample(dataset, index=0):
    # Extract the audio array and sampling rate for the given index
    audio_array = dataset[index]["audio"]["array"]
    sampling_rate = dataset[index]["audio"]["sampling_rate"]

    # Use IPython's Audio class to play the audio in the notebook
    return Audio(data=audio_array, rate=sampling_rate)


# 6. Transcribe the audio sample
def transcribe_audio_sample(pipe, dataset, index=0):
    print(f"Transcribing sample {index + 1}...")
    sample = dataset[index]["audio"]["array"]
    result = pipe(sample)
    print(f"Transcription: {result['text']}")
    return result["text"]


# Main function to run everything
def main():
    # Step 1: Device and dtype setup
    device, torch_dtype = get_device()
    print(f"Using device: {device}, dtype: {torch_dtype}")

    # Specify the language you want to transcribe
    language = "en"  # Change to your desired language code

    # Step 2: Load the Whisper model and set language
    model_id = "openai/whisper-large-v3-turbo"
    model, processor = load_whisper_model(model_id, device, torch_dtype, language=language)

    # Step 3: Set up the ASR pipeline
    pipe = setup_pipeline(model, processor, device, torch_dtype)

    # Step 4: Load the dataset and list audio samples
    dataset_name = "distil-whisper/librispeech_long"
    dataset = load_and_list_dataset(dataset_name)
    print(dataset)

    # Step 5: Play the first audio sample (adjust index as needed)
    # print("Playing the first audio sample:")
    # display(play_audio_sample(dataset, index=0))

    # Step 6: Transcribe the first audio sample (adjust index as needed)
    transcribe_audio_sample(pipe, dataset, index=0)


# Run the main function
if __name__ == "__main__":
    main()
