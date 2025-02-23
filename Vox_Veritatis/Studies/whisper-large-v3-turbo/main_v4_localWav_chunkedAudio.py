import textwrap
import warnings
import os
import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import numpy as np

warnings.filterwarnings("ignore")

# Set the environment variable to disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


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
    model.config.forced_decoder_ids = forced_decoder_ids

    return model, processor


# Function to chunk audio
def chunk_audio(audio_input, sampling_rate, chunk_duration=30):
    num_samples_per_chunk = int(sampling_rate * chunk_duration)
    num_chunks = int(np.ceil(len(audio_input) / num_samples_per_chunk))

    chunks = [
        audio_input[i * num_samples_per_chunk:(i + 1) * num_samples_per_chunk]
        for i in range(num_chunks)
    ]

    return chunks, num_chunks


# 3. Transcribe the local .wav file
def transcribe_audio_chunk(chunk, model, processor, device, torch_dtype, sampling_rate):
    # Convert audio input to torch tensor and the correct dtype
    chunk_input = torch.tensor(chunk, dtype=torch.float32 if torch_dtype == torch.float32 else torch.float16)

    # Preprocess the audio for the model
    inputs = processor(chunk_input, return_tensors="pt", sampling_rate=sampling_rate)
    inputs = inputs.to(device, dtype=torch_dtype)

    # Run inference with the model
    with torch.no_grad():
        predicted_ids = model.generate(**inputs)

    # Decode the predicted IDs to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


def transcribe_audio_file_in_chunks(file_path, model, processor, device, torch_dtype, chunk_duration=30):
    print(f"Transcribing audio file in chunks: {file_path}")

    # Load the audio file
    audio_input, sampling_rate = sf.read(file_path)

    # Split the audio into smaller chunks
    audio_chunks, num_chunks = chunk_audio(audio_input, sampling_rate, chunk_duration)

    # Transcribe each chunk
    full_transcription = ""
    for i, chunk in enumerate(audio_chunks):
        print(f"Transcribing chunk {i + 1}/{num_chunks}...")
        chunk_transcription = transcribe_audio_chunk(chunk, model, processor, device, torch_dtype, sampling_rate)
        full_transcription += chunk_transcription + " "

    # Use textwrap to format the transcription
    formatted_transcription = textwrap.fill(full_transcription, width=133)
    print(f"Full Transcription: {formatted_transcription}")
    return formatted_transcription


# Main function to run everything
def main():
    # Step 1: Device and dtype setup
    device, torch_dtype = get_device()
    print(f"Using device: {device}, dtype: {torch_dtype}")

    # Specify the language you want to transcribe
    language = "en"

    # Step 2: Load the Whisper model and set language
    model_id = "openai/whisper-large-v3-turbo"
    model, processor = load_whisper_model(model_id, device, torch_dtype, language=language)

    # Step 3: Provide the path to the local .wav file
    audio_file_path = r"C:\Users\baciu\Desktop\Neo Training\Text_Assistant\recordings\saghuru human experience beings after fullfilling the survival instinct.wav"  # Update with your file path

    # Step 4: Transcribe the local audio file in chunks
    transcribe_audio_file_in_chunks(audio_file_path, model, processor, device, torch_dtype)


# Run the main function
if __name__ == "__main__":
    main()
