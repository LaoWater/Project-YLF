import textwrap
import warnings

warnings.filterwarnings("ignore")
import os

# Set the environment variable to disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Reached bottle-neck cause of model limitation of ~30 sec - moved to chunked Audio.


# Model Precision: Since you're running the model in torch.float16 (which is common on GPUs for better performance), the model's parameters (including biases) are in half precision.
# Input Audio: The input audio data from the .wav file is likely in float32 (standard floating-point precision), which doesn’t match the model's float16 precision.
# Solution:
# We need to convert the audio input data to float16 to match the model's precision.


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


# 3. Transcribe the local .wav file and include timestamps
def transcribe_audio_file_with_timestamps(file_path, model, processor, device, torch_dtype):
    print(f"\nTranscribing audio file with timestamps: {file_path}")

    # Load the audio file
    audio_input, sampling_rate = sf.read(file_path)

    # Convert audio input to torch tensor and the correct dtype
    audio_input = torch.tensor(audio_input, dtype=torch.float32 if torch_dtype == torch.float32 else torch.float16)

    # Preprocess the audio for the model
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=sampling_rate)
    inputs = inputs.to(device, dtype=torch_dtype)

    # Run inference with the model and request timestamps
    with torch.no_grad():
        predicted_ids = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)

    # Decode the predicted IDs to text with timestamps
    transcription_with_timestamps = processor.batch_decode(predicted_ids.sequences, skip_special_tokens=True,
                                                           output_attentions=True)

    # Process the transcription and timestamps
    return transcription_with_timestamps


# 3. Transcribe the local .wav file
def transcribe_audio_file(file_path, model, processor, device, torch_dtype):
    print(f"Transcribing audio file: {file_path}")

    # Load the audio file
    audio_input, sampling_rate = sf.read(file_path)

    # Convert audio input to torch tensor and the correct dtype (float16 if needed)
    audio_input = torch.tensor(audio_input, dtype=torch.float32 if torch_dtype == torch.float32 else torch.float16)

    # Preprocess the audio for the model
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=sampling_rate)
    inputs = inputs.to(device, dtype=torch_dtype)

    # Run inference with the model
    with torch.no_grad():
        predicted_ids = model.generate(**inputs)

    # Decode the predicted IDs to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    # Use textwrap to format the transcription
    formatted_transcription = textwrap.fill(transcription, width=120)  # Adjust width as needed
    print(f"Transcription: {formatted_transcription}")

    # Step 4: Transcribe the local audio file with timestamps
    transcription_with_timestamps = transcribe_audio_file_with_timestamps(file_path, model, processor, device,
                                                                          torch_dtype)
    print(transcription_with_timestamps)
    return formatted_transcription


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

    # Step 3: Provide the path to the local .wav file
    audio_file_path = r"C:\Users\baciu\Desktop\Neo Training\Text_Assistant\recordings\saghuru human experience beings after fullfilling the survival instinct.wav"  # Update this with your local file path

    # Step 4: Transcribe the local audio file
    transcribe_audio_file(audio_file_path, model, processor, device, torch_dtype)


# Run the main function
if __name__ == "__main__":
    main()

