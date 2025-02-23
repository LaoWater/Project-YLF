#####################################################
# Loading HuggingFace Model and running it on GPU  ##
# ###################################################


import  whisper
import os
import textwrap
import re

# Local Whisper
# Customization: You have direct control over which model size to
# load (e.g., tiny, base, small, medium, or large) and can tweak parameters.
# Can further proceed in fine-tuning, training, RFL, and so on. It is clearly The Choice.


def transcribe_file(model, file_path, temperature, language="en", no_speech_threshold=0.1, logprob_threshold=-1.0):
    print(f"Transcribing with Temperature: {temperature}")

    # Transcribe the file with custom parameters
    result = model.transcribe(
        file_path,
        language=language,
        temperature=temperature,
        no_speech_threshold=no_speech_threshold,
        logprob_threshold=logprob_threshold,
        condition_on_previous_text=False
    )

    transcription = result['text']

    # Split the transcription into sentences based on punctuation followed by a space
    sentences = re.split(r'(?<=[.!?]) +', transcription)

    # Join each sentence with a newline
    formatted_transcription = "\n".join(sentences)

    # Print the formatted transcription
    print(f"Transcription\n{formatted_transcription}")
    print("-" * 50)  # Separator for readability

    return transcription


# New function to transcribe and save to file
def transcribe_and_save_to_file(model, file_path, temperature, output_file, language="en", no_speech_threshold=0.1,
                                logprob_threshold=-1.0):
    transcription = transcribe_file(model, file_path, temperature, language, no_speech_threshold, logprob_threshold)

    # Split the transcription into sentences based on punctuation followed by a space
    sentences = re.split(r'(?<=[.!?]) +', transcription)

    # Join each sentence with a newline
    formatted_transcription = "\n".join(sentences)

    # Write the formatted transcription to a text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(formatted_transcription)

    print(f"Transcription saved to {output_file}")


def main(audio_recording, model_temperature=0.55):
    print(f"Whisper-Base Local Model: (Temperature: {model_temperature})")
    # Load Whisper model
    device = "cuda"
    print(f"Device: {device}")
    model = whisper.load_model("base", device=device)

    # Path to the .wav file in the 'recordings' folder
    file_name = audio_recording
    file_path = os.path.join("recordings", file_name)

    # temperatures = [0.57, 0.1, 0.44, 0.88, 0.92]  # Adjust temperature values as needed

    # Output on console
    # transcribe_file(model, file_path, model_temperature)

    # Define the output text file path in the 'Transcripts' directory in the current directory
    output_file = os.path.join(os.getcwd(), "Transcripts", "BhikkuReading Part 1.txt")

    # Transcribe and save to file
    # transcribe_and_save_to_file(model, file_path, model_temperature, output_file)

    transcribe_and_save_to_file(model, file_path, model_temperature, r'Transcripts/BhikkuReading Part 1.txt')


# Run the main function
if __name__ == "__main__":
    main(audio_recording=r"Bhikku - dhamma language vs people language - part 2.wav")

# Main Parameters:
# Temperature: Lower values make the transcription more accurate and less creative (helpful for clear audio like English). Higher values increase diversity but may lead to hallucinations.
# No Speech Threshold: Use this if the audio contains background noise or pauses that Whisper mistakenly identifies as silence.
# Logprob Threshold: This filters out words with low confidence scores, which can help eliminate hallucinations or incorrect guesses.

#
# 2. Exploring Parameters for the Local Whisper Model:
# When using the local Whisper model, you can fine-tune the transcription with several parameters to improve accuracy or fit specific use cases. Here are some of the important parameters you can experiment with:
#
# Key Parameters:
# language: Force the model to transcribe in a specific language.
#
# Example: language="en" for English.
# temperature: Controls the randomness of the transcription. Lower values make the model more deterministic.
#
# Example: temperature=0.0 will make the model more confident but less creative (may help with hallucination).
# temperature_increment_on_fallback: Adjusts the temperature when fallback is triggered (fallback happens when the transcription is uncertain).
#
# Example: temperature_increment_on_fallback=0.2 will make the model more creative during fallback attempts.
# no_speech_threshold: A float that determines whether to classify audio as "no speech." Lowering this may reduce false positives.
#
# Example: no_speech_threshold=0.1 to make the model less likely to skip silent or very quiet parts.
# logprob_threshold: A threshold for word probabilities. It discards words with lower probabilities (useful for filtering hallucinated words).
#
# Example: logprob_threshold=-1.0.
# compression_ratio_threshold: Discards transcriptions with a high compression ratio (too repetitive).
#
# Example: compression_ratio_threshold=2.4.
# condition_on_previous_text: Whether or not the model should condition on the previous output text when transcribing. Default is True.
#
# Example: condition_on_previous_text=False might improve cases where hallucinations accumulate over time.


# Before parameters proper use

# Result from whisper call in live-recording processing
# Transcription: Părșa, dacă te ficate un lucruri în ages, Tuscurate kızım. Acestul starteeunific поч 어렵 este.

# Transcription 1:  plante pentru ca recori totacii azi nu ştiu gosh

# Transcription attempt 2:
# Transcription 2:  Mille obidă acolo în scoapau ceävereائăpa nu am înțe structure.
# --------------------------------------------------
# Transcription attempt 3:
# Transcription 3:  Și envie mai lucru care făusă în aceste værea internală Până instituți incharit conduction STACK
# --------------------------------------------------
# Transcription attempt 4:
# Transcription 4:  E umpltit si suntan in sa acasca de amintic sa-mi dur away?
# --------------------------------------------------
# Transcription attempt 5:
# Transcription 5:  Delicious.興 fromisparul in què colte a măzuta, hele smarter, Prツul in cabrion, mustina in Burch.
