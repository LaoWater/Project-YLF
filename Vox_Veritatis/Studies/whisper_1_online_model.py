import textwrap
from openai import OpenAI
import os


def main(audio_recording, model_temperature):
    print(f"Whisper-1 Online Model /w API: (Temperature {model_temperature})")
    # Create an api client
    client = OpenAI()

    # Load audio file
    file_name = audio_recording
    file_path = os.path.join("recordings", file_name)
    audio_file = open(file_path, 'rb')

    # Transcribe
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="en",  # Specify the language code for English
        temperature=model_temperature
    )

    # Extract the actual transcription text from the result
    transcription_text = result.text  # Assuming the transcription text is stored under 'text'

    # Format the transcription (wrap the text to 120 characters per line)
    formatted_transcription = textwrap.fill(transcription_text, width=120)

    # Print the formatted transcription
    print(f"Transcription\n{formatted_transcription}")
    print("-" * 50)  # Separator for readability


# Run the main function
if __name__ == "__main__":
    main(audio_recording=r"saghuru human experience beings after fullfilling the survival instinct.wav",
         model_temperature=0.44)
