import os
import math
import numpy as np
from pydub import AudioSegment
from spleeter.separator import Separator

# Run in venv_python3.6 due to spleeter dependencies.

def numpy_to_audiosegment(np_data, frame_rate):
    """
    Convert a numpy array (assumed to be float32 in range -1 to 1)
    into a pydub AudioSegment with 16-bit PCM encoding.
    """
    # Convert float32 [-1,1] to int16
    np_int16 = (np_data * 32767).clip(-32768, 32767).astype(np.int16)

    # Handle mono vs stereo
    if np_int16.ndim == 1:
        channels = 1
    else:
        channels = np_int16.shape[1]

    # Create AudioSegment from raw bytes.
    return AudioSegment(
        data=np_int16.tobytes(),
        sample_width=2,  # 2 bytes for 16-bit audio
        frame_rate=frame_rate,
        channels=channels
    )


def process_chunk(chunk, separator):
    """
    Process a pydub AudioSegment chunk with Spleeter and return the vocal-only AudioSegment.
    """
    # Convert the chunk to a numpy array.
    samples = np.array(chunk.get_array_of_samples())
    if chunk.channels == 2:
        # pydub stores stereo data interleaved, so reshape accordingly.
        samples = samples.reshape((-1, 2))
    # Normalize to float32 in range [-1, 1] (assuming 16-bit PCM)
    samples = samples.astype(np.float32) / 32768.0
    # Process with Spleeter – this returns a dict with keys 'vocals' and 'accompaniment'
    prediction = separator.separate(samples)
    vocals = prediction['vocals']
    # Convert the resulting numpy array back to an AudioSegment.
    vocal_segment = numpy_to_audiosegment(vocals, chunk.frame_rate)
    return vocal_segment


def main():
    # Define file paths (use raw string literals for Windows paths)
    input_path = r"C:\Users\baciu\Desktop\G. Lucas\Lucas Second Rebirth\split_audio_buffaloes.WAV"
    output_path = r"C:\Users\baciu\Desktop\G. Lucas\Lucas Second Rebirth\spleeter\splitted_buffaloes.wav"

    # Load the full audio file using pydub.
    print("Loading audio file...")
    audio = AudioSegment.from_wav(input_path)

    # Define the chunk length in milliseconds (e.g. 60 sec = 60000 ms)
    chunk_length_ms = 60000
    total_length_ms = len(audio)
    num_chunks = math.ceil(total_length_ms / chunk_length_ms)
    print(
        f"Audio length: {total_length_ms / 1000:.1f} seconds. Splitting into {num_chunks} chunks of ~{chunk_length_ms / 1000:.0f} seconds each.")

    # Initialize the Spleeter separator for 2 stems (vocals and accompaniment)
    separator = Separator('spleeter:2stems')

    vocal_chunks = []
    for i in range(num_chunks):
        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, total_length_ms)
        print(f"Processing chunk {i + 1}/{num_chunks} (from {start_ms / 1000:.1f}s to {end_ms / 1000:.1f}s)...")
        chunk = audio[start_ms:end_ms]
        vocal_segment = process_chunk(chunk, separator)
        vocal_chunks.append(vocal_segment)

    # Concatenate all the vocal segments.
    print("Concatenating chunks...")
    combined_vocals = vocal_chunks[0]
    for seg in vocal_chunks[1:]:
        combined_vocals += seg

    # Ensure the output directory exists.
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export the final vocal track.
    print(f"Exporting the vocal track to {output_path}...")
    combined_vocals.export(output_path, format="wav")
    print("Done!")


if __name__ == '__main__':
    main()
