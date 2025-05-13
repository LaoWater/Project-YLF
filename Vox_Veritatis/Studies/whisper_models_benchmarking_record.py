import time
import sys
import sounddevice as sd
import numpy as np
import wave
import keyboard
import threading
from datetime import datetime
import os  # Import os for checking if functions exist if needed
from whisper_base_local_model import main as base_local_main
from whisper_1_online_model import main as online_model_main
from whisper_large_v3turbo_local import main as large_v3turbo_local_main


# --- Recording Configuration ---
SAMPLERATE = 16000  # Whisper prefers 16kHz
CHANNELS = 1
DTYPE = 'int16'  # Common for WAV
FILENAME_TEMPLATE = "recorded_audio_{timestamp}.wav"
HOTKEY = 'ctrl+space'
# --- Global variables for recording state ---
is_recording = False
audio_data = []
stream = None
stop_event = threading.Event()  # To signal the main thread when recording is done


# --- Recording Functions ---

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    if is_recording:
        audio_data.append(indata.copy())


def save_wave_file(filename, data, samplerate, channels, dtype):
    """Saves the recorded audio data to a WAV file."""
    print(f"Saving audio to {filename}...")
    try:
        # Ensure data is a single numpy array
        if not data:
            print("Error: No audio data recorded.", file=sys.stderr)
            return False
        recording_np = np.concatenate(data, axis=0)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(np.dtype(dtype).itemsize)
            wf.setframerate(samplerate)
            wf.writeframes(recording_np.tobytes())
        print(f"Successfully saved {filename}")
        return True
    except Exception as e:
        print(f"Error saving WAV file: {e}", file=sys.stderr)
        return False


def toggle_recording():
    """Callback function triggered by the hotkey."""
    global is_recording, audio_data, stream

    if not is_recording:
        # --- Start Recording ---
        print(f"\nStarting recording... Press '{HOTKEY}' again to stop.")
        audio_data = []  # Clear previous recording data
        try:
            # Use InputStream for non-blocking recording
            stream = sd.InputStream(
                samplerate=SAMPLERATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=audio_callback
            )
            stream.start()
            is_recording = True
        except Exception as e:
            print(f"Error starting audio stream: {e}", file=sys.stderr)
            print("Please ensure you have a working microphone connected and selected.")
            # Signal main thread to potentially exit or handle error
            stop_event.set()  # Use stop_event to signal failure too
            is_recording = False  # Ensure state is correct
            stream = None  # Clear stream object

    else:
        # --- Stop Recording ---
        print("Stopping recording...")
        is_recording = False  # Signal callback to stop appending data
        if stream:
            stream.stop()
            stream.close()
            stream = None  # Clear stream object
        # Signal the main thread that recording is finished
        stop_event.set()


def record_audio_on_hotkey():
    """Manages the recording process using a hotkey."""
    global audio_data, stop_event
    stop_event.clear()  # Ensure event is clear initially

    # Setup the hotkey listener
    try:
        keyboard.add_hotkey(HOTKEY, toggle_recording)
        print("-" * 40)
        print(f"Press '{HOTKEY}' to start recording.")
        print(f"Press '{HOTKEY}' again to stop recording.")
        print("NOTE: This script might need admin/root privileges for the hotkey.")
        print("-" * 40)

        # Wait until recording is started and then stopped
        stop_event.wait()  # Wait here until toggle_recording sets the event on stop

        # Remove the hotkey listener after recording is done
        keyboard.remove_hotkey(HOTKEY)
        print("Hotkey listener removed.")

        # Generate filename and save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = FILENAME_TEMPLATE.format(timestamp=timestamp)

        if save_wave_file(output_filename, audio_data, SAMPLERATE, CHANNELS, DTYPE):
            return output_filename  # Return filename if save was successful
        else:
            return None  # Return None if saving failed

    except Exception as e:
        print(f"\nAn error occurred during hotkey setup or waiting: {e}", file=sys.stderr)
        print("Ensure the 'keyboard' library is installed and you have necessary permissions.")
        # Attempt to clean up if something went wrong
        if is_recording and stream:
            stream.stop()
            stream.close()
        keyboard.remove_hotkey(HOTKEY)  # Try to remove hotkey anyway
        return None
    except KeyboardInterrupt:
        print("\nRecording interrupted by user (Ctrl+C).")
        # Attempt to clean up
        if is_recording and stream:
            stream.stop()
            stream.close()
        try:  # Try removing hotkey, might fail if not added
            keyboard.remove_hotkey(HOTKEY)
        except Exception:
            pass
        return None


# --- Main Execution ---
if __name__ == "__main__":
    # --- Step 1: Record Audio ---
    recorded_audio_file = record_audio_on_hotkey()

    # --- Step 2: Transcribe if recording was successful ---
    if recorded_audio_file and os.path.exists(recorded_audio_file):
        print(f"\n--- Starting Transcription for {recorded_audio_file} ---")
        model_temp = 0.88

        print("\n1. Running Base Local Model...")
        try:
            start_time = time.time()
            base_local_main(recorded_audio_file, model_temperature=model_temp)
            print(f"   Base Local Model finished in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"   Error running base_local_main: {e}", file=sys.stderr)

        print("\n2. Running Online Model...")
        try:
            start_time = time.time()
            online_model_main(recorded_audio_file, model_temperature=model_temp)
            print(f"   Online Model finished in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"   Error running online_model_main: {e}", file=sys.stderr)

        print("\n3. Running Large v3 Turbo Local Model...")
        try:
            start_time = time.time()
            large_v3turbo_local_main(recorded_audio_file, model_temperature=model_temp)
            print(f"   Large v3 Turbo Local Model finished in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"   Error running large_v3turbo_local_main: {e}", file=sys.stderr)

        print("\n--- Transcription Complete ---")

    elif recorded_audio_file:
        print(f"\n--- Transcription Skipped: File '{recorded_audio_file}' not found after saving. ---")
    else:
        print("\n--- Transcription Skipped: Audio recording failed or was cancelled. ---")
