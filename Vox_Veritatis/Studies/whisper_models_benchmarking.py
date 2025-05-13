import time
import sys
from whisper_base_local_model import main as base_local_main
from whisper_1_online_model import main as online_model_main
from whisper_large_v3turbo_local import main as large_v3turbo_local_main

# The audio file to benchmark
audio_file_to_benchmark = "what did i want to say.wav"
model_temp = 0.88
# Call each main function
base_local_main(audio_file_to_benchmark, model_temperature=model_temp)
online_model_main(audio_file_to_benchmark, model_temperature=model_temp)
large_v3turbo_local_main(audio_file_to_benchmark, model_temperature=model_temp)


# Task: Create manual recording audio files, not loaded from disk