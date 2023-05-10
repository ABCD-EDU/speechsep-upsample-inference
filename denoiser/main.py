import os
from scipy.io import wavfile
import noisereduce as nr

# Specify the directory containing the noisy wav files
def run(directory):
   # Get a list of all WAV files in the directory
   wav_files = [file for file in os.listdir(directory) if file.endswith(".wav")]

   for file in wav_files:
      file_path = os.path.join(directory, file)
      rate, data = wavfile.read(file_path)
      reduced_noise = nr.reduce_noise(y=data, sr=rate)

      wavfile.write(f"noise_reduced_{file}",8000, reduced_noise)