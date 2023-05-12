import main
import importlib
importlib.reload(main)
from main import run
from glob import glob
import os
from datetime import datetime
from tqdm import tqdm

# absolute path pls
directory = "C:/Users/chris/Music/standard_model_3_speakers/audio_results"
# directory ="../speech-sep-metrics/results/audio_results"
wavs = glob(directory + "/*hat.wav")
outdir_name = directory + "/upsampled"

if not os.path.exists(outdir_name):
    os.makedirs(outdir_name)

start_time = datetime.now()
print("UPSCALING STARTED: ", start_time)
print("TOTAL NUMBER OF FILES:", len(wavs))

for wav in tqdm(wavs, total=len(wavs)):
    # current_time = datetime.now().strftime("%H:%M:%S %m-%d")
    # print(f"[{i}/{len(wavs)}] | START: {current_time}", end=" | ")
    file_name = os.path.basename(wav).split(".")[0]
    run(input_path=wav, output_path=outdir_name + "/" + file_name + "_up" + ".wav")
    # current_time = datetime.now().strftime("%H:%M:%S %m-%d")
    # print(f"ELAPSED: {datetime.now() - start_time} | FILE: {file_name}")

end_time = datetime.now()
print("UPSCALING ENDED: ", end_time)
print("TOTAL DURATION:", end_time - start_time)
