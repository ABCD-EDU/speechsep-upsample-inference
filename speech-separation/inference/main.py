from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import gdown

import os
import requests

def download_file(url, folder_path):
   if not os.path.exists(folder_path):
      os.makedirs(folder_path)

   file_id = url.split("/")[-2]
   filename = url.split("/")[-1]
   file_path = os.path.join(folder_path, filename)

   if os.path.exists(file_path):
      print("File already exists.")
      return

   download_url = f"https://drive.google.com/uc?id={file_id}"
   gdown.download(download_url, file_path, quiet=False)

# Example usage
folder_path = "ckpt_2_speakers"
url = "https://drive.google.com/drive/folders/17H9gpFsCBkUy2ZNRyi5YM8FSkRPUdPDR?usp=share_link"
download_file(url, folder_path)
folder_path = "ckpt_3_speakers"
url = "https://drive.google.com/drive/folders/1TfF491bHXKuZ39GjzDcoqjo1jZnNqcYW?usp=share_link"
download_file(url, folder_path)

model_2speakers = separator.from_hparams(source="speechbrain/sepformer-libri2mix", savedir='ckpt_2_speakers',run_opts={"device":"cuda"})
model_3speakers = separator.from_hparams(source="speechbrain/sepformer-libri3mix", savedir='ckpt_3_speakers',run_opts={"device":"cuda"})

def run(input_path, output_path, speaker_count):
   est_sources = 0
   if speaker_count==2:
      est_sources = model_2speakers.separate_file(path=input_path) 
   else:
      est_sources = model_3speakers.separate_file(path=input_path) 
      torchaudio.save(f"{output_path}/source3hat.wav", est_sources[:, :, 2].detach().cpu(), 8000)
    
   torchaudio.save(f"{output_path}/source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
   torchaudio.save(f"{output_path}/source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)

run('item0_mix.wav','output',3)