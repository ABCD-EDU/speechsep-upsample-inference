"# speechsep-inference"

### Install Necessary packages

`pip install -r requirements.txt`
`pip install speechbrain`
`pip uninstall torch`
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`


### Create a folder named model
`mkdir model`
### Download the models from this link:
https://drive.google.com/drive/folders/1E-TOzcKUNPuPnfzFsJ1H-j-Sd6y9LxoJ?usp=share_link
### Extract the zip within the folder
### Three ckpt files should appear (encoder.ckpt, decoder.ckpt, masknet.ckpt)

### Edit the following line of code to specify mix_audio_path and the output_path

line 92 | main.py -> run('mix_audio_path', 'output_path')

### Run the following command

python main.py hyperparams.yaml
