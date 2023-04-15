# speechsep-upsample-inference

## Instructions
### Speech Separation
Install Necessary packages

```
pip install -r requirements.txt
pip install speechbrain
pip uninstall torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

1. Create a folder named model

```
> mkdir model
> mkdir checkpoint_models
```

2. Download the models from this link:

```https://drive.google.com/drive/folders/1E-TOzcKUNPuPnfzFsJ1H-j-Sd6y9LxoJ?usp=share_link```

3. Extract the downloaded `ckpt` files to the folders `model` and `model_checkpoints`

> Three ckpt files should appear (encoder.ckpt, decoder.ckpt, masknet.ckpt)

4. Run the following command or use the `run(input_path: str, output_path: str)` inside main.py when integrating into APIs
```
python main.py hyperparams.yaml
```
