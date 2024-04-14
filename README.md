
# Variational Autoencoder for Nerf dataset

Decoding the latent space into images in the Nerf dataset.

## Installation
```bash
  pip install -r requirements.txt
```


## Running code
Change directory
```bash
  cd /Your path/variational_autoencoder
```
Create data:
```bash
  python dataset/save_vae_data.py
```
Training model:
```bash
  config_path = 'direction path'
  config_name = 'config name'
  HYDRA_FULL_ERROR=1 python run.py --config-path {config_path} --config-name {config_name}
``` 

## Dataset link

Synthesis Nerf: [Link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

Objaverse Eval: [Link](https://drive.google.com/drive/folders/1iQ7TlcqCbbKDBnEt4QBxaV2k1N1vQZzm?usp=drive_link)

