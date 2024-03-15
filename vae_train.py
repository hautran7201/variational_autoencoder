import os
import torch
from torch.utils.data import DataLoader
from utils import load_dataset, load_multi_dataset, load_vae_data_from_disk
from models.vae import Var_Autoencoder

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data = load_vae_data_from_disk().to(device)

# Dataloader
ratio = 0.9
data_length = len(data)
train_dataloader = DataLoader(
    data[:int(data_length*ratio)],
    batch_size=1,
    shuffle=True
)
test_dataloader = DataLoader(
    data[int(data_length*ratio):],
    batch_size=1
)

# Initialize model 
model = Var_Autoencoder(latent_dim=4, seed=42, device=device)
loss = model.train(data=train_dataloader, learning_rate=0.001, epochs=3)
torch.save(model.cpu(), 'pretrained/vae.pt')
print(loss)

