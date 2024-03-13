import sys
sys.path.append("sd")

import torch 
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from models.encoder import VAE_Encoder
from models.decoder import VAE_Decoder

class Var_Autoencoder(nn.Module):
    def __init__(self, latent_dim, seed=None, device='cpu'):
        super().__init__()

        self.encoder = VAE_Encoder(latent_dim, seed).to(device)
        self.decoder = VAE_Decoder(latent_dim).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, x: torch.Tensor):
        return self.decoder(x)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train(
            self, 
            data: DataLoader, 
            eval_data: DataLoader=None, 
            learning_rate: int=5e-4, 
            epochs: int=1,
            saving_path: str=None
        ):
        # Optimzier
        opt = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        # Loss variable
        total_loss_tracker = []
        reconstruction_loss_tracker = []
        kl_loss_tracker = []        

        # Train
        pbar = tqdm(range(epochs*len(data)))
        for epoch in range(epochs):
            for i, x in enumerate(data):
                opt.zero_grad()
                x_hat = self(x)

                # Calculate loss
                # reconstruction_loss = nn.CrossEntropyLoss(x, x_hat)                
                reconstruction_loss = ((x - x_hat)**2).sum()
                kl_loss = self.encoder.kl
                total_loss = reconstruction_loss + kl_loss

                loss = total_loss
                loss.backward()
                opt.step()

                # Save loss
                reconstruction_loss_tracker.append(reconstruction_loss)
                kl_loss_tracker.append(kl_loss)
                total_loss_tracker.append(total_loss)

                # Show loss
                pbar.set_description(
                    f"Epoch {epoch+1:02d}:"
                    + f" Iteration {i:04d}:"
                    + f" reconstruction_loss = {reconstruction_loss_tracker[-1]:.2f}"
                    + f" kl_loss = {kl_loss_tracker[-1]:.2f}"
                    + f" total_loss = {total_loss_tracker[-1]:.2f}"
                )
                pbar.update(1)
            
            # Test loss variable
            total_test_loss_tracker = []
            reconstruction_test_loss_tracker = []
            kl_test_loss_tracker = [] 
            if eval_data:
                with torch.no_grad():
                    test_pbar = tqdm(range(len(eval_data)))
                    for i, x in enumerate(eval_data):
                        x_hat = self(x)

                        # Calculate loss
                        reconstruction_loss = ((x - x_hat)**2).sum()
                        kl_loss = self.encoder.kl
                        total_loss = reconstruction_loss + kl_loss

                        reconstruction_test_loss_tracker.append(reconstruction_loss)
                        kl_test_loss_tracker.append(kl_loss)
                        total_test_loss_tracker.append(total_loss)

                        pbar.set_description(
                            f"Epoch {epoch:02d}:"
                            + f" Iteration {i:04d}:"
                            + f" reconstruction_loss = {reconstruction_test_loss_tracker[-1]:.2f}"
                            + f" kl_loss = {kl_test_loss_tracker[-1]:.2f}"
                            + f" total_loss = {total_test_loss_tracker[-1]:.2f}"
                        )

                        test_pbar.update(1)

        recon_loss_mean = torch.mean(torch.tensor(reconstruction_loss_tracker, dtype=torch.float32))
        kl_loss_mean = torch.mean(torch.tensor(kl_loss_tracker, dtype=torch.float32))
        total_loss_mean = torch.mean(torch.tensor(total_loss_tracker, dtype=torch.float32))

        recon_test_loss_mean = torch.mean(torch.tensor(reconstruction_test_loss_tracker, dtype=torch.float32))
        kl_test_loss_mean = torch.mean(torch.tensor(kl_test_loss_tracker, dtype=torch.float32))
        total_test_loss_mean = torch.mean(torch.tensor(total_test_loss_tracker, dtype=torch.float32))

        if saving_path:
            torch.save(self.detach().cpu(), os.path.join(saving_path, 'vae.pt'))

        return {
            'train': {
                'reconstruction_loss': recon_loss_mean, 
                'kl_loss': kl_loss_mean,
                'total_loss': total_loss_mean
            },
            'test': {
                'reconstruction_test_loss': recon_test_loss_mean, 
                'kl_test_loss': kl_test_loss_mean,
                'total_test_loss': total_test_loss_mean
            }
            
        }