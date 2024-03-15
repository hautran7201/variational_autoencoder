import torch
import numpy as np
import matplotlib.pyplot as plt

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Variable
model_path = 'pretrained/vae.pt'
model = torch.load(model_path)

# Predict
with torch.no_grad():
    latent = torch.randn(1, 4, int(512/8), int(512/8), device=device)
    decoder = model.decoder.to(device)
    output = decoder(latent).squeeze(0) 

# Convert to numpy
image = output.squeeze().permute(1, 2, 0).detach().cpu().numpy()
image = (image * 255).astype(np.uint8)

# Show image
plt.imshow(image)
plt.axis('off')
plt.show()