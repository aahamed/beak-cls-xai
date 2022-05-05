import torch

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
