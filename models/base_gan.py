import torch
from abc import ABC, abstractmethod
import os

class BaseGAN(ABC):
    """
    Abstract Base Class for all GAN variants.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        
    @abstractmethod
    def train(self, dataloader, epochs):
        pass
        
    @abstractmethod
    def generate(self, num_samples, seq_len):
        pass
        
    def save_models(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'generator': self.generator.state_dict() if self.generator else None,
            'discriminator': self.discriminator.state_dict() if self.discriminator else None
        }, path)
        print(f"Models saved to {path}")
        
    def load_models(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        if self.generator and 'generator' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator'])
        if self.discriminator and 'discriminator' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
        print(f"Models loaded from {path}")
