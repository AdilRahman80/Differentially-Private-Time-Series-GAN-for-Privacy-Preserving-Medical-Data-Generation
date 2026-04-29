import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from .base_gan import BaseGAN
from .components import LSTMRNN

class LSTMGAN(BaseGAN):
    """
    Standard LSTM-based GAN for time-series generation.
    Does not use the TimeGAN supervisor/embedding networks.
    """
    def __init__(self, seq_len, feature_dim, hidden_dim, num_layers, device='cpu', lr=1e-3):
        super().__init__(device)
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Generator: latent space (Z) -> feature space (X)
        # Latent dim we define to be same as feature_dim for simplicity
        self.generator = LSTMRNN(
            input_dim=feature_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            output_dim=feature_dim,
            use_sigmoid=True  # Normalised data is 0-1
        ).to(device)
        
        # Discriminator: feature space (X) -> probability (Y)
        self.discriminator = LSTMRNN(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1,
            use_sigmoid=True
        ).to(device)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        
    def train(self, dataloader, epochs):
        self.generator.train()
        self.discriminator.train()
        
        history = {'d_loss': [], 'g_loss': []}
        
        for epoch in range(epochs):
            d_loss_epoch = 0.0
            g_loss_epoch = 0.0
            batches = 0
            
            # Using progress bar only if user is running script interactively, handled in train.py
            # Here we just iterate
            for X_real in dataloader:
                batch_size = X_real.size(0)
                X_real = X_real.to(self.device)
                
                # Labels
                real_idx = torch.ones(batch_size, self.seq_len, 1).to(self.device)
                fake_idx = torch.zeros(batch_size, self.seq_len, 1).to(self.device)
                
                # --- Train Discriminator ---
                self.d_optimizer.zero_grad()
                
                # Real data loss
                d_real = self.discriminator(X_real)
                d_real_loss = self.criterion(d_real, real_idx)
                
                # Fake data loss
                Z = torch.rand(batch_size, self.seq_len, self.feature_dim).to(self.device)
                X_fake = self.generator(Z)
                d_fake = self.discriminator(X_fake.detach())
                d_fake_loss = self.criterion(d_fake, fake_idx)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                
                # --- Train Generator ---
                self.g_optimizer.zero_grad()
                
                # We want generator to fool discriminator
                Z = torch.rand(batch_size, self.seq_len, self.feature_dim).to(self.device)
                X_fake = self.generator(Z)
                d_fake = self.discriminator(X_fake)
                
                g_loss = self.criterion(d_fake, real_idx)
                g_loss.backward()
                self.g_optimizer.step()
                
                d_loss_epoch += d_loss.item()
                g_loss_epoch += g_loss.item()
                batches += 1
                
            history['d_loss'].append(d_loss_epoch / batches)
            history['g_loss'].append(g_loss_epoch / batches)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss_epoch/batches:.4f} | G Loss: {g_loss_epoch/batches:.4f}")
                
        return history
        
    def generate(self, num_samples, seq_len=None):
        if seq_len is None:
            seq_len = self.seq_len
            
        self.generator.eval()
        with torch.no_grad():
            Z = torch.rand(num_samples, seq_len, self.feature_dim).to(self.device)
            X_fake = self.generator(Z)
            
        return X_fake.cpu().numpy()
