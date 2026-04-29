import torch
import torch.nn as nn
import torch.optim as optim
import os

from .base_gan import BaseGAN
from .components import LSTMRNN

class TimeGAN(BaseGAN):
    """
    PyTorch implementation of TimeGAN (Yoon et al., 2019).
    Incorporates Embedding, Recovery, Supervisor, Generator, and Discriminator.
    """
    def __init__(self, seq_len, feature_dim, hidden_dim, num_layers, device='cpu', lr=1e-3):
        super().__init__(device)
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 1. Embedding Network: X -> H (latent space)
        self.embedder = LSTMRNN(
            input_dim=feature_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            output_dim=hidden_dim,
            use_sigmoid=True
        ).to(device)
        
        # 2. Recovery Network: H -> X (feature space)
        self.recovery = LSTMRNN(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            output_dim=feature_dim,
            use_sigmoid=True
        ).to(device)
        
        # 3. Generator Network: Z -> E (fake latent space)
        self.generator = LSTMRNN(
            input_dim=feature_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            output_dim=hidden_dim,
            use_sigmoid=True
        ).to(device)
        
        # 4. Supervisor Network: E -> H_supervised (captures temporal dynamics)
        self.supervisor = LSTMRNN(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers-1, 
            output_dim=hidden_dim,
            use_sigmoid=True
        ).to(device)
        
        # 5. Discriminator Network: H -> Y (real/fake probability)
        self.discriminator = LSTMRNN(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            output_dim=1,
            use_sigmoid=False # Logits output for BCEWithLogitsLoss
        ).to(device)
        
        # Optimizers
        # E0 optimizer: embedder and recovery
        self.e_opt = optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=lr)
        # G0 optimizer: generator and supervisor
        self.g_opt = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=lr)
        # D0 optimizer: discriminator
        self.d_opt = optim.Adam(self.discriminator.parameters(), lr=lr)
        
        # Loss functions
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        
    def train(self, dataloader, epochs):
        """
        TimeGAN Training has 3 phases:
        1. Autoencoder training
        2. Supervisor training
        3. Joint training
        """
        history = {'e_loss': [], 's_loss': [], 'g_loss': [], 'd_loss': []}
        
        print("Phase 1: Autoencoder training...")
        for _ in range(epochs // 2 + 1): # Standard practice is to train AE for half epochs
            e_loss_epoch = 0
            for X_real in dataloader:
                X_real = X_real.to(self.device)
                self.e_opt.zero_grad()
                
                H = self.embedder(X_real)
                X_tilde = self.recovery(H)
                
                e_loss = 10 * torch.sqrt(self.mse(X_real, X_tilde))
                e_loss.backward()
                self.e_opt.step()
                e_loss_epoch += e_loss.item()
            history['e_loss'].append(e_loss_epoch / len(dataloader))
            
        print("Phase 2: Supervisor training...")
        for _ in range(epochs // 2 + 1):
            s_loss_epoch = 0
            for X_real in dataloader:
                X_real = X_real.to(self.device)
                self.g_opt.zero_grad()
                
                H = self.embedder(X_real).detach() # Keep embedder fixed
                H_hat_supervise = self.supervisor(H[:, :-1, :])
                
                s_loss = self.mse(H[:, 1:, :], H_hat_supervise)
                s_loss.backward()
                self.g_opt.step()
                s_loss_epoch += s_loss.item()
            history['s_loss'].append(s_loss_epoch / len(dataloader))
                
        print("Phase 3: Joint training...")
        for epoch in range(epochs):
            g_loss_epoch = 0
            d_loss_epoch = 0
            
            for X_real in dataloader:
                batch_size = X_real.size(0)
                X_real = X_real.to(self.device)
                
                Z = torch.rand(batch_size, self.seq_len, self.feature_dim).to(self.device)
                
                # --- Train Generator/Supervisor/Embedder/Recovery ---
                for _ in range(2): # Generator is often trained more often than discriminator in TimeGAN
                    self.g_opt.zero_grad()
                    self.e_opt.zero_grad()
                    
                    H = self.embedder(X_real)
                    E_hat = self.generator(Z)
                    H_hat = self.supervisor(E_hat)
                    H_hat_supervise = self.supervisor(H[:, :-1, :])
                    
                    X_tilde = self.recovery(H)
                    X_hat = self.recovery(H_hat)
                    
                    # Losses
                    Y_fake = self.discriminator(H_hat)
                    Y_fake_e = self.discriminator(E_hat)
                    
                    g_loss_u = self.bce(Y_fake, torch.ones_like(Y_fake)) + self.bce(Y_fake_e, torch.ones_like(Y_fake_e))
                    g_loss_s = 100 * torch.sqrt(self.mse(H[:, 1:, :], H_hat_supervise))
                    g_loss_v1 = 100 * torch.mean(torch.abs(torch.sqrt(torch.var(X_hat, dim=0) + 1e-6) - torch.sqrt(torch.var(X_real, dim=0) + 1e-6)))
                    g_loss_v2 = 100 * torch.mean(torch.abs(torch.mean(X_hat, dim=0) - torch.mean(X_real, dim=0)))
                    
                    e_loss_t0 = 10 * torch.sqrt(self.mse(X_real, X_tilde))
                    e_loss0 = 0.1 * g_loss_s
                    
                    g_loss = g_loss_u + g_loss_s + g_loss_v1 + g_loss_v2
                    e_loss = e_loss_t0 + e_loss0
                    
                    loss = g_loss + e_loss
                    loss.backward()
                    self.g_opt.step()
                    self.e_opt.step()
                    
                # --- Train Discriminator ---
                self.d_opt.zero_grad()
                
                H = self.embedder(X_real).detach()
                E_hat = self.generator(Z).detach()
                H_hat = self.supervisor(E_hat).detach()
                
                Y_real = self.discriminator(H)
                Y_fake = self.discriminator(H_hat)
                Y_fake_e = self.discriminator(E_hat)
                
                d_loss_real = self.bce(Y_real, torch.ones_like(Y_real))
                d_loss_fake = self.bce(Y_fake, torch.zeros_like(Y_fake))
                d_loss_fake_e = self.bce(Y_fake_e, torch.zeros_like(Y_fake_e))
                
                d_loss = d_loss_real + d_loss_fake + d_loss_fake_e
                if d_loss.item() > 0.15: # Prevent discriminator from dominating
                    d_loss.backward()
                    self.d_opt.step()
                    
                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()
                
            history['g_loss'].append(g_loss_epoch / len(dataloader))
            history['d_loss'].append(d_loss_epoch / len(dataloader))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {history['d_loss'][-1]:.4f} | G Loss: {history['g_loss'][-1]:.4f}")
                
        return history
        
    def generate(self, num_samples, seq_len=None):
        if seq_len is None:
            seq_len = self.seq_len
            
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()
        
        dataset = []
        batch_size = 128
        
        # Batching generation to avoid memory issues
        with torch.no_grad():
            for _ in range(num_samples // batch_size + 1):
                Z = torch.rand(batch_size, seq_len, self.feature_dim).to(self.device)
                E_hat = self.generator(Z)
                H_hat = self.supervisor(E_hat)
                X_hat = self.recovery(H_hat)
                dataset.append(X_hat.cpu().numpy())
                
        generated_data = np.concatenate(dataset, axis=0)
        return generated_data[:num_samples] # Trim excess
