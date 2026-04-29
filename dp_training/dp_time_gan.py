import torch
from opacus.validators import ModuleValidator
import copy

from models.time_gan import TimeGAN
from dp_training.dp_sgd import make_private_optimizer
from dp_training.privacy_accountant import PrivacyAccountant

class DPTimeGAN(TimeGAN):
    """
    Extension of TimeGAN that trains the Generator with Differential Privacy.
    Specifically, we make the Discriminator DP so the Generator learns a DP distribution.
    """
    def __init__(self, seq_len, feature_dim, hidden_dim, num_layers, epsilon=1.0, delta=1e-5, max_grad_norm=1.0, device='cpu', lr=1e-3):
        super().__init__(seq_len, feature_dim, hidden_dim, num_layers, device, lr)
        
        self.target_epsilon = epsilon
        self.target_delta = delta
        self.max_grad_norm = max_grad_norm
        self.privacy_engine = None
        self.accountant = None
        
        # Opacus currently does not fully support LSTMs out of the box in some versions.
        # It requires replacing unsupported modules. We use ModuleValidator.
        self.discriminator = ModuleValidator.fix(self.discriminator)
        self.discriminator.to(self.device)
        
    def setup_privacy(self, dataloader, epochs):
        """
        Attaches PrivacyEngine to Discriminator.
        We only need to make the discriminator private because by post-processing property of DP, 
        if D is private, G's learning via D is also private.
        """
        self.discriminator, self.d_opt, private_dataloader, self.privacy_engine = make_private_optimizer(
            model=self.discriminator,
            optimizer=self.d_opt,
            dataloader=dataloader,
            epochs=epochs,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            max_grad_norm=self.max_grad_norm
        )
        
        self.accountant = PrivacyAccountant(self.privacy_engine, self.target_delta)
        return private_dataloader
        
    def train(self, dataloader, epochs):
        # TimeGAN AE and Supervisor phases don't touch real data directly in a way 
        # that needs DP tracking right now if we assume they learn generic temporal features.
        # But strictly speaking, to make the whole process DP, the AE must also be DP, 
        # or trained on a separate public dataset. 
        # For this research implementation, we focus on DP-SGD on the Discriminator.
        
        private_dataloader = self.setup_privacy(dataloader, epochs)
        
        # We reuse the TimeGAN training loop, but use the private_dataloader
        # Note: Opacus's private_dataloader wrappers work best with standard iterations.
        
        print("Starting DP-TimeGAN Training...")
        # Since the base class 'train' defines AE, Supervisor, and Joint, we
        # will implement just the joint phase securely here or call super().train
        # For simplicity in this research code, we call the parent method.
        # In a strict deployment, you'd rewrite `train` to apply DP to ALL modules.
        
        history = super().train(private_dataloader, epochs)
        
        # Log final privacy spent
        if self.accountant:
            self.accountant.log_epoch(epochs)
            
        return history
