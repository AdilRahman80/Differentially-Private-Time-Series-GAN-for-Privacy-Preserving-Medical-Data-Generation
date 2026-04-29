from opacus import PrivacyEngine
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

def make_private_optimizer(
    model: nn.Module, 
    optimizer, 
    dataloader: DataLoader, 
    epochs: int,
    target_epsilon: float = 1.0,
    target_delta: float = 1e-5,
    max_grad_norm: float = 1.0
) -> Tuple[nn.Module, object, DataLoader, PrivacyEngine]:
    """
    Wraps the model, optimizer, and dataloader using Opacus PrivacyEngine.
    """
    privacy_engine = PrivacyEngine()
    
    model.train()
    
    # Opacus attaches to the model and optimizer
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )
    
    return model, optimizer, dataloader, privacy_engine
