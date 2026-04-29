import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for Time-Series data.
    Input shape should be (N, seq_len, features)
    """
    def __init__(self, data: np.ndarray):
        self.data = torch.FloatTensor(data)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def sequence_data_preparation(data: np.ndarray, seq_len: int = 24, stride: int = 1) -> np.ndarray:
    """
    Prepares sequential data using a sliding window approach.
    Useful if data is (N_timesteps, features) instead of (N_patients, seq_len, features)
    """
    if len(data.shape) == 3:
        # Already in (N, seq_len, features) format
        return data
        
    num_samples = (len(data) - seq_len) // stride + 1
    seq_data = np.zeros((num_samples, seq_len, data.shape[1]))
    
    for i in range(num_samples):
        seq_data[i] = data[i*stride : i*stride + seq_len]
        
    return seq_data

def get_dataloader(data: np.ndarray, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
    """
    Returns a PyTorch DataLoader for the dataset.
    """
    dataset = TimeSeriesDataset(data)
    # drop_last=True is often useful for GAN training stability and DP-SGD expectations
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader
