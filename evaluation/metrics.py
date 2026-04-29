import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance

def calculate_rmse(real_data: np.ndarray, fake_data: np.ndarray) -> np.ndarray:
    """
    Calculate Root Mean Squared Error feature-wise.
    Reshapes (N, seq_len, features) to (N*seq_len, features)
    """
    real_flat = real_data.reshape(-1, real_data.shape[-1])
    fake_flat = fake_data.reshape(-1, fake_data.shape[-1])
    
    rmse = np.zeros(real_data.shape[-1])
    for i in range(real_data.shape[-1]):
        rmse[i] = np.sqrt(mean_squared_error(real_flat[:, i], fake_flat[:, i]))
    return rmse

def calculate_mae(real_data: np.ndarray, fake_data: np.ndarray) -> np.ndarray:
    """
    Calculate Mean Absolute Error feature-wise.
    """
    real_flat = real_data.reshape(-1, real_data.shape[-1])
    fake_flat = fake_data.reshape(-1, fake_data.shape[-1])
    
    mae = np.zeros(real_data.shape[-1])
    for i in range(real_data.shape[-1]):
        mae[i] = mean_absolute_error(real_flat[:, i], fake_flat[:, i])
    return mae

def calculate_wasserstein(real_data: np.ndarray, fake_data: np.ndarray) -> np.ndarray:
    """
    Calculate 1D Wasserstein distance (Earth Mover's Distance) for each feature.
    """
    real_flat = real_data.reshape(-1, real_data.shape[-1])
    fake_flat = fake_data.reshape(-1, fake_data.shape[-1])
    
    wd = np.zeros(real_data.shape[-1])
    for i in range(real_data.shape[-1]):
        wd[i] = wasserstein_distance(real_flat[:, i], fake_flat[:, i])
    return wd
