import numpy as np

class MinMaxNormalizer:
    """
    Min-max normalization mapped to [0, 1] range.
    Crucial for GAN stability and standard across TimeGAN implementations.
    """
    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.is_fit = False
        
    def fit(self, data: np.ndarray):
        """
        Data should be of shape (N, seq_len, features) or (N, features).
        We calculate min/max per feature.
        """
        if len(data.shape) == 3:
            # (N, seq_len, features) -> find min/max across N and seq_len for each feature
            self.min_val = np.min(np.min(data, axis=0), axis=0)
            self.max_val = np.max(np.max(data, axis=0), axis=0)
        else:
            self.min_val = np.min(data, axis=0)
            self.max_val = np.max(data, axis=0)
            
        self.is_fit = True
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise ValueError("Normalizer is not fit yet.")
            
        # Prevent division by zero
        range_val = self.max_val - self.min_val
        range_val[range_val == 0] = 1e-6
        
        norm_data = (data - self.min_val) / range_val
        return norm_data
        
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Revert back to original scale (useful for evaluation/dashboard logs)
        """
        if not self.is_fit:
            raise ValueError("Normalizer is not fit yet.")
            
        denorm_data = data * (self.max_val - self.min_val) + self.min_val
        return denorm_data
