import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import os

def plot_time_series(real_data: np.ndarray, fake_data: np.ndarray, sample_idx: int = 0, feature_names=None, save_path=None):
    """
    Plots real vs fake time series for a single sequence.
    """
    seq_len = real_data.shape[1]
    num_features = real_data.shape[2]
    
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(num_features)]
        
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 2*num_features))
    if num_features == 1: axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(real_data[sample_idx, :, i], label='Real', color='blue')
        ax.plot(fake_data[sample_idx, :, i], label='Synthetic', color='red', linestyle='--')
        ax.set_title(feature_names[i])
        ax.legend()
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def plot_tsne(real_data: np.ndarray, fake_data: np.ndarray, num_samples: int = 500, save_path=None):
    """
    t-SNE visualization to compare distributions.
    """
    # Subsample for faster t-SNE
    num_samples = min(num_samples, len(real_data), len(fake_data))
    
    real_sample = real_data[:num_samples].reshape(num_samples, -1)
    fake_sample = fake_data[:num_samples].reshape(num_samples, -1)
    
    combined = np.concatenate((real_sample, fake_sample), axis=0)
    labels = np.concatenate((np.ones(num_samples), np.zeros(num_samples)))
    
    tsne = TSNE(n_components=2, perplexity=40, max_iter=300)
    tsne_results = tsne.fit_transform(combined)
    
    fig = plt.figure(figsize=(8, 6))
    
    plt.scatter(tsne_results[labels==1, 0], tsne_results[labels==1, 1], 
                c='blue', alpha=0.5, label='Real')
    plt.scatter(tsne_results[labels==0, 0], tsne_results[labels==0, 1], 
                c='red', alpha=0.5, label='Synthetic')
                
    plt.legend()
    plt.title('t-SNE Projection: Real vs Synthetic')
    
    if save_path:
        plt.savefig(save_path)
    return fig
