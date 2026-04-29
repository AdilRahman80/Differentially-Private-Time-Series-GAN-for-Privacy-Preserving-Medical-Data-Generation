import argparse
import os
import pandas as pd
import numpy as np
import torch

from dataset.synthetic_generator import generate_synthetic_medical_data
from models import TimeGAN, LSTMGAN
from dp_training import DPTimeGAN
from preprocessing import MinMaxNormalizer
import config

def generate_base_data(args):
    """Generates the ground truth synthetic dataset to simulate real patient data."""
    print(f"Generating '{args.num_samples}' base samples...")
    data_array, df = generate_synthetic_medical_data(num_samples=args.num_samples, seq_len=config.SEQ_LEN)
    
    # Save raw array and dataframe
    np.save(os.path.join(config.DATA_DIR, "real_data.npy"), data_array)
    df.to_csv(os.path.join(config.DATA_DIR, "real_data.csv"), index=False)
    print(f"Saved base data to {config.DATA_DIR}")

def generate_from_model(args):
    """Generates synthetic data from a trained model checkpoint."""
    if not os.path.exists(args.model_path):
        print(f"Model path not found: {args.model_path}")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Needs to match the training config
    if 'dp_time_gan' in args.model_path:
        model = DPTimeGAN(
            seq_len=config.SEQ_LEN, 
            feature_dim=config.FEATURE_DIM, 
            hidden_dim=config.HIDDEN_DIM, 
            num_layers=config.NUM_LAYERS,
            device=device
        )
    elif 'time_gan' in args.model_path:
        model = TimeGAN(
            seq_len=config.SEQ_LEN, 
            feature_dim=config.FEATURE_DIM, 
            hidden_dim=config.HIDDEN_DIM, 
            num_layers=config.NUM_LAYERS,
            device=device
        )
    else:
        model = LSTMGAN(
            seq_len=config.SEQ_LEN, 
            feature_dim=config.FEATURE_DIM, 
            hidden_dim=config.HIDDEN_DIM, 
            num_layers=config.NUM_LAYERS,
            device=device
        )
        
    model.load_models(args.model_path)
    
    print(f"Generating {args.num_samples} synthetic samples...")
    generated_array = model.generate(args.num_samples, config.SEQ_LEN)
    
    # Try inverse transform if normalizer was saved or we reload base data 
    # For now, if we have real_data, fit a normalizer to invert
    real_data_path = os.path.join(config.DATA_DIR, "real_data.npy")
    if os.path.exists(real_data_path):
        real_data = np.load(real_data_path)
        normalizer = MinMaxNormalizer()
        normalizer.fit(real_data)
        generated_array = normalizer.inverse_transform(generated_array)
        
    # Convert to dataframe
    flat_data = []
    for i in range(args.num_samples):
        for t in range(config.SEQ_LEN):
            flat_data.append({
                'Patient_ID': i,
                'Time_Step': t,
                'HeartRate': generated_array[i, t, 0],
                'SystolicBP': generated_array[i, t, 1],
                'SpO2': generated_array[i, t, 2],
                'Temperature': generated_array[i, t, 3]
            })
            
    df = pd.DataFrame(flat_data)
    df.to_csv(os.path.join(config.RESULTS_DIR, "synthetic_data.csv"), index=False)
    np.save(os.path.join(config.RESULTS_DIR, "synthetic_data.npy"), generated_array)
    
    print(f"Generated data saved to {config.RESULTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["base", "synthetic"], required=True, help="Generate base dataset or synthetic predictions")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--model_path", type=str, default=os.path.join(config.RESULTS_DIR, "dp_time_gan.pt"))
    
    args = parser.parse_args()
    
    if args.mode == "base":
        generate_base_data(args)
    else:
        generate_from_model(args)
