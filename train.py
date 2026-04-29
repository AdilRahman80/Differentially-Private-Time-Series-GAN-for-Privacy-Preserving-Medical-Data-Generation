import argparse
import os
import torch
import numpy as np
import time

import config
from dataset import get_dataloader
from preprocessing import MinMaxNormalizer
from models import TimeGAN, LSTMGAN
from dp_training import DPTimeGAN
from dashboard.database import DatabaseManager

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    real_data_path = os.path.join(config.DATA_DIR, "real_data.npy")
    if not os.path.exists(real_data_path):
        print("Real data missing. Generating base data first...")
        os.system("python generate.py --mode base")
        
    real_data = np.load(real_data_path)
    if args.quick:
        real_data = real_data[:500] # smaller subset
        
    # 2. Preprocess
    normalizer = MinMaxNormalizer()
    normalized_data = normalizer.fit_transform(real_data)
    
    dataloader = get_dataloader(normalized_data, batch_size=config.BATCH_SIZE)
    
    # 3. Model Setup
    epochs = 5 if args.quick else args.epochs
    
    start_time = time.time()
    
    if args.model == 'dp_time_gan':
        model = DPTimeGAN(
            seq_len=config.SEQ_LEN,
            feature_dim=config.FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            epsilon=args.epsilon,
            device=device,
            lr=args.lr
        )
    elif args.model == 'time_gan':
        model = TimeGAN(
            seq_len=config.SEQ_LEN,
            feature_dim=config.FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            device=device,
            lr=args.lr
        )
    else:
        model = LSTMGAN(
            seq_len=config.SEQ_LEN,
            feature_dim=config.FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            device=device,
            lr=args.lr
        )
        
    # 4. Train
    print(f"Training {args.model} for {epochs} epochs...")
    history = model.train(dataloader, epochs)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # 5. Save model
    model_path = os.path.join(config.RESULTS_DIR, f"{args.model}.pt")
    model.save_models(model_path)
    
    # 6. Log to Dashboard DB
    db = DatabaseManager()
    exp_name = f"{args.model}_run_{int(time.time())}"
    exp_id = db.log_experiment(exp_name, args.model, args.epsilon if args.model == 'dp_time_gan' else 0.0, epochs)
    
    # Log last epoch losses
    final_metrics = {k: v[-1] for k, v in history.items() if len(v) > 0}
    final_metrics['train_time_sec'] = train_time
    if hasattr(model, 'accountant') and model.accountant:
        final_metrics['final_epsilon'] = model.accountant.get_history()[-1]['epsilon']
        
    db.log_metrics(exp_id, final_metrics)
    print("Experiment logged to database.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['lstm_gan', 'time_gan', 'dp_time_gan'], default='time_gan')
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--epsilon", type=float, default=config.TARGET_EPSILON)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test with 5 epochs and small data")
    
    args = parser.parse_args()
    main(args)
