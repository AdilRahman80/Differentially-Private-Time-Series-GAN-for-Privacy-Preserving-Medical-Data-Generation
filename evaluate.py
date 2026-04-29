import argparse
import os
import numpy as np
import json

from evaluation import calculate_rmse, calculate_mae, train_test_on_synthetic, calculate_wasserstein
from evaluation.visualizations import plot_time_series, plot_tsne
import config

def main(args):
    print("Loading Data...")
    if not os.path.exists(config.DATA_DIR / "real_data.npy") or not os.path.exists(config.RESULTS_DIR / "synthetic_data.npy"):
        print("Data not found. Run generate.py first.")
        return
        
    real_data = np.load(config.DATA_DIR / "real_data.npy")
    fake_data = np.load(config.RESULTS_DIR / "synthetic_data.npy")
    
    # Ensure same shape for metrics
    min_samples = min(len(real_data), len(fake_data))
    real_data = real_data[:min_samples]
    fake_data = fake_data[:min_samples]
    
    report = {}
    
    print("Calculating Metrics...")
    # 1. Predictive Score (TSTR)
    tstr_results = train_test_on_synthetic(real_data, fake_data)
    report.update(tstr_results)
    
    # 2. Similarity Metrics
    rmse = calculate_rmse(real_data, fake_data)
    mae = calculate_mae(real_data, fake_data)
    wd = calculate_wasserstein(real_data, fake_data)
    
    report['RMSE'] = rmse.tolist()
    report['MAE'] = mae.tolist()
    report['Wasserstein'] = wd.tolist()
    
    print("\\n--- Evaluation Results ---")
    for k, v in report.items():
        if isinstance(v, list):
            print(f"{k}: {np.mean(v):.4f} (mean across features)")
        else:
            print(f"{k}: {v:.4f}")
            
    # Save report
    with open(os.path.join(config.RESULTS_DIR, "evaluation_report.json"), 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\\nGenerating Visualizations...")
    plot_time_series(real_data, fake_data, save_path=os.path.join(config.RESULTS_DIR, "timeseries_plot.png"))
    plot_tsne(real_data, fake_data, save_path=os.path.join(config.RESULTS_DIR, "tsne_plot.png"))
    
    print(f"Evaluation complete. Reports and plots saved to {config.RESULTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
