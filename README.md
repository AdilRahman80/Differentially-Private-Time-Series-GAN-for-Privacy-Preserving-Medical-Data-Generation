# Differentially Private Time-Series GAN for Privacy-Preserving Medical Data Generation

This project generates realistic synthetic medical time-series data while protecting sensitive patient information with Differential Privacy (DP). It includes data generation, GAN training, synthetic sequence generation, evaluation metrics, and a Streamlit dashboard.

## Features

- Synthetic physiological time-series generation for heart rate, blood pressure, SpO2, and temperature.
- LSTM-GAN, TimeGAN, and DP-TimeGAN model workflows.
- Differential privacy training support through Opacus.
- Evaluation metrics including TSTR/TRTS accuracy, RMSE, MAE, and Wasserstein distance.
- Streamlit dashboard for data exploration, training logs, evaluation, and privacy analysis.

## Requirements

- Python 3.9+
- CUDA-compatible GPU recommended for longer training runs, but CPU works for quick tests.

The pinned `requirements.txt` may not install cleanly on very new Python versions. If `pip install -r requirements.txt` fails because of `torch`, use Python 3.9-3.12 or install compatible current package versions:

```bash
pip install torch opacus numpy pandas matplotlib seaborn scikit-learn streamlit scipy tqdm
```

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` fails on your Python version, run:

```powershell
pip install torch opacus numpy pandas matplotlib seaborn scikit-learn streamlit scipy tqdm
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run these commands from the project root.

### 1. Generate Base Data

Creates the dataset files used by training and the dashboard:

```bash
python generate.py --mode base
```

Outputs:

- `dataset/real_data.csv`
- `dataset/real_data.npy`

### 2. Train a Model

For a fast smoke test:

```bash
python train.py --model lstm_gan --quick
```

This creates:

- `results/lstm_gan.pt`

For a fuller differentially private run:

```bash
python train.py --model dp_time_gan --epochs 50 --epsilon 1.0
```

This creates:

- `results/dp_time_gan.pt`

### 3. Generate Synthetic Data

If you used the quick LSTM-GAN run:

```bash
python generate.py --mode synthetic --model_path results/lstm_gan.pt --num_samples 1000
```

If you trained DP-TimeGAN:

```bash
python generate.py --mode synthetic --model_path results/dp_time_gan.pt --num_samples 1000
```

Outputs:

- `results/synthetic_data.csv`
- `results/synthetic_data.npy`

### 4. Evaluate Results

```bash
python evaluate.py
```

The evaluation script reads:

- `dataset/real_data.npy`
- `results/synthetic_data.npy`

Outputs:

- `results/evaluation_report.json`
- `results/timeseries_plot.png`
- `results/tsne_plot.png`

### 5. Launch the Dashboard

```bash
streamlit run app.py
```

Or, if Streamlit is installed only inside the active Python environment:

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Dashboard Data Checklist

If the dashboard shows a missing file warning, run the matching command:

- Missing `dataset/real_data.csv`: run `python generate.py --mode base`
- Missing `results/synthetic_data.csv`: train a model, then run `python generate.py --mode synthetic --model_path results/lstm_gan.pt --num_samples 1000`
- Missing `results/evaluation_report.json`: run `python evaluate.py`

## Project Pipeline

1. Generate or load real time-series data.
2. Normalize and prepare training batches.
3. Train LSTM-GAN, TimeGAN, or DP-TimeGAN.
4. Generate synthetic time-series data.
5. Evaluate utility and distribution similarity.
6. Inspect data and metrics in the Streamlit dashboard.

## Notes

- The quick LSTM-GAN command is best for verifying the project runs end to end.
- DP-TimeGAN training is slower and is the intended privacy-preserving experiment path.
- The default data is simulated. To use real datasets such as MIMIC-III, place them in `dataset/` and adapt the loader in `dataset/physionet_loader.py`.

## Research Paper and Presentation

- Full methodology: `research_paper/paper.md`
- Presentation slides: `presentation/slides.md`
