import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_test_on_synthetic(real_data: np.ndarray, fake_data: np.ndarray) -> dict:
    """
    TSTR (Train on Synthetic, Test on Real) & TRTS (Train on Real, Test on Synthetic) Evaluator.
    Specifically, we treat this as a predictive task: given t=0 to t=N-1, predict t=N (e.g., HR > threshold).
    For simplicity in this 3D array: predict if the next step's first feature goes up or down.
    """
    def prepare_predictive_data(data):
        # Flatten temporal into features: X = data[:, :-1, :].reshape, Y = data[:, -1, 0] > median
        X = data[:, :-1, :].reshape(data.shape[0], -1)
        # Binary target: is the last step's first feature above the median?
        median_val = np.median(data[:, -1, 0])
        Y = (data[:, -1, 0] > median_val).astype(int)
        return X, Y

    X_real, Y_real = prepare_predictive_data(real_data)
    X_fake, Y_fake = prepare_predictive_data(fake_data)
    
    # Train on Real, Test on Real (Baseline)
    Xr_train, Xr_test, Yr_train, Yr_test = train_test_split(X_real, Y_real, test_size=0.2)
    model_trtr = LogisticRegression(max_iter=1000)
    model_trtr.fit(Xr_train, Yr_train)
    trtr_acc = accuracy_score(Yr_test, model_trtr.predict(Xr_test))
    
    # Train on Synthetic, Test on Real (TSTR)
    model_tstr = LogisticRegression(max_iter=1000)
    model_tstr.fit(X_fake, Y_fake)
    tstr_acc = accuracy_score(Y_real, model_tstr.predict(X_real))
    
    # Train on Real, Test on Synthetic (TRTS)
    model_trts = LogisticRegression(max_iter=1000)
    model_trts.fit(X_real, Y_real)
    trts_acc = accuracy_score(Y_fake, model_trts.predict(X_fake))

    return {
        "TRTR_Baseline_Accuracy": trtr_acc,
        "TSTR_Accuracy": tstr_acc,
        "TRTS_Accuracy": trts_acc
    }
