import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(data: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """
    Detects anomalous sequences using Isolation Forest.
    Useful for filtering out bad synthetic generations before using them.
    Returns: Array of boolean mask (True if anomalous)
    """
    # Flatten the time-series
    flat_data = data.reshape(data.shape[0], -1)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    preds = iso_forest.fit_predict(flat_data)
    
    # -1 is anomaly, 1 is normal
    return preds == -1
