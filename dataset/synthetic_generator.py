import numpy as np
import pandas as pd
from typing import Tuple

def generate_synthetic_medical_data(
    num_samples: int = 5000, 
    seq_len: int = 24
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generates realistic synthetic medical time-series data.
    Features: Heart Rate, Systolic BP, SpO2, Temperature
    """
    np.random.seed(42)
    
    # Pre-allocate array
    data = np.zeros((num_samples, seq_len, 4))
    
    for i in range(num_samples):
        # Base vital signs parameters for this patient
        base_hr = np.random.normal(75, 10) # Normal HR 60-100
        base_bp = np.random.normal(120, 15) # Normal Sys BP 90-140
        base_spo2 = np.random.normal(97, 2) # Normal SpO2 95-100, capped at 100
        base_temp = np.random.normal(37.0, 0.4) # Normal Temp 36.5-37.5
        
        # Add temporal dynamics (circadian rhythms, random walks, drift)
        t = np.linspace(0, 2*np.pi, seq_len)
        
        # 1. Heart Rate: smooth random walk + circadian rhythm
        hr = base_hr + 5 * np.sin(t) + np.cumsum(np.random.normal(0, 1.5, seq_len))
        
        # 2. Systolic BP: correlated with HR
        bp = base_bp + 0.3 * (hr - base_hr) + np.cumsum(np.random.normal(0, 2.0, seq_len))
        
        # 3. SpO2: mostly flat, occasional minor dips
        spo2 = np.clip(base_spo2 + np.random.normal(0, 0.5, seq_len), a_min=85, a_max=100)
        
        # 4. Temperature: very slow drift
        temp = base_temp + 0.2 * np.cos(t - np.pi/4) + np.cumsum(np.random.normal(0, 0.05, seq_len))
        
        # Put into array
        data[i, :, 0] = np.clip(hr, 40, 200)
        data[i, :, 1] = np.clip(bp, 70, 200)
        data[i, :, 2] = spo2
        data[i, :, 3] = np.clip(temp, 35.0, 41.0)
    
    # Create a flattened dataframe representation for easier EDA
    flat_data = []
    for i in range(num_samples):
        for t in range(seq_len):
            flat_data.append({
                'Patient_ID': i,
                'Time_Step': t,
                'HeartRate': data[i, t, 0],
                'SystolicBP': data[i, t, 1],
                'SpO2': data[i, t, 2],
                'Temperature': data[i, t, 3]
            })
            
    df = pd.DataFrame(flat_data)
    
    return data, df

if __name__ == '__main__':
    data, df = generate_synthetic_medical_data(num_samples=10, seq_len=24)
    print(f"Generated array shape: {data.shape}")
    print(f"DataFrame head:\n{df.head()}")
