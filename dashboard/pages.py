import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from evaluation.visualizations import plot_time_series
from dashboard.database import DatabaseManager
from config import RESULTS_DIR, DATA_DIR

db = DatabaseManager()

def page_data_explorer():
    st.header("Data Explorer")
    
    data_type = st.radio("Select Data Source", ["Real Data", "Synthetic Data (Results)"])
    
    if data_type == "Real Data":
        path = os.path.join(DATA_DIR, "real_data.csv")
    else:
        path = os.path.join(RESULTS_DIR, "synthetic_data.csv")
        
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.write(f"Loaded **{data_type}** with shape: {df.shape}")
        st.dataframe(df.head(100))
        
        # Simple plot
        if 'Patient_ID' in df.columns:
            st.subheader("Patient Trajectory")
            patient_id = st.selectbox("Select Patient ID", df['Patient_ID'].unique()[:50])
            patient_data = df[df['Patient_ID'] == patient_id]
            st.line_chart(patient_data.drop(['Patient_ID', 'Time_Step'], axis=1, errors='ignore'))
    else:
        st.warning(f"File not found: {path}. Please run generate scripts first.")

def page_training_monitor():
    st.header("Training Monitor")
    st.write("View logs from recent experiment runs.")
    
    experiments = db.get_experiments()
    if len(experiments) > 0:
        st.dataframe(experiments)
        
        selected_exp = st.selectbox("Select Experiment ID for details", experiments['id'].values)
        metrics = db.get_metrics(selected_exp)
        
        if len(metrics) > 0:
            st.subheader("Metrics & Losses")
            # Pivot table for plotting
            st.dataframe(metrics)
    else:
        st.info("No experiments logged yet.")

def page_evaluation():
    st.header("Evaluation Metrics")
    st.write("Compares predictive and discriminative scores across models.")
    
    # Check if evaluation summary exists
    eval_path = os.path.join(RESULTS_DIR, "evaluation_report.json")
    if os.path.exists(eval_path):
        import json
        with open(eval_path, 'r') as f:
            report = json.load(f)
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("TSTR Accuracy", f"{report.get('TSTR_Accuracy', 0):.2%}")
        with col2:
            st.metric("TRTS Accuracy", f"{report.get('TRTS_Accuracy', 0):.2%}")
        with col3:
            st.metric("TRTR Baseline", f"{report.get('TRTR_Baseline_Accuracy', 0):.2%}")
            
        st.subheader("Feature-wise RMSE")
        rmse = report.get('RMSE', [])
        st.bar_chart(pd.DataFrame(rmse, columns=['RMSE']))
    else:
        st.warning("Run evaluate.py to generate metrics report.")

def page_privacy_analysis():
    st.header("Privacy Analysis")
    st.write("Differential Privacy Budget Tracking")
    
    epsilon = st.slider("Target Epsilon (epsilon)", 0.1, 10.0, 1.0)
    delta = 1e-5
    st.latex(r"P(\mathcal{M}(D) \in S) \le e^{\epsilon} P(\mathcal{M}(D') \in S) + \delta")
    
    st.write(f"Current Budget Configuration:")
    st.write(f"**Epsilon**: {epsilon}")
    st.write(f"**Delta**: {delta}")
    
    st.info("Lower epsilon indicates stronger privacy guarantees, but may reduce the utility (fidelity) of generated time-series.")
