import streamlit as st
import pandas as pd
import numpy as np

def render_metric_card(title, value, description=None):
    desc_html = f'<p style="margin:0px;color:#7e8299;font-size:12px;">{description}</p>' if description else ''
    st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-bottom:10px;">
            <h4 style="margin:0px;color:#31333F;">{title}</h4>
            <h2 style="margin:0px;color:#FF4B4B;">{value}</h2>
            {desc_html}
        </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    st.sidebar.title("DP-TimeGAN Dashboard")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", [
        "Data Explorer",
        "Model Training Monitor",
        "Evaluation & metrics",
        "Privacy Analysis"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Differentially Private Time-Series GAN for Privacy-Preserving Medical Data Generation.\n\n"
        "Built with PyTorch, Opacus, and Streamlit."
    )
    
    return page
