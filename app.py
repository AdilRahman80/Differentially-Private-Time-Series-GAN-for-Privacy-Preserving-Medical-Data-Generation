import streamlit as st

from dashboard.components import render_sidebar
from dashboard.pages import (
    page_data_explorer,
    page_training_monitor,
    page_evaluation,
    page_privacy_analysis,
)

st.set_page_config(
    page_title="DP-TimeGAN Dashboard",
    page_icon="DP",
    layout="wide",
)


def main():
    page = render_sidebar()

    if page == "Data Explorer":
        page_data_explorer()
    elif page == "Model Training Monitor":
        page_training_monitor()
    elif page == "Evaluation & metrics":
        page_evaluation()
    elif page == "Privacy Analysis":
        page_privacy_analysis()


if __name__ == "__main__":
    main()
