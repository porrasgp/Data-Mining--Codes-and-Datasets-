import streamlit as st
from UI.ui import display_data, build_model, show_prediction
from config.model_config import DATASETS, MODELS

st.title("Scalable Data Analysis and Model Training App")

# Sidebar for dataset and model selection
st.sidebar.title("Settings")
dataset_name = st.sidebar.selectbox("Select Dataset", list(DATASETS.keys()))
model_name = st.sidebar.selectbox("Select Model", list(MODELS.keys()))

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Select an option", ("View Data", "Train Model", "Predict"))

# Navigation options
if option == "View Data":
    if dataset_name:
        display_data(dataset_name)
    else:
        st.error("Please select a valid dataset.")

elif option == "Train Model":
    if model_name and dataset_name:
        build_model(model_name, dataset_name)
    else:
        st.error("Please select both a valid model and dataset.")

elif option == "Predict":
    if model_name:
        show_prediction(model_name)
    else:
        st.error("Please select a valid model.")