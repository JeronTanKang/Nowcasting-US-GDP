import streamlit as st
import pandas as pd
from Backend.model_ADL_bridge import model_bridge  # Import the function from model_ADL_bridge.py
from Backend.model_AR import model_AR  # Import the function from model_AR.py
from Frontend.dashboard_layout import display_dashboard  # Import the UI components from the frontend

st.set_page_config(
    page_title="GDP Nowcasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
    )

st.write("Loading data...")  # Debug marker
data = pd.read_csv("Data/bridge_df.csv")
st.write("Data loaded:", data.shape)

# Load default dataset
data_path = "Data/bridge_df.csv"
data = pd.read_csv(data_path)

# Let user select model
model_choice = st.selectbox("Select a Model for GDP Nowcasting", ("ADL Model", "AR Model"))

# Generate nowcast
if model_choice == "ADL Model":
    st.info("Running ADL model...")
    nowcast = model_bridge(data)
    model_name = "ADL Model"

elif model_choice == "AR Model":
    st.info("Running AR model...")
    nowcast = model_AR(data)
    model_name = "AR Model"

# Display results
display_dashboard(nowcast, model_name)