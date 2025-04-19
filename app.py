"""
Main entry point for the GDP Nowcasting Streamlit Dashboard.

This script initializes the user interface, handles navigation between the
three main dashboard views, and loads pre-processed datasets for use across
modules. The dashboard supports:
1. Real-time nowcasting of GDP growth using benchmark and bridge models.
2. Retrospective 'Time Travel' simulation for backtesting model predictions.
3. Historical model comparison with and without the COVID-19 shock.

Key Modules:
- dashboard_layout.py: Contains UI and backend logic for each dashboard view
"""

import streamlit as st
import pandas as pd
import os
from Frontend.dashboard_layout import run_time_travel_dashboard, run_nowcast_dashboard, run_model_comparison_dashboard
from streamlit_option_menu import option_menu


# 1. Dashboard Configuration
st.set_page_config(
    page_title="GDP Nowcasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# 2. Data Loader with Caching 
@st.cache_data
def load_data():
    """
    Loads preprocessed datasets (linear and nonlinear indicators) from the Data directory.
    These are used by all models for forecasting GDP growth.

    Returns:
        df (pd.DataFrame): Stationary macro indicators for linear models (AR and ADL Bridge)
        df_nonlinear (pd.DataFrame): Feature set for non-linear models (RF and RF Bridge)
    """

    base_dir = os.path.dirname(__file__)
    bridge_df_path = os.path.join(base_dir, "Data", "bridge_df.csv")
    tree_df_path = os.path.join(base_dir, "Data", "tree_df.csv")
    df = pd.read_csv(bridge_df_path, parse_dates=["date"])
    df_nonlinear = pd.read_csv(tree_df_path, parse_dates=["date"])
    return df.sort_values(by="date"), df_nonlinear.sort_values(by="date")

# 3. Load Data Once (Session State Management) 
if 'df' not in st.session_state:
    df, df_nonlinear = load_data()
    st.session_state.df = df
    st.session_state.df_nonlinear = df_nonlinear
else:
    df = st.session_state.df
    df_nonlinear = st.session_state.df_nonlinear

# 4. Sidebar Navigation Panel
# Use session state to persist selected page
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Nowcast"

# Sidebar for page selection using option_menu
with st.sidebar:
    st.image("Frontend/logo.png", width=80)
    st.markdown("---")
    st.title("GDP Nowcasting Dashboard")
    st.markdown("##")

    selected_page = option_menu(
        menu_title=None,
        options=["Current Nowcast", "Time Travel", "Model Comparison"],
        icons=["graph-up", "clock-history", "bar-chart-line"],
        orientation="vertical"
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="color:#729FCF">
            <h4>Business Objective</h4>
            <p>
            This dashboard enables <b>GDP nowcasting</b> by leveraging high-frequency economic indicators.<br><br>
            It helps <b>policymakers, risk analysts</b>, and <b>investment strategists</b> monitor economic trends and react faster, bridging the gap caused by delayed official GDP releases.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

# 5. Page Router
if selected_page == "Current Nowcast":
    run_nowcast_dashboard(df, df_nonlinear)

if selected_page == "Time Travel":
    run_time_travel_dashboard(df, df_nonlinear)

if selected_page == "Model Comparison":
    run_model_comparison_dashboard()
