"""
This file defines the frontend logic for the GDP Nowcasting Streamlit Dashboard.

It contains the layout, user interactions, data loading, model execution, and visualization logic 
for 3 distinct dashboard views:

1. `run_nowcast_dashboard`: 
    - Displays real-time nowcasts for the current quarter based on incoming macroeconomic data.
    - Supports model comparison (AR, RF, ADL Bridge, RF Bridge).
    - Dynamically adjusts to current month of quarter to trigger multiple flash estimates using the bridge models.
    - Displays uncertainty intervals around forecasts using adjusted RMSFE values and skew/kurtosis statistics.

2. `run_time_travel_dashboard`: 
    - Simulates historical nowcast conditions using a “time travel” mode.
    - Lets users choose any historical quarter and model to visualize what forecasts looked like at that point in time.

3. `run_model_comparison_dashboard`:
    - Allows comparison of model accuracy across historical periods with and without COVID.
    - Visualizes forecast performance vs. actual GDP for each model type.

Helper Functions:
- `move_back_one_month`: Simulates monthly data unavailability by replacing the most recent non-NA value with NA.
- `nowcast_revisions`: Based on the current month, generates the appropriate dataframes (e.g., bridge_df_m1, tree_df_m2) used by bridge models.
- `create_labeled_forecast`: Formats model output with dynamic labels (`m1` to `m6`, `h1`, `h2`) and aligns them for plotting.
- `compute_interval_bounds`: Constructs forecast intervals using RMSFE, skewness, and kurtosis to reflect uncertainty bands around nowcasts.

All results are cached in `st.session_state` for efficiency and reusability.

Required files:
- Preprocessed macroeconomic datasets (`bridge_df.csv`, `tree_df.csv`)
- Model outputs generated within this script itself and stored in "..\Data\results_and_outputs" folder (`output_AR.csv`, `output_ADL_Bridge_combined.csv`, etc.)
- Monthly release metadata files for bridge models generated within this script itself and stored in "..\Data\results_and_outputs" folder (`forecast_months_ADL.json`, `forecast_months_RF.json`)
- Evaluation metrics (`rmsfe.csv`, `distribution.csv`, `rmsfe_dropped_covid.csv`, `distribution_no_covid.csv`, `row_error.csv`, `row_error_dropped_covid.csv`)

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
from Backend.forecast_evaluation import generate_oos_forecast
from Backend.model_AR import model_AR
from Backend.model_RF import model_RF
from Backend.model_ADL_bridge import model_ADL_bridge
from Backend.model_RF_bridge import model_RF_bridge
import json
import plotly.express as px
import scipy.stats as stats
from datetime import datetime



def run_time_travel_dashboard(df, df_nonlinear):
    """
    Interactive dashboard tab for time-travel GDP nowcasting.
    Allows users to select a past quarter and simulate model forecasts as if they were in that period.
    Forecasts are then compared against actual GDP outcomes for visual analysis.

    Args:
        df (pd.DataFrame): DataFrame with linear macroeconomic indicators.
        df_nonlinear (pd.DataFrame): DataFrame with non-linear macroeconomic indicators for RF and RF Bridge models.
    """

    # === Header ===
    st.header("🕰️ Time Travel GDP Growth Nowcasting")

    # === Step 1: Setup time window and user selection ===
    min_forecast_month = pd.Timestamp("2005-01-01")
    max_forecast_month = pd.Timestamp("2024-09-01")
    valid_dates = pd.date_range(start=min_forecast_month, end=max_forecast_month, freq='QS')
    valid_quarters = [(f"Q{(d.month - 1)//3 + 1}", d.year) for d in valid_dates]

    quarters = sorted(set(q for q, _ in valid_quarters))
    years = sorted(set(y for _, y in valid_quarters))

    col1, col2 = st.columns(2)
    with col1:
        selected_quarter = st.selectbox("Select Quarter", quarters)
    with col2:
        selected_year = st.selectbox("Select Year", years)

    # Model selector
    selected_model = st.multiselect(
        "Select Models",
        ["AR", "RF", "ADL Bridge", "RF Bridge"],
        default=["ADL Bridge"]
    )

    # Convert quarter selection to datetime object (end of quarter)
    quarter_map = {"Q1": "03-01", "Q2": "06-01", "Q3": "09-01", "Q4": "12-01"}
    selected_date = pd.to_datetime(f"{selected_year}-{quarter_map[selected_quarter]}")

    # === Step 2: Buttons to Generate/Reset Nowcast ===
    col3, col4 = st.columns([1, 1])
    with col3:
        if 'generate_clicked' not in st.session_state:
            st.session_state.generate_clicked = False
        if st.button("📤 Generate Nowcast", type="primary"):
            st.session_state.generate_clicked = True

    with col4:
        inner_col1, inner_col2 = st.columns([3, 1])  # Add padding space on the left
        with inner_col2:
            reset_clicked = st.button("🔁 Reset Nowcast", type="secondary")
    
    if reset_clicked:
        for key in ["forecast_df", "selected_quarter", "selected_year", "generate_clicked"]:
            st.session_state.pop(key, None)
        st.rerun()

    # === Step 3: Run Forecast if triggered ===
    if st.session_state.generate_clicked:

        if (
            'forecast_df' not in st.session_state or 
            st.session_state.selected_quarter != selected_quarter or 
            st.session_state.selected_year != selected_year
        ):
            with st.spinner('Nowcasting GDP growth...'):
                forecast_df = generate_oos_forecast(
                    df,
                    df_nonlinear,
                    time_travel_date=selected_date,
                    usage="single_period_forecast"
                )
                st.session_state.forecast_df = forecast_df
                st.session_state.selected_quarter = selected_quarter
                st.session_state.selected_year = selected_year
        else:
            forecast_df = st.session_state.forecast_df

        # === Step 4: Combine Actuals + Forecasts ===
        end_of_next_quarter = (selected_date + pd.offsets.QuarterEnd(2)).replace(day=1)
        actuals_df = df[
            (df['date'] >= pd.Timestamp("1995-04-01")) &
            (df['date'] <= end_of_next_quarter)
        ][['date', 'gdp_growth']].rename(columns={"gdp_growth": "actual_gdp_growth"})

        plot_df = pd.merge(actuals_df, forecast_df, on="date", how="left", suffixes=('_actual', '_forecast'))
        plot_df = plot_df.drop(columns=["actual_gdp_growth_forecast"])
        plot_df = plot_df.rename(columns={"actual_gdp_growth_actual": "actual_gdp_growth"})

        # === Step 5: Set NaNs for plot continuity (trailing unreleased forecasts) ===
        set_nan_indices = {
            'model_AR_h1': [-1],
            'model_ADL_bridge_m1': [-3],
            'model_ADL_bridge_m2': [-2],
            'model_ADL_bridge_m3': [-1],
            'model_RF_h1': [-1],
            'model_RF_bridge_m1': [-3],
            'model_RF_bridge_m2': [-2],
            'model_RF_bridge_m3': [-1]
        }
        for col, indices in set_nan_indices.items():
            for idx in indices:
                plot_df.loc[plot_df.tail(abs(idx)).head(1).index, col] = np.nan

        # === Step 6: Y-Axis Range Control ===
        min_y = plot_df.drop(columns=["date"]).min().min()
        max_y = plot_df.drop(columns=["date"]).max().max()
        pad = 0.1 * (max_y - min_y)
        y_range = st.slider("Select Y-Axis Range", int(min_y - pad), int(max_y + pad), (int(min_y - pad), int(max_y + pad)))

        # === Step 7: Plot Actuals + Forecasts ===
        if not plot_df.empty:
            st.subheader(f"GDP Growth Nowcast up to {selected_quarter} {selected_year}")

            # --- Initialize Plot ---
            fig = go.Figure()

            # Interpolate actual GDP growth line (for smoother line plotting)
            plot_df_actual = plot_df.copy()
            plot_df_actual['actual_gdp_growth'] = plot_df_actual['actual_gdp_growth'].interpolate(method='linear')

            # --- Add Actual GDP Growth ---
            fig.add_trace(go.Scatter(
                x=plot_df_actual['date'],
                y=plot_df_actual['actual_gdp_growth'],
                name="Actual GDP Growth",
                mode='lines',
                line=dict(color="#729FCF", width=2),
                hoverinfo='none'
            ))

            fig.add_trace(go.Scatter(
                x=plot_df['date'],
                y=plot_df['actual_gdp_growth'],
                name="Actual GDP Growth",
                mode='markers',
                marker=dict(color="#729FCF"),
                hovertemplate='%{x}: %{y:.2f}% <extra></extra>',
                showlegend=False
            ))

            # --- Forecast Column Mappings ---
            model_column_map = {
                "AR": ["model_AR_h1", "model_AR_h2"],
                "RF": ["model_RF_h1", "model_RF_h2"],
                "ADL Bridge": [f"model_ADL_bridge_m{i}" for i in range(1, 7)],
                "RF Bridge": [f"model_RF_bridge_m{i}" for i in range(1, 7)]
            }

            model_color_map = {
                "AR": "blue",
                "RF": "red",
                "ADL Bridge": "orange",
                "RF Bridge": "green"
            }

            # --- Add Forecasts by Model ---
            for model in selected_model:
                color = model_color_map.get(model)
                label = f"{model} Forecast (h1 - h2)" if model in ["AR", "RF"] else f"{model} Nowcast (m1 - m6)"
                combined_x, combined_y = [], []
                is_first_trace = True

                for col in model_column_map[model]:
                    forecast_vals = plot_df[col].dropna()

                    # Determine relevant dates (last 6 for bridge models, specific for benchmarks)
                    if model in ["ADL Bridge", "RF Bridge"]:
                        combined_x += plot_df['date'].tail(6).tolist()
                    elif model in ["AR", "RF"]:
                        combined_x += plot_df['date'].tail(4).head(1).tolist() + plot_df['date'].tail(1).tolist()

                    combined_y += forecast_vals.tolist()

                    # Plot individual forecast traces
                    fig.add_trace(go.Scatter(
                        x=plot_df['date'],
                        y=plot_df[col],
                        mode='lines+markers',
                        name=label if is_first_trace else None,
                        line=dict(color=color),
                        marker=dict(color=color),
                        hovertemplate='%{x}: %{y:.2f}% <extra></extra>',
                        showlegend=is_first_trace
                    ))
                    is_first_trace = False

                # Add a solid connecting line
                fig.add_trace(go.Scatter(
                    x=combined_x,
                    y=combined_y,
                    mode='lines',
                    name=f"{model} Continuous Forecast",
                    line=dict(color=color, width=2),
                    showlegend=False
                ))

            # --- Forecast Period Highlighting ---
            forecast_start_date = plot_df['date'].iloc[-6]  # Start of forecast period

            fig.add_trace(go.Scatter(
                x=[forecast_start_date, forecast_start_date],
                y=[min_y, max_y],
                mode='lines',
                line=dict(color="gray", dash="dash"),
                showlegend=False
            ))

            fig.add_annotation(
                x=forecast_start_date,
                y=plot_df['actual_gdp_growth'].max(),
                text="Start of Forecast Period",
                showarrow=True,
                arrowhead=2,
                arrowcolor="gray",
                ax=-50,
                ay=-40,
                font=dict(color="black", size=13)
            )

            fig.add_vrect(
                x0=forecast_start_date,
                x1=plot_df['date'].max() + pd.DateOffset(months=1),
                fillcolor="rgba(23, 50, 77, 0.2)",
                line_width=0,
                name="Forecast Period"
            )

            # --- Final Plot Styling ---
            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type='date',
                    tickformat="%b %Y"
                ),
                yaxis=dict(
                    range=y_range,
                    tickformat="%.2f"
                ),
                legend_title="Legend",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

            # === Step 8: Extract and Organize Forecast Data for Selected and Next Quarter ===

            forecast_period_df = forecast_df.copy()

            # Drop the trailing values for benchmark and bridge models to avoid partial forecasts
            forecast_period_df.loc[forecast_period_df.tail(1).index, 'model_AR_h1'] = np.nan
            forecast_period_df.loc[forecast_period_df.tail(3).head(1).index, 'model_ADL_bridge_m1'] = np.nan
            forecast_period_df.loc[forecast_period_df.tail(2).head(1).index, 'model_ADL_bridge_m2'] = np.nan
            forecast_period_df.loc[forecast_period_df.tail(1).index, 'model_ADL_bridge_m3'] = np.nan
            forecast_period_df.loc[forecast_period_df.tail(1).index, 'model_RF_h1'] = np.nan
            forecast_period_df.loc[forecast_period_df.tail(3).head(1).index, 'model_RF_bridge_m1'] = np.nan
            forecast_period_df.loc[forecast_period_df.tail(2).head(1).index, 'model_RF_bridge_m2'] = np.nan
            forecast_period_df.loc[forecast_period_df.tail(1).index, 'model_RF_bridge_m3'] = np.nan

            # Initialize forecast data storage
            forecast_period_data = {}

            # Map quarter strings to integers and vice versa
            quarter_map_num = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
            quarter_map_str = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}

            selected_quarter_num = quarter_map_num.get(selected_quarter)
            next_quarter_num = (selected_quarter_num % 4) + 1
            next_quarter = quarter_map_str.get(next_quarter_num)

            def drop_nans(array):
                """Helper function to remove NaNs from a list-like array."""
                return [x for x in array if pd.notna(x)]

            # Extract forecast data for the selected quarter (e.g., Q1)
            selected_quarter_data = forecast_period_df[forecast_period_df['date'].dt.quarter == selected_quarter_num]
            forecast_period_data[selected_quarter] = {
                'actual_gdp_growth': selected_quarter_data['actual_gdp_growth'].dropna().values[0] if not selected_quarter_data['actual_gdp_growth'].dropna().empty else None,
                'ADL_bridge': drop_nans(selected_quarter_data[['model_ADL_bridge_m1', 'model_ADL_bridge_m2', 'model_ADL_bridge_m3']].values.flatten()),
                'RF_bridge': drop_nans(selected_quarter_data[['model_RF_bridge_m1', 'model_RF_bridge_m2', 'model_RF_bridge_m3']].values.flatten()),
                'AR': selected_quarter_data['model_AR_h1'].dropna().values[0] if not selected_quarter_data['model_AR_h1'].dropna().empty else None,
                'RF': selected_quarter_data['model_RF_h1'].dropna().values[0] if not selected_quarter_data['model_RF_h1'].dropna().empty else None
            }

            # Extract forecast data for the next quarter (e.g., Q2)
            next_quarter_data = forecast_period_df[forecast_period_df['date'].dt.quarter == next_quarter_num]
            forecast_period_data[next_quarter] = {
                'actual_gdp_growth': next_quarter_data['actual_gdp_growth'].dropna().values[0] if not next_quarter_data['actual_gdp_growth'].dropna().empty else None,
                'ADL_bridge': drop_nans(next_quarter_data[['model_ADL_bridge_m4', 'model_ADL_bridge_m5', 'model_ADL_bridge_m6']].values.flatten()),
                'RF_bridge': drop_nans(next_quarter_data[['model_RF_bridge_m4', 'model_RF_bridge_m5', 'model_RF_bridge_m6']].values.flatten()),
                'AR': next_quarter_data['model_AR_h2'].dropna().values[0] if not next_quarter_data['model_AR_h2'].dropna().empty else None,
                'RF': next_quarter_data['model_RF_h2'].dropna().values[0] if not next_quarter_data['model_RF_h2'].dropna().empty else None
            }

            # === Step 9: Side-by-side Bar Charts for Each Quarter ===
            col5, col6 = st.columns(2)

            # --- Bar Chart for Selected Quarter ---
            with col5:
                st.subheader(f"Model Comparison for {selected_quarter}")
                fig_q = go.Figure()
                fig_q.add_trace(go.Bar(
                    x=['Actual GDP Growth', 'ADL Bridge (m3)', 'RF Bridge (m3)', 'AR (h1)', 'RF (h1)'],
                    y=[
                        forecast_period_data[selected_quarter]['actual_gdp_growth'],
                        forecast_period_data[selected_quarter]['ADL_bridge'][2],
                        forecast_period_data[selected_quarter]['RF_bridge'][2],
                        forecast_period_data[selected_quarter]['AR'],
                        forecast_period_data[selected_quarter]['RF']
                    ],
                    marker=dict(color=[
                        "gray", "#17375E", "#254F87", "#7F9DBE", "#B3CCE0"
                    ]),
                    hovertemplate='%{x}: %{y:.2f}% <extra></extra>'
                ))
                fig_q.update_layout(
                    xaxis_title="Models",
                    yaxis_title="GDP Growth (%)",
                    plot_bgcolor="#FAFAFA",
                    paper_bgcolor="#FAFAFA",
                    margin=dict(t=40, l=40, r=40, b=40),
                    showlegend=False
                )
                st.plotly_chart(fig_q, use_container_width=True)

            # --- Bar Chart for Next Quarter ---
            with col6:
                st.subheader(f"Model Comparison for {next_quarter}")
                fig_next_q = go.Figure()
                fig_next_q.add_trace(go.Bar(
                    x=['Actual GDP Growth', 'ADL Bridge (m6)', 'RF Bridge (m6)', 'AR (h2)', 'RF (h2)'],
                    y=[
                        forecast_period_data[next_quarter]['actual_gdp_growth'],
                        forecast_period_data[next_quarter]['ADL_bridge'][2],
                        forecast_period_data[next_quarter]['RF_bridge'][2],
                        forecast_period_data[next_quarter]['AR'],
                        forecast_period_data[next_quarter]['RF']
                    ],
                    marker=dict(color=[
                        "gray", "#17375E", "#254F87", "#7F9DBE", "#B3CCE0"
                    ]),
                    hovertemplate='%{x}: %{y:.2f}% <extra></extra>'
                ))
                fig_next_q.update_layout(
                    xaxis_title="Models",
                    yaxis_title="GDP Growth (%)",
                    plot_bgcolor="#FAFAFA",
                    paper_bgcolor="#FAFAFA",
                    margin=dict(t=40, l=40, r=40, b=40),
                    showlegend=False
                )
                st.plotly_chart(fig_next_q, use_container_width=True)

        else:
            st.warning("No forecast data available for the selected time travel period.")




def run_nowcast_dashboard(df, df_nonlinear):
    """
    This function renders the real-time GDP nowcasting dashboard.
    Includes current nowcast forecasts, interval plots, and model comparison tables.

    Args:
        df (pd.DataFrame): Preprocessed linear macroeconomic indicators.
        df_nonlinear (pd.DataFrame): Preprocessed nonlinear indicators for tree models.
    """

    # === Step 1: Setup and Current Quarter Metadata ===
    # Get current date
    now = datetime.now()

    # Determine current quarter and year
    current_year = now.year
    current_month = now.month
    current_quarter = (current_month - 1) // 3 + 1

    # Get current month name (e.g., "April")
    month_name = now.strftime("%B")

    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend'))
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'results_and_outputs'))


    # === Step 2: Helper Functions ===
    def move_back_one_month(df):
        df = df.copy()
        for col in df.columns:
            if col.startswith("date") or "gdp" in col.lower():
                continue
            last_valid_index = df[col][df[col].notna()].index.max()
            if last_valid_index is not None:
                df.at[last_valid_index, col] = np.nan
        return df
    
    def nowcast_revisions(data_dir, df, df_nonlinear):
        # Determine current month in quarter
        month = datetime.now().month
        month_position = {1: "first", 2: "second", 3: "third",
                        4: "first", 5: "second", 6: "third",
                        7: "first", 8: "second", 9: "third",
                        10: "first", 11: "second", 12: "third"}
        pos = month_position[month]

        # Prepare and save required CSVs
        if pos == "first":
            df.to_csv(os.path.join(data_dir, "bridge_df.csv"), index=False)
            df_nonlinear.to_csv(os.path.join(data_dir, "tree_df.csv"), index=False)
            
        if pos == "second":
            # Generate *_m2
            bridge_df_m2 = move_back_one_month(df)
            bridge_df_m2.to_csv(os.path.join(data_dir, "bridge_df_m2.csv"), index=False)

            tree_df_m2 = move_back_one_month(df_nonlinear)
            tree_df_m2.to_csv(os.path.join(data_dir, "tree_df_m2.csv"), index=False)

        elif pos == "third":
            # Generate *_m2
            bridge_df_m2 = move_back_one_month(df)
            bridge_df_m2.to_csv(os.path.join(data_dir, "bridge_df_m2.csv"), index=False)

            tree_df_m2 = move_back_one_month(df_nonlinear)
            tree_df_m2.to_csv(os.path.join(data_dir, "tree_df_m2.csv"), index=False)

            # Generate *_m1 from *_m2
            bridge_df_m1 = move_back_one_month(bridge_df_m2)
            bridge_df_m1.to_csv(os.path.join(data_dir, "bridge_df_m1.csv"), index=False)

            tree_df_m1 = move_back_one_month(tree_df_m2)
            tree_df_m1.to_csv(os.path.join(data_dir, "tree_df_m1.csv"), index=False)
    

    # === Step 3: Page Header ===
    st.header(f"📈 Current GDP Growth Nowcast for {current_year}:Q{current_quarter} (As of {month_name} {current_year})")

    st.markdown("""
        **GDP Nowcast** is best viewed as an estimate of real GDP growth based on available economic data for the current quarter and forecasts of explanatory variables for the remaining term.
    """)

    # Model filter
    selected_models = st.multiselect(
        "Select Models to Display",
        options=["AR", "RF", "ADL Bridge", "RF Bridge"],
        default=["ADL Bridge"],
        help="Choose which models you want to display on the nowcast plot."
    )


    # === Step 4: Generate Model Forecasts ===
    if "nowcast_forecast_df" not in st.session_state:

        try:
            model_AR_nowcast = model_AR(df)
            model_RF_nowcast = model_RF(df_nonlinear)
        except Exception as e:
            st.error(f"❌ Failed to run one of the model functions: {e}")

        # Save outputs from benchmark models
        model_AR_nowcast.to_csv(os.path.join(data_dir, "output_AR.csv"), index=False)
        df_ar = pd.read_csv(os.path.join(data_dir, "output_AR.csv"))

        model_RF_nowcast.to_csv(os.path.join(data_dir, "output_RF.csv"), index=False)
        df_rf = pd.read_csv(os.path.join(data_dir, "output_RF.csv"))

        # Helper to label forecast results (e.g., m1/m4, m2/m5)
        def create_labeled_forecast(df, label1, label2, offset1, offset2):
            df = df.copy()

            if len(df) == 3:
                df0 = pd.DataFrame({
                    "date": [pd.to_datetime(df.loc[0, "date"])],
                    "Nowcasted_GDP_Growth": [df.loc[0, "Nowcasted_GDP_Growth"]],
                    "label": ["m3"]  # You can adjust this label if needed
                })
                df1 = pd.DataFrame({
                    "date": [pd.to_datetime(df.loc[1, "date"]) + pd.DateOffset(months=offset1)],
                    "Nowcasted_GDP_Growth": [df.loc[1, "Nowcasted_GDP_Growth"]],
                    "label": [label1]
                })
                df2 = pd.DataFrame({
                    "date": [pd.to_datetime(df.loc[2, "date"]) + pd.DateOffset(months=offset2)],
                    "Nowcasted_GDP_Growth": [df.loc[2, "Nowcasted_GDP_Growth"]],
                    "label": [label2]
                })
                return pd.concat([df0, df1, df2], ignore_index=True)
            
            else:
                df1 = pd.DataFrame({
                    "date": [pd.to_datetime(df.loc[0, "date"]) + pd.DateOffset(months=offset1)],
                    "Nowcasted_GDP_Growth": [df.loc[0, "Nowcasted_GDP_Growth"]],
                    "label": [label1]
                })
                df2 = pd.DataFrame({
                    "date": [pd.to_datetime(df.loc[1, "date"]) + pd.DateOffset(months=offset2)],
                    "Nowcasted_GDP_Growth": [df.loc[1, "Nowcasted_GDP_Growth"]],
                    "label": [label2]
                })
                return pd.concat([df1, df2], ignore_index=True)

        # Detect current month of quarter
        month = datetime.now().month
        month_position = {1: "first", 2: "second", 3: "third",
                        4: "first", 5: "second", 6: "third",
                        7: "first", 8: "second", 9: "third",
                        10: "first", 11: "second", 12: "third"}
        quarter_pos = month_position[month]

        # Prepare bridge dataframes for simulation (drop latest data)
        nowcast_revisions(data_dir, df, df_nonlinear)

        # File mapping for each stage of quarter (used to construct forecasts)
        adl_mapping = {
            "first": [("bridge_df.csv", "m1", "m4", 0, 0)],
            "second": [("bridge_df_m2.csv", "m1", "m4", 0, 0),
                    ("bridge_df.csv", "m2", "m5", 1, 1)],
            "third": [("bridge_df_m1.csv", "m1", "m4", 0, 0),
                    ("bridge_df_m2.csv", "m2", "m5", 1, 1),
                    ("bridge_df.csv", "m3", "m6", 2, 2)]
        }

        rf_mapping = {
            "first": [("tree_df.csv", "m1", "m4", 0, 0)],
            "second": [("tree_df_m2.csv", "m1", "m4", 0, 0),
                    ("tree_df.csv", "m2", "m5", 1, 1)],
            "third": [("tree_df_m1.csv", "m1", "m4", 0, 0),
                    ("tree_df_m2.csv", "m2", "m5", 1, 1),
                    ("tree_df.csv", "m3", "m6", 2, 2)]
        }

        # === Run ADL Bridge model ===
        adl_dfs = []
        for file, l1, l2, o1, o2 in adl_mapping[quarter_pos]:
            df = pd.read_csv(os.path.join(data_dir, file))
            forecast_df = model_ADL_bridge(df)
            labeled = create_labeled_forecast(forecast_df, l1, l2, o1, o2)
            adl_dfs.append(labeled)

        df_adl_combined = pd.concat(adl_dfs, ignore_index=True).sort_values("date")
        df_adl_combined.to_csv(os.path.join(data_dir, "output_ADL_Bridge_combined.csv"), index=False)
        df_adl = pd.read_csv(os.path.join(data_dir, "output_ADL_Bridge_combined.csv"))

        # === Run RF Bridge model ===
        rf_dfs = []
        for file, l1, l2, o1, o2 in rf_mapping[quarter_pos]:
            df = pd.read_csv(os.path.join(data_dir, file))
            forecast_df = model_RF_bridge(df)
            labeled = create_labeled_forecast(forecast_df, l1, l2, o1, o2)
            rf_dfs.append(labeled)

        df_rf_combined = pd.concat(rf_dfs, ignore_index=True).sort_values("date")
        df_rf_combined.to_csv(os.path.join(data_dir, "output_RF_Bridge_combined.csv"), index=False)
        df_rf_bridge = pd.read_csv(os.path.join(data_dir, "output_RF_Bridge_combined.csv"))


        # === Step 5: Combine All Forecast Outputs ===
        # Add model and label columns if not present
        df_ar['Model'] = "AR"
        df_rf['Model'] = "RF"

        # Convert date column to datetime (if not already)
        df_ar['date'] = pd.to_datetime(df_ar['date'])
        df_rf['date'] = pd.to_datetime(df_rf['date'])

        # Label based on chronological order
        df_ar = df_ar.sort_values('date').reset_index(drop=True)
        df_rf = df_rf.sort_values('date').reset_index(drop=True)

        df_ar['label'] = ['h1', 'h2']
        df_rf['label'] = ['h1', 'h2']

        df_adl['Model'] = "ADL Bridge"
        df_rf_bridge['Model'] = "RF Bridge"

        # Combine all forecasts
        df_combined = pd.concat([df_ar, df_rf, df_adl, df_rf_bridge], ignore_index=True)
        df_combined['date'] = pd.to_datetime(df_combined['date'])

        st.session_state.nowcast_forecast_df = df_combined
        st.session_state.nowcast_last_run = pd.Timestamp.now()

    else:
        df_combined = st.session_state.nowcast_forecast_df

    #print(f"Combined DataFrame:\n{df_combined}")

    # === Step 6: Compute and Merge Forecast Interval Bounds ===
    def compute_interval_bounds(rmsfe_choice, df_combined, data_dir):
        """
        Computes upper and lower prediction intervals for each model's forecasts using Cornish-Fisher quantile adjustment.

        Parameters:
            rmsfe_choice (str): Toggle option for whether to exclude COVID period.
            df_combined (pd.DataFrame): Combined forecast output.
            data_dir (str): Path to directory where RMSFE and distribution files are stored.

        Returns:
            pd.DataFrame: Forecasts with corresponding upper and lower bounds merged in.
        """
        # Load RMSFE and distribution data
        if rmsfe_choice == "Normal (All Years)":
            rmsfe_df = pd.read_csv(os.path.join(data_dir, "rmsfe.csv"))
            dist_df = pd.read_csv(os.path.join(data_dir, "distribution.csv"))
        else:
            rmsfe_df = pd.read_csv(os.path.join(data_dir, "rmsfe_dropped_covid.csv"))
            dist_df = pd.read_csv(os.path.join(data_dir, "distribution_no_covid.csv"))

        rmsfe_df = rmsfe_df.melt(var_name="Model_Label", value_name="RMSFE")
        rmsfe_df["Model_Label"] = rmsfe_df["Model_Label"].str.replace("RMSFE_", "", regex=False)
        dist_df.columns = ["Model_Label", "Skewness", "Kurtosis"]
        interval_df = pd.merge(rmsfe_df, dist_df, on="Model_Label")

        # Cornish-Fisher 95% quantile adjustment for non-normal error distribution
        z_95 = stats.norm.ppf(0.975)
        interval_df["Adj_Upper"] = z_95 + \
            (1/6)*(z_95**2 - 1)*interval_df["Skewness"] + \
            (1/24)*(z_95**3 - 3*z_95)*interval_df["Kurtosis"] - \
            (1/36)*(2*z_95**3 - 5*z_95)*interval_df["Skewness"]**2
        interval_df["Adj_Lower"] = -interval_df["Adj_Upper"]

        # Calculate upper and lower bounds
        interval_df["UB"] = interval_df["RMSFE"] * interval_df["Adj_Upper"]
        interval_df["LB"] = interval_df["RMSFE"] * interval_df["Adj_Lower"]

        interval_df["Model"] = interval_df["Model_Label"].str.extract(r"model_(.*)_")[0].replace({
            "AR": "AR", "RF": "RF", "ADL_bridge": "ADL Bridge", "RF_bridge": "RF Bridge"
        })
        interval_df["label"] = interval_df["Model_Label"].str.extract(r"_([^_]+)$")

        # Merge bounds with forecast outputs
        return pd.merge(df_combined, interval_df, on=["Model", "label"], how="left")

    # === Toggle for user to select RMSFE source ===
    rmsfe_choice  = st.radio(
        "Forecast Interval Settings:",
        ["Exclude COVID Period", "Normal (All Years)"],
        index=0,
        horizontal=True
    )

    # Compute final interval-enhanced forecast table
    interval_df_plot = compute_interval_bounds(rmsfe_choice, df_combined, data_dir)

    # === Step 7: Merge Actual GDP Data and Prepare Plotting DataFrame ===
    # --- Get actual GDP data up to Q1 2025 ---
    df["date"] = pd.to_datetime(df["date"])
    actuals = df[df['date'] <= pd.to_datetime("2025-01-01")][['date', 'gdp_growth']]
    actuals = actuals.rename(columns={'gdp_growth': 'Actual_GDP_Growth'})

    actuals['Actual_GDP_Growth_interp'] = actuals['Actual_GDP_Growth'].interpolate(method='linear')

    # --- Merge actuals + forecasts for unified plot ---
    plot_df = pd.merge(actuals, df_combined, on="date", how="outer").sort_values("date")

    print(f"Plot DataFrame:\n{plot_df}")

    print(f"Interval DataFrame for Plotting:\n{interval_df_plot}")
    interval_df_plot.to_csv(os.path.join(data_dir, "interval_df_plot.csv"), index=False)

    # --- Timeframe toggle ---
    timeframes = ["6M", "YTD", "1Y", "5Y", "Max"]
    cols = st.columns([1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1])

    if "selected_timeframe" not in st.session_state:
        st.session_state.selected_timeframe = "Max"

    for i, label in enumerate(timeframes):
        if cols[i].button(label):
            st.session_state.selected_timeframe = label

    timeframe = st.session_state.selected_timeframe

    last_date = plot_df['date'].max()
    if timeframe == "6M":
        start_date = last_date - pd.DateOffset(months=6)
    elif timeframe == "YTD":
        start_date = pd.Timestamp(f"{last_date.year}-01-01")
    elif timeframe == "1Y":
        start_date = last_date - pd.DateOffset(years=1)
    elif timeframe == "5Y":
        start_date = last_date - pd.DateOffset(years=5)
    else:
        start_date = plot_df['date'].min()

    # Filter the dataframe
    plot_df = plot_df[plot_df['date'] >= start_date]
    actuals = actuals[actuals['date'] >= start_date]

    # --- Create a slider for the Y-axis range ---
    # Exclude 'date' and 'actual_gdp_growth' columns
    plot_columns = plot_df[["Actual_GDP_Growth", "Nowcasted_GDP_Growth"]]

    # Compute overall min and max across all forecast columns
    min_y = plot_columns.min().min()
    max_y = plot_columns.max().max()

    min_y = min_y - 5
    max_y = max_y + 5

    # --- Plot ---
    if not plot_df.empty:
        fig = go.Figure()

        # Actual GDP Growth (interpolated line)
        fig.add_trace(go.Scatter(
            x=actuals['date'],
            y=actuals['Actual_GDP_Growth_interp'],
            name="Actual GDP Growth",
            mode='lines',
            line=dict(color="#729FCF", width=2),
            hoverinfo='none'
        ))

        # Actual GDP Growth (markers)
        fig.add_trace(go.Scatter(
            x=actuals['date'],
            y=actuals['Actual_GDP_Growth'],
            name="Actual GDP Growth",
            mode='markers',
            marker=dict(color="#729FCF"),
            hovertemplate='%{x}: %{y:.2f}% <extra></extra>',
            showlegend=False
        ))

        # Forecasts
        color_map = {
            "AR": "#1f77b4",         # blue
            "RF": "#d62728",         # red
            "ADL Bridge": "#ff7f0e", # orange
            "RF Bridge": "#2ca02c"   # green
        }

        hex_to_rgba = lambda hex_color, alpha: f"rgba({int(hex_color[1:3], 16)}, {int(hex_color[3:5], 16)}, {int(hex_color[5:7], 16)}, {alpha})"

        for model in df_combined['Model'].unique():
            if model not in selected_models:
                continue  # Skip if not selected

            model_df = interval_df_plot[interval_df_plot['Model'] == model].copy()
            model_df = model_df.sort_values('date')

            model_color = color_map.get(model, "gray")
            band_color = hex_to_rgba(model_color, 0.15) if model_color.startswith('#') else 'rgba(0,0,0,0.15)'

            forecast_name = f"{model} Forecast (h1 - h2)" if model in ["AR", "RF"] else f"{model} Forecast (m1 - m6)"

            is_first_trace = True
            upper_band = []
            lower_band = []
            forecast_dates = []
            forecast_points = []

            for _, row in model_df.iterrows():

                forecast = row['Nowcasted_GDP_Growth']
                lb = forecast + row["LB"]
                ub = forecast + row["UB"]

                forecast_dates.append(row['date'])
                forecast_points.append(forecast)
                lower_band.append(lb)
                upper_band.append(ub)

            # Forecast point
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_points,
                mode='markers+lines',
                name=forecast_name,
                marker=dict(color=model_color, size=8),
                line=dict(color=model_color, width=2),
                hovertemplate='%{x}: %{y:.2f}% <extra></extra>',
                showlegend=True
            ))

            fig.add_trace(go.Scatter(
                x=forecast_dates + forecast_dates[::-1],
                y=upper_band + lower_band[::-1],
                fill='toself',
                fillcolor=hex_to_rgba(model_color, 0.15),
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                name=f"{model} Forecast Interval",
                showlegend=False
            ))

        # --- Highlight forecast period ---
        forecast_start_date = pd.Timestamp("2025-01-01")  # bridge model Q1 2025 forecast

        # Annotation for vertical line
        fig.add_annotation(
            x=forecast_start_date,
            y=max_y,
            text="Start of Forecast Period",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gray",
            ax=-50,
            ay=-40,
            font=dict(color="black", size=13)
        )

        # Shaded forecast region
        fig.add_vrect(
            x0=forecast_start_date,
            x1=plot_df['date'].max() + pd.DateOffset(months=1),
            fillcolor="rgba(23, 50, 77, 0.2)",  # dark navy transparent
            line_width=0,
            name="Forecast Period"
        )

        fig.update_layout(
            height=700,
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date',
                autorange=True,  # Auto-adjust range based on zoom level
                tickformat="%b %Y",  # Customize the date format for more granular ticks
            ),
            yaxis=dict(
                range=[min_y - max(lower_band)-5, max_y + max(upper_band)],  # Auto-adjust range for the y-axis as well
                tickformat="%.2f",  # Ensure two decimal places for granularity
            ),
            xaxis_title="Date",
            yaxis_title="GDP Growth (%)",
            template="plotly_white",
            legend_title="Legend"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div style="text-align: right; font-size: 0.85rem; color: #6c757d; margin-top: -15px;">
                💡 <i>Note:</i> The nowcast is based on the most recent data available and is subject to change as new data becomes available.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
            st.warning("No forecast data available for the selected time travel period.")

    col1, col2 = st.columns([1, 2])  # Wider column for the forecast table

    # === Left Column: Model Comparison Table ===
    with col1:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        st.markdown(f"### Q{current_quarter} {current_year} GDP Nowcast Comparison")

        # Determine label based on month position in quarter
        month_position = {1: "m1", 2: "m2", 3: "m3",
                        4: "m1", 5: "m2", 6: "m3",
                        7: "m1", 8: "m2", 9: "m3",
                        10: "m1", 11: "m2", 12: "m3"}

        month_label = month_position[current_month]

        # Convert date if needed
        interval_df_plot["date"] = pd.to_datetime(interval_df_plot["date"])
        current_quarter_start = pd.Timestamp(f"{current_year}-{(current_quarter - 1) * 3 + 1:02d}-01")

        def get_model_value(model, label):
            try:
                val = interval_df_plot[
                    (interval_df_plot["Model"] == model) &
                    (interval_df_plot["label"] == label) &
                    (interval_df_plot["date"] == current_quarter_start)
                ]["Nowcasted_GDP_Growth"].values[0]
                return round(val, 3)
            except IndexError:
                return None

        nowcast_values = {
            "AR Benchmark": get_model_value("AR", "h1"),
            "ADL Bridge": get_model_value("ADL Bridge", month_label),
            "RF Benchmark": get_model_value("RF", "h1"),
            "RF Bridge": get_model_value("RF Bridge", month_label)
        }

        nowcast_data = {
            "Model": list(nowcast_values.keys()),
            f"{current_year}:Q{current_quarter} Nowcast (SAAR)": list(nowcast_values.values())
        }

        df_nowcast = pd.DataFrame(nowcast_data)

        # Format to bold model names and color negatives
        def format_nowcast(val):
            color = 'red' if val < 0 else 'black'
            return f'<span style="color:{color}">{val:.2f}</span>'

        df_nowcast_styled = (
            df_nowcast.style
            .format({f'{current_year}:Q{current_quarter} Nowcast (SAAR)': format_nowcast}, escape="html")
            .set_table_styles([
                {'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', '#17375E'), ('color', 'white')]},
                {'selector': 'td', 'props': [('text-align', 'center')]}
            ])
            .hide(axis="index")
        )

        st.write(df_nowcast_styled.to_html(escape=False), unsafe_allow_html=True)

        st.caption(f"""
            1. SAAR: Seasonally adjusted annual rate 
            3. Model Forecasts based on data up to 31 {month_name} {current_year}
        """)


    # === Right Column: Forecast Month Table ===
    with col2:
        
        model_RF_bridge(df_nonlinear)
        model_ADL_bridge(df)

        json_paths = {
            "ADL Bridge": os.path.join(data_dir, "forecast_months_ADL.json"),
            "RF Bridge": os.path.join(data_dir, "forecast_months_RF.json")
        }

        # Sidebar or below plot toggle
        selected_model_for_table = st.radio(
            "Select Model for Forecast Table",  # Provide a meaningful label
            ["ADL Bridge", "RF Bridge"],
            horizontal=True,
            label_visibility="collapsed" 
        )

        # Load the selected file
        with open(json_paths[selected_model_for_table], 'r') as f:
            forecast_dict = json.load(f)

        # Finalized predictor lists
        finalized_features = {
            "ADL Bridge": [
                "Industrial_Production_lag1", "New_Orders_Durable_Goods", "junk_bond_spread",
                "Construction_Spending", "Housing_Starts", "Unemployment",
                "Trade_Balance", "Capacity_Utilization",
            ],
            "RF Bridge": [
                "Unemployment", "Capacity_Utilization", "Nonfarm_Payrolls",
                "New_Orders_Durable_Goods", "Housing_Starts",
                "New_Home_Sales"
            ]
        }

        # Build filtered table
        filtered_table_data = []
        for var in finalized_features[selected_model_for_table]:
            if var in forecast_dict and forecast_dict[var]:
                timestamps = [pd.to_datetime(ts) for ts in forecast_dict[var]]
                start_date = min(timestamps)
                end_date = max(timestamps)
                last_release = start_date - pd.DateOffset(months=1)

                filtered_table_data.append({
                    "Variable": var,
                    "Last Release Month": last_release.strftime('%Y-%m'),
                    "Start Forecast Month": start_date.strftime('%Y-%m'),
                    "End Forecast Month": end_date.strftime('%Y-%m')
                })

        # Display clean single table
        if filtered_table_data:
            df_filtered = pd.DataFrame(filtered_table_data).sort_values(by="Start Forecast Month").reset_index(drop=True)

            # Only apply renaming if ADL Bridge is selected
            if selected_model_for_table == "ADL Bridge":
                df_filtered["Variable"] = df_filtered["Variable"].replace({
                    "Industrial_Production_lag1": "Industrial_Production"
                })

            # Start index from 1
            df_filtered.index = df_filtered.index + 1

            st.markdown(f"### 📑 Monthly Variables Release Schedule for {selected_model_for_table}")
            st.dataframe(df_filtered, use_container_width=True)
        else:
            st.info("No forecasted variables available for this model.")



def run_model_comparison_dashboard():
    """
    This function renders compares actual GDP growth to model-predicted GDP growth
    over historical periods, including visualizations with and without the COVID-19 period.

    Models included:
    - AR Benchmark
    - ADL Bridge Model
    - RF Benchmark
    - RF Bridge Model
    """

    st.header("Model Performance Comparison")
    st.write("Compare GDP nowcasting results across different models against past actual GDP values.")

    model_types = st.multiselect("Select Model Types", ["AR", "ADL Bridge", "RF", "RF Bridge"], default=["ADL Bridge", "RF Bridge"])

    # === Step 1: Load Evaluation Datasets ===
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "../Data/results_and_outputs/row_error.csv")
    data = pd.read_csv(data_path)
    data_path_no_covid = os.path.join(base_dir, "../Data/results_and_outputs/row_error_dropped_covid.csv")
    data_no_covid = pd.read_csv(data_path_no_covid)

    data['date'] = pd.to_datetime(data['date'])
    data_no_covid['date'] = pd.to_datetime(data_no_covid['date'])

    # Only retain key evaluation columns
    df_eval = data[['date', 'actual_gdp_growth', 'model_AR_h1', 'model_ADL_bridge_m3', 'model_RF_h1', 'model_RF_bridge_m3']] 
    df_eval = df_eval.dropna()

    df_eval_no_covid = data_no_covid[['date', 'actual_gdp_growth', 'model_AR_h1', 'model_ADL_bridge_m3', 'model_RF_h1', 'model_RF_bridge_m3']]
    df_eval_no_covid = df_eval_no_covid.dropna()    

    # Plot Actual GDP Growth
    st.subheader("Actual GDP Growth Over Time")
    fig_actual = px.line(df_eval, 
                         x='date', 
                         y='actual_gdp_growth', 
                         labels={'date': 'Date', 'actual_gdp_growth': 'GDP Growth (%)'})
    fig_actual.update_traces(name="Actual GDP Growth", 
                             showlegend=True, 
                             hovertemplate='%{x}: %{y:.2f}% <extra></extra>') 
    fig_actual.update_layout(legend_title_text="Legend", 
                             template="plotly_white")  
    st.plotly_chart(fig_actual, use_container_width=True)

    # Mapping model names to column names
    model_map = {
        "AR": 'model_AR_h1',
        "ADL Bridge": 'model_ADL_bridge_m3',
        "RF": 'model_RF_h1',
        "RF Bridge": 'model_RF_bridge_m3'
    }

    if model_types:
        # --- Full Period Plot ---
        st.subheader("GDP Growth Comparison: Actual vs Predicted by Models")

        combined_df = df_eval[['date', 'actual_gdp_growth']].rename(columns={'actual_gdp_growth': 'Actual GDP Growth'})
        for model in model_types:
            combined_df[model] = df_eval[model_map[model]]

        fig_compare = px.line(combined_df, 
                              x='date', 
                              y=combined_df.columns[1:], 
                              labels={'date': 'Date','value': 'GDP Growth (%)'},)
        
        fig_compare.update_traces(
            hovertemplate='%{x}: %{y:.2f}% <extra></extra>'
        )

        fig_compare.update_layout(
            shapes=[
                dict(
                    type="rect",
                    x0="2019-10-01", x1="2021-04-01",
                    y0=combined_df.iloc[:, 1:].min().min(), y1=combined_df.iloc[:, 1:].max().max(),
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    line=dict(color="red", width=1),
                )
            ],
            legend_title_text="Legend",
            template="plotly_white"
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown(
            """
            <div style="color: gray; font-size: 0.85rem;">
                🛠️ <i>The shaded region indicates the COVID-19 period (2019:Q4–2021:Q1), where GDP growth showed significant volatility.</i>
            </div>
            """, unsafe_allow_html=True
        )

        # --- No COVID Period Plot ---
        st.subheader("GDP Growth Comparison Excluding COVID Period: Actual vs Predicted by Models")

        combined_df_no_covid = df_eval_no_covid[['date', 'actual_gdp_growth']].rename(columns={'actual_gdp_growth': 'Actual GDP Growth'})
        for model in model_types:
            combined_df_no_covid[model] = df_eval_no_covid[model_map[model]]

        fig_no_covid = px.line(combined_df_no_covid, 
                               x='date', 
                               y=combined_df_no_covid.columns[1:], 
                               labels={'date': 'Date', 'value': 'GDP Growth (%)'})
        
        fig_no_covid.update_traces(
            hovertemplate='%{x}: %{y:.2f}% <extra></extra>'
        )
 
        # Add shaded area for COVID period in the second plot 
        fig_no_covid.update_layout(
            shapes=[
                go.layout.Shape(
                    type="rect", 
                    x0="2019-10-01", x1="2021-04-01",
                    y0=-9, y1=7,  # Adjust these values to fit your graph's range
                    xref="x", yref="y",
                    fillcolor="rgba(255, 0, 0, 0.1)",  # Red color with some transparency
                    line=dict(color="red", width=1),  # Red border line
                    layer="below"  # Make sure it's below the graph lines,
                )
            ],
            legend_title_text="Legend",
            template="plotly_white"
        )
        st.plotly_chart(fig_no_covid, use_container_width=True)
