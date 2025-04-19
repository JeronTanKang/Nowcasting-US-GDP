# Project for DSE3101

# Real-Time GDP Nowcasting Dashboard

Front-end pls copy paste section 1 project overview here once u done

---

## Repository Structure
<pre>

DSE3101/
├── app.py                         # Main Streamlit dashboard entry point
├── README.md

├── Backend/                       # All backend processing and forecasting logic
│   ├── data_processing.py                 # Preprocessing and differencing
│   ├── forecast_bridge_indicators.py      # Forecast missing monthly data
│   ├── forecast_evaluation.py             # computes RMSFE, MAFE, Skew and Kurtosis
│   ├── model_AR.py                        # AR benchmark model
│   ├── model_ADL_bridge.py                # ADL Bridge model
│   ├── model_RF.py                        # Random Forest benchmark model
│   ├── model_RF_bridge.py                 # RF Bridge with hyperparameter tuning
│   ├── dm_test.py                         # Diebold-Mariano test
│   └── tuned_RF_bridge_model.joblib       # Pretrained RF Bridge model

├── Frontend/                    # UI components
│   ├── Frontend to add these
├── Data/                            # All data inputs, transformation scripts, and outputs
│   ├── api_keys.R                       # FRED API key loader
│   ├── bridge_df.csv                   # Linear model input
│   ├── tree_df.csv                     # Non-linear model input
│   ├── tree_df_test.csv                # Manual testing file (not part of main pipeline)
│   ├── final_df.csv                    # Combined evaluation set
│   ├── manual_testing.csv              # For manually testing 2024 out-of-sample
│   ├── important_indicators.r          # LASSO + intuition-based feature selection
│   ├── bridge_indicators.r             # Monthly to quarterly transformation
│   ├── tree_df.r                       # Script to generate tree_df
│
│   ├── results_and_outputs/            # Model performance metrics and evaluation results
│   │   ├── distribution.csv                    # Forecast error distribution (skew, kurtosis)
│   │   ├── distribution_no_covid.csv           # Distribution excluding COVID quarters
│   │   ├── mae_df.csv                          # Mean Absolute Forecast Error (full sample)
│   │   ├── mae_df_dropped_covid.csv            # MAE excluding COVID quarters
│   │   ├── rmsfe.csv                           # Root Mean Squared Forecast Error (full sample)
│   │   ├── rmsfe_dropped_covid.csv             # RMSFE excluding COVID quarters
│   │   ├── row_error.csv                       # Forecast errors (full sample)
│   │   └── row_error_dropped_covid.csv         # Forecast errors excluding COVID period
│
│   └── Plots/                         # Generated visualizations of forecast errors
│       ├── forecast_errors_1.png
│       ├── forecast_errors_2.png
│       ├── forecast_errors_3.png
│       ├── forecast_errors_excluding_covid_1.png
│       ├── forecast_errors_excluding_covid_2.png
│       └── forecast_errors_excluding_covid_3.png

</pre>

## Data & Methodology
We used a set of **23 macroeconomic indicators** obtained from the FRED-MD database via API. Indicators were aggregated to **quarterly frequency** using variable-specific rules (e.g., sum for flow variables, mean for stock variables). The target variable, **annualized GDP growth**, was calculated as the log-difference of real GDP scaled by 400.

**Preprocessing involved:**
- Differencing non-stationary indicators based on ADF test results  
- Lag creation to capture delayed effects  
- Feature selection using:
  - LASSO for linear models (e.g., ADL Bridge)  
  - RFECV (Recursive Feature Elimination with Cross-Validation) for Random Forest models

**We implemented four models:**
- AR (Autoregressive benchmark)  
- RF (Random Forest benchmark)  
- ADL Bridge (Linear model with high-frequency inputs)  
- RF Bridge (Non-linear with hyperparameter tuning and high-frequency inputs)

Bridge models constructed quarterly predictors using **forecasted monthly indicators**, simulating real-time data availability.

**Model performance was evaluated using:**
- RMSFE (Root Mean Squared Forecast Error)  
- MAFE (Mean Absolute Forecast Error)  
- Downturn Accuracy  
- Directional Accuracy  
- Diebold-Mariano Test for comparing predictive accuracy across models
##  Setup & Running the Dashboard

### 1. Clone the Repository
```bash
git clone https://github.com/JeronTanKang/Nowcasting-US-GDP.git
cd Nowcasting-US-GDP
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Then, create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add API Keys
You need a FRED API key to pull macroeconomic data.

Paste your API key into the Data/api_keys.R file like this:
