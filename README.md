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

├── Data/                        # Data files and R scripts
│   ├── tree_df.csv                        # Non-linear model input
│   ├── tree_df_test.csv                   # Not part of main workflow. For manually testing.
│   ├── bridge_df.csv                      # Linear model input
│   ├── final_df.csv                       # Combined evaluation set
│   ├── row_error.csv                      # Forecast errors
│   ├── row_error_dropped_covid.csv        # Forecast errors without covid period
│   ├── rmsfe.csv                          # Model performance metrics
│   ├── mae_df.csv                         # Model performance metrics
│   ├── *_dropped_covid.csv                # Also Model Performance metrics but these Excludes COVID quarters
│   ├── important_indicators.r             # LASSO + intuition-based selection
│   ├── bridge_indicators.r                # Monthly to quarterly transformation
│   ├── tree_df.r                          # Script for tree_df construction
│   └── api_keys.R                         # FRED API key loader

</pre>

## Data & Methodology

We used a set of 23 macroeconomic indicators obtained from the FRED-MD database via API. Indicators were aggregated to quarterly frequency using variable-specific rules (e.g., sum for flow variables, mean for stock variables). The target variable, annualized GDP growth, was calculated as the log-difference of real GDP scaled by 400.

Preprocessing involved:
	•	Differencing non-stationary indicators based on ADF test results
	•	Lag creation to capture delayed effects
	•	Feature selection, using:
	•	LASSO for linear models (e.g., ADL Bridge)
	•	RFECV (Recursive Feature Elimination with Cross-Validation) for Random Forest models

We implemented four models:
	•	AR (Autoregressive benchmark)
	•	RF (Random Forest benchmark)
	•	ADL Bridge (Linear model with high-frequency inputs)
	•	RF Bridge (Non-linear bridge model with hyperparameter tuning)

Bridge models constructed quarterly predictors using forecasted monthly indicators, simulating real-time availability constraints.

Model performance was evaluated using:
	•	RMSFE (Root Mean Squared Forecast Error)
	•	MAFE (Mean Absolute Forecast Error)
	•	Downturn Accuracy
	•	Directional Accuracy
	•	Diebold-Mariano Test for comparing predictive accuracy across models

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
