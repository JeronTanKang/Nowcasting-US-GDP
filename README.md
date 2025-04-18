# DSE3101

# Real-Time GDP Nowcasting Dashboard

This project implements a real-time GDP nowcasting system using statistical and machine learning models. It simulates what analysts would see in real time by limiting inputs to data that would have been available at the forecast date. The application features an interactive dashboard built in Streamlit and supports model comparison, flash estimate simulations, and recession-aware forecasting.

---

## Repository Structure

DSE3101/
│
├── app.py                     # Main Streamlit dashboard entry point
├── README.md
│
├── Backend/                   # All backend processing and forecasting logic
│   ├── data_processing.py             # Preprocessing and differencing
│   ├── forecast_bridge_indicators.py  # Forecast missing monthly data
│   ├── forecast_evaluation.py         # RMSFE, MAFE, and DM test
│   ├── model_AR.py                    # AR benchmark model
│   ├── model_ADL_bridge.py           # ADL Bridge model
│   ├── model_RF.py                   # Random Forest benchmark model
│   ├── model_RF_bridge.py            # RF Bridge with hyperparameter tuning
│   ├── model_DFM.py                  # (Optional) DFM baseline
│   ├── dm_test.py                    # Diebold-Mariano test
│   └── tuned_RF_bridge_model.joblib  # Pretrained RF Bridge model
│
├── Frontend/                 # UI components (if modularized)
│
├── Data/                     # Data files and R scripts
│   ├── tree_df.csv                     # Non-linear model input
│   ├── bridge_df.csv                   # Linear model input
│   ├── final_df.csv                    # Combined evaluation set
│   ├── row_error.csv                   # Forecast errors for DM test
│   ├── rmsfe.csv / mae_df.csv          # Model performance metrics
│   ├── *_dropped_covid.csv             # Excludes COVID quarters
│   ├── important_indicators.r          # LASSO + intuition-based selection
│   ├── bridge_indicators.r             # Monthly → quarterly transformation
│   └── api_keys.R                      # API key loader

---

##  Setup & Running the Dashboard

### 1. Clone the Repository
```bash
git clone https://github.com/JeronTanKang/stealth_bois.git
cd stealth_bois
