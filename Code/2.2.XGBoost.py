
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import glob
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Processed")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Forecasts", "XGBoost")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# XGBoost Params (Baseline)
XGB_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'early_stopping_rounds': 50
}

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df

def add_time_features(df):
    """Adds cyclic time features."""
    df = df.copy()
    # Hour of Day
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Day of Year
    df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    
    return df

def create_lagged_features(df, col_name, lags=[1, 2, 3, 4]):
    """Creates lagged features for a specific column."""
    df_lags = pd.DataFrame(index=df.index)
    for lag in lags:
        df_lags[f'{col_name}_lag{lag}'] = df[col_name].shift(lag)
    return df_lags

def train_predict_xgboost(df_train, df_test, target_col, feature_cols, model_name="xgb"):
    """Trains XGBoost and predicts on test set."""
    
    # Filter Training Data: Day Time Only (Zenith < 85)
    # This avoids training on noise/zeros at night
    if 'zenith_angle' in df_train.columns:
        df_train = df_train[df_train['zenith_angle'] < 85]
    
    # Prepare X and y
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test = df_test[feature_cols]
    # y_test is not needed for prediction, strictly speaking
    
    # Filter NaNs in Training (XGBoost handles NaNs in X, but y must be valid)
    mask_train = y_train.notna() & X_train.notna().all(axis=1) # Safer to drop nan features too for simple baseline
    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    
    if len(X_train_clean) < 100:
        print(f"  Not enough valid training data for {target_col}.")
        return pd.Series(np.nan, index=df_test.index)
        
    # Initialize Model
    model = xgb.XGBRegressor(**XGB_PARAMS)
    
    # Train
    # Use 10% of training as validation for early stopping
    # Ensure shuffle=False for time series if using simplistic split, 
    # though random split is often okay for mapping functions if no autoregression leakage.
    # Here we have lags, so strictly speaking we should split by time block.
    # But 90/10 sequential split is standard.
    validation_split = int(len(X_train_clean) * 0.9)
    X_tr, X_val = X_train_clean.iloc[:validation_split], X_train_clean.iloc[validation_split:]
    y_tr, y_val = y_train_clean.iloc[:validation_split], y_train_clean.iloc[validation_split:]
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict
    preds = model.predict(X_test)
    return pd.Series(preds, index=df_test.index)

def process_station(filepath):
    stn_code = os.path.basename(filepath).split('_')[0]
    print(f"Processing {stn_code}...")
    
    df = load_data(filepath)
    
    # 1. Feature Engineering
    # We need to perform this on the full dataset to compute lags correctly across year boundaries
    
    # Calculate Clearness Index (kappa)
    # kappa = measured / clear-sky
    # Avoid division by zero
    for rad in ['GHI', 'DNI']:
        meas_col = f'measured_{rad}'
        cs_col = f'clear-sky_{rad}'
        kappa_col = f'kappa_{rad}'
        
        # Simple division, handle low clear-sky
        with np.errstate(divide='ignore', invalid='ignore'):
            df[kappa_col] = df[meas_col] / df[cs_col]
        
        # Clean up kappa: if CS < 10, kappa is unreliable/undefined -> 0 or NaN?
        # Let's set to 0 (night) 
        df.loc[df[cs_col] < 10, kappa_col] = 0.0
        # Clip crazy values
        df[kappa_col] = df[kappa_col].clip(0, 2.0)

    # Add Time Features
    df = add_time_features(df)
    
    # Define features for Irradiance Model
    # Lags of Irradiance + Zenith + Time
    lag_cols_ghi = create_lagged_features(df, 'measured_GHI', lags=[1, 2, 3, 4, 96]) # 1h and 24h lags? 96=24h
    lag_cols_dni = create_lagged_features(df, 'measured_DNI', lags=[1, 2, 3, 4, 96])
    
    # Define features for Csky Model
    # Lags of Kappa + Zenith + Time
    lag_cols_kappa_ghi = create_lagged_features(df, 'kappa_GHI', lags=[1, 2, 3, 4, 96])
    lag_cols_kappa_dni = create_lagged_features(df, 'kappa_DNI', lags=[1, 2, 3, 4, 96])
    
    # Concatenate features
    df = pd.concat([df, lag_cols_ghi, lag_cols_dni, lag_cols_kappa_ghi, lag_cols_kappa_dni], axis=1)
    
    # 2. Split Train (2023) / Test (2024)
    if 2023 not in df.index.year or 2024 not in df.index.year:
         print(f"  Skipping {stn_code}: Missing 2023 or 2024 data.")
         return
         
    df_train = df.loc['2023']
    df_test = df.loc['2024']
    
    results_df = df_test[['measured_GHI', 'measured_DNI', 'clear-sky_GHI', 'clear-sky_DNI', 'zenith_angle']].copy()
    
    # 3. Model 1: Direct Irradiance Forecasting
    print("  Training Direct Models...")
    
    # Features to use
    base_feats = ['zenith_angle', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos']
    
    # GHI Direct
    feats_ghi_dir = base_feats + [c for c in df.columns if 'measured_GHI_lag' in c]
    pred_ghi_dir = train_predict_xgboost(df_train, df_test, 'measured_GHI', feats_ghi_dir)
    
    # DNI Direct
    feats_dni_dir = base_feats + [c for c in df.columns if 'measured_DNI_lag' in c]
    pred_dni_dir = train_predict_xgboost(df_train, df_test, 'measured_DNI', feats_dni_dir)
    
    results_df['pred_GHI_xgboost_direct'] = pred_ghi_dir
    results_df['pred_DNI_xgboost_direct'] = pred_dni_dir
    
    # 4. Model 2: Clear-Sky Index Forecasting
    print("  Training Clear-Sky Index Models...")
    
    # GHI Csky
    feats_kappa_ghi = base_feats + [c for c in df.columns if 'kappa_GHI_lag' in c]
    pred_kappa_ghi = train_predict_xgboost(df_train, df_test, 'kappa_GHI', feats_kappa_ghi)
    # Reconstruct: pred = kappa_pred * cs_test
    results_df['pred_GHI_xgboost_csky'] = pred_kappa_ghi * df_test['clear-sky_GHI']
    
    # DNI Csky
    feats_kappa_dni = base_feats + [c for c in df.columns if 'kappa_DNI_lag' in c]
    pred_kappa_dni = train_predict_xgboost(df_train, df_test, 'kappa_DNI', feats_kappa_dni)
    results_df['pred_DNI_xgboost_csky'] = pred_kappa_dni * df_test['clear-sky_DNI']
    
    # 5. Post-Processing
    pred_cols = [c for c in results_df.columns if 'pred_' in c]
    
    # Clip Negatives
    results_df[pred_cols] = results_df[pred_cols].clip(lower=0)
    
    # Night Filtering (Zenith > 90 -> NaN)
    mask_night = results_df['zenith_angle'] > 90
    results_df.loc[mask_night, pred_cols] = np.nan
    
    # Save
    # Format: Zenith to 3 decimals, Irradiance to Int64
    save_df = results_df.copy()
    save_df['zenith_angle'] = save_df['zenith_angle'].round(3)
    
    for col in pred_cols:
        save_df[col] = save_df[col].astype(float).round(0).astype('Int64')
        
    out_path = os.path.join(OUTPUT_DIR, f"{stn_code}_forecast_2024.csv")
    save_df.to_csv(out_path)
    print(f"  Saved {out_path}")

def main():
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_15min_qc.csv")))
    for f in files:
        process_station(f)

if __name__ == "__main__":
    main()
