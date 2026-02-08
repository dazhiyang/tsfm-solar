import pandas as pd
import numpy as np
import tabpfn_client as tc
import os
import glob
import warnings

# Initialize TabPFN Client
token = tc.get_access_token()
tc.set_access_token(token)

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Processed")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Forecasts", "TabPFN-2.5")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def train_predict_tabpfn(df_train, df_test, target_col, feature_cols):
    """Trains TabPFN and predicts on test set."""
    
    # Filter Training Data: Day Time Only (Zenith < 85)
    if 'zenith_angle' in df_train.columns:
        df_train = df_train[df_train['zenith_angle'] < 85]
    
    # Prepare X and y
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test = df_test[feature_cols]
    
    # Clean data (TabPFN handles some NaNs but y must be clean)
    mask_train = y_train.notna() & X_train.notna().all(axis=1)
    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    
    if len(X_train_clean) < 10:
        return pd.Series(np.nan, index=df_test.index)
    
    # TabPFN limit recommendation: if too many rows, subsample for fitting
    # Although v2 handles more, 30k+ might be slow on local device
    # Let's try to cap at 10,000 for training if needed, but for now we'll use all.
    # Note: fitting is very fast in TabPFN, it's just prediction that takes time.
    
    # Initialize Model (Client version)
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = tc.TabPFNRegressor()
            # Fit
            model.fit(X_train_clean, y_train_clean)
            # Predict
            preds = model.predict(X_test)
            return pd.Series(preds, index=df_test.index)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Warning: TabPFN error on attempt {attempt+1}. Retrying in 5s... ({e})")
                time.sleep(5)
            else:
                print(f"  Error: TabPFN failed after {max_retries} attempts.")
                raise e

def process_station(filepath):
    stn_code = os.path.basename(filepath).split('_')[0]
    out_path = os.path.join(OUTPUT_DIR, f"{stn_code}_forecast_2024.csv")
    
    if os.path.exists(out_path):
        print(f"Skipping {stn_code}: Already processed.")
        return
        
    print(f"Processing {stn_code}...")
    
    df = load_data(filepath)
    
    # 1. Feature Engineering (Same as XGBoost)
    for rad in ['GHI', 'DNI']:
        meas_col = f'measured_{rad}'
        cs_col = f'clear-sky_{rad}'
        kappa_col = f'kappa_{rad}'
        with np.errstate(divide='ignore', invalid='ignore'):
            df[kappa_col] = df[meas_col] / df[cs_col]
        df.loc[df[cs_col] < 10, kappa_col] = 0.0
        df[kappa_col] = df[kappa_col].clip(0, 2.0)

    df = add_time_features(df)
    
    lag_cols_ghi = create_lagged_features(df, 'measured_GHI', lags=[1, 2, 3, 4, 96])
    lag_cols_dni = create_lagged_features(df, 'measured_DNI', lags=[1, 2, 3, 4, 96])
    lag_cols_kappa_ghi = create_lagged_features(df, 'kappa_GHI', lags=[1, 2, 3, 4, 96])
    lag_cols_kappa_dni = create_lagged_features(df, 'kappa_DNI', lags=[1, 2, 3, 4, 96])
    
    df = pd.concat([df, lag_cols_ghi, lag_cols_dni, lag_cols_kappa_ghi, lag_cols_kappa_dni], axis=1)
    
    if 2023 not in df.index.year or 2024 not in df.index.year:
         print(f"  Skipping {stn_code}: Missing 2023 or 2024 data.")
         return
         
    df_train = df.loc['2023']
    df_test = df.loc['2024']
    
    results_df = df_test[['measured_GHI', 'measured_DNI', 'clear-sky_GHI', 'clear-sky_DNI', 'zenith_angle']].copy()
    
    # --- Training ---
    base_feats = ['zenith_angle', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos']
    
    # GHI Direct
    print("  GHI Direct...")
    feats_ghi_dir = base_feats + [c for c in df.columns if 'measured_GHI_lag' in c]
    results_df['pred_GHI_tabpfn_direct'] = train_predict_tabpfn(df_train, df_test, 'measured_GHI', feats_ghi_dir)
    
    # DNI Direct
    print("  DNI Direct...")
    feats_dni_dir = base_feats + [c for c in df.columns if 'measured_DNI_lag' in c]
    results_df['pred_DNI_tabpfn_direct'] = train_predict_tabpfn(df_train, df_test, 'measured_DNI', feats_dni_dir)
    
    # GHI Csky
    print("  GHI Csky...")
    feats_kappa_ghi = base_feats + [c for c in df.columns if 'kappa_GHI_lag' in c]
    pred_kappa_ghi = train_predict_tabpfn(df_train, df_test, 'kappa_GHI', feats_kappa_ghi)
    results_df['pred_GHI_tabpfn_csky'] = pred_kappa_ghi * df_test['clear-sky_GHI']
    
    # DNI Csky
    print("  DNI Csky...")
    feats_kappa_dni = base_feats + [c for c in df.columns if 'kappa_DNI_lag' in c]
    pred_kappa_dni = train_predict_tabpfn(df_train, df_test, 'kappa_DNI', feats_kappa_dni)
    results_df['pred_DNI_tabpfn_csky'] = pred_kappa_dni * df_test['clear-sky_DNI']
    
    # Post-Processing
    pred_cols = [c for c in results_df.columns if 'pred_' in c]
    results_df[pred_cols] = results_df[pred_cols].clip(lower=0)
    
    # Night Filtering (Zenith > 90 -> NaN)
    mask_night = results_df['zenith_angle'] > 90
    results_df.loc[mask_night, pred_cols] = np.nan
    
    # Final Formatting
    save_df = results_df.copy()
    save_df['zenith_angle'] = save_df['zenith_angle'].round(3)
    for col in pred_cols:
        save_df[col] = save_df[col].astype(float).round(0).astype('Int64')
        
    out_path = os.path.join(OUTPUT_DIR, f"{stn_code}_forecast_2024.csv")
    save_df.to_csv(out_path)
    print(f"  Saved {out_path}")

def main():
    # Only process 'dra' for now for speed/testing if many stations
    # User can adjust to all
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_15min_qc.csv")))
    # For now, let's process all as the user request implies they want results
    for f in files:
        process_station(f)

if __name__ == "__main__":
    main()
