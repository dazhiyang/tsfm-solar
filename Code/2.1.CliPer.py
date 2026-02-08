
import pandas as pd
import numpy as np
import os
import glob
import warnings

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Processed")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Forecasts", "CLIPER")

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df

def process_variable(df_train, df_test, var_meas, var_cs, pred_col_name):
    """
    Applies the specific partial-climatology partial-persistence logic.
    R Logic Transliteration:
      Clim <- mean(k) * McClear
      Pers <- k(t-h) * McClear(t)
      gamma <- cor(k(t), k(t+h))
      Comb <- (gamma * Pers/McClear + (1-gamma) * Clim/McClear) * McClear
           = (gamma * k_pers + (1-gamma) * k_mean) * McClear
    """
    # 1. Prepare Training Data for Parameters
    # Convert to numeric, forcing errors to NaN
    train_meas = pd.to_numeric(df_train[var_meas], errors='coerce')
    train_cs = pd.to_numeric(df_train[var_cs], errors='coerce')
    train_zen = df_train['zenith_angle']
    
    # Calculate Kappa (Clearness/Clear-sky Index)
    # Filter: Zenith < 85 AND CS > 10 to avoid noise
    mask_train = (train_zen < 85) & (train_cs > 10)
    
    train_kappa = (train_meas / train_cs).where(mask_train)
    
    # PARAMETER 1: Climatology Mean (Scalar)
    kappa_mean = train_kappa.mean()
    
    # PARAMETER 2: Gamma (Correlation)
    # Correlation between kappa(t) and kappa(t+1) (h=1 step)
    # We drop NaNs to compute valid correlation
    # Create lagged series
    kappa_t = train_kappa
    kappa_t1 = train_kappa.shift(-1) # We align t and t+1 on the same row index t
    
    valid_corr_idx = kappa_t.notna() & kappa_t1.notna()
    if valid_corr_idx.sum() > 10:
        gamma = np.corrcoef(kappa_t[valid_corr_idx], kappa_t1[valid_corr_idx])[0, 1]
    else:
        gamma = 0.5 # Fallback
        
    print(f"    {var_meas}: Mean Kappa={kappa_mean:.3f}, Gamma={gamma:.3f}")
    
    # 2. Apply to Test Data (Step-by-step to match logic)
    # We need full context to allow persistence from end of 2023 if continuous, 
    # but simplest is to just shift within test df (first point lost/using Clim).
    
    test_meas = pd.to_numeric(df_test[var_meas], errors='coerce')
    test_cs = pd.to_numeric(df_test[var_cs], errors='coerce')
    test_zen = df_test['zenith_angle']
    
    # Calculate Test Kappa
    mask_test = (test_zen < 85) & (test_cs > 10)
    test_kappa = (test_meas / test_cs).where(mask_test)
    
    # Persistence Component: k_pers = k(t-1)
    # shift(1) moves t-1 to t
    kappa_pers = test_kappa.shift(1)
    
    # Fill missing persistence (e.g. first hour, or after night) with mean
    # Or strict implementation: if pers missing, fallback to clim?
    # The formula `gamma*k_pers + (1-gamma)*k_mean` works best if we fillna k_pers with k_mean
    # effectively reducing it to pure climatology when persistence is unknown.
    kappa_pers_filled = kappa_pers.fillna(kappa_mean)
    
    # Combination Kappa
    kappa_comb = gamma * kappa_pers_filled + (1 - gamma) * kappa_mean
    
    # Reconstruct Irradiance
    pred_comb = kappa_comb * test_cs
    
    # 3. Post-Processing
    # Set Night Time (Zenith > 90) to NaN
    pred_comb.loc[test_zen > 90] = np.nan
    # Clip negatives
    pred_comb = pred_comb.clip(lower=0)
    
    return pred_comb

def process_station(filepath):
    filename = os.path.basename(filepath)
    stn_code = filename.split('_')[0]
    print(f"Processing {stn_code}...")
    
    df = load_data(filepath)
    
    if 2023 not in df.index.year:
        print(f"  Skipping {stn_code}: No 2023 data.")
        return

    # Use 2023 for Training (stat calculation)
    train_df = df.loc['2023']
    
    # Use 2024 for Testing
    # To have the "persistence" for 2024-01-01 00:00, we technically need the last point of 2023.
    # Let's create a "test context" df that includes the last day of 2023.
    last_2023 = df.loc['2023'].iloc[-1:]
    test_df_raw = df.loc['2024']
    
    test_context = pd.concat([last_2023, test_df_raw])
    
    # GHI
    pred_ghi = process_variable(train_df, test_context, 
                                'measured_GHI', 'clear-sky_GHI', 'pred_GHI_comb')
    
    # DNI
    pred_dni = process_variable(train_df, test_context, 
                                'measured_DNI', 'clear-sky_DNI', 'pred_DNI_comb')
    
    # Trim back to 2024
    pred_ghi = pred_ghi.loc['2024']
    pred_dni = pred_dni.loc['2024']
    test_df = test_df_raw.copy() # Ensure clean frame
    
    test_df['pred_GHI_comb'] = pred_ghi
    test_df['pred_DNI_comb'] = pred_dni
    
    # Save
    final_cols = ['measured_GHI', 'measured_DNI', 'clear-sky_GHI', 'clear-sky_DNI', 
                  'zenith_angle', 'pred_GHI_comb', 'pred_DNI_comb']
    
    save_df = test_df[final_cols].copy()
    
    # Format: Zenith to 3 decimals
    save_df['zenith_angle'] = save_df['zenith_angle'].round(3)
    
    # Format: Irradiance to Integers (nullable Int64 to handle NaNs)
    irr_cols = [c for c in final_cols if c != 'zenith_angle']
    for col in irr_cols:
        save_df[col] = save_df[col].astype(float).round(0).astype('Int64')

    out_path = os.path.join(OUTPUT_DIR, f"{stn_code}_forecast_2024.csv")
    save_df.to_csv(out_path)
    print(f"  Saved to {out_path}")

def main():
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_15min_qc.csv")))
    for f in files:
        process_station(f)

if __name__ == "__main__":
    main()
