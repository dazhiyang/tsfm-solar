
import pandas as pd
import numpy as np
import timesfm
import torch
from tqdm import tqdm
import os
import glob
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CONTEXT_LENGTH = 1024 # User requested
PREDICTION_LENGTH = 1 # 1-step ahead rolling
BATCH_SIZE = 1024      # Batch size for inference (Increased for speed)
MODEL_NAME = "google/timesfm-2.5-200m-pytorch"
DEVICE = "mps" # User confirmed MPS is the way, just needs warm-up

# Paths (Adjust based on script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Processed")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Forecasts", "TimesFM-2.5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_processed_data(stn_code):
    pattern = os.path.join(PROCESSED_DIR, f"{stn_code}_15min_qc.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    df = pd.read_csv(files[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df

def run_forecast_station(stn_code, model):
    out_path = os.path.join(OUTPUT_DIR, f"{stn_code}_forecast_2024.csv")
    
    if os.path.exists(out_path):
        print(f"Skipping {stn_code}: Already processed.")
        return

    print(f"\nForecasting for {stn_code} using {MODEL_NAME}...")

    df = load_processed_data(stn_code)
    if df is None: return
    
    # Define Test Period
    test_start = pd.Timestamp("2024-01-01 00:00:00")
    test_end = pd.Timestamp("2024-12-31 23:45:00")
    
    # Needs sufficient history for context
    context_start = pd.Timestamp("2023-01-01 00:00:00")
    
    data_subset = df[context_start : test_end].copy()
    
    if data_subset.empty:
        print(f"  Not enough data for {stn_code}")
        return

    # Prepare Columns
    for col in ['pred_GHI_direct', 'pred_DNI_direct', 'pred_GHI_csky', 'pred_DNI_csky']:
        data_subset[col] = np.nan

    # Target Indices
    # We predict only for daytime
    day_target_indices = data_subset.index[(data_subset.index >= test_start) & (data_subset['zenith_angle'] < 90)]
    
    
    if len(day_target_indices) == 0:
        print("No target indices found.")
        return

    # Create sliding windows efficiently
    def create_sliding_windows(data, window_size):
        shape = (data.size - window_size + 1, window_size)
        strides = (data.strides[0], data.strides[0])
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    # 1. Direct Forecasting
    print("  Mode: Direct Forecasting")
    
    # Pre-calculate indices we need
    # day_target_indices contains the timestamps we want to PREDICT
    # The input window for predicting time T starts at T - context_len
    target_ilocs = np.array([data_subset.index.get_loc(t) for t in day_target_indices])
    
    # Filter out any that don't have enough history
    valid_mask = target_ilocs >= CONTEXT_LENGTH
    valid_target_ilocs = target_ilocs[valid_mask]
    valid_timestamps = day_target_indices[valid_mask]
    
    # Input window index for a target at 'loc' is 'loc - CONTEXT_LENGTH'
    # This index corresponds to the row in the sliding_window view
    window_indices = valid_target_ilocs - CONTEXT_LENGTH

    for col, pred_col in [('measured_GHI', 'pred_GHI_direct'), ('measured_DNI', 'pred_DNI_direct')]:
        series = np.nan_to_num(data_subset[col].values).astype(np.float32)
        all_windows = create_sliding_windows(series, CONTEXT_LENGTH)
        
        # Select ONLY the windows we need (daytime)
        # This is a view/slice, very fast
        needed_windows = all_windows[window_indices]
        
        preds_list = []
        
        for i in tqdm(range(0, len(needed_windows), BATCH_SIZE), desc=f"    {col}", leave=False):
            batch_arr = needed_windows[i : i + BATCH_SIZE]
            # Direct Tensor-based call to compiled_decode (Bypass slow forecast() wrapper)
            mask_arr = np.zeros_like(batch_arr, dtype=bool)
            
            # Returns (point_forecast, full_forecast) where point_forecast is (Batch, Horizon)
            point_forecast, _ = model.compiled_decode(
                horizon=PREDICTION_LENGTH,
                inputs=batch_arr,
                masks=mask_arr
            )
            
            # point_forecast is already a numpy array from compiled_decode
            preds_list.extend(point_forecast[:, 0])
            
        # Assign back
        if preds_list:
            data_subset.loc[valid_timestamps, pred_col] = preds_list

    # 2. Clear-Sky Index Forecasting
    print("  Mode: Clear-Sky Index")
    
    cs_ghi = data_subset['clear-sky_GHI'].values
    meas_ghi = data_subset['measured_GHI'].values
    kappa_ghi = np.divide(meas_ghi, cs_ghi, out=np.zeros_like(meas_ghi, dtype=float), where=cs_ghi>10)
    
    cs_dni = data_subset['clear-sky_DNI'].values
    meas_dni = data_subset['measured_DNI'].values
    kappa_dni = np.divide(meas_dni, cs_dni, out=np.zeros_like(meas_dni, dtype=float), where=cs_dni>10)
    
    
    for kappa_series, cs_series, pred_col in [(kappa_ghi, cs_ghi, 'pred_GHI_csky'), (kappa_dni, cs_dni, 'pred_DNI_csky')]:
        all_windows = create_sliding_windows(kappa_series.astype(np.float32), CONTEXT_LENGTH)
        needed_windows = all_windows[window_indices]
        
        preds_list = []
        
        for i in tqdm(range(0, len(needed_windows), BATCH_SIZE), desc=f"    kappa {pred_col}", leave=False):
            batch_arr = needed_windows[i : i + BATCH_SIZE]
            mask_arr = np.zeros_like(batch_arr, dtype=bool)
            
            point_forecast, _ = model.compiled_decode(
                horizon=PREDICTION_LENGTH,
                inputs=batch_arr,
                masks=mask_arr
            )
            preds_list.extend(point_forecast[:, 0])
            
        if preds_list:
            kappa_preds = np.array(preds_list)
            # Retrieve corresponding Clearsky values for the TARGET timestamps
            cs_targets = cs_series[valid_target_ilocs]
            data_subset.loc[valid_timestamps, pred_col] = kappa_preds * cs_targets

    # --- Post-Processing ---
    pred_cols = ['pred_GHI_direct', 'pred_DNI_direct', 'pred_GHI_csky', 'pred_DNI_csky']
    data_subset[pred_cols] = data_subset[pred_cols].clip(lower=0)

    if 'zenith_angle' in data_subset.columns:
        night_mask = data_subset['zenith_angle'] > 90
        data_subset.loc[night_mask, pred_cols] = np.nan
    
    # Save ONLY 2024
    save_df = data_subset.loc[test_start : test_end].copy()
    
    save_df['zenith_angle'] = save_df['zenith_angle'].round(3)
    for col in pred_cols:
        save_df[col] = save_df[col].astype(float).round(0).astype('Int64')
        
    final_cols = ['measured_GHI', 'measured_DNI', 'clear-sky_GHI', 'clear-sky_DNI', 'zenith_angle'] + pred_cols
    save_df[final_cols].to_csv(out_path)
    print(f"  Saved forecast to {out_path}")

def main():
    print(f"Initializing TimesFM-2.5 Pipeline ({MODEL_NAME}) on {DEVICE}...")

    # Initialize implementation
    # Note: torch_compile=True drastically fails on MPS (Illegal Instruction). Disabling for stability.
    try:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            MODEL_NAME,
            device=DEVICE,
            torch_compile=False # MPS does not support torch.compile well yet
        )
        if hasattr(model, 'model'):
            model.model.to(DEVICE)
            model.model.device = torch.device(DEVICE) # Patch because library lacks MPS detection
            model.model.eval() # Ensure eval mode for speed
            
            # Optimization: Direct Tensor Forward Pass (No torch.compile needed)
            # We will use model.model.decode() directly in the loop
            pass
        
        # Configure
        model.compile(
            timesfm.ForecastConfig(
                max_context=CONTEXT_LENGTH,
                max_horizon=128, 
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    station_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_15min_qc.csv")))
    stations = [os.path.basename(f).split('_')[0] for f in station_files]


    for stn in stations:
        run_forecast_station(stn, model)

if __name__ == "__main__":
    main()
