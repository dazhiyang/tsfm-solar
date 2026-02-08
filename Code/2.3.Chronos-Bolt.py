
import pandas as pd
import numpy as np
import torch
from chronos import ChronosBoltPipeline
from tqdm import tqdm
import os
import glob
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CONTEXT_LENGTH = 512  # Optimal for Bolt
PREDICTION_LENGTH = 1 # 1-step ahead
BATCH_SIZE = 64       
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "amazon/chronos-bolt-small" 

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Processed")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Forecasts", "Chronos-Bolt")
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

def run_forecast_station(stn_code, pipeline):
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
    
    # Needs sufficient history for 512 context steps.
    # 512 steps of DAYLIGHT can take 2-3 weeks in winter.
    # Loading from 2023-01-01 to be safe and consistent.
    context_start = pd.Timestamp("2023-01-01 00:00:00")
    
    data_subset = df[context_start : test_end].copy()
    
    if data_subset.empty:
        print(f"  Not enough data for {stn_code}")
        return

    # Prepare Columns
    # Prepare Columns
    for col in ['pred_GHI_direct', 'pred_DNI_direct', 'pred_GHI_csky', 'pred_DNI_csky']:
        data_subset[col] = np.nan

    # Target Indices (Predict FOR these times)
    target_indices = data_subset.index[data_subset.index >= test_start]
    # We still predict only for daytime to save compute (no sun at night anyway)
    day_target_indices = data_subset.index[(data_subset.index >= test_start) & (data_subset['zenith_angle'] < 90)]
    
    if len(day_target_indices) == 0:
        print("No target indices found.")
        return

    # Batch generator function (operates on FULL data_subset)
    def generate_batches(series_data, target_timestamps):
        batch_contexts = []
        batch_times = []
        target_ilocs = [data_subset.index.get_loc(t) for t in target_timestamps]
        
        for i in target_ilocs:
            if i < CONTEXT_LENGTH: continue
            context = series_data[i-CONTEXT_LENGTH : i]
            batch_contexts.append(torch.tensor(context))
            batch_times.append(data_subset.index[i])
            
            if len(batch_contexts) == BATCH_SIZE:
                yield torch.stack(batch_contexts).to(torch.float32), batch_times
                batch_contexts = []
                batch_times = []
        if batch_contexts:
            yield torch.stack(batch_contexts).to(torch.float32), batch_times
 
    # --- 1. Direct Forecasting ---
    print("  Mode: Direct Forecasting (Full Context)")
    
    for col, pred_col in [('measured_GHI', 'pred_GHI_direct'), ('measured_DNI', 'pred_DNI_direct')]:
        series = np.nan_to_num(data_subset[col].values)
        
        for batch_tensor, batch_ts in tqdm(generate_batches(series, day_target_indices), 
                                           total=len(day_target_indices)//BATCH_SIZE, 
                                           desc=f"    {col}", leave=False):
            forecast = pipeline.predict(batch_tensor, prediction_length=PREDICTION_LENGTH, limit_prediction_length=False)
            median_pred = forecast[:, 4, :].flatten().cpu().numpy()
            data_subset.loc[batch_ts, pred_col] = median_pred

    # --- 2. Clear-Sky Index Forecasting ---
    print("  Mode: Clear-Sky Index (Full Context)")
    
    # Calculate kappa on FULL data_subset
    cs_ghi = data_subset['clear-sky_GHI'].values
    meas_ghi = data_subset['measured_GHI'].values
    kappa_ghi = np.divide(meas_ghi, cs_ghi, out=np.zeros_like(meas_ghi, dtype=float), where=cs_ghi>10)
    
    cs_dni = data_subset['clear-sky_DNI'].values
    meas_dni = data_subset['measured_DNI'].values
    kappa_dni = np.divide(meas_dni, cs_dni, out=np.zeros_like(meas_dni, dtype=float), where=cs_dni>10)
    
    for kappa_series, cs_series, pred_col in [(kappa_ghi, cs_ghi, 'pred_GHI_csky'), (kappa_dni, cs_dni, 'pred_DNI_csky')]:
        
        batch_contexts = []
        # We need to manually batch because we need 'i' to get CS target
        target_ilocs = [data_subset.index.get_loc(t) for t in day_target_indices]
        
        batch_contexts = []
        batch_indices = []
        batch_ts = []

        for i in tqdm(target_ilocs, desc=f"    kappa {pred_col}", leave=False):
            if i < CONTEXT_LENGTH: continue
            
            context = kappa_series[i-CONTEXT_LENGTH : i]
            batch_contexts.append(torch.tensor(context))
            batch_indices.append(i)
            batch_ts.append(data_subset.index[i])
            
            if len(batch_contexts) == BATCH_SIZE:
                 batch_input = torch.stack(batch_contexts).to(torch.float32)
                 forecast = pipeline.predict(batch_input, prediction_length=1)
                 kappa_pred = forecast[:, 4, :].flatten().cpu().numpy()
                 cs_targets = cs_series[batch_indices]
                 data_subset.loc[batch_ts, pred_col] = kappa_pred * cs_targets
                 batch_contexts, batch_indices, batch_ts = [], [], []

        if batch_contexts:
             batch_input = torch.stack(batch_contexts).to(torch.float32)
             forecast = pipeline.predict(batch_input, prediction_length=1)
             kappa_pred = forecast[:, 4, :].flatten().cpu().numpy()
             cs_targets = cs_series[batch_indices]
             data_subset.loc[batch_ts, pred_col] = kappa_pred * cs_targets

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
    print(f"Initializing Chronos Bolt Pipeline ({MODEL_NAME}) on {DEVICE}...")
    
    dtype = torch.float32 if (DEVICE == "mps" or DEVICE == "cpu") else torch.bfloat16
    
    try:
        pipeline = ChronosBoltPipeline.from_pretrained(
            MODEL_NAME,
            device_map=DEVICE,
            dtype=dtype,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    station_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_15min_qc.csv")))
    stations = [os.path.basename(f).split('_')[0] for f in station_files]

    for stn in stations:
        run_forecast_station(stn, pipeline)

if __name__ == "__main__":
    main()
