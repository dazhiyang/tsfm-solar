import pandas as pd
import numpy as np
import torch
import os
import glob
import warnings
from tqdm import tqdm
import transformers
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CONTEXT_LENGTH = 512 # TTM default
PREDICTION_LENGTH = 96 # TTM is fixed to 96 usually, we can handle this.
BATCH_SIZE = 64
MODEL_NAME = "ibm-granite/granite-timeseries-ttm-r1" 
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Processed")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Forecasts", "TTM-R1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_processed_data(stn_code):
    pattern = os.path.join(PROCESSED_DIR, f"{stn_code}_15min_qc.csv")
    files = glob.glob(pattern)
    if not files: return None
    df = pd.read_csv(files[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    float_cols = ['measured_GHI', 'measured_DNI', 'clear-sky_GHI', 'clear-sky_DNI', 'zenith_angle']
    for c in float_cols:
        if c in df.columns:
             df[c] = df[c].astype(np.float32)
            
    df = df.set_index('timestamp').sort_index()
    return df

def run_forecast_station(stn_code, model=None):
    out_path = os.path.join(OUTPUT_DIR, f"{stn_code}_forecast_2024.csv")
    if os.path.exists(out_path):
        print(f"Skipping {stn_code}: Already processed.")
        return

    print(f"\nForecasting for {stn_code} using {MODEL_NAME}...")
    df = load_processed_data(stn_code)
    if df is None: return

    # Time Setup
    test_start = pd.Timestamp("2024-01-01 00:00:00")
    test_end = pd.Timestamp("2024-12-31 23:45:00")
    context_start = pd.Timestamp("2023-01-01 00:00:00")
    data_full = df[context_start : test_end].copy()
    
    if data_full.empty: return
    
    # Indices to predict (2024 only, day only)
    # Actually we predict all of 2024, post-process mask night
    target_indices = data_full.index[data_full.index >= test_start]
    
    # We need integers for indexing numpy arrays
    target_int_indices = [data_full.index.get_loc(t) for t in target_indices]
    
    # Filter valid context: We need at least CONTEXT_LENGTH history
    valid_int_indices = [t for t in target_int_indices if t >= CONTEXT_LENGTH]
    
    if not valid_int_indices:
        print("Not enough context.")
        return

    def run_inference_for_column(col_name, pred_col_name):
        series = data_full[col_name].values
        preds_map = {}
        
        # Batching
        for i in tqdm(range(0, len(valid_int_indices), BATCH_SIZE), desc=f"    {col_name}", leave=False):
            batch_target_indices = valid_int_indices[i : i + BATCH_SIZE]
            
            batch_input = []
            
            for t_idx in batch_target_indices:
                # TTM requires input shape (Batch, Seq, Channels)
                # Univariate: (1, 512, 1)
                
                # Context window: [t-Ctx : t]
                window = series[t_idx - CONTEXT_LENGTH : t_idx]
                
                # Handle NaNs
                if np.isnan(window).any():
                    window = np.nan_to_num(window)
                
                batch_input.append(window)
            
            # Stack: (Batch, Ctx)
            batch_arr = np.array(batch_input, dtype=np.float32)
            # Reshape for TTM: (Batch, Ctx, 1)
            batch_arr = batch_arr[..., np.newaxis]
            
            # To Tensor
            batch_tensor = torch.tensor(batch_arr).to(DEVICE)
            
            # Inference
            with torch.no_grad():
                # model forward returns object. prediction_outputs usually.
                # It might output (Batch, Horizon, Channels)
                outputs = model(past_values=batch_tensor)
                # TTM output is typically .prediction_outputs or .logits?
                # Check TSModelOutput structure. 
                # Usually .prediction_logits or .forecast
                
                if hasattr(outputs, 'prediction_outputs'):
                    forecast = outputs.prediction_outputs # (B, H, C)
                elif hasattr(outputs, 'forecast'):
                    forecast = outputs.forecast
                else:
                    # Fallback check attributes
                    # print(outputs.keys())
                    forecast = outputs[0] # Tuple?

            # We want simple 1-step forecast at horizon 1? 
            # Or the first step of the forecast?
            # TTM predicts [T, T+H]. So index 0 is T.
            
            batch_preds = forecast[:, 0, 0].cpu().numpy() # (Batch,)
            
            # Map back
            for j, val in enumerate(batch_preds):
                t_idx = batch_target_indices[j]
                ts = data_full.index[t_idx]
                preds_map[ts] = val
        
        # Assign
        pred_series = pd.Series(preds_map)
        data_full.loc[pred_series.index, pred_col_name] = pred_series

    # 1. Direct
    print("  Mode: Direct Forecasting")
    run_inference_for_column('measured_GHI', 'pred_GHI_direct')
    run_inference_for_column('measured_DNI', 'pred_DNI_direct')

    # 2. Kappa
    print("  Mode: Clear-Sky Index")
    cs_ghi = data_full['clear-sky_GHI'].values
    meas_ghi = data_full['measured_GHI'].values
    kappa_ghi = np.divide(meas_ghi, cs_ghi, out=np.zeros_like(meas_ghi), where=cs_ghi>10)
    data_full['kappa_GHI'] = kappa_ghi.astype(np.float32)
    
    cs_dni = data_full['clear-sky_DNI'].values
    meas_dni = data_full['measured_DNI'].values
    kappa_dni = np.divide(meas_dni, cs_dni, out=np.zeros_like(meas_dni), where=cs_dni>10)
    data_full['kappa_DNI'] = kappa_dni.astype(np.float32)

    run_inference_for_column('kappa_GHI', 'pred_GHI_csky')
    run_inference_for_column('kappa_DNI', 'pred_DNI_csky')
    
    # Decode Kappa
    data_full['pred_GHI_csky'] *= data_full['clear-sky_GHI']
    data_full['pred_DNI_csky'] *= data_full['clear-sky_DNI']

    # --- Post Processing --- 
    pred_cols = ['pred_GHI_direct', 'pred_DNI_direct', 'pred_GHI_csky', 'pred_DNI_csky']
    data_full[pred_cols] = data_full[pred_cols].clip(lower=0)
    
    if 'zenith_angle' in data_full.columns:
        night_mask = data_full['zenith_angle'] > 90
        # data_full.loc[night_mask, pred_cols] = 0 # Or Nan? Standard is 0 or Nan. 
        # Previous scripts used NaN for night in final output usually? Or 0?
        # Let's clean up:
        data_full.loc[night_mask, pred_cols] = 0.0

    save_df = data_full.loc[test_start : test_end].copy()
    
    if 'zenith_angle' in save_df.columns:
        save_df['zenith_angle'] = save_df['zenith_angle'].round(3)
        
    for col in pred_cols:
        if col in save_df.columns:
            # fillna(0)
            save_df[col] = save_df[col].fillna(0).astype(float).round(0).astype('Int64')

    final_cols = ['measured_GHI', 'measured_DNI', 'clear-sky_GHI', 'clear-sky_DNI', 'zenith_angle'] + pred_cols
    final_cols = [c for c in final_cols if c in save_df.columns]
    
    save_df[final_cols].to_csv(out_path)
    print(f"  Saved to {out_path}")

def main():
    print(f"Initializing TinyTimeMixer ({MODEL_NAME}) on {DEVICE}...")
    try:
        model = TinyTimeMixerForPrediction.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Failed to load TTM: {e}")
        return

    station_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_15min_qc.csv")))
    stations = [os.path.basename(f).split('_')[0] for f in station_files]

    # Run all stations
    for stn in stations:
        run_forecast_station(stn, model=model)

if __name__ == "__main__":
    main()
