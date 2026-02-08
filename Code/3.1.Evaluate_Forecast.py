
import pandas as pd
import numpy as np
import os
import glob
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Processed")
FORECAST_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Forecasts")
TEX_DIR = os.path.join(SCRIPT_DIR, "..", "tex") # Still defined if needed

# Define Baseline Model Name
BASELINE_MODEL = "CLIPER"

def load_processed_data(stn_code):
    """Loads the ground truth processed data."""
    pattern = os.path.join(PROCESSED_DIR, f"{stn_code}_15min_qc.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    df = pd.read_csv(files[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df

def get_model_forecasts(stn_code):
    """
    Scans FORECAST_DIR for all subdirectories (Models).
    Standardizes names based on folder names: Folder-Direct or Folder-Kappa.
    """
    results = {'GHI': {}, 'DNI': {}}
    
    if not os.path.exists(FORECAST_DIR):
        return results

    model_dirs = [d for d in os.listdir(FORECAST_DIR) if os.path.isdir(os.path.join(FORECAST_DIR, d))]
    
    for m_dir in model_dirs:
        file_path = os.path.join(FORECAST_DIR, m_dir, f"{stn_code}_forecast_2024.csv")
        if not os.path.exists(file_path):
            continue
            
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            for col in df.columns:
                # Variable identification
                if "pred_GHI_" in col: var = "GHI"
                elif "pred_DNI_" in col: var = "DNI"
                else: continue
                
                # Model Naming based on folder
                if m_dir == "CLIPER":
                    m_name = "CLIPER"
                else:
                    if "direct" in col: suffix = "Direct"
                    elif "csky" in col or "kappa" in col: suffix = "Kappa"
                    else: suffix = col.split("_")[-1].capitalize()
                    
                    m_name = f"{m_dir}-{suffix}"
                
                results[var][m_name] = df[col]
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return results

def calculate_metrics(y_true, y_pred, zenith):
    """Calculates RMSE, nRMSE, MBE, nMBE for valid daytime data."""
    # Align indices
    common_idx = y_true.index.intersection(y_pred.index)
    t = y_true.loc[common_idx]
    p = y_pred.loc[common_idx]
    z = zenith.loc[common_idx]
    
    # Filter Daytime (Zenith < 85)
    mask = (z < 85) & t.notna() & p.notna()
    
    t_clean = t[mask]
    p_clean = p[mask]
    
    if len(t_clean) == 0:
        return np.nan, np.nan, np.nan, np.nan
        
    # Stats
    diff = p_clean - t_clean
    mse = (diff ** 2).mean()
    rmse = np.sqrt(mse)
    mbe = diff.mean()
    
    # Normalize
    mean_val = t_clean.mean()
    if mean_val != 0:
        nrmse = (rmse / mean_val) * 100
        nmbe = (mbe / mean_val) * 100
    else:
        nrmse = np.nan
        nmbe = np.nan
    
    return rmse, nrmse, mbe, nmbe

def main():
    print("--- Starting Forecast Evaluation ---")
    
    # Get List of Stations
    station_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_15min_qc.csv")))
    raw_stations = [os.path.basename(f).split('_')[0] for f in station_files]
    
    # Mapping for display (Change GCM/GCW to GWN and capitalize)
    STN_MAPPING = {"gcm": "GWN", "gcw": "GWN"}
    stations = [STN_MAPPING.get(s, s).upper() for s in raw_stations]
    
    all_stations_metrics = {} # nested dict: stn -> var -> model -> {'rmse':, 'nrmse':, ...}
    
    all_models = set()
    
    for i, raw_stn in enumerate(raw_stations):
        stn = stations[i]
        print(f"Processing {stn} (from {raw_stn})...")
        
        # Load Ground Truth
        df_obs = load_processed_data(raw_stn)
        if df_obs is None: continue
            
        # Filter for 2024
        try:
            df_obs = df_obs.loc['2024']
        except KeyError: continue
            
        # Load Forecasts
        forecasts = get_model_forecasts(raw_stn) # {'GHI': {mdl: series}, ...}
        
        metrics = {'GHI': {}, 'DNI': {}}
        
        # Calculate Metrics
        for var in ['GHI', 'DNI']:
            obs_col = f"measured_{var}"
            if obs_col not in df_obs.columns:
                continue
            
            y_true = df_obs[obs_col]
            zenith = df_obs['zenith_angle']
            
            for m_name, y_pred in forecasts[var].items():
                all_models.add(m_name)
                rmse, nrmse, mbe, nmbe = calculate_metrics(y_true, y_pred, zenith)
                
                metrics[var][m_name] = {
                    'rmse': rmse, 'nrmse': nrmse, 
                    'mbe': mbe, 'nmbe': nmbe
                }
        
        all_stations_metrics[stn] = metrics

    # --- Build Tables ---
    # Desired Format: Two tables (GHI, DNI) in one file.
    # Structure: Rows = Models, Columns = Stations.
    
    # Sort models by preferred order: CLIPER, XGBoost, etc.
    # We use a helper to define weight
    def model_sort_key(m):
        # Priority mapping
        if m.startswith("CLIPER"): return (0, m)
        if m.startswith("XGBoost"): return (1, m)
        if m.startswith("Chronos-Bolt"): return (2, m)
        if m.startswith("Chronos-2"): return (3, m)
        if m.startswith("TimesFM-2.5"): return (4, m)
        if m.startswith("TTM-R1"): return (5, m)
        if m.startswith("TTM-R2"): return (6, m)
        if m.startswith("TabPFN-2.5"): return (7, m)
        if m.startswith("TiRex"): return (8, m)
        
        return (99, m)
    
    sorted_models = sorted(list(all_models), key=model_sort_key)
        
    variables = ['GHI', 'DNI']
    
    # Store numeric DataFrames for Skill Score calculation later
    # Structure: { 'GHI': pd.DataFrame(index=Models, columns=Stations), ... }
    rmse_numeric = {}
    
    # helper to build numeric DF
    def build_numeric_df(var, metric_key, nmetric_key=None):
        # Rows: Models, Cols: Stations
        df = pd.DataFrame(index=sorted_models, columns=stations)
        
        for mdl in sorted_models:
            for stn in stations:
                res = all_stations_metrics.get(stn, {}).get(var, {}).get(mdl, {})
                val = res.get(metric_key, np.nan)
                df.at[mdl, stn] = val
                
        return df.astype(float)

    # Helper to build display DF (formatted string) from numeric DFs
    def build_display_df(df_main, df_norm=None, best_mode='min'):
        """
        Builds a display DataFrame with LaTeX bolding for best results.
        best_mode: 'min', 'max', or 'abs_min'
        """
        # Rows: Models, Cols: Stations
        display_df = pd.DataFrame(index=sorted_models, columns=stations + ['Average'])
        
        # Identify "Best" per station (column)
        best_indices = {}
        for col in stations + ['Average']:
            if col == 'Average':
                # Calculate average column numeric values
                series = df_main.mean(axis=1) if col not in df_main.columns else df_main[col]
            else:
                series = df_main[col]
            
            valid_series = series.dropna()
            if valid_series.empty:
                best_indices[col] = None
                continue
                
            if best_mode == 'min':
                best_indices[col] = valid_series.idxmin()
            elif best_mode == 'max':
                best_indices[col] = valid_series.idxmax()
            elif best_mode == 'abs_min':
                best_indices[col] = valid_series.abs().idxmin()
        
        for mdl in sorted_models:
            vals_main = df_main.loc[mdl].values.astype(float)
            
            # Per Station
            for stn in stations + ['Average']:
                if stn == 'Average':
                    v = np.nanmean(vals_main)
                    if df_norm is not None:
                        vals_norm = df_norm.loc[mdl].values.astype(float)
                        nv = np.nanmean(vals_norm)
                else:
                    v = df_main.at[mdl, stn]
                    if df_norm is not None:
                        nv = df_norm.at[mdl, stn]
                
                if np.isnan(v):
                    display_df.at[mdl, stn] = ""
                    continue
                
                # Format value string
                if df_norm is not None:
                    # Format: "Val (nVal)"
                    val_str = f"{v:.1f} ({nv:.1f})"
                else:
                    # Just Val
                    val_str = f"{v:.1f}"
                
                # Highlight if best
                if best_indices[stn] == mdl:
                    val_str = f"\\textbf{{{val_str}}}"
                
                # Replace minus sign with LaTeX minus
                val_str = val_str.replace("-", "$-$")
                
                display_df.at[mdl, stn] = val_str
                
        return display_df

    # 1. RMSE Analysis
    rmse_displays = {}
    for var in variables:
        df_rmse = build_numeric_df(var, 'rmse')
        df_nrmse = build_numeric_df(var, 'nrmse')
        rmse_numeric[var] = df_rmse # Store for Skill
        
        rmse_displays[var] = build_display_df(df_rmse, df_nrmse, best_mode='min')

    # 2. MBE Analysis
    mbe_displays = {}
    for var in variables:
        df_mbe = build_numeric_df(var, 'mbe')
        df_nmbe = build_numeric_df(var, 'nmbe')
        
        mbe_displays[var] = build_display_df(df_mbe, df_nmbe, best_mode='abs_min')
        
    # 3. Skill Score Analysis
    # Skill = 1 - (RMSE_Model / RMSE_Baseline)
    skill_displays = {}
    for var in variables:
        df_rmse = rmse_numeric[var]
        
        # Check if baseline exists
        if BASELINE_MODEL not in df_rmse.index:
            print(f"Warning: Baseline {BASELINE_MODEL} not found in {var}. Skipping Skill.")
            continue
            
        rmse_base = df_rmse.loc[BASELINE_MODEL] # Series (Stations)
        
        # DataFrame division (Row-wise? No, Broadcast)
        # We want Model_Row / Baseline_Row
        # df_rmse is [Models x Stations]
        # rmse_base is [Stations]
        
        # div(axis=1) matches columns (Stations)
        ratio = df_rmse.div(rmse_base, axis=1)
        skill = (1 - ratio) * 100
        
        skill_displays[var] = build_display_df(skill, df_norm=None, best_mode='max')


    # --- Save Output ---
    
    os.makedirs(TEX_DIR, exist_ok=True)
    
    def save_latex_combined(dict_displays, filename, caption_prefix):
        """Saves GHI and DNI tables into one .tex file"""
        path = os.path.join(TEX_DIR, filename)
        with open(path, "w") as f:
            for var in ['GHI', 'DNI']:
                if var not in dict_displays: continue
                df = dict_displays[var]
                
                f.write(f"% --- {caption_prefix} {var} ---\n")
                f.write(f"\\section*{{{caption_prefix} - {var}}}\n")
                f.write(df.to_latex(escape=False))
                f.write("\n\\vspace{1cm}\n\n")
        print(f"Saved {path}")

    # Save RMSE
    save_latex_combined(rmse_displays, "overall_rmse.tex", "RMSE (nRMSE %)")

    # Save MBE
    save_latex_combined(mbe_displays, "overall_mbe.tex", "MBE (nMBE %)")
    
    # Save Skill
    save_latex_combined(skill_displays, "overall_skill.tex", "Skill Score (%)")
    
    print("\nSample GHI RMSE Table:")
    print(rmse_displays['GHI'].iloc[:, :4])

if __name__ == "__main__":
    main()
