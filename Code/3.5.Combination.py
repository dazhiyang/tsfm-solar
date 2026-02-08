
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
TEX_DIR = os.path.join(SCRIPT_DIR, "..", "tex")

# Baseline Model Name
BASELINE_MODEL = "CLIPER"

# Define Groups
GROUPS = {
    'Regression': ['XGBoost-Kappa', 'TabPFN-2.5-Kappa'],
    'TSFMs': [
        'Chronos-Bolt-Kappa', 'Chronos-2-Kappa', 
        'TimesFM-2.5-Kappa', 'TTM-R1-Kappa', 
        'TTM-R2-Kappa', 'TiRex-Kappa'
    ],
    'All': [
        'XGBoost-Kappa', 'TabPFN-2.5-Kappa',
        'Chronos-Bolt-Kappa', 'Chronos-2-Kappa', 
        'TimesFM-2.5-Kappa', 'TTM-R1-Kappa', 
        'TTM-R2-Kappa', 'TiRex-Kappa'
    ]
}

def load_processed_data(stn_code):
    """Loads the ground truth processed data."""
    pattern = os.path.join(PROCESSED_DIR, f"{stn_code}_15min_qc.csv")
    files = glob.glob(pattern)
    if not files: return None
    df = pd.read_csv(files[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df

def get_model_forecasts(stn_code):
    """
    Standard load logic from 3.1.Evaluate_Forecast.py
    """
    results = {'GHI': {}, 'DNI': {}}
    if not os.path.exists(FORECAST_DIR): return results
    model_dirs = [d for d in os.listdir(FORECAST_DIR) if os.path.isdir(os.path.join(FORECAST_DIR, d))]
    
    for m_dir in model_dirs:
        file_path = os.path.join(FORECAST_DIR, m_dir, f"{stn_code}_forecast_2024.csv")
        if not os.path.exists(file_path): continue
            
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            for col in df.columns:
                if "pred_GHI_" in col: var = "GHI"
                elif "pred_DNI_" in col: var = "DNI"
                else: continue
                
                if m_dir == "CLIPER": m_name = "CLIPER"
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
    common_idx = y_true.index.intersection(y_pred.index)
    if common_idx.empty: return np.nan, np.nan, np.nan, np.nan
    
    t = y_true.loc[common_idx]
    p = y_pred.loc[common_idx]
    z = zenith.loc[common_idx]
    
    mask = (z < 85) & t.notna() & p.notna()
    t_clean = t[mask]
    p_clean = p[mask]
    
    if len(t_clean) == 0: return np.nan, np.nan, np.nan, np.nan
        
    diff = p_clean - t_clean
    rmse = np.sqrt((diff ** 2).mean())
    mbe = diff.mean()
    
    mean_obs = t_clean.mean()
    if mean_obs != 0:
        nrmse = (rmse / mean_obs) * 100
        nmbe = (mbe / mean_obs) * 100
    else:
        nrmse, nmbe = np.nan, np.nan
        
    return rmse, nrmse, mbe, nmbe

def build_display_df(df_main, df_norm, stations, group_names, best_mode='min'):
    """
    Builds a display DataFrame with LaTeX bolding for best results.
    Format: "Val (nVal)"
    """
    display_df = pd.DataFrame(index=group_names, columns=stations + ['Average'])
    
    # Identify "Best" per station (column)
    best_indices = {}
    for col in stations + ['Average']:
        if col == 'Average':
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
    
    for grp in group_names:
        if grp not in df_main.index: 
            for stn in stations + ['Average']: display_df.at[grp, stn] = ""
            continue
            
        vals_main = df_main.loc[grp].values.astype(float)
        if df_norm is not None:
            vals_norm = df_norm.loc[grp].values.astype(float)
        
        for stn in stations + ['Average']:
            if stn == 'Average':
                v = np.nanmean(vals_main)
                nv = np.nanmean(vals_norm) if df_norm is not None else None
            else:
                v = df_main.at[grp, stn]
                nv = df_norm.at[grp, stn] if df_norm is not None else None
            
            if np.isnan(v):
                display_df.at[grp, stn] = ""
                continue
            
            if nv is not None:
                val_str = f"{v:.1f} ({nv:.1f})"
            else:
                val_str = f"{v:.1f}" if best_mode != 'max' else f"{v:.2f}"
            
            if best_indices[stn] == grp:
                val_str = f"\\textbf{{{val_str}}}"
            
            val_str = val_str.replace("-", "$-$")
            display_df.at[grp, stn] = val_str
            
    return display_df

def main():
    print("--- Starting Forecast Combination Evaluation ---")
    station_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_15min_qc.csv")))
    raw_stations = [os.path.basename(f).split('_')[0] for f in station_files]
    STN_MAPPING = {"gcm": "GWN", "gcw": "GWN"}
    stations_display = [STN_MAPPING.get(s, s).upper() for s in raw_stations]
    
    results_list = []
    # Targeted Order: Regression, TSFMs, All, Oracle
    group_order = ['Regression', 'TSFMs', 'All', 'Oracle']

    for i, raw_stn in enumerate(raw_stations):
        stn_disp = stations_display[i]
        print(f"Processing {stn_disp}...")
        df_obs = load_processed_data(raw_stn)
        if df_obs is None: continue
        try: df_obs = df_obs.loc['2024']
        except KeyError: continue
            
        forecasts = get_model_forecasts(raw_stn)
        for var in ['GHI', 'DNI']:
            obs_col = f"measured_{var}"
            if obs_col not in df_obs.columns: continue
            y_true = df_obs[obs_col]
            zenith = df_obs['zenith_angle']
            
            baseline_rmse = np.nan
            if BASELINE_MODEL in forecasts[var]:
                baseline_rmse, _, _, _ = calculate_metrics(y_true, forecasts[var][BASELINE_MODEL], zenith)
            
            # 1. Standard Ensembles (Regression, TSFMs, All)
            for group_name, model_list in GROUPS.items():
                preds = []
                for m in model_list:
                    if m in forecasts[var]:
                        preds.append(forecasts[var][m])
                
                if not preds: continue
                combined_df = pd.concat(preds, axis=1)
                # Deduplicate columns if any
                combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
                group_pred = combined_df.mean(axis=1)
                
                rmse, nrmse, mbe, nmbe = calculate_metrics(y_true, group_pred, zenith)
                skill = (1 - (rmse / baseline_rmse)) * 100 if not np.isnan(baseline_rmse) else np.nan
                results_list.append({
                    'Station': stn_disp, 'Variable': var, 'Group': group_name,
                    'RMSE': rmse, 'nRMSE': nrmse, 'MBE': mbe, 'nMBE': nmbe, 'Skill': skill
                })
            
            # 2. 'Oracle' model
            # Use all unique models from 'All' pool
            all_list = GROUPS['All']
            preds = [forecasts[var][m] for m in all_list if m in forecasts[var]]
            if preds:
                all_df = pd.concat(preds, axis=1)
                all_df = all_df.loc[:, ~all_df.columns.duplicated()]
                
                common_idx = y_true.index.intersection(all_df.index)
                t_slice = y_true.loc[common_idx]
                f_slice = all_df.loc[common_idx]
                
                # abs error per model
                abs_err = f_slice.sub(t_slice, axis=0).abs()
                best_mdls_indices = abs_err.to_numpy().argmin(axis=1)
                
                f_np = f_slice.to_numpy()
                oracle_values = f_np[np.arange(len(f_np)), best_mdls_indices]
                oracle_pred = pd.Series(oracle_values, index=common_idx)
                
                rmse, nrmse, mbe, nmbe = calculate_metrics(y_true, oracle_pred, zenith)
                skill = (1 - (rmse / baseline_rmse)) * 100 if not np.isnan(baseline_rmse) else np.nan
                results_list.append({
                    'Station': stn_disp, 'Variable': var, 'Group': 'Oracle',
                    'RMSE': rmse, 'nRMSE': nrmse, 'MBE': mbe, 'nMBE': nmbe, 'Skill': skill
                })
    
    df_results = pd.DataFrame(results_list)
    os.makedirs(TEX_DIR, exist_ok=True)
    COMBINATION_TEX = os.path.join(TEX_DIR, "combination_performance.tex")
    
    with open(COMBINATION_TEX, "w") as f:
        # RMSE Section
        for var in ['GHI', 'DNI']:
            df_var = df_results[df_results['Variable'] == var]
            if df_var.empty: continue
            
            df_rmse = df_var.pivot_table(index='Group', columns='Station', values='RMSE').reindex(group_order)
            df_nrmse = df_var.pivot_table(index='Group', columns='Station', values='nRMSE').reindex(group_order)
            display_rmse = build_display_df(df_rmse, df_nrmse, stations_display, group_order, best_mode='min')
            
            f.write(f"\\section*{{Combination Performance - {var} - RMSE (nRMSE \\%)}}\n")
            f.write(display_rmse.to_latex(escape=False, caption=f"{var} RMSE and nRMSE for Ensembles"))
            f.write("\n\\vspace{1cm}\n\n")
            
        # Skill Section
        for var in ['GHI', 'DNI']:
            df_var = df_results[df_results['Variable'] == var]
            if df_var.empty: continue
            
            df_skill = df_var.pivot_table(index='Group', columns='Station', values='Skill').reindex(group_order)
            display_skill = build_display_df(df_skill, None, stations_display, group_order, best_mode='max')
            
            f.write(f"\\section*{{Combination Performance - {var} - Skill Score (\\%)}}\n")
            f.write(display_skill.to_latex(escape=False, caption=f"{var} Skill for Ensembles"))
            f.write("\n\\vspace{1cm}\n\n")

        # MBE Section
        for var in ['GHI', 'DNI']:
            df_var = df_results[df_results['Variable'] == var]
            if df_var.empty: continue
            
            df_mbe = df_var.pivot_table(index='Group', columns='Station', values='MBE').reindex(group_order)
            df_nmbe = df_var.pivot_table(index='Group', columns='Station', values='nMBE').reindex(group_order)
            display_mbe = build_display_df(df_mbe, df_nmbe, stations_display, group_order, best_mode='abs_min')
            
            f.write(f"\\section*{{Combination Performance - {var} - MBE (nMBE \\%)}}\n")
            f.write(display_mbe.to_latex(escape=False, caption=f"{var} MBE and nMBE for Ensembles"))
            f.write("\n\\vspace{1cm}\n\n")

    print(f"Saved results to {COMBINATION_TEX}")

if __name__ == "__main__":
    main()
