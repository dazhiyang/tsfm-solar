
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Configuration
STATION = "dra" # Choose a representative station
YEAR = "2024"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Processed")
FORECAST_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Forecasts")
PLOT_DIR = os.path.join(SCRIPT_DIR, "..", "tex")
os.makedirs(PLOT_DIR, exist_ok=True)

def load_data(stn):
    # Load Ground Truth
    gt_path = os.path.join(PROCESSED_DIR, f"{stn}_15min_qc.csv")
    df_gt = pd.read_csv(gt_path)
    df_gt['timestamp'] = pd.to_datetime(df_gt['timestamp'])
    df_gt = df_gt.set_index('timestamp').sort_index()
    df_gt = df_gt.loc[YEAR]
    return df_gt

def load_forecast(stn, model_subfolder):
    path = os.path.join(FORECAST_DIR, model_subfolder, f"{stn}_forecast_2024.csv")
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df

def main():
    print(f"Generating diagnostic plots for {STATION}...")
    
    # 1. Load Data
    df_gt = load_data(STATION)
    
    models = {
        'ChronosBolt': 'ChronosBolt', 
        'Chronos2': 'Chronos2',
        'TabPFN': 'TabPFN'
    }
    
    dfs = {}
    for name, folder in models.items():
        res = load_forecast(STATION, folder)
        if res is not None:
            dfs[name] = res
            
    if not dfs:
        print("No forecast data found.")
        return

    # 2. Scatter Plots (Kappa vs Observed Kappa)
    plt.figure(figsize=(15, 5))
    
    n_models = len(dfs)
    if n_models == 0: return

    for i, (name, df_pred) in enumerate(dfs.items()):
        plt.subplot(1, n_models, i+1)
        
        # Align
        common_idx = df_gt.index.intersection(df_pred.index)
        
        # Get Ground Truth Kappa
        meas_ghi = df_gt.loc[common_idx, 'measured_GHI']
        cs_ghi = df_gt.loc[common_idx, 'clear-sky_GHI']
        
        # Safe Divide for Kappa
        # Filter where CS > 10 to avoid noise at sunrise/sunset
        valid_cs = cs_ghi > 10
        obs_kappa = np.divide(meas_ghi, cs_ghi, out=np.zeros_like(meas_ghi, dtype=float), where=valid_cs)
        
        # Get Predicted Kappa (Reversing the reconstruction)
        # Prediction column is Reconstructed GHI
        col = 'pred_GHI_csky'
        if col not in df_pred.columns:
            if 'pred_GHI_tabpfn_csky' in df_pred.columns:
                col = 'pred_GHI_tabpfn_csky'
            else:
                # Fallback
                cols = [c for c in df_pred.columns if 'csky' in c and 'GHI' in c]
                if cols: col = cols[0]
            
        pred_ghi = df_pred.loc[common_idx, col]
        pred_kappa = np.divide(pred_ghi, cs_ghi, out=np.zeros_like(pred_ghi, dtype=float), where=valid_cs)
        
        # Filter Day for Plotting (Zenith < 85)
        zenith = df_gt.loc[common_idx, 'zenith_angle']
        mask = (zenith < 85) & meas_ghi.notna() & pred_ghi.notna() & valid_cs
        
        y_true = obs_kappa[mask]
        y_pred = pred_kappa[mask]
        
        # Plot
        plt.scatter(y_true, y_pred, alpha=0.1, s=1)
        plt.plot([0, 1.5], [0, 1.5], 'r--')
        plt.xlabel("Observed Kappa")
        plt.ylabel(f"Predicted Kappa ({name})")
        plt.title(f"{name} - Kappa Scatter")
        plt.xlim(0, 1.2)
        plt.ylim(0, 1.2)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{STATION}_kappa_scatter_diagnostic.png"))
    print(f"Saved Kappa scatter plot to {PLOT_DIR}")
    
    # 3. Time Series Zoom (Kappa)
    start_zoom = pd.Timestamp(f"{YEAR}-06-01")
    end_zoom = pd.Timestamp(f"{YEAR}-06-05")
    
    plt.figure(figsize=(15, 8))
    
    # Plot GT Kappa
    zoom_gt = df_gt[start_zoom:end_zoom]
    
    # Calculate Zoom Kappa
    z_meas = zoom_gt['measured_GHI']
    z_cs = zoom_gt['clear-sky_GHI']
    z_valid = z_cs > 10
    z_kappa = np.divide(z_meas, z_cs, out=np.zeros_like(z_meas, dtype=float), where=z_valid)
    z_kappa[~z_valid] = np.nan # Hide night
    
    plt.plot(zoom_gt.index, z_kappa, color='black', label='Observed Kappa', linewidth=2)

    colors = {'ChronosBolt': 'blue', 'Chronos2': 'green', 'TabPFN': 'orange'}
    
    for name, df_pred in dfs.items():
        zoom_pred = df_pred[start_zoom:end_zoom]
        common = zoom_gt.index.intersection(zoom_pred.index)
        
        # Resolve col
        col = 'pred_GHI_csky'
        if col not in df_pred.columns:
            if 'pred_GHI_tabpfn_csky' in df_pred.columns:
                col = 'pred_GHI_tabpfn_csky'
            else:
                cols = [c for c in df_pred.columns if 'csky' in c and 'GHI' in c]
                if cols: col = cols[0]

        # Calculate Pred Kappa
        p_ghi = zoom_pred.loc[common, col]
        p_cs = df_gt.loc[common, 'clear-sky_GHI'] # Ensure alignment using global GT
        p_valid = p_cs > 10
        
        p_kappa = np.divide(p_ghi, p_cs, out=np.zeros_like(p_ghi, dtype=float), where=p_valid)
        p_kappa[~p_valid] = np.nan
        
        plt.plot(common, p_kappa, label=f"{name}", color=colors.get(name, 'red'), alpha=0.8)
        
    plt.title(f"Kappa Forecast Zoom (June 1-5, {STATION})")
    plt.ylabel("Clear-Sky Index (Kappa)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.2) # Cap at 1.2 usually
    
    plt.savefig(os.path.join(PLOT_DIR, f"{STATION}_kappa_timeseries_zoom.png"))
    print(f"Saved Kappa time-series zoom to {PLOT_DIR}")

if __name__ == "__main__":
    main()
