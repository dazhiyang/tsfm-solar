
import pandas as pd
import numpy as np
import os
from plotnine import *
from pypalettes import load_cmap
from mizani.formatters import date_format
from matplotlib import rcParams

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Processed")
FORECAST_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Forecasts")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "tex")
STN = "dra"

# Days
CLEAR_DAY = "2024-06-30"
CLOUDY_DAY = "2024-12-22"

# Color Palette (OkabeIto)
cmap = load_cmap("OkabeIto")
# Explicit Mapping: Ground Truth = Black, others use OkabeIto colors.
# We ensure Ground Truth is always black and models receive distinct colors.
COLORS = {"Ground Truth": "#000000"}
OTHER_MODELS = [
    "CLIPER", "XGBoost-Kappa", "Chronos-2-Kappa", 
    "TimesFM-2.5-Kappa", "TTM-R2-Kappa", "TabPFN-2.5-Kappa", "TiRex-Kappa"
]
for i, m_name in enumerate(OTHER_MODELS):
    COLORS[m_name] = cmap.colors[i % len(cmap.colors)]

# Model Standard Names
MODEL_ORDER = ["Ground Truth"] + OTHER_MODELS

def load_data():
    # 1. Load Ground Truth
    gt_path = os.path.join(PROCESSED_DIR, f"{STN}_15min_qc.csv")
    df_gt = pd.read_csv(gt_path)
    df_gt['timestamp'] = pd.to_datetime(df_gt['timestamp'])
    
    # Shift UTC to Local (UTC-8 for DRA)
    df_gt['timestamp'] = df_gt['timestamp'] - pd.Timedelta(hours=8)
    df_gt = df_gt.set_index('timestamp').sort_index()
    
    # 2. Scan Forecasts
    model_dirs = [d for d in os.listdir(FORECAST_DIR) if os.path.isdir(os.path.join(FORECAST_DIR, d))]
    
    plot_data = []
    days = [CLEAR_DAY, CLOUDY_DAY]
    
    for d_str in days:
        # In local time, we want the full day
        start = pd.Timestamp(d_str)
        end = start + pd.Timedelta(hours=24)
        
        subset_gt = df_gt.loc[start : end].copy()
        subset_gt['Condition'] = f"Clear-sky ({CLEAR_DAY})" if d_str == CLEAR_DAY else f"Cloudy ({CLOUDY_DAY})"
        
        # Ground Truth
        for var in ['GHI', 'DNI']:
            display_var = "BNI" if var == "DNI" else var
            temp = subset_gt[['measured_' + var, 'zenith_angle', 'Condition']].copy()
            temp = temp.rename(columns={'measured_' + var: 'Value'})
            temp['Variable'] = display_var
            temp['Model'] = "Ground Truth"
            temp['Timestamp'] = temp.index
            plot_data.append(temp)

        # Forecasts
        for m_name in MODEL_ORDER:
            if m_name == "Ground Truth": continue
            
            # Directory mapping
            if m_name == "CLIPER": m_dir = "CLIPER"
            else: m_dir = m_name.replace("-Kappa", "")
            
            f_path = os.path.join(FORECAST_DIR, m_dir, f"{STN}_forecast_2024.csv")
            if not os.path.exists(f_path): continue
            
            df_f = pd.read_csv(f_path)
            df_f['timestamp'] = pd.to_datetime(df_f['timestamp'])
            
            # Apply same UTC-8 shift
            df_f['timestamp'] = df_f['timestamp'] - pd.Timedelta(hours=8)
            df_f = df_f.set_index('timestamp').sort_index()
            
            cols = [c for c in df_f.columns if ("csky" in c or "kappa" in c or "comb" in c)]
            
            for col in cols:
                var = "GHI" if "_GHI_" in col else "DNI"
                display_var = "BNI" if var == "DNI" else var
                
                f_subset = df_f.loc[start : end].copy()
                if f_subset.empty: continue
                
                temp = f_subset[[col]].copy()
                temp = temp.rename(columns={col: 'Value'})
                temp['Variable'] = display_var
                temp['Model'] = m_name
                temp['Condition'] = f"Clear-sky ({CLEAR_DAY})" if d_str == CLEAR_DAY else f"Cloudy ({CLOUDY_DAY})"
                temp['Timestamp'] = temp.index
                plot_data.append(temp)

    df_final = pd.concat(plot_data)
    
    # Daytime filter (Zenith < 90)
    # Note: Zenith in the files is already matched to the timestamp
    # Since we shifted both timestamp and index consistently, the mapping remains correct.
    zenith_map = df_gt['zenith_angle'].to_dict()
    df_final['Zenith'] = df_final['Timestamp'].map(zenith_map)
    df_final = df_final[df_final['Zenith'] < 90].copy()

    return df_final

def main():
    df = load_data()
    
    # Factor levels
    available_models = [m for m in MODEL_ORDER if m in df['Model'].unique()]
    df['Model'] = pd.Categorical(df['Model'], categories=available_models, ordered=True)
    
    # Ensure GHI comes before BNI
    df['Variable'] = pd.Categorical(df['Variable'], categories=['GHI', 'BNI'], ordered=True)
    
    # Dimensions: 160 x 85 mm
    mm_to_in = 0.03937
    width_in = 160 * mm_to_in
    height_in = 85 * mm_to_in
    
    # Font settings for plotnine (via matplotlib)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Times']
    
    p = (
        ggplot(df, aes(x='Timestamp', y='Value', color='Model'))
        + geom_line(data=df[df['Model'] == 'Ground Truth'], size=0.4)
        + geom_line(data=df[df['Model'] != 'Ground Truth'], size=0.2)
        + facet_wrap('~Condition + Variable', scales='free', ncol=2)
        + scale_color_manual(values=COLORS)
        + theme_minimal()
        + labs(x=None, y="Irradiance (W m$^{-2}$)", color=None)
        + theme(
            text=element_text(family="Times New Roman", size=8),
            legend_position="bottom",
            legend_box_margin=0,
            legend_key_size=10,
            panel_grid_minor=element_blank(),
            strip_text=element_text(family="Times New Roman", size=8),
            axis_text=element_text(family="Times New Roman", size=8),
            axis_title=element_text(family="Times New Roman", size=8),
            figure_size=(width_in, height_in)
        )
        + scale_x_datetime(labels=date_format("%H:%M"))
    )

    out_file = os.path.join(OUTPUT_DIR, "dra_example.pdf")
    p.save(out_file, dpi=300)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
