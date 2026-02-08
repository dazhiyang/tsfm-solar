
import pandas as pd
import numpy as np
import os
import glob
import warnings
from plotnine import *
from matplotlib import rcParams
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Processed")
FORECAST_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Forecasts")
TEX_DIR = os.path.join(SCRIPT_DIR, "..", "tex")
SKILL_COND_OUTPUT = os.path.join(TEX_DIR, "sky_condition_skill.pdf")
SKILL_TABLE_OUTPUT = os.path.join(TEX_DIR, "sky_condition_skill_table.tex")
INDEX_COMBINED_OUTPUT = os.path.join(TEX_DIR, "sky_condition_index_combined.pdf")

# Constants
BASELINE_MODEL = "CLIPER"
MM_TO_IN = 0.03937

# Plotting Style
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 8

def load_all_data():
    """
    Loads data for all stations in 2024.
    """
    station_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_15min_qc.csv")))
    all_obs = []
    all_forecast_series = {'GHI': {}, 'DNI': {}} # model -> list of series
    
    print("Loading data...")
    for f in station_files:
        raw_stn = os.path.basename(f).split('_')[0]
        
        # 1. Load Observations
        df = pd.read_csv(f)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        try:
            df = df.loc['2024']
        except KeyError:
            continue
        df['station'] = raw_stn
        df = df.set_index('station', append=True).swaplevel(0, 1)
        all_obs.append(df)
        
        # 2. Load Forecasts
        model_dirs = [d for d in os.listdir(FORECAST_DIR) if os.path.isdir(os.path.join(FORECAST_DIR, d))]
        for m_dir in model_dirs:
            fc_path = os.path.join(FORECAST_DIR, m_dir, f"{raw_stn}_forecast_2024.csv")
            if not os.path.exists(fc_path): continue
            try:
                fc_df = pd.read_csv(fc_path)
                fc_df['timestamp'] = pd.to_datetime(fc_df['timestamp'])
                fc_df = fc_df.set_index('timestamp').sort_index()
                for col in fc_df.columns:
                    var = None
                    if "pred_GHI_" in col: var = "GHI"
                    elif "pred_DNI_" in col: var = "DNI"
                    
                    if var:
                        if m_dir == "CLIPER": m_name = "CLIPER"
                        else:
                            if "direct" in col: suffix = "Direct"
                            elif "csky" in col or "kappa" in col: suffix = "Kappa"
                            else: suffix = col.split("_")[-1].capitalize()
                            m_name = f"{m_dir}-{suffix}"
                        
                        s_df = fc_df[col].to_frame()
                        s_df['station'] = raw_stn
                        s_df = s_df.set_index('station', append=True).swaplevel(0, 1)
                        if m_name not in all_forecast_series[var]: all_forecast_series[var][m_name] = []
                        all_forecast_series[var][m_name].append(s_df[col])
            except Exception as e:
                print(f"Error loading {fc_path}: {e}")

    if not all_obs: return None, None
    obs_concat = pd.concat(all_obs)
    final_forecasts = {'GHI': {}, 'DNI': {}}
    for var in ['GHI', 'DNI']:
        for m_name, series_list in all_forecast_series[var].items():
            final_forecasts[var][m_name] = pd.concat(series_list)
    return obs_concat, final_forecasts

def classify_sky_condition(df):
    """
    Classifies sky condition using the Perez Clearness Index (epsilon)
    Reference: Perez et al. (1990)
    """
    ghi = df['measured_GHI']
    dni = df['measured_DNI']
    z_deg = df['zenith_angle']
    z_rad = np.radians(z_deg)
    
    # Calculate Diffuse Horizontal Irradiance (DHI)
    # BHI = DNI * cos(Z)
    bhi = dni * np.cos(z_rad)
    dhi = ghi - bhi
    
    # Perez epsilon formula
    # epsilon = [ ( (Idif + Idir) / Idif ) + 1.041 * Z^3 ] / [ 1 + 1.041 * Z^3 ]
    # (Idif + Idir) = GHI
    # We clip DHI to avoid division by zero or negative values due to sensor noise
    dhi_min = 1.0 # Minimum 1 W/m2 for calculation stability
    ghi_safe = ghi.clip(lower=dhi_min)
    dhi_safe = dhi.clip(lower=dhi_min)
    
    # Formula setup
    # epsilon = [ ( (Idif + Idir) / Idif ) + 1.041 * Z^3 ] / [ 1 + 1.041 * Z^3 ]
    # The user corrected that Idir is "simple DNI" (Direct Normal), not projected.
    # So numerator ratio is (DHI + DNI) / DHI
    z3 = z_rad**3
    num = ((dhi_safe + dni.clip(lower=0)) / dhi_safe) + 1.041 * z3
    den = 1 + 1.041 * z3
    eps = num / den
    
    # Classification logic based on user ranges
    # eps = 1 to 1.065: Overcast
    # eps = 1.065 to 4.5: Intermediate/Partly Cloudy
    # eps > 4.5: Clear
    cond = pd.Series(np.nan, index=df.index)
    valid_mask = (z_deg < 85) & (ghi > 10) # Day-time clear signal
    
    is_overcast = (eps >= 1) & (eps <= 1.065)
    is_intermediate = (eps > 1.065) & (eps <= 4.5)
    is_clear = (eps > 4.5)
    
    cond[is_overcast] = 'Overcast'
    cond[is_intermediate] = 'Cloudy' # Mapping 'Intermediate' to the 'Cloudy' label used in plot
    cond[is_clear] = 'Clear'
    
    cond[~valid_mask] = np.nan
    return cond

def calculate_index_data(obs_df, forecasts, conditions, selected_models, var, sample_n=500):
    print(f"Calculating index data for {var}...")
    obs_col = 'measured_GHI' if var == 'GHI' else 'measured_DNI'
    cs_col = 'clear-sky_GHI' if var == 'GHI' else 'clear-sky_DNI'
    
    mask = conditions.notna() & obs_df[obs_col].notna() & (obs_df[cs_col] > 20) # Avoid low sun
    y_true = obs_df.loc[mask, obs_col]
    y_cs = obs_df.loc[mask, cs_col]
    c_valid = conditions[mask]
    
    k_obs = y_true / y_cs
    
    all_sampled = []
    
    for m in selected_models:
        if m not in forecasts[var]: continue
        y_pred = forecasts[var][m].reindex(y_true.index)
        k_pred = y_pred / y_cs
        
        # Clamp k to reasonable range
        k_pred = k_pred.clip(0, 1.5)
        
        df_m = pd.DataFrame({
            'Model': m,
            'Condition': c_valid,
            'k_obs': k_obs,
            'k_pred': k_pred
        })
        
        # Sampling for geom_point
        # Sample per condition to ensure visibility of all states
        sampled_m = []
        for cond_name in ['Clear', 'Cloudy', 'Overcast']:
            df_c = df_m[df_m['Condition'] == cond_name]
            if len(df_c) > sample_n:
                sampled_m.append(df_c.sample(sample_n))
            else:
                sampled_m.append(df_c)
        all_sampled.append(pd.concat(sampled_m))
        
    return pd.concat(all_sampled)

def plot_index_comparison(df_plot, selected_models, width_in, height_in):
    # Colors: Clear (#E69F00), Cloudy (#56B4E9), Overcast (#009E73)
    cond_colors = {"Clear": "#E69F00", "Cloudy": "#56B4E9", "Overcast": "#009E73"}
    
    # Use only model names for the stripes as requested
    df_plot['Facet_Label'] = df_plot['Model']
    
    # We still need a unique identifier for 18 panels if using facet_wrap
    # Actually, if we have duplicate names, facet_wrap might combine them.
    # To keep the 6x3 stack (GHI top, BNI bottom), we use a hidden grouping or nested facets if possible.
    # Since plotnine facet_wrap needs unique labels for unique panels:
    # We will use the model names but ensure the 18 panels are distinct by adding invisible spaces 
    # for the BNI group to keep labels visually "the same" but programmatically different.
    
    facet_categories = []
    # GHI panels (Top 3x3)
    for m in selected_models:
        facet_categories.append(m)
    # BNI panels (Bottom 3x3) - add a trailing zero-width space or space
    for m in selected_models:
        facet_categories.append(m + " ") 
        
    df_plot.loc[df_plot['Variable'] == 'BNI', 'Facet_Label'] = df_plot.loc[df_plot['Variable'] == 'BNI', 'Model'] + " "
    
    df_plot['Facet_Label'] = pd.Categorical(df_plot['Facet_Label'], categories=facet_categories, ordered=True)
    df_plot['Condition'] = pd.Categorical(df_plot['Condition'], categories=['Clear', 'Cloudy', 'Overcast'], ordered=True)
    
    # Ensure "Clear" points are on top
    plot_order = {'Overcast': 0, 'Cloudy': 1, 'Clear': 2}
    df_plot['z_order'] = df_plot['Condition'].map(plot_order)
    df_plot = df_plot.sort_values(['Facet_Label', 'z_order'])

    p = (
        ggplot(df_plot, aes(x='k_obs', color='Condition', fill='Condition'))
        + geom_abline(slope=1, intercept=0, linetype='dashed', color='black', alpha=0.5)
        + geom_point(aes(y='k_pred'), size=0.4, alpha=0.4, stroke=0)
        + facet_wrap('~Facet_Label', ncol=3)
        + labs(x="Observed $\kappa$ (dimensionless)", 
               y="Forecast $\kappa$ (dimensionless)")
        + scale_x_continuous(breaks=[0.0, 0.5, 1.0], limits=[0, 1.2])
        + scale_y_continuous(breaks=[0.0, 0.5, 1.0], limits=[0, 1.2])
        + scale_color_manual(values=cond_colors)
        + scale_fill_manual(values=cond_colors)
        + guides(
            color=guide_legend(override_aes={'size': 4, 'alpha': 1}), # Force larger, opaque markers in legend
            fill=guide_legend(override_aes={'alpha': 1})
        )
        + theme_minimal()
        + theme(
            text=element_text(family="Times New Roman", size=8),
            strip_text=element_text(size=7, weight='bold'),
            legend_position="bottom",
            legend_title=element_blank(),
            legend_key_size=15, # Box size
            legend_entry_spacing_x=15, 
            figure_size=(width_in, height_in),
            panel_spacing=0.015,
            plot_background=element_rect(fill='white'),
            panel_background=element_rect(fill='white'),
            legend_margin=0
        )
    )
    return p

def plot_skill_bars(df_metrics, selected_models, colors, width_in, height_in):
    df_plot = df_metrics.copy()
    df_plot['Condition'] = pd.Categorical(df_plot['Condition'], categories=['Clear', 'Cloudy', 'Overcast'], ordered=True)
    df_plot['Model'] = pd.Categorical(df_plot['Model'], categories=selected_models, ordered=True)
    df_plot['Variable'] = pd.Categorical(df_plot['Variable'], categories=['GHI', 'BNI'], ordered=True)
    
    p = (
        ggplot(df_plot, aes(x='Condition', y='Skill', fill='Model'))
        + geom_bar(stat='identity', position=position_dodge(width=0.8), width=0.7)
        + geom_hline(yintercept=0, color="black", size=0.5)
        + facet_wrap('~Variable', ncol=1, scales='free_y') 
        + labs(x = "Sky condition", y="Skill score ($s$, %)")
        + scale_y_continuous(trans='pseudo_log') # Refine scale to handle outliers/negatives
        + scale_fill_manual(values=colors)
        + guides(fill=guide_legend(nrow=3))
        + theme_minimal()
        + theme(
            text=element_text(family="Times New Roman", size=8),
            axis_text=element_text(size=8),
            strip_text=element_text(size=8), 
            legend_position="bottom",
            legend_title=element_blank(),
            legend_key_size=5, 
            legend_entry_spacing_x=2, 
            figure_size=(width_in, height_in),
            panel_spacing=0.02, 
            legend_box_spacing=0.01,
            legend_margin=0
        )
    )
    return p

def main():
    print("--- Starting Sky Condition Evaluation ---")
    obs_df, forecasts = load_all_data()
    if obs_df is None: return
    
    selected_models = [
        'CLIPER', 'XGBoost-Kappa', 'Chronos-Bolt-Kappa', 'Chronos-2-Kappa', 
        'TimesFM-2.5-Kappa', 'TTM-R1-Kappa', 'TTM-R2-Kappa', 'TabPFN-2.5-Kappa', 'TiRex-Kappa'
    ]
    model_colors = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#4b0082"]

    conditions = classify_sky_condition(obs_df)
    
    # 1. Skill Plot (Preserved)
    results = []
    for var, obs_col in [('GHI', 'measured_GHI'), ('DNI', 'measured_DNI')]:
        mask = conditions.notna()
        y_true = obs_df.loc[mask, obs_col]
        c_valid = conditions[mask]
        baseline_rmse = {}
        if BASELINE_MODEL in forecasts[var]:
            y_base = forecasts[var][BASELINE_MODEL].reindex(y_true.index)
            for c in ['Clear', 'Cloudy', 'Overcast']:
                cm = c_valid == c
                if cm.sum() > 0:
                    baseline_rmse[c] = np.sqrt(((y_base[cm]-y_true[cm])**2).mean())
        for m in selected_models:
            if m not in forecasts[var]: continue
            y_p = forecasts[var][m].reindex(y_true.index)
            for c in ['Clear', 'Cloudy', 'Overcast']:
                cm = c_valid == c
                if cm.sum() == 0: continue
                rmse = np.sqrt(((y_p[cm]-y_true[cm])**2).mean())
                skill = (1 - (rmse/baseline_rmse[c]))*100 if m != BASELINE_MODEL and c in baseline_rmse else 0.0
                results.append({'Variable': 'GHI' if var == 'GHI' else 'BNI', 'Model': m, 'Condition': c, 'Skill': skill})
    
    print("Generating skill bars...")
    df_results = pd.DataFrame(results)
    p1 = plot_skill_bars(df_results, selected_models, model_colors, 90 * MM_TO_IN, 90 * MM_TO_IN)
    p1.save(SKILL_COND_OUTPUT, dpi=300)

    # 1.1 Generate LaTeX Table
    print("Generating LaTeX table...")
    # Pivot to: Rows=Model, Columns=[Variable, Condition]
    df_table = df_results.pivot_table(index='Model', columns=['Variable', 'Condition'], values='Skill')
    # Reorder columns and index for logical flow
    df_table = df_table.reindex(selected_models)
    cols = []
    for v in ['GHI', 'BNI']:
        for c in ['Clear', 'Cloudy', 'Overcast']:
            if (v, c) in df_table.columns:
                cols.append((v, c))
    df_table = df_table[cols]
    
    with open(SKILL_TABLE_OUTPUT, 'w') as f:
        f.write(df_table.to_latex(float_format="%.2f", na_rep="-", caption="Skill Score (s, %) by Sky Condition", label="tab:skill_condition"))
    
    # 2. Combined Clear-sky Index Plots
    all_index_data = []
    for var in ['GHI', 'DNI']:
        df_v = calculate_index_data(obs_df, forecasts, conditions, selected_models, var, sample_n=2000)
        df_v['Variable'] = 'GHI' if var == 'GHI' else 'BNI'
        all_index_data.append(df_v)
    
    df_combined = pd.concat(all_index_data)
    print("Generating Combined Index Plot...")
    p_combined = plot_index_comparison(df_combined, selected_models, 80 * MM_TO_IN, 160 * MM_TO_IN)
    p_combined.save(INDEX_COMBINED_OUTPUT, dpi=300)

if __name__ == "__main__":
    main()
