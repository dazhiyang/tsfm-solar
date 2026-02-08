
import pandas as pd
import numpy as np
import os
import glob
import pvlib
import warnings
from tqdm import tqdm

# Constants
YEARS = ["2023", "2024"]
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
SURFRAD_DIR = os.path.join(script_dir, "..", "Data", "SURFRAD")
METADATA_PATH = os.path.join(script_dir, "..", "Data", "metadata.csv")

# Suppress warnings
warnings.filterwarnings("ignore")

# SURFRAD Columns (Custom header handling)
COL_NAMES = ['year', 'jday', 'month', 'day', 'hour', 'min', 'dt', 'zen', 
             'ghi', 'qc_ghi', 'sw_up', 'qc_sw_up', 'dni', 'qc_dni', 'dhi', 'qc_dhi', 
             'uvb', 'qc_uvb', 'par', 'qc_par', 'netsolar', 'qc_netsolar', 
             'uw_solar', 'qc_uw_solar', 'lw_down', 'qc_lw_down', 'lw_up', 'qc_lw_up', 
             't_air', 'qc_t_air', 'rh', 'qc_rh', 'wind_spd', 'qc_wind_spd', 
             'wind_dir', 'qc_wind_dir', 'pressure', 'qc_pressure']

def load_surfrad_data(station_code, years=YEARS):
    """Loads all SURFRAD daily files for a station for specified years."""
    dfs = []
    for year in years:
        search_path = os.path.join(SURFRAD_DIR, station_code, str(year), f"{station_code}*.dat")
        files = sorted(glob.glob(search_path))
        
        if not files:
            continue

        for f in tqdm(files, desc=f"Reading {station_code} {year}", leave=False):
            try:
                # Skip 2 header lines, use whitespace delimiter
                df = pd.read_csv(f, delim_whitespace=True, skiprows=2, header=None, 
                                 names=COL_NAMES, usecols=range(38), engine='python')
                dfs.append(df)
            except Exception as e:
                pass
    
    if not dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Create Timestamp (UTC)
    full_df.rename(columns={'min': 'minute'}, inplace=True)
    full_df['timestamp'] = pd.to_datetime(full_df[['year', 'month', 'day', 'hour', 'minute']])
    full_df = full_df.set_index('timestamp').sort_index()
    
    # Replace -9999.9 with NaN
    full_df = full_df.replace(-9999.9, np.nan)
    
    return full_df

def apply_qc(df, lat, lon, elev):
    """Applies BSRN QC checks and filters data."""
    if df.empty:
        return df

    times = df.index
    
    # Solar Geometry
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    zenith = solpos['zenith']
    
    # Cosine Zenith
    cos_zen = np.cos(np.deg2rad(zenith))
    cos_zen_clipped = cos_zen.clip(lower=0.0001) 
    
    # Extraterrestrial Irradiance
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    
    # --- QC MASKS (True = Valid) ---
    # 1. Extremely rare limits (ERL)
    ghi_max = 1.2 * dni_extra * np.power(cos_zen.clip(lower=0), 1.2) + 50
    mask_ghi_erl = (df['ghi'] >= -2) & (df['ghi'] <= ghi_max)
    
    dhi_max = 0.75 * dni_extra * np.power(cos_zen.clip(lower=0), 1.2) + 30
    mask_dhi_erl = (df['dhi'] >= -2) & (df['dhi'] <= dhi_max)
    
    dni_max = 0.95 * dni_extra * np.power(cos_zen.clip(lower=0), 0.2) + 10
    mask_dni_erl = (df['dni'] >= -2) & (df['dni'] <= dni_max)
    
    # 2. Closure equation tests
    sum_sw = df['dhi'] + df['dni'] * cos_zen
    diff_abs = np.abs(df['ghi'] - sum_sw)
    closure_g = diff_abs / df['ghi']
    
    cond_cl_1 = (zenith < 75) & (df['ghi'] > 50)
    mask_cl_1 = ~(cond_cl_1 & (closure_g > 0.08))
    
    cond_cl_2 = (zenith >= 75) & (zenith < 93) & (df['ghi'] > 50)
    mask_cl_2 = ~(cond_cl_2 & (closure_g > 0.15))
    
    mask_closure = mask_cl_1 & mask_cl_2
    
    # 3. K-index tests
    kt = df['ghi'] / (dni_extra * cos_zen_clipped)
    kn = df['dni'] / dni_extra
    kd = df['dhi'] / df['ghi']
    
    mask_k1 = ~((df['ghi'] > 50) & (kt > 0) & (kn > 0) & (kn >= kt))
    kn_limit = (1100 + 0.03 * elev) / dni_extra
    mask_k2 = ~((df['ghi'] > 50) & (kn > 0) & (kn >= kn_limit))
    mask_k3 = ~((df['ghi'] > 50) & (kt > 0) & (kt >= 1.35))
    mask_k4 = ~((zenith < 75) & (df['ghi'] > 50) & (kd >= 1.05))
    mask_k5 = ~((zenith >= 75) & (df['ghi'] > 50) & (kd >= 1.10))
    mask_k6 = ~((kt > 0.6) & (zenith < 85) & (df['ghi'] > 150) & (kd > 0) & (kd >= 0.96))
    
    mask_k = mask_k1 & mask_k2 & mask_k3 & mask_k4 & mask_k5 & mask_k6
    
    # Apply Masks (Set bad data to NaN)
    valid_ghi = mask_ghi_erl & mask_closure & mask_k
    valid_dni = mask_dni_erl & mask_closure & mask_k
    valid_dhi = mask_dhi_erl & mask_closure & mask_k
    
    df.loc[~valid_ghi, 'ghi'] = np.nan
    df.loc[~valid_dni, 'dni'] = np.nan
    df.loc[~valid_dhi, 'dhi'] = np.nan
    
    return df

def generate_table():
    if not os.path.exists(METADATA_PATH):
        print("Metadata file not found.")
        return

    meta = pd.read_csv(METADATA_PATH)
    
    # Prepare result structure
    results = {}
    
    for _, row in tqdm(meta.iterrows(), total=meta.shape[0], desc="Analyzing Stations"):
        stn_code = row['stn']
        lat = row['lat']
        lon = row['lon']
        elev = row['elev']
        
        # Load Raw Data
        df_sf = load_surfrad_data(stn_code)
        
        if df_sf.empty:
            results[stn_code] = 100.0
            continue
            
        # QC
        df_qc = apply_qc(df_sf, lat, lon, elev)
        
        # Resample to 15min (with 7-point threshold)
        resampler = df_qc[['ghi']].resample('15min', closed='right', label='right')
        df_15 = resampler.mean()
        counts = resampler.count()
        df_15.loc[counts['ghi'] < 7, 'ghi'] = np.nan
        
        # Create Full Skeleton Index
        full_idx = pd.date_range(start="2023-01-01 00:15:00", end="2025-01-01 00:00:00", freq="15min")
        # Ensure we cover exactly 2023-2024
        full_idx = full_idx[full_idx.year.isin([2023, 2024])]
        
        # Standardize index
        df_15 = df_15.reindex(full_idx)
        
        # Calculate Zenith for midpoints (consistency with Arrange.py)
        midpoints = df_15.index - pd.Timedelta(minutes=7.5)
        solpos = pvlib.solarposition.get_solarposition(midpoints, lat, lon)
        zenith = solpos['zenith']
        
        # Filter Day Time (Zenith <= 90)
        day_mask = zenith.values <= 90
        
        expected_points = day_mask.sum()
        valid_points = df_15['ghi'].values[day_mask]
        valid_count = np.count_nonzero(~np.isnan(valid_points))
        
        missing_pct = 100 * (1 - (valid_count / expected_points))
        results[stn_code] = missing_pct

    # --- Generate LaTeX Table ---
    # Transposed format: Stations as columns
    
    stations = meta['stn'].tolist()
    full_names = meta['full'].tolist()
    lats = meta['lat'].tolist()
    lons = meta['lon'].tolist()
    elevs = meta['elev'].tolist()
    missings = [results.get(stn, 100) for stn in stations]
    
    # Header
    latex = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * len(stations) + "}\n\\toprule\n"
    
    # Station Codes
    latex += "Station & " + " & ".join([s.upper() for s in stations]) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Full Names (Optional, might be too long for columns)
    # latex += "Name & " + " & ".join(full_names) + " \\\\\n"
    
    # Latitude
    latex += "Latitude ($^\\circ$) & " + " & ".join([f"{x:.2f}" for x in lats]) + " \\\\\n"
    
    # Longitude
    latex += "Longitude ($^\\circ$) & " + " & ".join([f"{x:.2f}" for x in lons]) + " \\\\\n"
    
    # Elevation
    latex += "Elevation (m) & " + " & ".join([f"{int(x)}" for x in elevs]) + " \\\\\n"
    
    # Missing Percentage
    latex += "Missing (\\%) & " + " & ".join([f"{x:.1f}" for x in missings]) + " \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\caption{Station metadata and missing data percentage (daytime GHI).}\n\\label{tab:metadata}\n\\end{table}"
    
    print("\n" + "="*30)
    print("LATEX TABLE OUTPUT")
    print("="*30)
    print(latex)
    print("="*30)

if __name__ == "__main__":
    generate_table()
