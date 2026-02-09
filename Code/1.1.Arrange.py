"""
Created on Mon Feb 02 12:53:00 2026

@author: Dazhi Yang & Gemini 3 Pro (High)
@institute: Harbin Institute of Technology
"""
import pandas as pd
import requests
import os
import time
import numpy as np
import pvlib
import glob
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
API_KEY = "YOUR_API_KEY"  # Replace with your NREL API Key
EMAIL = "YOUR_EMAIL"  # Replace with your registered email
BASE_URL = f"https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv?api_key={API_KEY}"
YEARS = ["2023", "2024"]
ATTRIBUTES = "ghi,dni,dhi,clearsky_ghi,clearsky_dni,clearsky_dhi,air_temperature,wind_speed"

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script
NSRDB_DIR = os.path.join(script_dir, "..", "Data", "NSRDB")
SURFRAD_DIR = os.path.join(script_dir, "..", "Data", "SURFRAD")
PROCESSED_DIR = os.path.join(script_dir, "..", "Data", "Processed")
METADATA_PATH = os.path.join(script_dir, "..", "Data", "metadata.csv")

os.makedirs(NSRDB_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# SURFRAD Columns (Custom header handling)
COL_NAMES = ['year', 'jday', 'month', 'day', 'hour', 'min', 'dt', 'zen', 
             'ghi', 'qc_ghi', 'sw_up', 'qc_sw_up', 'dni', 'qc_dni', 'dhi', 'qc_dhi', 
             'uvb', 'qc_uvb', 'par', 'qc_par', 'netsolar', 'qc_netsolar', 
             'uw_solar', 'qc_uw_solar', 'lw_down', 'qc_lw_down', 'lw_up', 'qc_lw_up', 
             't_air', 'qc_t_air', 'rh', 'qc_rh', 'wind_spd', 'qc_wind_spd', 
             'wind_dir', 'qc_wind_dir', 'pressure', 'qc_pressure']

# --- HELPER FUNCTIONS ---

def download_nsrdb():
    stations = pd.read_csv(METADATA_PATH)
    print(f"Saving data to: {NSRDB_DIR}")
    for _, row in tqdm(stations.iterrows(), total=stations.shape[0], desc="Downloading Stations"):
        stn_id = row['stn']
        lat = row['lat']
        lon = row['lon']
        
        # Create station directory
        station_dir = os.path.join(NSRDB_DIR, stn_id)
        os.makedirs(station_dir, exist_ok=True)
        
        for year in YEARS:
            filename = f"nsrdb_{stn_id}_{year}.csv"
            save_path = os.path.join(station_dir, filename)
            
            if os.path.exists(save_path):
                print(f"Skipping {filename}, already exists.")
                continue

            # Construct the parameters
            params = {
                'api_key': API_KEY,
                'wkt': f'POINT({lon} {lat})',
                'attributes': ATTRIBUTES,
                'names': year,
                'utc': 'true',
                'full_name': 'Researcher',
                'email': EMAIL,
                'affiliation': 'Research',
                'reason': 'Academic Paper',
                'mailing_list': 'false',
                'interval': '5'
            }

            print(f"Downloading NSRDB for {stn_id} ({year})...")
            headers = {
                'content-type': "application/x-www-form-urlencoded",
                'cache-control': "no-cache"
            }
            
            # Retry logic
            session = requests.Session()
            retry = requests.adapters.HTTPAdapter(max_retries=5)
            session.mount('https://', retry)
            
            try:
                response = session.post(BASE_URL, data=params, headers=headers, timeout=300)

                if response.status_code == 200:
                    with open(save_path, 'w') as f:
                        f.write(response.text)
                else:
                    print(f"Error {response.status_code} for {stn_id}: {response.text[:200]}")
            except Exception as e:
                print(f"Failed to download {stn_id} ({year}): {e}")
            
            time.sleep(2)
            
def load_surfrad_data(station_code, years=YEARS):
    """Loads all SURFRAD daily files for a station for specified years."""
    dfs = []
    for year in years:
        search_path = os.path.join(SURFRAD_DIR, station_code, str(year), f"{station_code}*.dat")
        files = sorted(glob.glob(search_path))
        
        if not files:
            print(f"  No info found for {station_code} {year}")
            continue

        print(f"  Loading {len(files)} files for {station_code} {year}...")
        for f in tqdm(files, desc=f"Reading {year}", leave=False):
            try:
                # Skip 2 header lines, use whitespace delimiter
                df = pd.read_csv(f, delim_whitespace=True, skiprows=2, header=None, 
                                 names=COL_NAMES, usecols=range(38), engine='python')
                dfs.append(df)
            except Exception as e:
                print(f"    Error reading {os.path.basename(f)}: {e}")
    
    if not dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Create Timestamp (UTC)
    # Rename 'min' to 'minute' for standard pd.to_datetime parsing
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
    
    # Solar Geometry and E0n
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
    # Mask ALL components if closure fails, otherwise individual ERL
    # Using strictly the user request "put any bad data points as NA"
    valid_ghi = mask_ghi_erl & mask_closure & mask_k
    valid_dni = mask_dni_erl & mask_closure & mask_k
    valid_dhi = mask_dhi_erl & mask_closure & mask_k
    
    df.loc[~valid_ghi, 'ghi'] = np.nan
    df.loc[~valid_dni, 'dni'] = np.nan
    df.loc[~valid_dhi, 'dhi'] = np.nan
    
    return df

def process_station(stn_row):
    stn_code = stn_row['stn']
    lat = stn_row['lat']
    lon = stn_row['lon']
    elev = stn_row['elev']
    
    print(f"\nProcessing {stn_code}...")

    # 1. Load SURFRAD data
    df_sf = load_surfrad_data(stn_code)
            
    if df_sf.empty:
        print(f"Skipping {stn_code}: No SURFRAD data found.")
        return

    # Data already concatenated and index sorted in load_surfrad_data
    
    # 2. Apply QC (1-min)
    print("  Applying QC...")
    df_qc = apply_qc(df_sf, lat, lon, elev)
    
    # 3. Aggregate to 15-min
    # Use end time stamp for each period
    resampler = df_qc[['ghi', 'dni', 'dhi']].resample('15min', closed='right', label='right')
    df_15 = resampler.mean()
    
    # Require at least 7 data points (1-min resolution)
    counts = resampler.count()
    for col in ['ghi', 'dni', 'dhi']:
        df_15.loc[counts[col] < 7, col] = np.nan
    
    # 4. Load NSRDB
    nsrdb_dfs = []
    for y in [2023, 2024]:
        fpath = os.path.join(NSRDB_DIR, stn_code, f"nsrdb_{stn_code}_{y}.csv")
        if os.path.exists(fpath):
            try:
                # NSRDB files check: Skip 2 lines of metadata
                ndf = pd.read_csv(fpath, skiprows=2)
                ndf['timestamp'] = pd.to_datetime(ndf[['Year', 'Month', 'Day', 'Hour', 'Minute']])
                ndf = ndf.set_index('timestamp').sort_index()
                nsrdb_dfs.append(ndf)
            except Exception as e:
                print(f"  Error reading NSRDB {fpath}: {e}")
    
    if not nsrdb_dfs:
        print(f"  Warning: No NSRDB data found for {stn_code}. Filling will not occur.")
        # Create empty DF with expected columns to ensure merge compatibility
        df_ns_15 = pd.DataFrame(columns=['ghi', 'dni', 'dhi', 'clearsky_ghi', 'clearsky_dni'])
    else:
        df_ns = pd.concat(nsrdb_dfs)
        # Resample NSRDB to 15min
        cols_to_agg = ['ghi', 'dni', 'dhi', 'clearsky_ghi', 'clearsky_dni']
        # Ensure lowercase cols and replace spaces with underscores (e.g. 'clearsky ghi' -> 'clearsky_ghi')
        df_ns.columns = [c.lower().replace(' ', '_') for c in df_ns.columns]
        df_ns_15 = df_ns[cols_to_agg].resample('15min', closed='right', label='right').mean()

    # 5. Merge and Fill
    combined = pd.merge(df_15, df_ns_15, left_index=True, right_index=True, how='outer', suffixes=('_mes', '_mod'))
    
    if not df_ns_15.empty:
        combined['ghi_measured'] = combined['ghi_mes'].fillna(combined['ghi_mod'])
        combined['dni_measured'] = combined['dni_mes'].fillna(combined['dni_mod'])
        combined['modeled_GHI'] = combined['ghi_mod']
        combined['modeled_DNI'] = combined['dni_mod']
        combined['clearsky_GHI'] = combined['clearsky_ghi']
        combined['clearsky_DNI'] = combined['clearsky_dni']
    else:
        combined['ghi_measured'] = combined['ghi_mes']
        combined['dni_measured'] = combined['dni_mes']
        combined['modeled_GHI'] = np.nan
        combined['modeled_DNI'] = np.nan
        combined['clearsky_GHI'] = np.nan
        combined['clearsky_DNI'] = np.nan

    # 6. Recalculate Zenith for complete timeline
    # Use midpoint of the 15-min interval for solar position (timestamp is right-labeled)
    midpoints = combined.index - pd.Timedelta(minutes=7.5)
    solpos = pvlib.solarposition.get_solarposition(midpoints, lat, lon)
    combined['zenith_angle'] = solpos['zenith'].values
    
    # Final Selection and Rename
    out_df = combined[['ghi_measured', 'dni_measured', 'modeled_GHI', 'modeled_DNI', 
                       'zenith_angle', 'clearsky_GHI', 'clearsky_DNI']]
    
    out_df.columns = ['measured_GHI', 'measured_DNI', 'modeled_GHI', 'modeled_DNI', 
                      'zenith_angle', 'clear-sky_GHI', 'clear-sky_DNI']
    
    # Night Time Zeroing (Zenith > 90)
    irr_cols = ['measured_GHI', 'measured_DNI', 'modeled_GHI', 'modeled_DNI', 'clear-sky_GHI', 'clear-sky_DNI']
    out_df.loc[out_df['zenith_angle'] > 90, irr_cols] = 0
    
    # Formatting
    out_df['zenith_angle'] = out_df['zenith_angle'].round(3)
    for col in irr_cols:
        out_df[col] = out_df[col].round(0).astype('Int64')
    
    out_path = os.path.join(PROCESSED_DIR, f"{stn_code}_15min_qc.csv")
    out_df.to_csv(out_path)
    print(f"  Saved to {out_path}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # download NSRDB, only needs to be run once
    # download_nsrdb()

    # arrange the data into 15-min resolution with no missing values
    print("Starting SURFRAD processing...")
    
    if not os.path.exists(METADATA_PATH):
        print("Metadata file not found!")
    else:
        meta = pd.read_csv(METADATA_PATH)
        
        # Process all stations
        for _, row in tqdm(meta.iterrows(), total=meta.shape[0], desc="Processing Stations"):
            process_station(row)
