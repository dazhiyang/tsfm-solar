
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import glob
from tqdm import tqdm
import warnings

# Use a style for better aesthetics
plt.style.use('ggplot') 
# Or seaborn-v0_8-whitegrid if ggplot isn't premium enough, but ggplot is standard decent.
# Let's try to make it look clean.

warnings.filterwarnings("ignore")

# Define Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "Data", "Processed")
TEX_DIR = os.path.join(SCRIPT_DIR, "..", "tex")

os.makedirs(TEX_DIR, exist_ok=True)

def plot_station_monthly(file_path):
    filename = os.path.basename(file_path)
    stn_code = filename.split('_')[0]
    
    print(f"Processing {stn_code}...")
    
    # Read Data
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    # Filter for GHI and DNI columns
    ghi_cols = ['measured_GHI', 'modeled_GHI', 'clear-sky_GHI']
    dni_cols = ['measured_DNI', 'modeled_DNI', 'clear-sky_DNI']
    
    # Colors: pypalettes 'Arches' (MetBrewer)
    # Measured (Dark Blue), Modeled (Burnt Orange), Clear-sky (Teal)
    colors = ['#182236', '#d75f0b', '#416b71']
    styles = ['-', '-', '--']
    alphas = [0.9, 0.8, 0.7] # Slightly higher alpha for visibility
    linewidths = [1.2, 1.2, 1.0]

    # Prepare PDF Output
    pdf_path = os.path.join(TEX_DIR, f"{stn_code}_monthly.pdf")
    
    with PdfPages(pdf_path) as pdf:
        # Group by Month (Year-Month)
        # Use PeriodIndex for easy iteration
        months = df.index.to_period('M').unique()
        
        for m in tqdm(months, desc=f"Plotting {stn_code}", leave=False):
            # Extract data for this month
            monthly_data = df[df.index.to_period('M') == m]
            
            if monthly_data.empty:
                continue
                
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
            fig.suptitle(f"Station: {stn_code.upper()} - {m}", fontsize=16)
            
            # 1. GHI Plot
            ax_ghi = axes[0]
            for col, color, style, alpha, lw in zip(ghi_cols, colors, styles, alphas, linewidths):
                if col in monthly_data.columns:
                    ax_ghi.plot(monthly_data.index, monthly_data[col], 
                                label=col.replace('_', ' ').title(), 
                                color=color, linestyle=style, alpha=alpha, linewidth=lw)
            
            ax_ghi.set_ylabel("Irradiance (W/m$^2$)")
            ax_ghi.set_title("Global Horizontal Irradiance (GHI)")
            ax_ghi.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
            ax_ghi.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            # 2. DNI Plot
            ax_dni = axes[1]
            for col, color, style, alpha, lw in zip(dni_cols, colors, styles, alphas, linewidths):
                if col in monthly_data.columns:
                    ax_dni.plot(monthly_data.index, monthly_data[col], 
                                label=col.replace('_', ' ').title(), 
                                color=color, linestyle=style, alpha=alpha, linewidth=lw)
            
            ax_dni.set_ylabel("Irradiance (W/m$^2$)")
            ax_dni.set_title("Direct Normal Irradiance (DNI)")
            ax_dni.set_xlabel("Date")
            ax_dni.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
            ax_dni.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            # Formatting x-axis to be nice
            # Major ticks every few days
            # ax_dni.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            # ax_dni.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
            # Auto formatting usually works well for pandas timestamps
            plt.xticks(rotation=45)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
            
            pdf.savefig(fig)
            plt.close(fig)
            
    print(f"  Saved booklet to {pdf_path}")

def main():
    # Find all processed files
    # pattern: <code_15min_qc.csv>
    search_pattern = os.path.join(PROCESSED_DIR, "*_15min_qc.csv")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"No processed files found in {PROCESSED_DIR}")
        return
        
    print(f"Found {len(files)} stations to plot.")
    
    for f in files:
        plot_station_monthly(f)

if __name__ == "__main__":
    main()
