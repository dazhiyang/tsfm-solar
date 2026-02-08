
import pandas as pd
import numpy as np
import os
import re
from plotnine import *
from matplotlib import rcParams
from mizani.bounds import rescale_mid as mid_rescaler

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEX_DIR = os.path.join(SCRIPT_DIR, "..", "tex")
INPUT_FILE = os.path.join(TEX_DIR, "overall_skill.tex")
HEATMAP_OUTPUT = os.path.join(TEX_DIR, "overall_skill_heatmap.pdf")
EFFICIENCY_OUTPUT = os.path.join(TEX_DIR, "skill_efficiency.pdf")

# Highlight model
HIGHLIGHT_MODEL = "TabPFN-2.5-Kappa"

def parse_latex_table(content, section_name):
    """Parses a specific section of the overall_skill.tex file into a DataFrame."""
    # Find the section
    pattern = rf"% --- Skill Score \(%\) {section_name} ---(.*?)\\bottomrule"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    
    table_text = match.group(1).strip()
    lines = [line.strip() for line in table_text.split("\n") if line.strip()]
    
    # 1. Extract headers (The line that starts with &)
    headers = []
    header_line_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("&"):
            header_line_idx = i
            # Headers are between the ampersands: & bon & dra ... \\
            row_clean = line.replace("\\\\", "").replace("\\toprule", "").strip()
            parts = [p.strip() for p in row_clean.split("&")]
            headers = [p for p in parts if p] # Skip the first empty part
            break
    
    if not headers:
        return None
        
    # 2. Extract rows (Everything after \midrule)
    rows = []
    start_data = False
    for line in lines:
        if "\\midrule" in line:
            start_data = True
            continue
        if not start_data:
            continue
            
        # Parse data row: Model & val & val ... \\
        if " & " in line:
            row_clean = line.replace("\\\\", "").strip()
            parts = [p.strip() for p in row_clean.split("&")]
            if len(parts) < 2: continue
            
            model = parts[0]
            vals_raw = parts[1:]
            
            vals = []
            for v in vals_raw:
                v = v.replace("\\textbf{", "").replace("}", "")
                v = v.replace("$-$", "-")
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(np.nan)
            
            # Match header length
            if len(vals) >= len(headers):
                rows.append([model] + vals[:len(headers)])
    
    df = pd.DataFrame(rows, columns=["Model"] + headers)
    return df

def plot_heatmap(df_final, width_in, height_in):
    # Highlight indices
    model_cat = df_final["Model"].cat.categories
    highlight_idx = model_cat.get_loc(HIGHLIGHT_MODEL)
    ymin = highlight_idx + 0.5
    ymax = highlight_idx + 1.5

    p = (
        ggplot(df_final, aes(x='Station', y='Model', fill='SkillClipped'))
        + geom_tile(color="white", size=0.1)
        # Highlight border for specific model
        + geom_rect(xmin=0.5, xmax=8.5, ymin=ymin, ymax=ymax,
                  fill=None, color="black", size=0.6, linetype="solid")
        + geom_text(aes(label='Skill.round(1)'), family="Times New Roman", size=8)
        + facet_wrap("~Variable", ncol=1) 
        + scale_fill_gradientn(
            colors=["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"],
            rescaler=lambda x, **kwargs: mid_rescaler(x, mid=0, **kwargs),
            name="$s$ (%)"
        )
        + labs(x=None, y=None)
        + theme_minimal()
        + theme(
            text=element_text(family="Times New Roman", size=8),
            axis_text_x=element_text(angle=45, hjust=1, size=8),
            axis_text_y=element_text(size=8),
            legend_position="bottom",
            legend_key_width=100,
            legend_key_height=8,
            legend_text=element_text(size=8),
            legend_title=element_text(size=8),
            panel_grid=element_blank(),
            strip_text=element_text(size=8, family="Times New Roman"),
            figure_size=(width_in, height_in)
        )
    )
    return p

def plot_efficiency(ghi_df, width_in, height_in):
    # Base data dictionary for Type and Complexity (mapping components)
    # We will derive the full list from ghi_df to ensure Direct models are included
    model_list = ghi_df['Model'].tolist()
    
    rows = []
    for m in model_list:
        m_lower = m.lower()
        if "cliper" in m_lower:
            m_type = "Baseline"
            complexity = 1
        elif "xgboost" in m_lower:
            m_type = "Regression"
            complexity = 2
        elif "tabpfn" in m_lower:
            m_type = "Regression"
            complexity = 3
        elif "ttm" in m_lower:
            m_type = "TSFM"
            complexity = 4
        else:
            # Chronos, TimesFM, TiRex, etc.
            m_type = "TSFM"
            complexity = 5
            
        rows.append({'Model': m, 'Complexity': complexity, 'Type': m_type})
        
    df_eff = pd.DataFrame(rows)
    
    # Map Skill from ghi_df (Average column)
    if 'Average' in ghi_df.columns:
        skill_map = ghi_df.set_index('Model')['Average'].to_dict()
        df_eff['Skill'] = df_eff['Model'].map(skill_map)
    else:
        # Emergency fallback
        df_eff['Skill'] = 0.0
    
    # Drop rows where skill is missing
    df_eff = df_eff.dropna(subset=['Skill'])

    # Clip skill for consistency with heatmap (and to avoid huge axis scale)
    df_eff['Skill'] = df_eff['Skill'].clip(lower=-6.5)

    # Colors and shapes
    colors = {'Baseline': 'gray', 'Regression': '#1f77b4', 'TSFM': '#d62728'}
    
    # Categorize Types for legend order
    df_eff['Type'] = pd.Categorical(df_eff['Type'], categories=['Baseline', 'Regression', 'TSFM'], ordered=True)

    # Complexity labels
    complexity_labels = ['Baseline', 'Shallow\nregression', 'Deep\ntabular', 'Medium\nTSFM', 'Large\nTSFM']
    
    # Multi-group labeling for precise horizontal positioning
    # Left: XGBoost and TTM-R1 versions
    df_left = df_eff[df_eff['Model'].str.contains('TTM-R1|XGBoost', case=False)]
    df_cliper = df_eff[df_eff['Model'].str.contains('CLIPER', case=False)]
    # Right: The rest (including TTM-R2 versions, TabPFN, etc.)
    df_right = df_eff[~df_eff['Model'].str.contains('TTM-R1|XGBoost|CLIPER', case=False)]

    p = (
        ggplot(df_eff, aes(x='Complexity', y='Skill', color='Type', shape='Type'))
        + geom_hline(yintercept=0, color='black', linetype='dashed', alpha=0.3)
        + geom_point(size=2.5, alpha=0.8, fill='white', stroke=0.8) # Smaller points
        # Group 1: Standard Right
        + geom_text(df_right, aes(label='Model'), family="Times New Roman", size=8, 
                  ha='left', va='center', nudge_x=0.15, show_legend=False)
        # Group 2: Left (TTM and XGB)
        + geom_text(df_left, aes(label='Model'), family="Times New Roman", size=8, 
                  ha='right', va='center', nudge_x=-0.15, show_legend=False)
        # Group 3: CLIPER (Right)
        + geom_text(df_cliper, aes(label='Model'), family="Times New Roman", size=8, 
                  ha='left', va='center', nudge_x=0.15, show_legend=False)
        + scale_color_manual(values=colors)
        + scale_shape_manual(values=['D', 'o', 's'])
        + scale_x_continuous(breaks=[1, 2, 3, 4, 5], labels=complexity_labels)
        # Use expand to ensure labels and points aren't cut off at the edges
        + scale_y_continuous(expand=(0.1, 0.2))
        + expand_limits(x=(0, 7.2)) # Space for labels on the right
        + labs(
            x='Model complexity',
            y='Average skill ($s$, %)',
            color='Category',
            shape='Category'
        )
        + theme_minimal()
        + theme(
            text=element_text(family="Times New Roman", size=8),
            axis_text=element_text(size=8),
            legend_position="bottom",
            legend_text=element_text(size=8),
            legend_title=element_text(size=8),
            panel_grid_minor=element_blank(),
            figure_size=(width_in, height_in)
        )
    )
    return p

def process_and_plot():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r") as f:
        content = f.read()

    ghi_df = parse_latex_table(content, "GHI")
    dni_df = parse_latex_table(content, "DNI")

    if ghi_df is None or dni_df is None:
        print("Error parsing tables.")
        return

    # --- HEATMAP ---
    plot_data = []
    for var, df in [("GHI", ghi_df), ("DNI", dni_df)]:
        df_long = df.melt(id_vars=["Model"], var_name="Station", value_name="Skill")
        df_long["Variable"] = var
        plot_data.append(df_long)
    
    df_final = pd.concat(plot_data)
    df_final["Station"] = df_final["Station"].apply(lambda x: x.upper() if x.lower() != "average" else "Average")
    
    model_order = ghi_df["Model"].tolist()[::-1]
    df_final["Model"] = pd.Categorical(df_final["Model"], categories=model_order, ordered=True)
    
    station_order = [s.upper() if s.lower() != "average" else "Average" for s in ghi_df.columns[1:].tolist()]
    df_final["Station"] = pd.Categorical(df_final["Station"], categories=station_order, ordered=True)
    
    df_final["Variable"] = df_final["Variable"].replace("DNI", "BNI")
    df_final["Variable"] = pd.Categorical(df_final["Variable"], categories=["GHI", "BNI"], ordered=True)
    
    df_final["SkillClipped"] = df_final["Skill"].clip(lower=-6.5)

    mm_to_in = 0.03937
    p_heatmap = plot_heatmap(df_final, 85 * mm_to_in, 120 * mm_to_in)
    p_heatmap.save(HEATMAP_OUTPUT, dpi=300)
    print(f"Heatmap saved to {HEATMAP_OUTPUT}")

    # --- EFFICIENCY PLOT ---
    # Final preferred dimensions: 160 mm width, 40 mm height
    p_eff = plot_efficiency(ghi_df, 90 * mm_to_in, 95 * mm_to_in)
    p_eff.save(EFFICIENCY_OUTPUT, dpi=300)
    print(f"Efficiency plot saved to {EFFICIENCY_OUTPUT}")

if __name__ == "__main__":
    process_and_plot()
