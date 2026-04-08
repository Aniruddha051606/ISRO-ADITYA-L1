import os
import time
import logging
import pandas as pd
import numpy as np
import matplotlib
# Headless mode for server rendering
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_DIR, "aditya_l1_catalog.csv")
# UI specifically looks here for the graphs
OUTPUT_DIR = os.path.join(PROJECT_DIR, "web_gallery", "eda_plots")
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "eda_analyzer.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- THEME SETTINGS (Matches UI exactly) ---
BG_COLOR = "#080f18"       # UI Panel Background
TEXT_COLOR = "#c9d1d9"     # UI Muted Text
ACCENT_BLUE = "#1f6feb"    # ISRO Blue
ACCENT_ORANGE = "#F99D2A"  # ISRO Orange

def setup_theme():
    """Applies the Mission Control dark theme to all plots."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": "#1e3a5f",
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": "#1e3a5f",
        "grid.alpha": 0.5,
        "font.family": "sans-serif"
    })

def safe_numeric(df, col_name):
    """Safely converts a column to numeric, dropping NaNs."""
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors='coerce').dropna()
    return pd.Series(dtype=float)

def generate_plots():
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        logging.warning("CSV not found or empty. Skipping EDA generation.")
        return False

    try:
        # Load the catalog, skipping bad lines
        df = pd.read_csv(CSV_PATH, low_memory=False, on_bad_lines='skip')
        if df.empty:
            return False

        setup_theme()

        # ---------------------------------------------------------
        # PLOT 1: Observation Timeline (Images captured over time)
        # ---------------------------------------------------------
        if 'DATE-OBS' in df.columns:
            try:
                df['DATE_PARSED'] = pd.to_datetime(df['DATE-OBS'], errors='coerce')
                daily_counts = df['DATE_PARSED'].dt.date.value_counts().sort_index()
                
                if not daily_counts.empty:
                    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
                    ax.plot(daily_counts.index, daily_counts.values, color=ACCENT_BLUE, linewidth=2, marker='o', markersize=4)
                    ax.set_title("Aditya-L1 Temporal Observation Frequency", color=TEXT_COLOR, pad=15)
                    ax.set_ylabel("Images Captured")
                    ax.grid(True, linestyle='--')
                    fig.autofmt_xdate()
                    fig.savefig(os.path.join(OUTPUT_DIR, "01_timeline.png"), bbox_inches='tight', facecolor=BG_COLOR)
                    plt.close(fig)
            except Exception as e:
                logging.warning(f"Failed to plot timeline: {e}")

        # ---------------------------------------------------------
        # PLOT 2: Exposure Time Distribution
        # ---------------------------------------------------------
        exptime = safe_numeric(df, 'EXPTIME')
        if not exptime.empty:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
            # Clip extreme outliers (e.g., test frames with 999s exposure)
            exptime = exptime[exptime < exptime.quantile(0.99)] 
            
            sns.histplot(exptime, bins=40, color="#8b5cf6", edgecolor=BG_COLOR, ax=ax)
            ax.axvline(exptime.median(), color=ACCENT_ORANGE, linestyle='--', linewidth=1.5, label=f"Median: {exptime.median():.2f}s")
            ax.set_title("Distribution of Camera Exposure Times", color=TEXT_COLOR, pad=15)
            ax.set_xlabel("Exposure Time (Seconds)")
            ax.set_ylabel("Count")
            ax.legend(facecolor=BG_COLOR, edgecolor="#1e3a5f", labelcolor=TEXT_COLOR)
            ax.grid(True, linestyle='--', axis='y')
            fig.savefig(os.path.join(OUTPUT_DIR, "02_exposure_dist.png"), bbox_inches='tight', facecolor=BG_COLOR)
            plt.close(fig)

        # ---------------------------------------------------------
        # PLOT 3: Filter / Wavelength Distribution
        # ---------------------------------------------------------
        filter_col = 'WAVELNTH' if 'WAVELNTH' in df.columns else 'FILTER' if 'FILTER' in df.columns else None
        if filter_col and filter_col in df.columns:
            counts = df[filter_col].value_counts().head(10)
            if not counts.empty:
                fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
                sns.barplot(y=counts.index.astype(str), x=counts.values, palette="light:b_r", ax=ax)
                ax.set_title(f"Top Data Volume by {filter_col}", color=TEXT_COLOR, pad=15)
                ax.set_xlabel("Total Images Captured")
                ax.grid(True, linestyle='--', axis='x')
                fig.savefig(os.path.join(OUTPUT_DIR, "03_wavelengths.png"), bbox_inches='tight', facecolor=BG_COLOR)
                plt.close(fig)

        # ---------------------------------------------------------
        # PLOT 4: Missing Data Heatmap
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        # Only plot the first 50 columns to avoid a massive unreadable blob
        subset_df = df.iloc[:, :50]
        sns.heatmap(subset_df.isnull(), yticklabels=False, xticklabels=False, cbar=False, cmap="viridis", ax=ax)
        ax.set_title("Telemetry Completeness Heatmap (Yellow = Missing)", color=TEXT_COLOR, pad=15)
        fig.savefig(os.path.join(OUTPUT_DIR, "04_missing_data.png"), bbox_inches='tight', facecolor=BG_COLOR)
        plt.close(fig)

        # ---------------------------------------------------------
        # PLOT 5: Correlation Matrix (Physics & Telemetry)
        # ---------------------------------------------------------
        # Extract numeric physics columns for correlation
        physics_cols = ['EXPTIME', 'SUN_CX', 'SUN_CY', 'R_SUN', 'NAXIS1', 'NAXIS2', 'DATAMEAN', 'DATARMS', 'DATAMIN', 'DATAMAX']
        available_cols = [c for c in physics_cols if c in df.columns]
        
        if len(available_cols) > 1:
            corr_df = df[available_cols].apply(pd.to_numeric, errors='coerce').corr()
            # Mask the upper triangle for a cleaner look
            mask = np.triu(np.ones_like(corr_df, dtype=bool))
            
            fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
            sns.heatmap(corr_df, mask=mask, cmap="vlag", center=0, annot=False, 
                        square=True, linewidths=.5, cbar_kws={"shrink": .7}, ax=ax)
            ax.set_title("Correlation Matrix of Telemetry Features", color=TEXT_COLOR, pad=15)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            fig.savefig(os.path.join(OUTPUT_DIR, "05_correlation.png"), bbox_inches='tight', facecolor=BG_COLOR)
            plt.close(fig)

        logging.info("SUCCESS: Updated 5 EDA Graphs.")
        return True

    except Exception as e:
        logging.error(f"Failed to generate EDA plots: {e}")
        return False

if __name__ == "__main__":
    logging.info("Starting EDA Analyzer Daemon...")
    print("📊 EDA Engine Online. Generating web gallery...")
    
    while True:
        try:
            generate_plots()
            # Sleep for 60 seconds before updating graphs again
            time.sleep(60) 
        except Exception as e:
            logging.error(f"DAEMON CRASH: {e}")
            time.sleep(60)
