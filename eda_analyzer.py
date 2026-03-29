import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# --- CONFIGURATION ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_DIR, "aditya_l1_catalog.csv")
EDA_DIR = os.path.join(PROJECT_DIR, "web_gallery", "eda_plots")

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def run_eda():
    if not os.path.exists(CSV_PATH):
        logging.error("Master CSV not found. Let the pipeline download some data first!")
        return

    os.makedirs(EDA_DIR, exist_ok=True)
    logging.info("Loading master catalog for Exploratory Data Analysis...")
    df = pd.read_csv(CSV_PATH)

    # Basic data cleaning: convert strings to proper dates and numbers
    df['Observation_Date'] = pd.to_datetime(df['Observation_Date'], errors='coerce')
    df['Exposure_Time_sec'] = pd.to_numeric(df['Exposure_Time_sec'], errors='coerce')

    if df.empty or len(df) < 5:
        logging.warning("Not enough rows in the CSV to generate meaningful graphs yet.")
        return

    # --- VISUAL STYLING ---
    plt.style.use('dark_background')
    sns.set_palette("husl")

    # --- PLOT 1: TIMELINE OF OBSERVATIONS ---
    logging.info("Generating Observation Timeline...")
    plt.figure(figsize=(10, 5))
    df_time = df.dropna(subset=['Observation_Date']).copy()
    if not df_time.empty:
        # Group by hour to see the rhythm of data collection
        df_time.set_index('Observation_Date').resample('h').size().plot(color='cyan', linewidth=2)
        plt.title('Aditya-L1 Observation Frequency', color='white')
        plt.ylabel('Images Captured', color='white')
        plt.xlabel('Timeline', color='white')
        plt.grid(color='#30363d', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, '01_timeline.png'), dpi=150, facecolor='#0d1117')
    plt.close()

    # --- PLOT 2: EXPOSURE TIME DISTRIBUTION ---
    logging.info("Generating Exposure Time Distribution...")
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Exposure_Time_sec'].dropna(), bins=30, color='magenta', kde=True)
    plt.title('Distribution of Camera Exposure Times', color='white')
    plt.xlabel('Exposure Time (Seconds)', color='white')
    plt.ylabel('Count', color='white')
    plt.grid(color='#30363d', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '02_exposure_dist.png'), dpi=150, facecolor='#0d1117')
    plt.close()

    # --- PLOT 3: DETECTOR BREAKDOWN ---
    logging.info("Generating Detector Breakdown...")
    plt.figure(figsize=(10, 5))
    # Filter out UNKNOWNs if they exist
    clean_detectors = df[df['Detector'] != 'UNKNOWN']
    sns.countplot(data=clean_detectors, y='Detector', order=clean_detectors['Detector'].value_counts().index, color='#f0abfc')
    plt.title('Data Volume by Sensor/Detector', color='white')
    plt.xlabel('Total Images', color='white')
    plt.ylabel('')
    plt.grid(color='#30363d', linestyle='-', linewidth=0.5, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '03_detectors.png'), dpi=150, facecolor='#0d1117')
    plt.close()

    # --- PLOT 4: WAVELENGTHS ---
    logging.info("Generating Wavelength Analysis...")
    plt.figure(figsize=(10, 5))
    clean_waves = df[df['Wavelength'] != 'UNKNOWN']
    sns.countplot(data=clean_waves, y='Wavelength', order=clean_waves['Wavelength'].value_counts().index, color='#34d399')
    plt.title('Observation Count by Wavelength', color='white')
    plt.xlabel('Total Images', color='white')
    plt.ylabel('Wavelength Filter', color='white')
    plt.grid(color='#30363d', linestyle='-', linewidth=0.5, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '04_wavelengths.png'), dpi=150, facecolor='#0d1117')
    plt.close()

    logging.info(f"EDA Complete! 4 statistical plots saved to {EDA_DIR}")

if __name__ == "__main__":
    run_eda()
