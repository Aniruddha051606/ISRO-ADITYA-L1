import os
import glob
import time
import zipfile
import pandas as pd
import numpy as np
import logging
import subprocess
import sys
from astropy.io import fits
import matplotlib.pyplot as plt

# --- CONFIG ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_DIR = os.path.join(PROJECT_DIR, "data")
EXTRACT_DIR = os.path.join(PROJECT_DIR, "master_archive")
IMAGE_DIR = os.path.join(PROJECT_DIR, "processed_images")
CSV_PATH = os.path.join(PROJECT_DIR, "aditya_l1_catalog.csv")
ML_SCRIPT = os.path.join(PROJECT_DIR, "scripts", "ml_forecaster.py")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FITS → PNG ---
def process_fits_to_png(fits_path, filename):
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data

            if data is None:
                return None

            # Normalize image (VERY IMPORTANT)
            data = np.nan_to_num(data)
            data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

            png_name = filename.replace(".fits", ".png")
            png_path = os.path.join(IMAGE_DIR, png_name)

            plt.imsave(png_path, data, cmap='magma')
            return png_path

    except Exception as e:
        logging.error(f"Image Error on {filename}: {e}")
        return None

# --- MAIN PIPELINE ---
def run_pipeline():
    os.makedirs(ZIP_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    while True:
        zip_files = glob.glob(os.path.join(ZIP_DIR, "*.zip"))

        if not zip_files:
            logging.info("Scanning data/... no ZIPs. Waiting 60s...")
            time.sleep(60)
            continue

        logging.info(f"Found {len(zip_files)} new ZIPs. Processing...")

        new_records = []

        for zip_path in zip_files:
            try:
                # --- Extract ---
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(EXTRACT_DIR)

                # --- Process FITS ---
                fits_files = glob.glob(os.path.join(EXTRACT_DIR, "**/*.fits"), recursive=True)

                for fits_file in fits_files:
                    fname = os.path.basename(fits_file)

                    try:
                        with fits.open(fits_file) as hdul:
                            header = hdul[0].header

                            # Instrument detection (safe)
                            if fname.startswith("SUT"):
                                instrument = "SUIT"
                            else:
                                instrument = header.get("INSTRUME", "UNKNOWN")

                            # Safe numeric conversion
                            wavelength = header.get("WAVELNTH", np.nan)
                            try:
                                wavelength = float(wavelength)
                            except:
                                wavelength = np.nan

                            exposure = header.get("EXPTIME", 0.0)

                        # Convert image
                        img_path = process_fits_to_png(fits_file, fname)

                        if img_path:
                            new_records.append({
                                "Filename": fname,
                                "Observation_Date": header.get("DATE-OBS", "UNKNOWN"),
                                "Instrument": instrument,
                                "Exposure_Time_sec": exposure,
                                "Wavelength": wavelength,
                                "Resolution_X": header.get("NAXIS1", 0),
                                "Resolution_Y": header.get("NAXIS2", 0)
                            })

                    except Exception as e:
                        logging.error(f"Error processing FITS {fname}: {e}")

                # --- Cleanup ---
                os.remove(zip_path)

                for f in fits_files:
                    if os.path.exists(f):
                        os.remove(f)

            except Exception as e:
                logging.error(f"ZIP processing error {zip_path}: {e}")

        # --- Save CSV ---
        if new_records:
            df_new = pd.DataFrame(new_records)

            if os.path.exists(CSV_PATH):
                df_old = pd.read_csv(CSV_PATH)
                df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=["Filename"])
            else:
                df_final = df_new

            df_final.to_csv(CSV_PATH, index=False)
            logging.info(f"Saved {len(new_records)} new records")

        # --- Trigger ML ---
        try:
            logging.info("Triggering ML script...")
            subprocess.run([sys.executable, ML_SCRIPT])
        except Exception as e:
            logging.error(f"ML trigger failed: {e}")

        time.sleep(10)


if __name__ == "__main__":
    run_pipeline()
