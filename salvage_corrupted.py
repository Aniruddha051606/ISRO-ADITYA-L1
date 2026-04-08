import os
import sys
import logging
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from astropy.io import fits

try:
    import cdflib
    HAS_CDF = True
except ImportError:
    HAS_CDF = False

# --- DYNAMIC DIRECTORY SETUP ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORRUPT_DIR = os.path.join(PROJECT_DIR, "master_archive", "corrupt_files")
SALVAGED_IMG_DIR = os.path.join(PROJECT_DIR, "salvaged_images")
SALVAGED_CSV = os.path.join(PROJECT_DIR, "salvaged_catalog.csv")
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "salvage.log")

for directory in [SALVAGED_IMG_DIR, os.path.dirname(LOG_FILE)]:
    Path(directory).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def salvage_fits(filepath, filename):
    """Aggressively forces FITS files to yield whatever data survived."""
    salvaged_data = {"Filename": filename, "Status": "Salvaged"}
    
    try:
        # THE MAGIC FLAGS: ignore_missing_end and silentfix force it to bypass corruption
        with fits.open(filepath, memmap=False, ignore_missing_end=True, output_verify='silentfix') as hdul:
            
            # 1. Salvage Header (Even if partial)
            try:
                header = hdul[0].header
                for key, val in header.items():
                    if key not in ['COMMENT', 'HISTORY', ''] and not pd.isna(val):
                        salvaged_data[key] = str(val)
            except Exception as e:
                logging.warning(f"Could not salvage header for {filename}: {e}")

            # 2. Salvage Image / Data Arrays
            for i, hdu in enumerate(hdul):
                try:
                    data = hdu.data
                    if data is None: continue
                    
                    # Salvage Image
                    if getattr(data, 'ndim', 0) >= 2:
                        img_data = np.squeeze(data)
                        if img_data.ndim > 2: img_data = img_data[0]
                        
                        png_path = os.path.join(SALVAGED_IMG_DIR, filename.replace('.fits', f'_salvaged_HDU{i}.png'))
                        
                        img_clean = np.nan_to_num(img_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        img_norm = cv2.normalize(img_clean, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        img_magma = cv2.applyColorMap(img_norm, cv2.COLORMAP_MAGMA)
                        cv2.imwrite(png_path, img_magma)
                        salvaged_data["Image_Salvaged"] = True
                        logging.info(f"Salvaged image from corrupt FITS: {filename}")
                        break # Only grab the first surviving image
                        
                    # Salvage Tabular Data length
                    elif isinstance(hdu, fits.BinTableHDU):
                        salvaged_data["Has_Tabular_Data"] = True
                        salvaged_data["Num_Telemetry_Rows"] = len(data)
                except Exception as array_err:
                    logging.warning(f"Failed to salvage HDU {i} in {filename}: {array_err}")
                    continue
                    
        return salvaged_data
    except Exception as e:
        logging.error(f"Complete salvage failure on {filename}. File is utterly destroyed: {e}")
        return None

def salvage_cdf(filepath, filename):
    """Attempts to read partial CDF variables."""
    if not HAS_CDF: return None
    salvaged_data = {"Filename": filename, "Status": "Salvaged_CDF"}
    
    try:
        # CDF files are harder to salvage if the footer is missing, but we try
        cdf = cdflib.CDF(filepath)
        attrs = cdf.globalattsget()
        for k, v in attrs.items():
            if isinstance(v, (list, np.ndarray)): v = v[0] if len(v)>0 else ""
            salvaged_data[f"CDF_{k}"] = str(v)[:500]
        cdf.close()
        return salvaged_data
    except Exception as e:
        logging.error(f"CDF salvage failed for {filename}: {e}")
        return None

def run_salvage():
    if not os.path.exists(CORRUPT_DIR):
        print("No corrupt_files directory found. Nothing to salvage!")
        return

    corrupt_files = os.listdir(CORRUPT_DIR)
    if not corrupt_files:
        print("No files currently in the quarantine folder. Your pipeline is healthy!")
        return

    print(f"🚑 Attempting to salvage {len(corrupt_files)} corrupted files...")
    
    salvaged_records = []
    for f in corrupt_files:
        filepath = os.path.join(CORRUPT_DIR, f)
        if f.endswith('.fits'):
            res = salvage_fits(filepath, f)
        elif f.endswith('.cdf'):
            res = salvage_cdf(filepath, f)
        else:
            continue
            
        if res:
            salvaged_records.append(res)

    if salvaged_records:
        df = pd.DataFrame(salvaged_records)
        df.to_csv(SALVAGED_CSV, index=False)
        print(f"✅ Successfully extracted partial data from {len(salvaged_records)} files!")
        print(f"📁 Check {SALVAGED_CSV} for the recovered telemetry.")
        print(f"🖼️ Check {SALVAGED_IMG_DIR}/ for recovered images.")
    else:
        print("❌ Salvage complete. All files were too heavily corrupted to recover any bytes.")

if __name__ == "__main__":
    run_salvage()
