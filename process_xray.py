import os
import time
import pandas as pd
import logging
import concurrent.futures
from astropy.io import fits

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_DIR = os.path.join(PROJECT_DIR, "master_archive")
CSV_OUTPUT_PATH = os.path.join(PROJECT_DIR, "aditya_l1_xray_catalog.csv")
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "process_xray.log")

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_single_xray(filename):
    filepath = os.path.join(MASTER_DIR, filename)
    file_metadata = {"Filename": filename}
    IGNORE_KEYS = ['COMMENT', 'HISTORY', '']
    try:
        with fits.open(filepath) as hdul:
            header = hdul[1].header if len(hdul) > 1 else hdul[0].header
            for key, value in header.items():
                if key not in IGNORE_KEYS and not pd.isna(value):
                    file_metadata[key] = str(value) if not isinstance(value, (int, float, str, bool)) else value
        return file_metadata
    except Exception as e:
        logging.error(f"Failed {filename}: {e}")
        try: os.remove(filepath)
        except OSError: pass
        return None

def process_new_xrays():
    valid_exts = ('.lc', '.pi', '.gti')
    xray_files = [f for f in os.listdir(MASTER_DIR) if f.endswith(valid_exts)]
    if not xray_files: return False

    processed = set(pd.read_csv(CSV_OUTPUT_PATH, usecols=['Filename'], low_memory=False, on_bad_lines='skip')['Filename']) if os.path.exists(CSV_OUTPUT_PATH) else set()
    new_files = [f for f in xray_files if f not in processed]
    if not new_files: return False 
        
    logging.info(f"Processing {len(new_files)} X-Ray files...")
    metadata_catalog = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        for res in executor.map(process_single_xray, new_files):
            if res: metadata_catalog.append(res)

    if metadata_catalog:
        df = pd.DataFrame(metadata_catalog)
        if os.path.exists(CSV_OUTPUT_PATH):
            existing_df = pd.read_csv(CSV_OUTPUT_PATH, low_memory=False, on_bad_lines='skip')
            df = pd.concat([existing_df, df], ignore_index=True)
        cols = df.columns.tolist()
        if 'Filename' in cols: cols.insert(0, cols.pop(cols.index('Filename')))
        df.reindex(columns=cols).to_csv(CSV_OUTPUT_PATH, index=False)
        logging.info(f"SUCCESS! X-Ray CSV updated. Total: {len(df)}")
        return True

if __name__ == "__main__":
    while True:
        try: process_new_xrays(); time.sleep(15)
        except Exception as e: logging.error(f"CRASH: {e}"); time.sleep(60)
