import os
import time
import pandas as pd
import logging
import cdflib
import concurrent.futures

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_DIR = os.path.join(PROJECT_DIR, "master_archive")
CSV_OUTPUT_PATH = os.path.join(PROJECT_DIR, "aditya_l1_cdf_catalog.csv")
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "process_cdf.log")

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_single_cdf(filename):
    filepath = os.path.join(MASTER_DIR, filename)
    file_metadata = {"Filename": filename}
    try:
        cdf_file = cdflib.CDF(filepath)
        for key, value in cdf_file.globalattsget().items():
            file_metadata[key] = str(value[0] if isinstance(value, (list, tuple)) else value)
        return file_metadata
    except Exception as e:
        logging.error(f"Failed {filename}: {e}")
        try: os.remove(filepath)
        except OSError: pass
        return None

def process_new_cdfs():
    cdf_files = [f for f in os.listdir(MASTER_DIR) if f.endswith('.cdf')]
    if not cdf_files: return False

    processed = set(pd.read_csv(CSV_OUTPUT_PATH, usecols=['Filename'], low_memory=False, on_bad_lines='skip')['Filename']) if os.path.exists(CSV_OUTPUT_PATH) else set()
    new_files = [f for f in cdf_files if f not in processed]
    if not new_files: return False 
        
    logging.info(f"Processing {len(new_files)} CDF files...")
    metadata_catalog = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        for res in executor.map(process_single_cdf, new_files):
            if res: metadata_catalog.append(res)

    if metadata_catalog:
        df = pd.DataFrame(metadata_catalog)
        if os.path.exists(CSV_OUTPUT_PATH):
            existing_df = pd.read_csv(CSV_OUTPUT_PATH, low_memory=False, on_bad_lines='skip')
            df = pd.concat([existing_df, df], ignore_index=True)
        cols = df.columns.tolist()
        if 'Filename' in cols: cols.insert(0, cols.pop(cols.index('Filename')))
        df.reindex(columns=cols).to_csv(CSV_OUTPUT_PATH, index=False)
        logging.info(f"SUCCESS! CDF CSV updated. Total: {len(df)}")
        return True

if __name__ == "__main__":
    while True:
        try: process_new_cdfs(); time.sleep(15)
        except Exception as e: logging.error(f"CRASH: {e}"); time.sleep(60)
