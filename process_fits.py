import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import logging

# --- CONFIGURATION ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR = os.path.join(PROJECT_DIR, "master_archive")
OUTPUT_IMG_DIR = os.path.join(PROJECT_DIR, "processed_images")
CSV_OUTPUT_PATH = os.path.join(PROJECT_DIR, "aditya_l1_catalog.csv")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_new_fits():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    
    fits_files = [f for f in os.listdir(MASTER_DIR) if f.endswith('.fits')]
    if not fits_files:
        return False # Nothing to process, tell loop to wait

    # --- MEMORY: Only process what isn't already in the CSV ---
    processed_files = set()
    if os.path.exists(CSV_OUTPUT_PATH):
        try:
            # Read only the Filename column to be incredibly fast and save RAM
            existing_df = pd.read_csv(CSV_OUTPUT_PATH, usecols=['Filename'], low_memory=False)
            processed_files = set(existing_df['Filename'].tolist())
        except Exception as e:
            pass # CSV is probably empty or brand new
            
    new_fits_files = [f for f in fits_files if f not in processed_files]
    
    if not new_fits_files:
        return False # All caught up
        
    logging.info(f"Found {len(new_fits_files)} NEW files to process.")
    
    metadata_catalog = []
    IGNORE_KEYS = ['COMMENT', 'HISTORY', '']

    for filename in new_fits_files:
        filepath = os.path.join(MASTER_DIR, filename)
        
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                file_metadata = {"Filename": filename}
                
                # Dynamic Meta Extraction
                for key, value in header.items():
                    if key not in IGNORE_KEYS and not pd.isna(value):
                        if not isinstance(value, (int, float, str, bool)):
                            value = str(value)
                        file_metadata[key] = value

                metadata_catalog.append(file_metadata)

                # Render PNG
                png_filename = filename.replace('.fits', '.png')
                png_path = os.path.join(OUTPUT_IMG_DIR, png_filename)
                
                if not os.path.exists(png_path):
                    img_data = None
                    for hdu in hdul:
                        if hdu.data is not None and len(hdu.data.shape) >= 2:
                            img_data = hdu.data
                            break
                    
                    if img_data is not None:
                        plt.figure(figsize=(8, 8))
                        plt.imshow(img_data, cmap='magma', origin='lower')
                        plt.axis('off') 
                        plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=150)
                        plt.close()
                        logging.info(f"Generated PNG: {png_filename}")

        except Exception as e:
            logging.error(f"Failed to process {filename}: {str(e)}")

    # --- SYNCHRONIZED CSV SAVER ---
    # This ensures the CSV is updated at the exact same time the images are generated
    if metadata_catalog:
        logging.info(f"Adding {len(metadata_catalog)} new records to CSV...")
        new_df = pd.DataFrame(metadata_catalog)
        
        if os.path.exists(CSV_OUTPUT_PATH):
            existing_df = pd.read_csv(CSV_OUTPUT_PATH, low_memory=False)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Keep Filename first
            cols = combined_df.columns.tolist()
            if 'Filename' in cols:
                cols.insert(0, cols.pop(cols.index('Filename')))
            combined_df = combined_df.reindex(columns=cols)
            
            combined_df.to_csv(CSV_OUTPUT_PATH, index=False)
        else:
            cols = new_df.columns.tolist()
            if 'Filename' in cols:
                cols.insert(0, cols.pop(cols.index('Filename')))
            new_df = new_df.reindex(columns=cols)
            new_df.to_csv(CSV_OUTPUT_PATH, index=False)
            
        logging.info(f"SUCCESS! CSV updated. Total columns now: {len(pd.read_csv(CSV_OUTPUT_PATH).columns)}")
        return True

if __name__ == "__main__":
    logging.info("Starting Process Fits Continuous Daemon...")
    while True:
        try:
            did_work = process_new_fits()
            # If no new files, sleep for 15 seconds to wait for unzipper
            time.sleep(15) 
        except Exception as e:
            logging.error(f"DAEMON CRASH: {e}")
            time.sleep(60)
