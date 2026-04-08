import os
import time
import logging
import concurrent.futures
from astropy.io import fits
import numpy as np
import cv2
import shutil

# --- DYNAMIC DIRECTORY SETUP ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_DIR = os.path.join(PROJECT_DIR, "master_archive")
OUTPUT_IMG_DIR = os.path.join(PROJECT_DIR, "processed_images")
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "process_fits.log")
CORRUPT_DIR = os.path.join(MASTER_DIR, "corrupt_files")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_single_fits(filename):
    filepath = os.path.join(MASTER_DIR, filename)
    png_filename = filename.replace('.fits', '.png')
    png_path = os.path.join(OUTPUT_IMG_DIR, png_filename)
    
    # 1. SKIP IF ALREADY RENDERED
    if os.path.exists(png_path):
        return False
        
    # 2. MATURATION CHECK (Wait for unzipper to finish)
    if not os.path.exists(filepath): return False
    if time.time() - os.path.getmtime(filepath) < 10.0:
        return False 
    
    try:
        # THE FIX: memmap=False universally handles Level-1 AND Level-2 (BZERO/BSCALE) safely in RAM.
        with fits.open(filepath, memmap=False) as hdul:
            
            # Extract Image Data safely
            img_data = None
            for hdu in hdul:
                if hdu.data is not None and getattr(hdu.data, 'ndim', 0) >= 2:
                    img_data = np.squeeze(hdu.data)
                    if img_data.ndim > 2:
                        img_data = img_data[0]
                    break
            
            if img_data is not None:
                # SANITIZE NANS
                img_clean = np.nan_to_num(img_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                
                # NORMALIZE & RENDER (Magma Colormap)
                img_norm = cv2.normalize(img_clean, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                img_magma = cv2.applyColorMap(img_norm, cv2.COLORMAP_MAGMA)
                cv2.imwrite(png_path, img_magma)
                logging.info(f"Successfully generated image: {png_filename}")
                return True
                
        return False
        
    except Exception as e:
        logging.error(f"Failed to process {filename}: {str(e)}")
        # QUARANTINE
        try:
            os.makedirs(CORRUPT_DIR, exist_ok=True)
            shutil.move(filepath, os.path.join(CORRUPT_DIR, filename))
            logging.info(f"☣️ QUARANTINED corrupt file: {filename}")
        except Exception:
            pass
        return False

def process_new_fits():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    fits_files = [f for f in os.listdir(MASTER_DIR) if f.endswith('.fits')]
    
    # Fast filtering: only process FITS that don't have a matching PNG yet
    new_fits_files = [f for f in fits_files if not os.path.exists(os.path.join(OUTPUT_IMG_DIR, f.replace('.fits', '.png')))]
    
    if not new_fits_files:
        return False 
        
    logging.info(f"Found {len(new_fits_files)} unprocessed FITS files. Rendering images...")
    
    did_work = False
    # Process Pool for true multi-core speed
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = list(executor.map(process_single_fits, new_fits_files))
        if any(results):
            did_work = True

    return did_work

if __name__ == "__main__":
    logging.info("Starting Process Fits (Image-Only) Daemon...")
    while True:
        try:
            did_work = process_new_fits()
            if not did_work:
                time.sleep(5) 
        except Exception as e:
            logging.error(f"DAEMON CRASH: {e}")
            time.sleep(60)
