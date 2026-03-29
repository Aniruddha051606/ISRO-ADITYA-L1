import os
import zipfile
import logging
import shutil
import time

# --- CONFIGURATION ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_DIR = os.path.join(PROJECT_DIR, "data")
MASTER_DIR = os.path.join(PROJECT_DIR, "master_archive")
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "unzipper.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_zips():
    os.makedirs(MASTER_DIR, exist_ok=True)

    zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith('.zip')]
    
    if not zip_files:
        return False # No files, tell loop to wait

    logging.info(f"Found {len(zip_files)} zip files to process.")

    for zip_filename in zip_files:
        zip_path = os.path.join(ZIP_DIR, zip_filename)
        extraction_successful = True
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for internal_file in zip_ref.namelist():
                    # Ignore directories
                    if internal_file.endswith('/'):
                        continue
                        
                    # Flatten the structure directly into master_archive
                    file_basename = os.path.basename(internal_file)
                    target_path = os.path.join(MASTER_DIR, file_basename)
                    
                    # Deduplication
                    if os.path.exists(target_path):
                        continue 
                    
                    logging.info(f"  -> Extracting: {file_basename}")
                    with zip_ref.open(internal_file) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                        
        except zipfile.BadZipFile:
            logging.error(f"CORRUPTED ZIP: {zip_filename}. Download likely failed mid-stream.")
            extraction_successful = False
        except Exception as e:
            logging.error(f"Error processing {zip_filename}: {e}")
            extraction_successful = False

        # Cleanup: Only delete if extraction was 100% successful
        if extraction_successful:
            try:
                os.remove(zip_path)
                logging.info(f"SUCCESS: Extracted and deleted -> {zip_filename}")
            except OSError as e:
                logging.error(f"Failed to delete {zip_filename}: {e}")
                
    return True # Work was done

if __name__ == "__main__":
    logging.info("Starting Continuous ZIP Extraction Daemon (unzipper.py)...")
    while True:
        try:
            did_work = process_zips()
            time.sleep(10) # Wait 10 seconds before checking for new downloads
        except Exception as e:
            logging.error(f"CRITICAL DAEMON CRASH: {e}")
            time.sleep(60) # Back off for a minute if something major breaks
