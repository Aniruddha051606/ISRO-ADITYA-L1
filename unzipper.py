import os
import time
import logging
import zipfile

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_DIR = os.path.join(PROJECT_DIR, "master_archive")
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "unzipper.log")

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

def extract_archives():
    zips = [f for f in os.listdir(MASTER_DIR) if f.endswith('.zip')]
    for zip_name in zips:
        zip_path = os.path.join(MASTER_DIR, zip_name)
        logging.info(f"Unzipping: {zip_name}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(MASTER_DIR)
            os.remove(zip_path)
            logging.info(f"Extracted and deleted: {zip_name}")
        except zipfile.BadZipFile:
            logging.error(f"Corrupted zip file detected: {zip_name}")
            os.remove(zip_path)
        except Exception as e:
            logging.error(f"Failed to extract {zip_name}: {e}")

if __name__ == "__main__":
    logging.info("Unzipper daemon started.")
    while True:
        extract_archives()
        time.sleep(15)
