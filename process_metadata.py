import os
import sys
import time
import logging
import concurrent.futures
import pandas as pd
import numpy as np
from astropy.io import fits
import fcntl
from contextlib import contextmanager
from pathlib import Path

# Optional imports for secondary space weather formats
try:
    import cdflib
    HAS_CDF = True
except ImportError:
    HAS_CDF = False

try:
    import netCDF4
    HAS_NC = True
except ImportError:
    HAS_NC = False

# --- DYNAMIC DIRECTORY SETUP ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_DIR = os.path.join(PROJECT_DIR, "master_archive")
OUTPUT_IMG_DIR = os.path.join(PROJECT_DIR, "processed_images")
CSV_OUTPUT_PATH = os.path.join(PROJECT_DIR, "aditya_l1_catalog.csv")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "process_metadata.log")

# Ensure all directories exist BEFORE logging setup
for directory in [MASTER_DIR, OUTPUT_IMG_DIR, LOG_DIR]:
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"FATAL: Cannot create directory {directory}: {e}", file=sys.stderr)
        sys.exit(1)

# Safe logging setup
try:
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
except Exception as e:
    print(f"WARNING: Cannot setup file logging: {e}. Using console only.", file=sys.stderr)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

@contextmanager
def file_lock(file_path, timeout=30):
    """
    Cross-process file lock to prevent CSV corruption.
    Uses fcntl on Unix systems.
    """
    lock_file = f"{file_path}.lock"
    lock_fd = None
    
    try:
        # Create lock file
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR)
        
        # Try to acquire lock with timeout
        start_time = time.time()
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Could not acquire lock on {file_path}")
                time.sleep(0.1)
        
        yield lock_fd
        
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            except Exception:
                pass
            
            # Clean up lock file
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except Exception:
                pass

def safe_fits_close(hdul):
    """Safely close FITS file even if already closed."""
    try:
        if hdul is not None and not hdul._file.closed:
            hdul.close()
    except Exception:
        pass

def extract_fits_meta(filepath, filename):
    """Extract FITS metadata with comprehensive error handling."""
    meta = {"Filename": filename, "FileType": "FITS"}
    IGNORE_KEYS = ['COMMENT', 'HISTORY', '']
    hdul = None
    
    try:
        # Verify file exists and is readable
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not os.access(filepath, os.R_OK):
            raise PermissionError(f"Cannot read file: {filepath}")
        
        # Open FITS with error handling
        try:
            hdul = fits.open(filepath, memmap=False, ignore_missing_end=True)
        except Exception as e:
            logging.error(f"FITS open failed for {filename}: {e}")
            return meta
        
        # Extract header safely
        try:
            header = hdul[0].header
            for key, value in header.items():
                if key not in IGNORE_KEYS and not pd.isna(value):
                    # Sanitize value
                    if not isinstance(value, (int, float, str, bool, type(None))):
                        value = str(value)
                    # Truncate extremely long strings
                    if isinstance(value, str) and len(value) > 1000:
                        value = value[:1000] + "..."
                    meta[key] = value
        except Exception as e:
            logging.warning(f"Header extraction partial failure for {filename}: {e}")
        
        # Check for tabular data
        try:
            if len(hdul) > 1 and isinstance(hdul[1], fits.BinTableHDU):
                meta["Has_Tabular_Data"] = True
                meta["Num_Telemetry_Rows"] = len(hdul[1].data)
        except Exception as e:
            logging.warning(f"Tabular data check failed for {filename}: {e}")
            
    except Exception as e:
        logging.error(f"FITS extraction failed for {filename}: {e}")
        
    finally:
        safe_fits_close(hdul)
    
    return meta

def extract_cdf_meta(filepath, filename):
    """Extract CDF metadata with error handling."""
    meta = {"Filename": filename, "FileType": "CDF"}
    
    if not HAS_CDF:
        meta["Error"] = "cdflib not installed"
        return meta
    
    cdf_file = None
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        cdf_file = cdflib.CDF(filepath)
        global_attrs = cdf_file.globalattsget()
        
        for key, val in global_attrs.items():
            try:
                if isinstance(val, (list, np.ndarray)):
                    val = val[0] if len(val) > 0 else ""
                # Sanitize and truncate
                val_str = str(val)
                if len(val_str) > 1000:
                    val_str = val_str[:1000] + "..."
                meta[f"CDF_{key}"] = val_str
            except Exception as e:
                logging.warning(f"CDF attribute {key} extraction failed: {e}")
                
    except Exception as e:
        logging.error(f"CDF extraction failed for {filename}: {e}")
        meta["Error"] = str(e)
        
    finally:
        if cdf_file is not None:
            try:
                cdf_file.close()
            except Exception:
                pass
    
    return meta

def extract_nc_meta(filepath, filename):
    """Extract NetCDF metadata with error handling."""
    meta = {"Filename": filename, "FileType": "NetCDF"}
    
    if not HAS_NC:
        meta["Error"] = "netCDF4 not installed"
        return meta
    
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with netCDF4.Dataset(filepath, 'r') as nc:
            for attr_name in nc.ncattrs():
                try:
                    val = str(getattr(nc, attr_name))
                    if len(val) > 1000:
                        val = val[:1000] + "..."
                    meta[f"NC_{attr_name}"] = val
                except Exception as e:
                    logging.warning(f"NetCDF attribute {attr_name} extraction failed: {e}")
                    
    except Exception as e:
        logging.error(f"NetCDF extraction failed for {filename}: {e}")
        meta["Error"] = str(e)
    
    return meta

def process_single_file(filename):
    """
    Process a single file with complete error isolation.
    Returns metadata dict or None on failure.
    """
    try:
        filepath = os.path.join(MASTER_DIR, filename)
        
        # Safety checks
        if not os.path.exists(filepath):
            return None
            
        # Skip files modified in last 10 seconds (might still be downloading)
        try:
            if time.time() - os.path.getmtime(filepath) < 10.0:
                return None
        except OSError:
            return None
        
        # Route to appropriate extractor
        if filename.endswith('.fits'):
            return extract_fits_meta(filepath, filename)
        elif filename.endswith('.cdf'):
            return extract_cdf_meta(filepath, filename)
        elif filename.endswith('.nc'):
            return extract_nc_meta(filepath, filename)
        else:
            return None
            
    except Exception as e:
        logging.error(f"Process file failed for {filename}: {e}")
        return None

def safe_read_csv(csv_path):
    """Safely read CSV with error handling."""
    try:
        if not os.path.exists(csv_path):
            return pd.DataFrame()
        
        # Try reading with error recovery
        df = pd.read_csv(csv_path, low_memory=False)
        return df
        
    except pd.errors.EmptyDataError:
        logging.warning(f"Empty CSV file: {csv_path}")
        return pd.DataFrame()
        
    except Exception as e:
        logging.error(f"CSV read failed: {e}")
        # Try to read with more lenient settings
        try:
            df = pd.read_csv(csv_path, low_memory=False, on_bad_lines='skip')
            logging.warning(f"CSV read with skipped bad lines")
            return df
        except Exception:
            return pd.DataFrame()

def process_metadata_loop():
    """Main processing loop with comprehensive error handling."""
    try:
        # Ensure master directory exists
        if not os.path.exists(MASTER_DIR):
            logging.warning(f"Master directory does not exist: {MASTER_DIR}")
            return False
        
        # Get all valid files
        valid_exts = ('.fits', '.cdf', '.nc')
        try:
            all_files = [f for f in os.listdir(MASTER_DIR) 
                        if f.endswith(valid_exts) and os.path.isfile(os.path.join(MASTER_DIR, f))]
        except Exception as e:
            logging.error(f"Cannot list master directory: {e}")
            return False
        
        if not all_files:
            return False

        # Get already processed files
        processed_files = set()
        try:
            existing_df = safe_read_csv(CSV_OUTPUT_PATH)
            if not existing_df.empty and 'Filename' in existing_df.columns:
                processed_files = set(existing_df['Filename'].dropna().astype(str).tolist())
        except Exception as e:
            logging.warning(f"Could not load existing catalog: {e}")
        
        # Find new files
        new_files = [f for f in all_files if f not in processed_files]
        if not new_files:
            return False
        
        logging.info(f"Processing {len(new_files)} new files...")
        
        # Process files in parallel with error isolation
        metadata_catalog = []
        try:
            max_workers = min(os.cpu_count() or 4, 8)  # Cap at 8 to avoid resource exhaustion
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Use timeout to prevent hanging
                future_to_file = {executor.submit(process_single_file, f): f for f in new_files}
                
                for future in concurrent.futures.as_completed(future_to_file, timeout=300):
                    try:
                        res = future.result(timeout=30)
                        if res:
                            metadata_catalog.append(res)
                    except concurrent.futures.TimeoutError:
                        file = future_to_file[future]
                        logging.error(f"Processing timeout for {file}")
                    except Exception as e:
                        file = future_to_file[future]
                        logging.error(f"Processing error for {file}: {e}")
                        
        except Exception as e:
            logging.error(f"Parallel processing failed: {e}")
            # Fallback to serial processing
            logging.info("Falling back to serial processing...")
            for f in new_files:
                try:
                    res = process_single_file(f)
                    if res:
                        metadata_catalog.append(res)
                except Exception as file_err:
                    logging.error(f"Serial processing failed for {f}: {file_err}")

        # Save results with file locking
        if metadata_catalog:
            try:
                logging.info(f"Saving {len(metadata_catalog)} new records...")
                new_df = pd.DataFrame(metadata_catalog)
                
                # Use file lock to prevent corruption
                with file_lock(CSV_OUTPUT_PATH):
                    existing_df = safe_read_csv(CSV_OUTPUT_PATH)
                    
                    if not existing_df.empty:
                        # Combine dataframes
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    else:
                        combined_df = new_df
                    
                    # Reorder columns
                    cols = combined_df.columns.tolist()
                    priority_cols = ['Filename', 'FileType']
                    for col in reversed(priority_cols):
                        if col in cols:
                            cols.insert(0, cols.pop(cols.index(col)))
                    
                    combined_df = combined_df.reindex(columns=cols)
                    
                    # Atomic write using temp file
                    temp_csv = f"{CSV_OUTPUT_PATH}.tmp"
                    combined_df.to_csv(temp_csv, index=False)
                    
                    # Atomic rename
                    os.replace(temp_csv, CSV_OUTPUT_PATH)
                
                logging.info(f"✓ Database updated. Total rows: {len(combined_df)}")
                return True
                
            except Exception as e:
                logging.error(f"Failed to save catalog: {e}")
                # Clean up temp file
                try:
                    if os.path.exists(f"{CSV_OUTPUT_PATH}.tmp"):
                        os.remove(f"{CSV_OUTPUT_PATH}.tmp")
                except Exception:
                    pass
                return False
        
        return False
        
    except Exception as e:
        logging.error(f"Metadata loop crashed: {e}")
        return False

def garbage_collection():
    """
    Safely deletes raw files ONLY if they are fully processed.
    Includes comprehensive error handling.
    """
    try:
        if not os.path.exists(CSV_OUTPUT_PATH):
            return
        
        # Read catalog safely
        df = safe_read_csv(CSV_OUTPUT_PATH)
        if df.empty or 'Filename' not in df.columns:
            return
        
        db_files = set(df['Filename'].dropna().astype(str).tolist())
        
        # Get all files in master directory
        try:
            all_files = os.listdir(MASTER_DIR)
        except Exception as e:
            logging.error(f"Cannot list master directory for GC: {e}")
            return
        
        deleted_count = 0
        
        for f in all_files:
            try:
                filepath = os.path.join(MASTER_DIR, f)
                
                # Safety checks
                if not os.path.isfile(filepath):
                    continue
                
                if f not in db_files:
                    continue
                
                # FITS files: only delete if PNG exists
                if f.endswith('.fits'):
                    png_path = os.path.join(OUTPUT_IMG_DIR, f.replace('.fits', '.png'))
                    if os.path.exists(png_path):
                        try:
                            os.remove(filepath)
                            deleted_count += 1
                        except Exception as e:
                            logging.warning(f"Could not delete {f}: {e}")
                
                # CDF and NC: delete immediately after cataloging
                elif f.endswith(('.cdf', '.nc')):
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                    except Exception as e:
                        logging.warning(f"Could not delete {f}: {e}")
                        
            except Exception as e:
                logging.warning(f"GC error for {f}: {e}")
                continue
        
        if deleted_count > 0:
            logging.info(f"🗑️ GARBAGE COLLECTION: Cleaned up {deleted_count} files")
            
    except Exception as e:
        logging.error(f"Garbage collection crashed: {e}")

def main():
    """Main daemon loop with crash recovery."""
    logging.info("=" * 80)
    logging.info("Starting Hardened Metadata Extractor + Garbage Collector")
    logging.info(f"Python: {sys.version}")
    logging.info(f"CDF support: {HAS_CDF}")
    logging.info(f"NetCDF support: {HAS_NC}")
    logging.info("=" * 80)
    
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    while True:
        try:
            did_work = process_metadata_loop()
            
            # Run garbage collection
            try:
                garbage_collection()
            except Exception as e:
                logging.error(f"GC failed: {e}")
            
            # Reset error counter on success
            if did_work or consecutive_errors > 0:
                consecutive_errors = 0
            
            # Sleep interval
            sleep_time = 10 if not did_work else 2
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logging.info("Received shutdown signal. Exiting gracefully...")
            break
            
        except Exception as e:
            consecutive_errors += 1
            logging.error(f"DAEMON ERROR #{consecutive_errors}: {e}")
            
            # If too many consecutive errors, increase sleep time
            if consecutive_errors >= max_consecutive_errors:
                logging.critical(f"Too many consecutive errors. Sleeping for 5 minutes...")
                time.sleep(300)
                consecutive_errors = 0  # Reset after long sleep
            else:
                time.sleep(60)
    
    logging.info("Daemon shutdown complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"FATAL CRASH: {e}")
        sys.exit(1)
