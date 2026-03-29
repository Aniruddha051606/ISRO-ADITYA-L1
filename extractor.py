import os
import time
import logging
import traceback
from playwright.sync_api import sync_playwright

# --- CONFIGURATION ---
LOGIN_URL = "https://pradan.issdc.gov.in/al1/"
USER_EMAIL = "Aniruddha0516"  
USER_PASS = "@ZAEusDkYtVw2KV"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "extraction.log")
PROGRESS_FILE = os.path.join(PROJECT_DIR, "logs", "progress.txt")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_start_page():
    """Reads the progress file to know where to resume after a crash."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            content = f.read().strip()
            if content.isdigit():
                return int(content)
    return 1 # Default to page 1 if no progress file exists

def save_progress(page_num):
    """Saves the current page number to disk."""
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(page_num))

def run_extraction_session():
    current_page = get_start_page()
    logging.info(f"=== Starting Session. Target Start Page: {current_page} ===")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True) 
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        # 1. AUTHENTICATION (Unchanged)
        logging.info("Authenticating...")
        page.goto(LOGIN_URL)
        page.click("text='Login/Signup'") 
        page.wait_for_url("**/auth/realms/issdc/protocol/openid-connect/**", timeout=30000)
        page.wait_for_selector("input[name='username']", timeout=15000)
        page.fill("input[name='username']", USER_EMAIL) 
        page.fill("input[name='password']", USER_PASS)
        page.click("input[type='submit'], button[type='submit'], input[name='login']") 
        page.wait_for_url("**/protected/payload.xhtml", timeout=30000)
        logging.info("Authentication Successful!")

        # 2. NAVIGATE TO ALL PAYLOADS
        logging.info("Selecting ALL Payloads...")
        page.locator("div.ui-panel", has_text="ALL Payloads").locator("button").click()
        page.wait_for_selector(".ui-datatable", timeout=15000) 

        # 3. FAST-FORWARD RECOVERY LOGIC
        if current_page > 1:
            logging.info(f"Fast-forwarding to page {current_page} without downloading...")
            for i in range(1, current_page):
                next_button = page.locator(".ui-paginator-next")
                next_button.click()
                # Wait 2 seconds for the PrimeFaces AJAX swap before clicking next again
                page.wait_for_timeout(2000) 
            logging.info(f"Successfully caught up to page {current_page}. Resuming downloads.")

        # 4. THE MASTER DOWNLOAD LOOP
        MAX_PAGES = 78228 

        while current_page <= MAX_PAGES:
            logging.info(f"--- Processing Page {current_page} ---")
            
            # Select files
            page.locator(".ui-chkbox-box").nth(0).click()
            page.wait_for_timeout(1000)

            # Download Zip
            download_button = page.locator("span", has_text="DOWNLOAD SELECTED as zip")
            
            if download_button.count() > 0:
                with page.expect_download(timeout=120000) as download_info: # 2-minute timeout for safety
                    download_button.click()
                
                download = download_info.value
                safe_filename = f"page_{current_page}_{download.suggested_filename}"
                file_path = os.path.join(DATA_DIR, safe_filename)
                download.save_as(file_path)
                logging.info(f"Saved: {file_path}")
                
                # CRITICAL: Save progress to disk only AFTER a successful download
                save_progress(current_page)
            else:
                logging.warning("No download button found. Skipping page.")

            # Move to next page
            if current_page < MAX_PAGES:
                next_button = page.locator(".ui-paginator-next")
                if "ui-state-disabled" in next_button.get_attribute("class", timeout=5000) or "":
                    logging.info("Reached the final page of the dataset.")
                    break
                    
                next_button.click()
                page.wait_for_timeout(5000) # Give ISRO time to breathe
            
            current_page += 1
            
        logging.info("ENTIRE ARCHIVE DOWNLOADED SUCCESSFULLY!")
        browser.close()

if __name__ == "__main__":
    # THE ZOMBIE LOOP: Keep the script alive forever until it finishes
    while True:
        try:
            run_extraction_session()
            break # If run_extraction_session() finishes without crashing, break the infinite loop
        except Exception as e:
            logging.error(f"CRITICAL SESSION CRASH: {e}")
            logging.error(traceback.format_exc())
            logging.info("ISRO likely terminated the session. Sleeping for 60 seconds, then restarting pipeline...")
            time.sleep(60)
            # The loop will automatically go back to the top and run_extraction_session() again
