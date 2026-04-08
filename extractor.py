import os
import time
import logging
import traceback
import urllib.parse
from playwright.sync_api import sync_playwright, Route

# --- CONFIGURATION ---
LOGIN_URL = "https://pradan.issdc.gov.in/al1/"
USER_EMAIL = "Aniruddha0516"  
USER_PASS = "@ZAEusDkYtVw2KV"

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_DIR = os.path.join(PROJECT_DIR, "master_archive")
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "extraction.log")
PROGRESS_FILE = os.path.join(PROJECT_DIR, "logs", "progress.txt")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
os.makedirs(MASTER_DIR, exist_ok=True)

def get_start_page():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            content = f.read().strip()
            if content.isdigit():
                return int(content)
    return 1

def save_progress(page_num):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(page_num))

def run_extraction_session():
    current_page = get_start_page()
    logging.info(f"=== Starting Session. Target Start Page: {current_page} ===")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True) 
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        # --- 1. AUTHENTICATION ---
        logging.info("Authenticating...")
        page.goto(LOGIN_URL)
        page.click("text='Login/Signup'") 
        
        page.wait_for_selector("input[name='username']", timeout=30000)
        page.fill("input[name='username']", USER_EMAIL) 
        page.fill("input[name='password']", USER_PASS)
        page.click("input[type='submit'], button[type='submit'], input[name='login']") 
        
        page.wait_for_url("**/protected/payload.xhtml*", timeout=60000)
        logging.info("Authentication Successful!")

        # --- 2. NAVIGATE TO ALL PAYLOADS ---
        logging.info("Selecting ALL Payloads...")
        page.locator("div.ui-panel", has_text="ALL Payloads").locator("button").click()
        page.wait_for_selector(".ui-datatable", timeout=15000) 

        # --- 3. NETWORK INTERCEPTION JUMP ---
        if current_page > 1:
            logging.info(f"Initiating Network Interception to jump to page {current_page}...")
            
            # The row offset is (page - 1) * 20 (assuming 20 items per page)
            target_first_row = (current_page - 1) * 20
            jump_successful = {"status": False}

            def intercept_pagination(route: Route):
                request = route.request
                if request.method == "POST" and "javax.faces.partial.ajax=true" in request.post_data:
                    # We found a PrimeFaces AJAX request! Let's rewrite it.
                    post_data = request.post_data
                    parsed_data = urllib.parse.parse_qs(post_data)
                    
                    # Look for the pagination parameter
                    for key in parsed_data:
                        if key.endswith("_first"): 
                            # Rewrite the row offset to force the server to jump
                            parsed_data[key] = [str(target_first_row)]
                            
                            # Re-encode the modified payload
                            new_post_data = urllib.parse.urlencode(parsed_data, doseq=True)
                            logging.info(f"Intercepted and rewrote pagination request: {key} -> {target_first_row}")
                            
                            route.continue_(post_data=new_post_data)
                            jump_successful["status"] = True
                            return
                route.continue_()

            try:
                # 1. Start intercepting network traffic
                page.route("**/*", intercept_pagination)
                
                # 2. Trigger ANY pagination click to force an AJAX request we can hijack
                page.locator(".ui-paginator-next").click()
                
                # 3. Wait for the server to process our hijacked request
                page.wait_for_timeout(8000) 
                
                # 4. Stop intercepting so normal downloads work again
                page.unroute("**/*")
                
                if jump_successful["status"]:
                    logging.info(f"Network Hijack Successful! Teleported to page {current_page}.")
                else:
                    raise Exception("Failed to intercept pagination POST request.")
                    
            except Exception as e:
                logging.warning(f"Network jump failed. Falling back to slow clicks. Error: {e}")
                page.unroute("**/*") # Ensure interception is off
                for i in range(1, current_page):
                    page.locator(".ui-paginator-next").click()
                    page.wait_for_timeout(1500) 

        # --- 4. THE MASTER DOWNLOAD LOOP ---
        MAX_PAGES = 78228 

        while current_page <= MAX_PAGES:
            logging.info(f"--- Processing Page {current_page} ---")
            
            page.locator(".ui-chkbox-box").nth(0).click()
            page.wait_for_timeout(1000)

            download_button = page.locator("span", has_text="DOWNLOAD SELECTED as zip")
            
            if download_button.count() > 0:
                with page.expect_download(timeout=300000) as download_info: 
                    download_button.click()
                
                download = download_info.value
                safe_filename = f"page_{current_page}_{download.suggested_filename}"
                file_path = os.path.join(MASTER_DIR, safe_filename)
                download.save_as(file_path)
                logging.info(f"Saved: {file_path}")
                
                save_progress(current_page)
            else:
                logging.warning("No download button found. Skipping page.")

            if current_page < MAX_PAGES:
                next_button = page.locator(".ui-paginator-next")
                if "ui-state-disabled" in next_button.get_attribute("class", timeout=5000) or "":
                    logging.info("Reached the final page of the dataset.")
                    break
                    
                next_button.click()
                page.wait_for_timeout(5000) 
            
            current_page += 1
            
        logging.info("ENTIRE ARCHIVE DOWNLOADED SUCCESSFULLY!")
        browser.close()

if __name__ == "__main__":
    while True:
        try:
            run_extraction_session()
            break 
        except Exception as e:
            logging.error(f"CRITICAL SESSION CRASH: {e}")
            logging.error(traceback.format_exc())
            logging.info("Sleeping for 60 seconds, then restarting pipeline...")
            time.sleep(60)
