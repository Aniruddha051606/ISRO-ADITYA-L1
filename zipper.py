import shutil
import os

# 1. The exact name or path of the folder you want to compress
folder_to_zip = "scripts" 

# 2. The name of the zip file you want to create 
# Note: Don't add '.zip' to the end, Python will do it automatically!
output_zip_name = "my_script_update" 

print(f"Attempting to zip the '{folder_to_zip}' folder...")

# Check if the folder actually exists before trying to zip it
if os.path.exists(folder_to_zip):
    
    # This single line does all the heavy lifting!
    # Format: shutil.make_archive(output_filename, format, folder_to_compress)
    shutil.make_archive(output_zip_name, 'zip', folder_to_zip)
    
    print(f"Success! '{output_zip_name}.zip' is ready.")
else:
    print(f"Error: Could not find a folder named '{folder_to_zip}'.")
    print("Make sure the folder is in the exact same location as this Python script!")
