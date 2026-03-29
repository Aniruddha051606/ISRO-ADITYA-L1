import os
import pandas as pd
import numpy as np
import logging
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
# This line allows PIL to attempt to load even slightly broken images 
# but our try-except block will still catch the fatal ones.
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIG ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_DIR, "aditya_l1_catalog.csv")
IMG_DIR = os.path.join(PROJECT_DIR, "processed_images")

def extract_solar_intensity(df):
    """Safely extracts pixel mean from PNGs with integrity checks."""
    intensities = []
    for filename in df['Filename']:
        png_path = os.path.join(IMG_DIR, filename.replace('.fits', '.png'))
        
        try:
            # Check 1: Does file exist?
            if not os.path.exists(png_path):
                intensities.append(np.nan)
                continue
                
            # Check 2: Is the file empty (0 bytes)?
            if os.path.getsize(png_path) < 100: # Minimum PNG header size
                intensities.append(np.nan)
                continue

            with Image.open(png_path) as img:
                # Check 3: Force a load to verify the buffer isn't truncated
                img.load() 
                img_gray = img.convert('L')
                # Normalization: Convert 0-255 to 0.0-1.0
                pixel_avg = np.mean(np.array(img_gray)) / 255.0
                intensities.append(pixel_avg)
                
        except (OSError, ValueError, UnboundLocalError) as e:
            logging.warning(f"Skipping incomplete image: {filename} - {e}")
            intensities.append(np.nan)
            
    df['Intensity'] = intensities
    return df.dropna(subset=['Intensity'])

def train_and_predict():
    if not os.path.exists(CSV_PATH):
        return

    df = pd.read_csv(CSV_PATH)
    
    # 1. SCIENTIFIC FILTERING
    # Only use 2796 (Mg II) to ensure we are comparing apples to apples
    df = df[df['Wavelength'] == "2796"].copy()
    
    if len(df) < 20:
        logging.info("Waiting for more 2796Å data to train ML...")
        return

    # 2. FEATURE EXTRACTION
    df = extract_solar_intensity(df)
    
    # 3. TIME-SERIES PREP
    df['Observation_Date'] = pd.to_datetime(df['Observation_Date'])
    df = df.sort_values('Observation_Date')
    
    # Create 'Lag' features (look at the previous 3 intensities)
    df['Lag_1'] = df['Intensity'].shift(1)
    df['Lag_2'] = df['Intensity'].shift(2)
    df['Lag_3'] = df['Intensity'].shift(3)
    df = df.dropna()

    X = df[['Lag_1', 'Lag_2', 'Lag_3']]
    y = df['Intensity']

    # 4. RANDOM FOREST MODEL
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict the next step
    last_row = df.iloc[-1]
    next_input = np.array([[last_row['Intensity'], last_row['Lag_1'], last_row['Lag_2']]])
    prediction = model.predict(next_input)[0]
    
    logging.info(f"ML UPDATE: Current Intensity: {last_row['Intensity']:.4f} | Predicted: {prediction:.4f}")

if __name__ == "__main__":
    train_and_predict()
