import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging

class AdityaL1Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        logging.info("Loading Aditya-L1 Catalog into memory...")
        # low_memory=False prevents Pandas from panicking over the 246+ columns
        self.data_frame = pd.read_csv(csv_file, low_memory=False)
        
        # --- FEATURE SELECTION (Phase 1 Base Features) ---
        # We start with the absolute core physics parameters. We can add more later.
        self.tabular_cols = ['EXPTIME', 'SUN_CX', 'SUN_CY', 'R_SUN']
        
        # Clean the data: Fill missing values with 0.0 so PyTorch doesn't crash on NaNs
        for col in self.tabular_cols:
            if col not in self.data_frame.columns:
                self.data_frame[col] = 0.0 
        
        self.tabular_data = self.data_frame[self.tabular_cols].fillna(0.0).astype(float)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # --- 1. LOAD THE IMAGE ---
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx]['Filename'].replace('.fits', '.png'))
        
        try:
            # Convert to RGB (3 channels) as most modern CNNs/ViTs expect 3 channels
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            # Safety net: If image is missing, return a blank black image
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # --- 2. LOAD THE METADATA ---
        tabular_features = torch.tensor(self.tabular_data.iloc[idx].values, dtype=torch.float32)

        # --- 3. THE TARGET LABEL (Flare Detection) ---
        # For now, we will look for 'FLR_TRIG' (Flare Triggered). If it's missing, default to 0.
        label = 0
        if 'FLR_TRIG' in self.data_frame.columns:
            val = self.data_frame.iloc[idx]['FLR_TRIG']
            # Basic logic: If the value is 1, True, or YES, flag it as a flare
            if str(val) == '1' or str(val).lower() == 'true':
                label = 1
                
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image, tabular_features, label_tensor

# --- TEST THE DATALOADER ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Image transformations: Resize to 224x224 (Standard for ResNet/ViT) and convert to Tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Standard ImageNet normalization (Helps the model converge faster)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize the dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = AdityaL1Dataset(
        csv_file=os.path.join(project_root, 'aditya_l1_catalog.csv'),
        img_dir=os.path.join(project_root, 'processed_images'),
        transform=transform
    )

    # Load a batch of 4 images at a time
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Grab the very first batch to prove it works
    images, tabular, labels = next(iter(dataloader))
    
    print("\n🚀 DATALOADER TEST SUCCESSFUL!")
    print(f"Image Batch Shape: {images.shape} (Batch, Channels, Height, Width)")
    print(f"Tabular Batch Shape: {tabular.shape} (Batch, Features)")
    print(f"Labels Batch Shape: {labels.shape} (Batch)")
