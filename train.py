import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import GradScaler, autocast # PyTorch 2.0+ AMP
import logging
import time

# Import your custom modules
from dataset import AdityaL1Dataset
from model import SolarHybridModel

# --- CONFIGURATION ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_DIR, 'aditya_l1_catalog.csv')
IMG_DIR = os.path.join(PROJECT_DIR, 'processed_images')
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, 'solar_hybrid_v1.pth')

BATCH_SIZE = 16  # Adjust based on your GPU VRAM
EPOCHS = 10
LEARNING_RATE = 1e-4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- ADVANCED PHYSICS: FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # inputs: raw logits from model, targets: ground truth 0 or 1
        bce_loss = self.bce_with_logits(inputs.view(-1), targets.view(-1))
        pt = torch.exp(-bce_loss) # Probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def train_model():
    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🔥 Training Engine Online. Device detected: {device}")

    # 2. Data Pipeline Pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = AdityaL1Dataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=transform)
    # pin_memory speeds up CPU to GPU data transfer
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # 3. Initialize Architecture
    model = SolarHybridModel(num_tabular_features=4).to(device)
    
    # 4. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.75, gamma=2.0) # Alpha > 0.5 weights the rare positive class heavier
    scaler = GradScaler() # For Mixed Precision (AMP)

    logging.info(f"Initiating Training for {EPOCHS} Epochs on {len(dataset)} samples...")

    # 5. THE MASTER TRAINING LOOP
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (images, tabular, labels) in enumerate(dataloader):
            # Move data to GPU
            images = images.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass with Automatic Mixed Precision
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images, tabular)
                loss = criterion(outputs, labels)
                
            # Backward pass and optimization (Scaled for AMP)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(dataloader)
        logging.info(f"✅ Epoch {epoch+1} Completed in {epoch_time:.1f}s | Average Loss: {avg_loss:.4f}")

    # 6. Save the trained brain
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logging.info(f"💾 Model weights saved securely to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
