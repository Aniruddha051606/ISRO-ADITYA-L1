import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SolarViTHybrid  # Our new Transformer model
from dataset import AdityaL1Dataset
import logging

# --- CONFIGURATION ---
CSV_PATH = '/root/aditya_l1_project/aditya_l1_catalog.csv'
IMG_DIR = '/root/aditya_l1_project/processed_images/'
BATCH_SIZE = 32
EPOCHS = 15 # Transformers sometimes need a few more epochs to settle
LEARNING_RATE = 1e-4 # Lower learning rate is safer for ViTs

def train_v2():
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 ViT Training Engine Online. Device: {device}")

    # 1. Data Augmentation (Crucial for ViT with small data)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Sun looks the same from many angles
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = AdityaL1Dataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize the Transformer Model
    model = SolarViTHybrid(num_tabular_features=4).to(device)
    
    # Loss & Optimizer
    # Using BCEWithLogitsLoss because it's stable for binary classification
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 3. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (images, tabular, labels) in enumerate(train_loader):
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images, tabular)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{i}/{len(train_loader)}] | Loss: {loss.item():.4f}")

    # 4. Save the "New Brain"
    torch.save(model.state_dict(), '/root/aditya_l1_project/solar_vit_v2.pth')
    logging.info("💾 ViT Model weights saved as solar_vit_v2.pth")

if __name__ == "__main__":
    train_v2()O
