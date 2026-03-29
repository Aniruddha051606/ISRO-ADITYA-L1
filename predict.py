import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Import your model architecture from your local scripts folder
from model import SolarHybridModel

# --- CONFIGURATION ---
MODEL_PATH = '/home/aniruddha0516/aditya_l1_project/solar_hybrid_v1.pth'
# Picking one of the images you just uploaded as a test
TEST_IMAGE = '/home/aniruddha0516/aditya_l1_project/processed_images/SUT_T26_0473_001978_Lev1.0_2026-03-25T22.31.21.391_08B3NB01.png'

def predict_flare(image_path, tabular_features):
    device = torch.device("cpu") # Running on ISRO CPU
    
    # 1. Load Architecture & Weights
    model = SolarHybridModel(num_tabular_features=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    # 3. Physics Data (EXPTIME, SUN_CX, SUN_CY, R_SUN)
    tab_tensor = torch.tensor([tabular_features], dtype=torch.float32)

    # 4. Inference
    with torch.no_grad():
        logits = model(image_tensor, tab_tensor)
        probability = torch.sigmoid(logits).item()

    return probability

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        # Example physics values (replace with actual values from your CSV if needed)
        physics_values = [1.0, 0.0, 0.0, 960.0] 
        
        prob = predict_flare(TEST_IMAGE, physics_values)
        
        print("\n" + "="*40)
        print("☀️  ADITYA-L1 REAL-TIME FLARE ANALYSIS")
        print("="*40)
        print(f"Target: {os.path.basename(TEST_IMAGE)}")
        print(f"Flare Probability: {prob * 100:.4f}%")
        
        if prob > 0.8:
            print("\n🚨 ALERT: HIGH INTENSITY SOLAR ACTIVITY DETECTED!")
        else:
            print("\n✅ STATUS: Solar Activity Normal.")
        print("="*40)
    else:
        print(f"Error: Brain not found at {MODEL_PATH}")
