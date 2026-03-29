import torch
import torch.nn as nn
from torchvision import models
import logging

class SolarHybridModel(nn.Module):
    def __init__(self, num_tabular_features=4, dropout_rate=0.3):
        super(SolarHybridModel, self).__init__()
        
        logging.info("Initializing Solar Multi-Modal Network...")
        
        # --- 1. THE VISION HEAD (Spatial Encoder) ---
        # We use ResNet18 pre-trained on ImageNet to give it a massive head start
        # on detecting edges, curves, and textures in the solar plasma.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the final classification layer to get the raw 512-dimensional embedding
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_feature_dim = resnet.fc.in_features # This will be 512
        
        # --- 2. THE METADATA HEAD (Tabular Encoder) ---
        # A smart dense network to process your CSV physics data
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.tabular_feature_dim = 32
        
        # --- 3. THE FUSION CLASSIFIER ---
        # Concatenate 512 (Image) + 32 (Tabular) = 544 Total Features
        combined_dim = self.image_feature_dim + self.tabular_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1) # Outputting a single raw number (Logit) for Flare/No Flare
        )

    def forward(self, images, tabular_data):
        # 1. Process the Image
        # Output shape from ResNet is (Batch, 512, 1, 1), so we flatten it to (Batch, 512)
        img_features = self.image_encoder(images)
        img_features = torch.flatten(img_features, 1)
        
        # 2. Process the Metadata
        tab_features = self.tabular_encoder(tabular_data)
        
        # 3. Fuse them together
        # We stack the two feature vectors side-by-side
        combined_features = torch.cat((img_features, tab_features), dim=1)
        
        # 4. Make the final prediction
        output = self.classifier(combined_features)
        
        # We return raw logits (no sigmoid here) because PyTorch's BCEWithLogitsLoss 
        # is mathematically much more stable for training.
        return output

# --- TEST THE ARCHITECTURE ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize our new model
    model = SolarHybridModel(num_tabular_features=4)
    
    # Create fake "Dummy Tensors" that match the shapes your Dataloader just outputted
    # This proves the math inside the network aligns perfectly with your dataset
    dummy_images = torch.randn(4, 3, 224, 224) 
    dummy_tabular = torch.randn(4, 4)          
    
    # Pass the fake data through the network
    predictions = model(dummy_images, dummy_tabular)
    
    print("\n🧠 MODEL ARCHITECTURE TEST SUCCESSFUL!")
    print(f"Input Image Shape: {dummy_images.shape}")
    print(f"Input Tabular Shape: {dummy_tabular.shape}")
    print(f"Output Prediction Shape: {predictions.shape} -> Ready for Loss Calculation")

