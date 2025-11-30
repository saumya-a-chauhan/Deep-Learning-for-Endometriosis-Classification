import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from final_lib import run_final_training, CBAM

# ==========================================
# ResNet50 + CBAM Builder
# ==========================================
def build_resnet_cbam(num_classes):
    print("Building ResNet50 + CBAM Model...")
    # Load Pretrained Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Inject CBAM into Bottleneck blocks
    # We iterate through layers 1-4 and wrap the final conv/bn
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            # The standard Bottleneck output channels
            ch = block.conv3.out_channels
            
            # Inject CBAM after the last batch norm (bn3)
            # This places attention right before the residual connection
            block.bn3 = nn.Sequential(
                block.bn3,
                CBAM(ch)
            )
            
    # Modify Classification Head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # Settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = "dataset_final" # Ensure this folder exists
    OUTPUT_FILE = "final_results_cbam.txt"
    
    print(f"Running Final ResNet50+CBAM Training on {DEVICE}")
    
    # Run Pipeline
    run_final_training(
        model_builder=build_resnet_cbam,
        data_dir=DATA_DIR,
        output_file=OUTPUT_FILE,
        device=DEVICE
    )