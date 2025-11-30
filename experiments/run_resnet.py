import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from final_lib import *

def build_resnet(num_classes, attn_type):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    if attn_type == 'Baseline':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def get_attn(ch):
        if attn_type == 'CBAM': return CBAM(ch)
        if attn_type == 'ECA': return ECANet(ch)
        if attn_type == 'SimAM': return SimAM()
        if attn_type == 'Coord': return CoordinateAttention(ch)
        if attn_type == 'scSE': return scSE(ch)
        if attn_type == 'Triplet': return TripletAttention()
        if attn_type == 'StatAware': return StatAwareBlock(ch)
        return nn.Identity()

    # Inject into layers 1, 2, 3, 4 (Bottleneck blocks)
    # Inject after the 3rd conv (bn3) before residual connection
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            # Output channels of the block is expansion*planes
            ch = block.conv3.out_channels
            
            # Wrap the bn3 with attention
            block.bn3 = nn.Sequential(
                block.bn3,
                get_attn(ch)
            )
            
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = "dataset_final" # Update path if needed
    
    print(f"Running ResNet50 Experiment on {device}")
    run_experiment("ResNet50", build_resnet, DATA_DIR, "results_resnet.txt", device)