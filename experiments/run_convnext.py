import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from final_lib import *

# ==========================================
# ConvNeXt Model Builder
# ==========================================
def get_convnext_model(num_classes, attention_type):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    
    if attention_type == 'Baseline':
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    def get_attn_layer(channels):
        if attention_type == 'CBAM': return CBAM(channels)
        if attention_type == 'Coord': return CoordinateAttention(channels)
        if attention_type == 'SimAM': return SimAM()
        if attention_type == 'ECA': return ECANet(channels)
        if attention_type == 'scSE': return scSE(channels)
        if attention_type == 'Triplet': return TripletAttention()
        if attention_type == 'StatAware': return StatAwareBlock(channels)
        return nn.Identity()

    # ConvNeXt features: [0, 2, 4, 6] are CNBlocks, [1, 3, 5] are Downsample layers (Norm+Conv).
    # structure: features[0] -> stage 1
    #            features[1] -> downsample
    #            features[2] -> stage 2 ...
    
    # We will inject attention after each main Stage (0, 2, 4, 6).
    
    new_features = []
    dummy = torch.randn(1, 3, 224, 224)
    
    for i, stage in enumerate(model.features):
        new_features.append(stage)
        
        # Stages 0, 2, 4, 6 contain the blocks
        if i in [0, 2, 4, 6]:
            # Infer channel size
            seq = nn.Sequential(*new_features)
            with torch.no_grad():
                out = seq(dummy)
            channels = out.shape[1]
            new_features.append(get_attn_layer(channels))
            
    model.features = nn.Sequential(*new_features)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./dataset_final"
    
    variants = ['Baseline', 'CBAM', 'Coord', 'SimAM', 'ECA', 'scSE', 'Triplet', 'StatAware']
    
    runner = ExperimentRunner("ConvNeXt-Tiny", device, data_dir, "results_convnext.txt")
    runner.run(variants, get_convnext_model)