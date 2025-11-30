import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from final_lib import *

# ==========================================
# ResNet Model Builder with Attention
# ==========================================
def get_resnet_model(num_classes, attention_type):
    """
    Wraps ResNet50 and injects the specified attention block into Bottlenecks.
    """
    # 1. Load Pretrained
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # 2. Define Attention Constructor
    def get_attn_layer(channels):
        if attention_type == 'CBAM': return CBAM(channels)
        if attention_type == 'Coord': return CoordinateAttention(channels)
        if attention_type == 'SimAM': return SimAM()
        if attention_type == 'ECA': return ECANet(channels)
        if attention_type == 'scSE': return scSE(channels)
        if attention_type == 'Triplet': return TripletAttention()
        if attention_type == 'StatAware': return StatAwareBlock(channels)
        return nn.Identity() # Baseline

    # 3. Inject into Bottleneck
    # Standard ResNet Bottleneck has expansion=4.
    # We inject attention after the last conv (conv3) but before residual addition in the original,
    # but here we can wrap existing blocks or monkey-patch.
    # Monkey-patching existing instances is easiest for torchvision models.
    
    for module in model.modules():
        if isinstance(module, torch.nn.modules.container.Sequential):
            # Iterate through blocks in layer1, layer2, etc.
            continue
    
    # A more robust way for ResNet is iterating named modules
    # We look for 'layer1', 'layer2', 'layer3', 'layer4'
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    
    if attention_type != 'Baseline':
        for layer in layers:
            for block in layer:
                # The Bottleneck in torchvision has: conv1, bn1, conv2, bn2, conv3, bn3, relu, downsample
                # We want to insert attention after bn3, before the final relu/residual add.
                # Since we can't easily rewrite the forward method of an instantiated object, 
                # we will Replace the `bn3` with a Sequential(bn3, Attention).
                # This works because bn3 output shape is (B, C, H, W).
                
                # Output channels of the block is block.conv3.out_channels
                channels = block.conv3.out_channels
                attn_layer = get_attn_layer(channels)
                
                # Replace bn3
                block.bn3 = nn.Sequential(
                    block.bn3,
                    attn_layer
                )

    # 4. Modify Head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./dataset_final" # Update this path
    
    variants = ['Baseline', 'CBAM', 'Coord', 'SimAM', 'ECA', 'scSE', 'Triplet', 'StatAware']
    
    runner = ExperimentRunner("ResNet50", device, data_dir, "results_resnet50.txt")
    runner.run(variants, get_resnet_model)