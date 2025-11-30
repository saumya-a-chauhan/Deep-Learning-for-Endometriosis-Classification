import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from torch.cuda.amp import autocast, GradScaler

# ==========================================
# 1. The Champion Mechanism: CBAM
# ==========================================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction)
        self.spatial_gate = SpatialGate()
        
    def forward(self, x):
        x_out = self.channel_gate(x)
        x_out = self.spatial_gate(x_out)
        return x_out

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        scale = torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.spatial(torch.cat([max_out, avg_out], dim=1)))
        return x * scale

# ==========================================
# 2. Grad-CAM Visualization Tool
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        
        score = output[0, class_idx]
        score.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0]
        
        # Weight the channels by the gradients
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        return heatmap, output

def visualize_gradcam(model, target_layer, img_tensor, img_path, save_name):
    grad_cam = GradCAM(model, target_layer)
    heatmap, _ = grad_cam(img_tensor.unsqueeze(0))
    
    # Read original image for overlay
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Colorize heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(save_name, superimposed_img)

# ==========================================
# 3. Data & Training Engine
# ==========================================

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'w') as f:
            f.write("Final ResNet50+CBAM Experiment Log\n")
            
    def log(self, text):
        print(text)
        with open(self.filepath, 'a') as f:
            f.write(text + "\n")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_dataloaders(data_dir, split_ratio, seed, batch_size=16):
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_ds = datasets.ImageFolder(data_dir)
    targets = full_ds.targets
    
    # 20% reserved for testing
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_pool_idx, test_idx = next(sss_test.split(np.zeros(len(targets)), targets))
    
    # Data Scarcity logic
    if split_ratio < 1.0:
        train_pool_targets = [targets[i] for i in train_pool_idx]
        sss_scarce = StratifiedShuffleSplit(n_splits=1, train_size=split_ratio, random_state=seed)
        used_idx, _ = next(sss_scarce.split(np.zeros(len(train_pool_idx)), train_pool_targets))
        final_train_idx = [train_pool_idx[i] for i in used_idx]
    else:
        final_train_idx = train_pool_idx

    train_ds = Subset(full_ds, final_train_idx)
    test_ds = Subset(full_ds, test_idx)
    
    # Transforms
    train_ds.dataset.transform = train_tfm
    test_ds.dataset.transform = val_tfm

    # Balanced Sampler
    train_targets_subset = [targets[i] for i in final_train_idx]
    class_counts = np.bincount(train_targets_subset)
    class_weights = 1. / (class_counts + 1e-6) # Avoid div zero
    sample_weights = [class_weights[t] for t in train_targets_subset]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # CRITICAL FIX: drop_last=True prevents single-sample batch crash
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader, len(full_ds.classes), full_ds.classes, test_ds

def run_final_training(model_builder, data_dir, output_file, device):
    logger = Logger(output_file)
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)
    
    # Configuration
    data_splits = [0.1, 0.2, 0.5, 0.8]
    best_overall_acc = 0.0
    best_overall_model_state = None
    best_class_names = []
    
    for split in data_splits:
        # Robustness for 10%, single run for others
        seeds = [42, 10, 20, 30, 40] if split == 0.1 else [42]
        
        split_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
        
        logger.log(f"\n{'='*40}\nRunning Split: {int(split*100)}%\n{'='*40}")
        
        for seed in seeds:
            logger.log(f"  > Seed: {seed} ...")
            
            try:
                # Load
                train_loader, test_loader, num_classes, class_names, test_subset = get_dataloaders(data_dir, split, seed)
                best_class_names = class_names
                
                # Build
                model = model_builder(num_classes).to(device)
                
                # Setup
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Smooth labels for better generalization
                logger.log(f"    Loss: CrossEntropy(smoothing=0.1)")
                optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
                
                # Train Loop (25 Epochs)
                model.train()
                for epoch in range(25):
                    run_loss = 0.0
                    run_correct = 0
                    total_samples = 0
                    
                    for imgs, lbls in train_loader:
                        imgs, lbls = imgs.to(device), lbls.to(device)
                        optimizer.zero_grad()
                        
                        with autocast(enabled=use_amp):
                            outputs = model(imgs)
                            loss = criterion(outputs, lbls)
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                        run_loss += loss.item() * imgs.size(0)
                        _, preds = torch.max(outputs, 1)
                        run_correct += (preds == lbls).sum().item()
                        total_samples += lbls.size(0)
                        
                    epoch_loss = run_loss / total_samples
                    epoch_acc = 100 * run_correct / total_samples
                    logger.log(f"    Epoch {epoch+1}/25 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
                
                # Evaluate
                model.eval()
                all_preds = []
                all_lbls = []
                with torch.no_grad():
                    for imgs, lbls in test_loader:
                        imgs, lbls = imgs.to(device), lbls.to(device)
                        outputs = model(imgs)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_lbls.extend(lbls.cpu().numpy())
                
                # Metrics
                acc = accuracy_score(all_lbls, all_preds)
                p = precision_score(all_lbls, all_preds, average='macro', zero_division=0)
                r = recall_score(all_lbls, all_preds, average='macro', zero_division=0)
                f1 = f1_score(all_lbls, all_preds, average='macro', zero_division=0)
                
                split_metrics['acc'].append(acc)
                split_metrics['prec'].append(p)
                split_metrics['rec'].append(r)
                split_metrics['f1'].append(f1)
                
                # Save Best Model logic (Keep the single best model from all runs)
                if acc > best_overall_acc:
                    best_overall_acc = acc
                    best_overall_model_state = model.state_dict()
                    # Generate Confusion Matrix for the best run
                    cm = confusion_matrix(all_lbls, all_preds)
                    plt.figure(figsize=(6,5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
                    plt.title(f"Confusion Matrix (Best Run - {int(split*100)}%)")
                    plt.ylabel('True')
                    plt.xlabel('Predicted')
                    plt.savefig(f"best_confusion_matrix_{int(split*100)}.png")
                    plt.close()
                    logger.log(f"    -> Saved Best Confusion Matrix (Acc: {acc:.4f})")
                    
                    # GradCAM on first test image
                    # Find a target layer: ResNet layer4 last block conv3
                    target_layer = list(model.layer4)[-1].conv3
                    sample_img_idx = test_subset.indices[0] # Get index in full dataset
                    # We need the path. full_ds.samples[idx] = (path, label)
                    full_ds_path = test_subset.dataset.samples[sample_img_idx][0]
                    # We need tensor
                    sample_tensor, _ = test_subset[0] # This gets transformed tensor
                    
                    visualize_gradcam(model, target_layer, sample_tensor.to(device), full_ds_path, "gradcam_best.png")
                    logger.log(f"    -> Generated GradCAM Visualization")

            except Exception as e:
                logger.log(f"    ERROR: {e}")
        
        # Log Summary for Split
        mean_acc = np.mean(split_metrics['acc'])
        std_acc = np.std(split_metrics['acc'])
        mean_rec = np.mean(split_metrics['rec'])
        mean_f1 = np.mean(split_metrics['f1'])
        
        logger.log(f"\nSUMMARY | Split {int(split*100)}% | Acc: {mean_acc:.4f} (+/- {std_acc:.4f}) | Recall: {mean_rec:.4f} | F1: {mean_f1:.4f}")

    # Save Final Best Model
    if best_overall_model_state:
        torch.save(best_overall_model_state, "final_resnet_cbam_best.pth")
        logger.log("\nSaved final best model weights to 'final_resnet_cbam_best.pth'")