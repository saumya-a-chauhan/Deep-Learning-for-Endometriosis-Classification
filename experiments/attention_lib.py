import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random
import os
import math
from torch.cuda.amp import autocast, GradScaler

# ==========================================
# 1. NOVELTY: Statistical Awareness Block
# ==========================================
class StatAwareBlock(nn.Module):
    """
    NOVELTY: Utilizes higher-order statistics (Std/Variance) in addition to Mean
    to capture texture irregularities common in medical lesions.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        mid_channels = max(1, channels // reduction)
        # Input dim is channels * 2 because we concat Mean and Std
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, mid_channels, bias=False), 
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 1. Mean Feature (Global Average Pooling)
        y_mean = self.avg_pool(x).view(b, c)
        
        # 2. Std Feature (Global Standard Deviation)
        y_std = torch.std(x.view(b, c, -1), dim=2)
        
        # 3. Concatenate Statistics
        y_stats = torch.cat([y_mean, y_std], dim=1)
        
        # 4. Excitation
        y = self.fc(y_stats).view(b, c, 1, 1)
        
        return x * y.expand_as(x)

# ==========================================
# 2. Standard Attention Mechanisms
# ==========================================

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        mc = self.sigmoid(avg_out + max_out)
        x = x * mc
        # Spatial
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        ms = self.sigmoid(self.conv_spatial(torch.cat([avg_pool, max_pool], dim=1)))
        return x * ms

class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, 1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid, channels, 1)
        self.conv_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w

class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class ECANet(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)

class scSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(channels, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        chn_se = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        spa_se = self.sigmoid(self.conv(x))
        return (x * chn_se) + (x * spa_se)

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super().__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        if self.relu: x = self.relu(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

# ==========================================
# 3. Data & Training Engine
# ==========================================

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'w') as f:
            f.write("Experiment Log Started\n")
            
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
    # Check if GPU is available to determine pin_memory
    use_gpu = torch.cuda.is_available()
    
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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
    
    # 1. Stratified Split (Train Pool vs Test)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_pool_idx, test_idx = next(sss_test.split(np.zeros(len(targets)), targets))
    
    # 2. Data Scarcity (Split Ratio on Train Pool)
    train_pool_targets = [targets[i] for i in train_pool_idx]
    if split_ratio < 1.0:
        sss_scarce = StratifiedShuffleSplit(n_splits=1, train_size=split_ratio, random_state=seed)
        used_idx, _ = next(sss_scarce.split(np.zeros(len(train_pool_idx)), train_pool_targets))
        final_train_idx = [train_pool_idx[i] for i in used_idx]
    else:
        final_train_idx = train_pool_idx

    # 3. Create Subsets
    train_ds = Subset(full_ds, final_train_idx)
    test_ds = Subset(full_ds, test_idx)
    
    train_ds.dataset.transform = train_tfm
    test_ds.dataset.transform = val_tfm

    # 4. Balanced Sampling
    train_targets_subset = [targets[i] for i in final_train_idx]
    class_counts = np.bincount(train_targets_subset)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in train_targets_subset]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=use_gpu)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_gpu)
    
    return train_loader, test_loader, len(full_ds.classes)

def run_experiment(model_name, model_builder, data_dir, output_file, device):
    logger = Logger(output_file)
    
    # Enable AMP only if CUDA is available
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)
    
    attn_mechanisms = ['Baseline', 'StatAware', 'CBAM', 'ECA', 'SimAM', 'CoordAtt', 'scSE', 'Triplet']
    data_splits = [0.1, 0.2, 0.5, 0.8]
    
    results = {}

    for attn in attn_mechanisms:
        logger.log(f"\n{'='*40}\nRunning Architecture: {model_name} + {attn}\n{'='*40}")
        
        for split in data_splits:
            # Robustness: 5 runs for 10% data
            seeds = [42, 10, 20, 30, 40] if split == 0.1 else [42]
            
            run_metrics = {'acc': [], 'rec': [], 'f1': []}
            
            for seed in seeds:
                logger.log(f"  > Split: {int(split*100)}% | Seed: {seed} ...")
                
                try:
                    train_loader, test_loader, num_classes = get_dataloaders(data_dir, split, seed)
                    model = model_builder(num_classes, attn).to(device)
                    
                    criterion = nn.CrossEntropyLoss()
                    logger.log(f"    Loss Function: {criterion}")
                    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                    
                    model.train()
                    for epoch in range(25):
                        epoch_loss = 0.0
                        epoch_correct = 0
                        epoch_total = 0
                        
                        for imgs, lbls in train_loader:
                            imgs, lbls = imgs.to(device), lbls.to(device)
                            optimizer.zero_grad()
                            
                            # Only use autocast if on GPU
                            with autocast(enabled=use_amp):
                                outputs = model(imgs)
                                loss = criterion(outputs, lbls)
                            
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            
                            # Stats
                            epoch_loss += loss.item() * imgs.size(0)
                            _, predicted = torch.max(outputs.data, 1)
                            epoch_total += lbls.size(0)
                            epoch_correct += (predicted == lbls).sum().item()
                        
                        avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0
                        avg_acc = 100 * epoch_correct / epoch_total if epoch_total > 0 else 0
                        logger.log(f"    Epoch {epoch+1}/25 | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.2f}%")
                    
                    model.eval()
                    all_preds, all_lbls = [], []
                    with torch.no_grad():
                        for imgs, lbls in test_loader:
                            imgs, lbls = imgs.to(device), lbls.to(device)
                            outputs = model(imgs)
                            _, preds = torch.max(outputs, 1)
                            all_preds.extend(preds.cpu().numpy())
                            all_lbls.extend(lbls.cpu().numpy())
                    
                    acc = accuracy_score(all_lbls, all_preds)
                    r = recall_score(all_lbls, all_preds, average='macro', zero_division=0)
                    f1 = f1_score(all_lbls, all_preds, average='macro', zero_division=0)
                    
                    run_metrics['acc'].append(acc)
                    run_metrics['rec'].append(r)
                    run_metrics['f1'].append(f1)
                except Exception as e:
                    logger.log(f"Error in run: {e}")
                    run_metrics['acc'].append(0) 
            
            mean_acc = np.mean(run_metrics['acc'])
            std_acc = np.std(run_metrics['acc'])
            
            log_str = f"RESULT | {attn} | Split {int(split*100)}% | Acc: {mean_acc:.4f} (+/- {std_acc:.4f}) | Recall: {np.mean(run_metrics['rec']):.4f}"
            logger.log(log_str)
            
            results[(attn, split)] = {'acc': mean_acc, 'std': std_acc, 'rec': np.mean(run_metrics['rec']), 'f1': np.mean(run_metrics['f1'])}

    # Summary
    logger.log("\nFINAL WINNERS")
    for split in data_splits:
        best_attn = max(attn_mechanisms, key=lambda x: results.get((x, split), {'acc': -1})['acc'])
        stats = results[(best_attn, split)]
        logger.log(f"Split {int(split*100)}%: BEST -> {best_attn} (Acc: {stats['acc']:.4f}, Rec: {stats['rec']:.4f})")