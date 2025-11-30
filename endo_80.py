# Usage:
# 1. Make sure you've already run 'train.py' and have a 'model_results.txt' file.
# 2. Run this script from your terminal:
#    python train_80.py
#
# 3. This will APPEND the 80% results to your existing 'model_results.txt' file.

import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from contextlib import redirect_stdout

# Sklearn imports
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import train_test_split

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# -------- Model Definitions (ResNet50 + RPL-CBAM) --------
# (These are identical to the previous script)

class RPL_CBAM(nn.Module):
    """Residual-Path Lightweight CBAM"""
    def __init__(self, channels, reduction=8, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        mid = max(1, channels // reduction)
        
        # Channel attention (lightweight)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention (light)
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca_w = self.ca(x)
        x_ca = x * ca_w
        
        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        sa_map = self.sa(torch.cat([avg, mx], dim=1))
        
        # Fusion with residual path
        fused = x_ca * sa_map
        return x + self.alpha * fused

class ResNet50_RPLCBAM(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        self.cbam1 = RPL_CBAM(256, reduction=16)
        self.cbam2 = RPL_CBAM(512, reduction=16)
        self.cbam3 = RPL_CBAM(1024, reduction=16)
        self.cbam4 = RPL_CBAM(2048, reduction=16)
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        # Replace the final fully connected layer
        self.backbone.fc = nn.Identity() 
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Forward through ResNet backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x); x = self.cbam1(x)
        x = self.backbone.layer2(x); x = self.cbam2(x)
        x = self.backbone.layer3(x); x = self.cbam3(x)
        x = self.backbone.layer4(x); x = self.cbam4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# -------- Helper Functions --------
# (These are identical to the previous script)

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, file_logger):
    """Training and validation loop."""
    best_auc = -1.0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        
        # --- Training Phase ---
        model.train()
        train_losses, train_labels, train_probs = [], [], []
        
        # Using tqdm for progress bar, writing to stderr to avoid file log clutter
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} (Train)", file=sys.stderr, leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            train_probs.extend(probs.tolist())
            train_labels.extend(labels.detach().cpu().numpy().tolist())
            
            pbar.set_postfix({"loss": f"{np.mean(train_losses):.4f}"})
        
        train_acc = accuracy_score(train_labels, np.array(train_probs) >= 0.5)
        try:
            train_auc = roc_auc_score(train_labels, train_probs)
        except ValueError:
            train_auc = 0.0 # Handle case with only one class in batch

        # --- Validation Phase ---
        model.eval()
        val_losses, val_labels, val_probs = [], [], []
        
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} (Val)", file=sys.stderr, leave=False)
            for imgs, labels in vbar:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                
                val_losses.append(loss.item())
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs.tolist())
                val_labels.extend(labels.cpu().numpy().tolist())
        
        val_acc = accuracy_score(val_labels, np.array(val_probs) >= 0.5)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = 0.0

        scheduler.step(val_auc)
        dt = time.time() - t0

        # Log to file
        print(f"\nEpoch {epoch}/{num_epochs} - {dt:.1f}s", file=file_logger)
        print(f"  Train loss: {np.mean(train_losses):.4f} | Train Acc: {train_acc*100:.2f}% | Train AUC: {train_auc:.4f}", file=file_logger)
        print(f"  Val   loss: {np.mean(val_losses):.4f} | Val   Acc: {val_acc*100:.2f}% | Val   AUC: {val_auc:.4f}", file=file_logger)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            print(f"  -> New best model found (val_auc={val_auc:.4f})", file=file_logger)
            
    print("\nTraining finished.", file=file_logger)
    return best_model_state

def evaluate_model(model, loader, criterion, device, class_names):
    """Evaluate model on a dataset (val or test) and return metrics."""
    model.eval()
    all_losses, all_labels, all_probs, all_preds = [], [], [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating", file=sys.stderr, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            all_losses.append(loss.item())
            
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
        
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary') # Assuming positive class is '1'
    
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        "loss": np.mean(all_losses),
        "accuracy": acc,
        "auc": auc,
        "f1_score": f1,
        "report": report,
        "cm": cm,
    }
    return metrics

# -------- Main Execution --------

def main():
    # --- Configuration ---
    DATA_DIR = Path("dataset_final") 
    OUTPUT_FILE = "model_results.txt"
    
    # <<<!!!>>> CHANGE 1: Only run for 80%
    TRAIN_PROPORTIONS = [0.8] 
    
    # Hyperparameters
    BATCH_SIZE = 8
    IMAGE_SIZE = 224
    NUM_EPOCHS = 18
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    SEED = 42
    NUM_WORKERS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    print(f"Appending 80% results to: {OUTPUT_FILE}")

    # --- Image Transforms ---
    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # --- Load Full Datasets ---
    try:
        dataset_full_train = datasets.ImageFolder(str(DATA_DIR), transform=train_tfms)
        dataset_full_eval = datasets.ImageFolder(str(DATA_DIR), transform=val_tfms)
    except FileNotFoundError:
        print(f"ERROR: Data directory not found at '{DATA_DIR}'")
        print("Please make sure your 'dataset_final' folder is in the same directory as this script.")
        return

    class_names = dataset_full_train.classes
    targets = [s[1] for s in dataset_full_train.samples]
    indices = list(range(len(targets)))

    # --- Open output file and run experiments ---
    
    # <<<!!!>>> CHANGE 2: Open file in 'a' (append) mode.
    with open(OUTPUT_FILE, 'a') as f:
        # Redirect all print statements to this file
        with redirect_stdout(f):
            
            # Add a separator to the file
            print("\n\n" + "="*50)
            print("APPENDING 80% EXPERIMENT RUN")
            print("="*50 + "\n")

            for prop in TRAIN_PROPORTIONS:
                header = f"--- RUNNING EXPERIMENT: {prop*100:.0f}% Training Data ---"
                print("\n" + "="*len(header))
                print(header)
                print("="*len(header) + "\n")
                
                # --- 1. Create Train/Test Split ---
                train_val_idx, test_idx = train_test_split(
                    indices, 
                    train_size=prop,
                    stratify=targets, 
                    random_state=SEED, 
                    shuffle=True
                )
                
                # --- 2. Create Train/Validation Split ---
                train_val_targets = [targets[i] for i in train_val_idx]
                if len(set(train_val_targets)) < 2:
                     print(f"Warning: Only one class present in {prop*100}% split. Skipping stratified split for train/val.")
                     train_idx, val_idx = train_test_split(
                        train_val_idx,
                        test_size=0.20, 
                        random_state=SEED,
                        shuffle=True
                     )
                else:
                    train_idx, val_idx = train_test_split(
                        train_val_idx,
                        test_size=0.20, 
                        stratify=train_val_targets,
                        random_state=SEED,
                        shuffle=True
                    )

                # --- 3. Create Subsets and DataLoaders ---
                train_ds = Subset(dataset_full_train, train_idx) 
                val_ds = Subset(dataset_full_eval, val_idx)       
                test_ds = Subset(dataset_full_eval, test_idx)      
                
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
                test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
                
                print("Dataset Split:")
                print(f"  Total images:   {len(targets)}")
                print(f"  Train pool:     {len(train_val_idx)} ({prop*100:.0f}%)")
                print(f"  Test pool:      {len(test_idx)} ({(1-prop)*100:.0f}%)")
                print(f"    -> Final Train: {len(train_ds)}")
                print(f"    -> Final Val:   {len(val_ds)}")
                
                if len(train_ds) == 0:
                    print("\nERROR: Training set has 0 samples. Skipping this proportion.")
                    continue

                # --- 4. Initialize Model and Train ---
                print("\nInitializing model...")
                model = ResNet50_RPLCBAM(num_classes=len(class_names), pretrained=True).to(DEVICE)
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
                
                print("Starting training...")
                best_model_state = train_model(
                    model, train_loader, val_loader, criterion, optimizer, 
                    scheduler, NUM_EPOCHS, DEVICE, file_logger=f
                )
                
                # --- 5. Load Best Model and Evaluate ---
                if best_model_state:
                    print("\nLoading best model for final evaluation...")
                    model.load_state_dict(best_model_state)
                else:
                    print("\nWarning: No best model saved (training may have failed). Evaluating last epoch model.")

                # --- 6. Evaluate on VALIDATION Set ---
                print("\n" + "-"*30)
                print("FINAL VALIDATION SET METRICS")
                print("-"*30)
                val_metrics = evaluate_model(model, val_loader, criterion, DEVICE, class_names)
                
                print(f"Validation Loss:     {val_metrics['loss']:.4f}")
                print(f"Validation Accuracy: {val_metrics['accuracy'] * 100:.2f}%")
                print(f"Validation F1-Score: {val_metrics['f1_score']:.4f}")
                print(f"Validation AUC:      {val_metrics['auc']:.4f}")
                print("\nValidation Classification Report:")
                print(val_metrics['report'])
                print("\nValidation Confusion Matrix:")
                print(val_metrics['cm'])
                print(f"(Labels: {', '.join(class_names)})")

                # --- 7. Evaluate on TEST Set ---
                print("\n" + "-"*30)
                print("FINAL TEST SET METRICS")
                print("-"*30)
                test_metrics = evaluate_model(model, test_loader, criterion, DEVICE, class_names)
                
                print(f"Test Loss:     {test_metrics['loss']:.4f}")
                print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
                print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
                print(f"Test AUC:      {test_metrics['auc']:.4f}")
                print("\nTest Classification Report:")
                print(test_metrics['report'])
                print("\nTest Confusion Matrix:")
                print(test_metrics['cm'])
                print(f"(Labels: {', '.join(class_names)})")
                print("\n" + "="*len(header))

    print(f"\n80% experiment complete. Results appended to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()

