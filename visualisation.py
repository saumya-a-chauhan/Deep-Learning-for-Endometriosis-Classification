import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ==========================================
# 1. DATA ENTRY (Extracted from your logs)
# ==========================================

# Data Splits
splits = ['10%', '20%', '50%', '80%']

# Test Accuracy Results (%)
# Extracted from resnet_baseline.txt
acc_baseline = [86.09, 91.93, 93.35, 95.61] 

# Extracted from model_results.txt
acc_cbam =     [86.74, 89.73, 96.48, 96.59]

# Best Confusion Matrix (CBAM 80%)
# [[TP, FP], [FN, TN]]
# Note: Your logs showed [[101, 6], [1, 97]]
# Row 0: Actual Endo (101 correct, 6 wrong/FN) -> Wait, let's re-read the log matrix carefully
# Log says:
# [[101   6]   <-- Row 0: Actual Endometriosis. 101 predicted Endo, 6 predicted Non
#  [  1  97]]  <-- Row 1: Actual Non-Endo. 1 predicted Endo, 97 predicted Non
#
# So: TP=101, FN=6, FP=1, TN=97
cm_data = np.array([[101, 6], [1, 97]])
labels = ['Endometriosis', 'Non-Endometriosis']

# ==========================================
# 2. GENERATE COMPARISON BAR CHART
# ==========================================

def plot_comparison():
    # Set academic style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create DataFrame for easier plotting with Seaborn
    df = pd.DataFrame({
        'Data Volume': splits * 2,
        'Accuracy (%)': acc_baseline + acc_cbam,
        'Model': ['ResNet50 (Baseline)'] * 4 + ['ResNet50 + RPL-CBAM (Ours)'] * 4
    })

    plt.figure(figsize=(10, 6), dpi=300) # High DPI for printing
    
    # Create Bar Plot
    ax = sns.barplot(
        data=df, 
        x='Data Volume', 
        y='Accuracy (%)', 
        hue='Model',
        palette=['#95a5a6', '#e74c3c'] # Grey for baseline, Red for yours (highlights your work)
    )

    # Add titles and labels
    plt.title('Model Accuracy vs. Training Data Volume', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Percentage of Training Data Used', fontsize=12)
    plt.ylabel('Test Set Accuracy (%)', fontsize=12)
    plt.ylim(80, 100) # Zoom in to show the differences clearly
    plt.legend(loc='lower right', frameon=True)

    # Add exact numbers on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', padding=3, fontsize=10, fontweight='bold')

    # Save
    save_path = "poster_comparison_chart.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Generated Comparison Chart: {save_path}")
    plt.close()

# ==========================================
# 3. GENERATE CONFUSION MATRIX HEATMAP
# ==========================================

def plot_confusion_matrix():
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Custom annotations with counts and percentages
    group_names = ['True Pos','False Neg','False Pos','True Neg']
    group_counts = ["{0:0.0f}".format(value) for value in cm_data.flatten()]
    
    # Calculate percentages
    group_percentages = ["{0:.2%}".format(value) for value in cm_data.flatten()/np.sum(cm_data)]
    
    labels_annot = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels_annot = np.asarray(labels_annot).reshape(2,2)

    # Create Heatmap
    sns.heatmap(
        cm_data, 
        annot=labels_annot, 
        fmt='', 
        cmap='Blues', 
        xticklabels=labels, 
        yticklabels=labels,
        cbar=False,
        square=True,
        annot_kws={"size": 14, "weight": "bold"},
        linewidths=2,
        linecolor='black'
    )

    plt.title('Confusion Matrix (Best Model)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')

    # Save
    save_path = "poster_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Generated Confusion Matrix: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_comparison()
    plot_confusion_matrix()