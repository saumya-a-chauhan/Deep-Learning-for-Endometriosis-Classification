
# **Endometriosis Classification using Deep Learning**

### **Automated Diagnosis of Endometriosis from Laparoscopic Images using Attention-Enhanced CNN Architectures**

---

## **📖 Table of Contents**

* [Project Overview](#-project-overview)
* [Motivation](#-motivation)
* [Methodology](#-methodology)
* [Dataset](#dataset)
* [Architectures](#architectures)
* [Attention Mechanisms](#attention-mechanisms)
* [Novelty: StatAware Block](#-novelty-stataware-block)
* [Experimental Setup](#-experimental-setup)
* [Key Results](#-key-results)
* [Visualizations](#-visualizations)
* [Installation & Usage](#-installation--usage)
* [Folder Structure](#-folder-structure)
* [Contributors](#-contributors)

---

## **📌 Project Overview**

Endometriosis is a chronic gynecological condition affecting **~10% of reproductive-age women worldwide**. Diagnosis is often invasive and delayed by **7–10 years** due to reliance on laparoscopy and high lesion variability.

This project builds a **Computer-Aided Diagnosis (CAD)** system to classify laparoscopic images as **Endometriosis** or **Non-Endometriosis**.
We compare modern deep learning architectures and integrate attention blocks to identify the most effective model under **data-scarce medical settings**.

---

## **🎯 Motivation**

### **Clinical Need**

Assist surgeons during laparoscopy and reduce diagnosis delays.

### **Technical Challenge**

Medical datasets are small, imbalanced, and contain high texture variability. Standard CNNs often struggle without explicit feature enhancement.

### **Solution**

Integrate **Attention Mechanisms** into RESNET-50 backbones for improved lesion detection, interpretability, and robustness.

---

## **🧠 Methodology**

### **Dataset**

* **Source:** Private/Public laparoscopic image dataset
* **Size:** ~1,000 images
* **Classes:** *Endometriosis* vs. *Normal*
* **Preprocessing:**

  * Resize to **224×224**
  * Normalize (ImageNet)
  * Augmentation: flips, color jitter, rotations

---

## **Architectures**

Three state-of-the-art backbones were benchmarked:

* **ResNet50** — industry-standard residual network
* **EfficientNetV2-S** — optimized for speed & efficiency
* **ConvNeXt-Tiny** — CNN architecture inspired by ViTs

---

## **Attention Mechanisms**

A modular library integrates multiple attention blocks:

* **CBAM** – Channel + spatial attention
* **ECA-Net** – Lightweight channel attention
* **SimAM** – Parameter-free attention
* **Coordinate Attention** – Direction-aware spatial attention
* **scSE / Triplet Attention** – Additional variants tested

---


### **Innovation**

➡️ Feed both into the attention module
➡️ Improve texture-sensitive classification

**Result:** Outperformed ResNet baseline in **3/4 experiments**.

---

## **🧪 Experimental Setup**

* **Data Scarcity Study:** 10%, 20%, 50%, 80% splits
* **Robustness Tests:** 5 random seeds for the 10% split
* **Training Setup:**

  * Optimizer: **AdamW (1e-4)**
  * Loss: **CrossEntropy + Label Smoothing (0.1)**
  * Epochs: **25**
  * AMP (Mixed Precision)

---

## **📊 Key Results**

### **1. 🏆 Best Overall Model: ResNet50 + CBAM**

| Metric       | Score      |
| ------------ | ---------- |
| **Accuracy** | **97.07%** |
| **Recall**   | **97.07%** |
| **F1-Score** | **0.9705** |

---

### **2. 🌟 Data-Scarcity Winner: ConvNeXt (10% Data)**

| Model                   | Accuracy   |
| ----------------------- | ---------- |
| **ConvNeXt (Baseline)** | **90.05%** |
| EfficientNetV2          | 81.85%     |
| ResNet50                | 78.15%     |

---

---

## **📈 Visualizations**

### **Confusion Matrix**

ResNet50+CBAM shows minimal false negatives.

### **Grad-CAM Heatmaps**

The model accurately focuses on lesion regions instead of backgrounds.
(Generated in `results/` after running code.)

---

## **💻 Installation & Usage**

### **Prerequisites**

```
Python 3.8+
PyTorch
Torchvision
Numpy
Scikit-learn
Matplotlib
OpenCV
```

### **Step 1: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 2: Train the Final Model**

```bash
python final_model/final_resnet_cbam.py
```

### **Step 3: Reproduce Experiments**

Example:

```bash
python experiments/run_convnext.py
```

---

## **📂 Folder Structure**

```
Endometriosis-ResNet-CBAM/
│
├── data/                       
│   └── dataset_final/
│
├── experiments/                
│   ├── attention_lib.py        
│   ├── run_resnet.py           
│   ├── run_efficientnet.py     
│   └── run_convnext.py         
│
├── final_model/                
│   ├── final_lib.py            
│   └── final_resnet_cbam.py    
│
├── results/                    
│   ├── confusion_matrix.png
│   ├── gradcam_best.png
│   └── final_results_cbam.txt
│
├── requirements.txt
└── README.md
```

---

## **👥 Contributors**

* **Saumya (U20230016)**
* **Bhavya Pathak (U20230136)**
* **Harmannat Kaur (U20230066)**



