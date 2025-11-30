Endometriosis Classification using Deep Learning

Automated Diagnosis of Endometriosis from Laparoscopic Images using Attention-Enhanced CNN Architectures

📖 Table of Contents

Project Overview

Motivation

Methodology

Dataset

Architectures

Attention Mechanisms

Novelty: StatAware Block

Experimental Setup

Key Results

Visualizations

Installation & Usage

Folder Structure

Contributors

📌 Project Overview

Endometriosis is a chronic gynecological condition affecting approximately 10% of reproductive-age women globally. Diagnosis is notoriously difficult, often delayed by 7-10 years due to the reliance on invasive laparoscopic surgery and the high variability in lesion appearance.

This project develops a Computer-Aided Diagnosis (CAD) system to automatically classify laparoscopic images as "Endometriosis" or "Non-Endometriosis". We conducted a rigorous comparative study of modern deep learning architectures and attention mechanisms to identify the most robust solution for medical image classification, especially under data-scarce conditions.

🎯 Motivation

Clinical Need: To reduce diagnostic delay and assist surgeons in identifying subtle lesions during laparoscopy.

Technical Challenge: Medical datasets are often small, unbalanced, and contain high textural variance. Standard CNNs (like ResNet) may struggle to distinguish pathological tissue from healthy tissue without specific feature enhancement.

Solution: We propose integrating Attention Mechanisms (which mimic human visual focus) into backbone networks to improve feature extraction and model interpretability.

🧠 Methodology

Dataset

Source: Private/Public Laparoscopic Image Dataset (Specify source if public).

Size: Total of ~1,000 images.

Classes: Binary Classification (Endometriosis vs. Normal).

Preprocessing: Resized to 224x224, Normalized to ImageNet standards, Augmentation (Flip, Rotation, Color Jitter).

Architectures

We benchmarked three state-of-the-art backbones:

ResNet50: A robust, deep residual network (The industry standard).

EfficientNetV2-S: Optimized for training speed and parameter efficiency.

ConvNeXt-Tiny: A modern architecture inspired by Vision Transformers.

Attention Mechanisms

We implemented a modular library to inject the following attention blocks into the backbones:

CBAM (Convolutional Block Attention Module): Sequentially infers attention maps along two separate dimensions (channel and spatial).

ECA-Net (Efficient Channel Attention): Uses 1D convolution for lightweight channel attention.

SimAM: A parameter-free, 3D attention module based on neuroscience energy functions.

Coordinate Attention: Factorizes spatial attention into horizontal and vertical directions for precise localization.

scSE & Triplet Attention: Other variants tested for completeness.

Novelty: StatAware Block

We designed and implemented a custom "StatAware" attention block.

Hypothesis: Medical lesions often manifest as texture irregularities. Standard Global Average Pooling (GAP) only captures the mean intensity, ignoring texture variance.

Innovation: Our block explicitly calculates the Global Standard Deviation (Variance) alongside the Mean. It feeds both statistics into the attention network to better capture pathological textures.

Outcome: This novelty successfully outperformed the ResNet Baseline in 3 out of 4 experimental scenarios.

🧪 Experimental Setup

We designed a rigorous study to simulate real-world medical data constraints:

Data Scarcity Analysis: Models were trained on 10%, 20%, 50%, and 80% of the dataset to measure performance scaling.

Robustness Testing: For the critical 10% split (simulating rare disease data), we ran 5 independent trials with different random seeds and reported the Mean Accuracy ± Standard Deviation.

Training:

Optimizer: AdamW (Learning Rate: 1e-4)

Loss Function: CrossEntropyLoss with Label Smoothing (0.1)

Epochs: 25 per run

Hardware: Mixed Precision Training (AMP) for GPU optimization.

📊 Key Results

1. The Grand Champion: ResNet50 + CBAM

When trained on the full dataset (80%), this architecture achieved the highest performance metrics of the entire study.

Metric

Score

Accuracy

97.07%

Recall

97.07%

F1-Score

0.9705

2. The Low-Data Hero: ConvNeXt

In scenarios with extreme data scarcity (only 10% data used), ConvNeXt proved to be far superior to older architectures.

Model (10% Data)

Accuracy

ConvNeXt (Baseline)

90.05%

EfficientNetV2

81.85%

ResNet50 (Baseline)

78.15%

3. Novelty Validation

Our custom StatAware block demonstrated clear improvements over the standard ResNet baseline.

Data Split

ResNet Baseline Acc

ResNet + StatAware Acc

Improvement

20%

84.39%

88.78%

+4.39%

50%

95.12%

91.22%

-3.90%

80%

95.12%

95.61%

+0.49%

📈 Visualizations

Confusion Matrix

The best model (ResNet50+CBAM) showed exceptional ability to distinguish classes with minimal false negatives.

Grad-CAM Analysis

We utilized Gradient-weighted Class Activation Mapping (Grad-CAM) to verify interpretability. The heatmaps confirm that the model focuses on the actual lesions (red regions) rather than background artifacts.

(Note: These images will be generated in your results/ folder after running the code).

💻 Installation & Usage

Prerequisites

Python 3.8+

PyTorch, Torchvision

Numpy, Scikit-learn, Matplotlib, OpenCV

Step 1: Install Dependencies

pip install -r requirements.txt


Step 2: Run the Final Model

To train the best-performing architecture (ResNet50 + CBAM) and generate all plots:

python final_model/final_resnet_cbam.py


Step 3: Reproduce Experiments

To re-run the comparative study for other architectures (e.g., ConvNeXt):

python experiments/run_convnext.py


📂 Folder Structure

Endometriosis-ResNet-CBAM/
│
├── data/                       # Data (Not uploaded to git)
│   └── dataset_final/
│
├── experiments/                # Research Code
│   ├── attention_lib.py        # Shared Library (Novelty is here!)
│   ├── run_resnet.py           # ResNet Experiment Runner
│   ├── run_efficientnet.py     # EfficientNet Experiment Runner
│   └── run_convnext.py         # ConvNeXt Experiment Runner
│
├── final_model/                # Deployment Code
│   ├── final_lib.py            # Cleaned Library
│   └── final_resnet_cbam.py    # Main Training Script
│
├── results/                    # Logs & Graphs
│   ├── confusion_matrix.png
│   ├── gradcam_best.png
│   └── final_results_cbam.txt
│
├── requirements.txt
└── README.md


👥 Contributors

Saumya (U20230016) , Bhavya Pathak (U20230136) , Harmannat Kaur (U20230066)
