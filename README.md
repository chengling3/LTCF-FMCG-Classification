markdown
# Label-Matching Text-Visual Collaborative Fusion (LTCF) for FMCG Fine-Grained Classification

This is the official PyTorch implementation of LTCF (Label-Matching Text-Visual Collaborative Fusion), a multimodal fusion model designed for high-accuracy fine-grained classification of fast-moving consumer goods (FMCG). The framework adaptively fuses visual features from product images and text matching scores derived from packaging text, enabling stronger discrimination between highly similar products.

---

## Project Structure
LTCF-FMCG/├── LTCF_code/ # LTCF main model code│ ├── model.py # Core LTCF fusion model (SimilarityFusionNet)│ ├── datasets.py # Dataset loader for images and similarity scores│ └── train.py # LTCF training and validation├── Dataset_code/ # Dataset processing pipeline│ ├── 1-OCR_json.py│ ├── 2-OCR_txt.py│ ├── 3-Area_position.py│ ├── 4-rename.py│ ├── 5-final_score.py│ └── README.md├── base_code/ # Baseline model (GoogLeNet)│ ├── googlenet.py│ └── train.py├── requirements.txt└── README.md # This file
plaintext

---

## Overview

The LTCF model utilizes a pre-trained backbone model as initial weights to enhance visual feature extraction. The pre-trained model is first trained independently on the target FMCG dataset, and its well-trained parameters are loaded into the LTCF framework to provide robust visual representations. The LTCF model then adaptively fuses visual features and multi-dimensional text matching scores at the decision layer to achieve optimal classification performance.

---

## Environment Requirements
python >= 3.8torch >= 1.8.0torchvision >= 0.9.0Pillow >= 8.0.0numpy >= 1.19.0tqdm >= 4.60.0opencv-pythonpaddleocr
plaintext

Install dependencies:
```bash
pip install -r requirements.txt
Usage
Step 1: Generate Similarity Scores
Run all scripts in Dataset_code in order to perform OCR, text extraction, and score generation:
bash
运行
cd Dataset_code
python 1-OCR_json.py
python 2-OCR_txt.py
python 3-Area_position.py
python 4-rename.py
python 5-final_score.py
Step 2: Train the Baseline Model
Run the baseline training to obtain the pre-trained backbone base.pth:
bash
运行
cd base_code
python train.py
Step 3: Train the LTCF Fusion Model
Use the pre-trained base.pth to initialize the LTCF model:
bash
运行
cd LTCF_code
python train.py
Step 4: Configure Paths
Before training, modify the following paths in train.py:
data_dir: root path of training/test images
similarity_dir: directory of generated similarity scores
class_indices.json: category index mapping file
pretrained_backbones: path to the pre-trained baseline model
Model Overview
The core model SimilarityFusionNet adopts a well-trained visual backbone initialized by pre-trained baseline models. It adaptively fuses image features and four types of text matching scores using learnable weights and a dynamic fusion factor.
Key features include:
Strong visual initialization from pre-trained base models
Learnable weights for four text matching rules
Adaptive alpha parameter to balance visual and textual contributions
Built-in numerical stability and NaN handling
Stable training with batch normalization and gradient clipping
Dataset Structure
plaintext
data_root/
├── train/
│   ├── class_001/
│   │   ├── image_001.jpg
│   │   ├── image_001.json
│   │   └── image_001.txt
├── test/
├── similarity_scores/
│   ├── train/
│   │   ├── class_001/
│   │   │   └── image_001.txt
│   └── test/
└── class_indices.json
Training Features
Initialization from pre-trained backbone models
Separate learning rates for backbone and fusion parameters
Gradient clipping to prevent gradient explosion
ReduceLROnPlateau learning rate scheduler
Early stopping based on validation accuracy
Automatic logging and best model saving
Real-time weight monitoring for modality fusion
Notes
The provided .pth files are pre-trained baseline models for initializing the LTCF model.
Each image must have a corresponding .txt similarity score file.
GPU is highly recommended for faster training and OCR processing.
Adjust batch size according to your GPU memory.
All file paths must be correctly configured before running scripts.
License
This project is for academic research purposes only.
