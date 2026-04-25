# Label-Matching Text-Visual Collaborative Fusion (LTCF) for FMCG Fine-Grained Classification
This is the official PyTorch implementation of LTCF (Label-Matching Text-Visual Collaborative Fusion), a multimodal fusion model designed for high-accuracy fine-grained classification of fast-moving consumer goods (FMCG). The framework adaptively fuses visual features from product images and text matching scores derived from packaging text, enabling stronger discrimination between highly similar products.

## Project Structure
 ├── LTCF_code/
 │ ├── model.py # Core LTCF fusion model (SimilarityFusionNet)
 │ ├── datasets.py # Dataset loader for images and multi-rule similarity scores
 │ └── train.py # Training, validation, logging, and model saving
 ├──Dataset_code/
│ ├── 1-OCR_json.py
│ ├── 2-OCR_txt.py
│ ├── 3-Area_position.py
│ ├── 4-rename.py
│ ├── 5-final_score.py
│ └── README.md
├──base_code/
│ ├── googlenet.py
│ └── train.py
├── requirements.txt
└── README.md

## Overview
The LTCF model utilizes a pre-trained backbone model as initial weights to enhance visual feature extraction capabilities. The pre-trained model is first trained independently on a target FMCG dataset, and its well-trained parameters are subsequently loaded into the LTCF framework, thereby providing robust and stable visual representations. Subsequently, the LTCF model performs adaptive fusion of visual features and multi-dimensional text matching scores at the decision layer to achieve optimal classification performance.

## Environment Requirements
python >= 3.8
torch >= 1.8.0
torchvision >= 0.9.0
Pillow >= 8.0.0
numpy >= 1.19.0
tqdm >= 4.60.0
opencv-python
paddleocr

Install dependencies:
```bash
pip install -r requirements.txt

Usage
Step 1: Generate Similarity Scores
Run all scripts in the data_processing folder in order to perform batch OCR, text extraction, text region importance calculation, and final label matching score generation:
cd Dataset_code
python 1-OCR_json.py
python 2-OCR_txt.py
python 3-Area_position.py
python 4-rename.py
python 5-final_score.py

Step 2: Prepare the pre-trained model using base_code
Run `train.py` to start training and obtain `base.pth`:
cd base_code
python train.py

Step 3: Train the LTCF Fusion Model
Use the `base.pth` file obtained in Step 2 as the pre-trained model for LTCF:
cd LTCF_code
python train.py

Step 4: Configure Paths
Before starting training, please modify the following paths in the `train.py` file:
`data_dir`: The root path for the training and testing image datasets
`similarity_dir`: The directory where generated similarity score files are stored
`class_indices.json`: The class index mapping file
`pretrained_backbones`: The path to the pretrained baseline models

Model Overview
The core model SimilarityFusionNet adopts a well-trained visual backbone initialized by a pre-trained baseline models. It adaptively fuses image features and four types of text matching scores using learnable weights and a dynamic fusion factor. Key features include:
Strong visual initialization from two independently trained base models
Learnable weights for four text matching rules
Adaptive alpha parameter to balance visual and textual modality contributions
Built-in numerical stability and NaN handling
Stable training with batch normalization and gradient clipping

Dataset Structure
data_root/
├── train/
│   ├── class_001/
│   │   ├── image_001.jpg
│   │   ├── image_001.json
│   │   └── image_001.txt 
│   └── ...
├── test/
│   └── ...
├── similarity_scores/
│   ├── train/
│   │   ├──  class_001
│   │   │   ├── image_001.txt
│   │   └── ...
│   └── test/
│   │   └── ...
└── class_indices.json

Training Features
Initialization from a pre-trained backbone models
Separate learning rates for backbone network and fusion parameters
Gradient clipping to prevent gradient explosion
ReduceLROnPlateau learning rate scheduler
Early stopping based on validation accuracy
Automatic logging and best model saving
Real-time monitoring of modality fusion weights

Notes
The  provided .pth file are pre-trained baseline models used to initialize the LTCF model.
Each image must have a corresponding .txt similarity score file.
GPU is highly recommended for faster training and OCR processing.
Adjust batch size according to your GPU memory capacity.
All file paths must be correctly configured before running scripts.

License
This project is for academic research purposes only.