# Label-Matching Text-Visual Collaborative Fusion (LTCF) for FMCG Fine-Grained Classification

## Important Notice
This code corresponds to the manuscript submitted to **The Visual Computer**.
If you use this code, dataset, or any part of our work in your research, **please cite our paper**.

This work proposes LTCF (Label-Matching Text-Visual Collaborative Fusion), a multimodal fusion model designed for high-accuracy fine-grained classification of fast-moving consumer goods (FMCG). The framework combines visual features extracted from product images and label matching scores derived from packaging text, enabling stronger discriminative power between highly similar products.

## Project Structure
 ├── LTCF_code/                # Core implementation of LTCF
 │   ├── model.py              # Core LTCF fusion network (SimilarityFusionNet)
 │   ├── datasets.py           # Dataset loader for images and multi-rule similarity scores
 │   └── train.py              # Training, validation, logging, and model checkpoint saving
 ├── Dataset_code/             # Dataset preprocessing pipeline
 │   ├── 1-OCR_json.py
 │   ├── 2-OCR_txt.py
 │   ├── 3-Area_position.py
 │   ├── 4-rename.py
 │   └── 5-final_score.py
 ├── requirements.txt          # Environment dependencies
 └── README.md                 # Project documentation

## Overview
LTCF adopts a mainstream visual backbone network for high-quality visual feature extraction, and generates multi-dimensional label matching scores based on text from product packaging. The model performs adaptive fusion of visual features and multi-dimensional text matching scores at the decision layer, which enhances the discriminative ability between highly similar products, and ultimately achieves state-of-the-art performance for FMCG fine-grained classification.

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

Step 2: Configure Paths
Before starting training, please modify the following path parameters in the train.py file:
data_dir: Root path of the training and test image datasets
similarity_dir: Directory for storing the generated similarity score files
class_indices.json: Class index mapping file

Step 3: Train the LTCF Fusion Model
After completing the path configuration, start the training of the multimodal fusion model:
cd LTCF_code
python train.py

Model Overview
The core network SimilarityFusionNet is built on a mainstream visual backbone, and adaptively fuses image features and four types of text matching scores via learnable weights and a dynamic fusion factor. Key features include:
High-quality visual feature extraction guaranteed by a mainstream visual backbone
Independent learnable weights for four text matching rules
Adaptive balancing of the contribution of visual and textual modalities
Built-in numerical stability and NaN handling mechanism
Stable training supported by batch normalization and gradient clipping

Dataset Structure
data_root/
├── train/
│   ├── class_001/
│   │   ├── image_001.jpg
│   │   ├── image_001.json
│   │   └── image_001.txt
│   └── ...
├── test/
│   └── ... (same structure as the training set)
├── similarity_scores/
│   ├── train/
│   │   ├── class_001/
│   │   │   └── image_001.txt
│   │   └── ...
│   └── test/
│       └── ...
└── class_indices.json

Training Features
Dual learning rates for the backbone network and fusion branch for optimized training
Gradient clipping to prevent gradient explosion
ReduceLROnPlateau learning rate scheduler
Early stopping based on validation accuracy
Automatic logging and best model checkpoint saving
Real-time monitoring of modality fusion weights

Notes
Each image must have a corresponding .txt similarity score file.
GPU is highly recommended for faster OCR processing and model training.
Adjust the batch size according to your GPU memory capacity.
All file paths must be correctly configured before running the scripts.

License
This project is for academic research purposes only.
