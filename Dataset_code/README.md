# FMCG Product Dataset Processing Pipeline
This repository contains the complete data processing scripts for the FMCG fine-grained classification dataset, including OCR extraction, text processing, position & area weighting, and final similarity scoring.

## Overview
This pipeline processes product image datasets to:
- Run batch OCR on product images
- Extract and filter text from OCR results
- Calculate text region importance (position + area)
- Compute final text matching scores for classification
- Support reproducible data preprocessing for multimodal models

## Pipeline Steps
All scripts must be executed in the following order:

1. `1-OCR_json.py` – Run OCR and save results to JSON files
2. `2-OCR_txt.py` – Extract text from OCR JSON files
3. `3-Area_position.py` – Compute text region position and area weights
4. `4-rename.py` – Rename output files for organization
5. `5-final_score.py` – Calculate final text matching scores
