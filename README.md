# RSNA Knee Abnormality Detection: DINOv2 Solution

This repository contains a full 5-Fold Cross Validation pipeline for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) competition on Kaggle.

## Overview
The goal of this competition is to detect and classify 12 different knee abnormalities (e.g., ACL tears, Meniscus tears, Osteoarthritis) from multi-planar MRI scans. 

This solution uses a state-of-the-art **DINOv2** Vision Transformer backbone, fine-tuned using a highly optimized spatial cropping strategy and community-sourced soft labels.

## Core Methodology

1. **Geometric DICOM Preprocessing**: Standard resizing destroys small focal lesions (1-3mm). Instead, we use a robust geometry engine (`ImagePositionPatient` & `ImageOrientationPatient`) to sort slices anatomically and extract a **130mm physical center-crop** to isolate the knee joint perfectly across all MRI machines and resolutions.
2. **3-Slice RGB Triplets**: DINOv2 requires 3-channel (RGB) input. We sample 3 equidistant slices from the center of the crop to form an RGB volume, representing roughly 10mm of continuous physical tissue.
3. **Soft LLM Labels for Regularization**: The official training data only contains 58 expert-labeled studies. We utilize soft probabilities (LLM-extracted from the radiology reports) for the remaining 4,400+ studies. These soft labels provide massive natural regularization (label smoothing) during training.
4. **Intelligent Pooling Head**: We attach a simple `Max-Pooling` head to the DINOv2 backbone. Max-pooling naturally favors the detection of highly focal, isolated abnormalities (like a 2mm meniscus tear) across the input triplets.
5. **5-Fold Stratified Ensembling**: The training loop utilizes `StratifiedGroupKFold` across 5 epochs per fold to guarantee even distribution and prevent patient leakage, resulting in 5 heavily generalized models.

## Repository Structure
* `notebooks/RSNA_Knee_DINOv2_Baseline.ipynb`: The primary Kaggle Notebook containing the full training pipeline, DataLoaders, and execution loops.
* `research-report.md`: Detailed technical analysis of the competition landscape, metric ceilings, and strategies employed by top Grandmasters.

## Usage
1. Upload the `RSNA_Knee_DINOv2_Baseline.ipynb` notebook to Kaggle.
2. Attach the official competition dataset.
3. Attach the community dataset: `pilkwang/rsna-knee-llm-labels`.
4. Attach the community dataset: `pilkwang/rsna-knee-weights` (for pre-trained initialization).
5. Run the notebook to generate the 5 `.pth` model weights.

## Requirements
* `torch`, `torchvision`, `timm`
* `pydicom`, `cv2`, `albumentations`
* `pandas`, `scikit-learn`
