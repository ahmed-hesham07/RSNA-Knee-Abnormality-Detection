# RSNA Knee Abnormality Detection: Multi-Planar DINOv2 Solution

An end-to-end Grandmaster-grade deep learning solution for the **[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)** competition on Kaggle.

---

## 🔬 Architectural Overview

Knee MRI examinations are 3D multi-planar studies where specific pathologies are localized across distinct acquisition planes:
- **Sagittal View (Fluid-Sensitive / T2)**: Ideal for Cruciate Ligaments (ACL/PCL), Joint Effusion, and Baker's Cysts.
- **Coronal View (Fat-Suppressed)**: Ideal for Collateral Ligaments (MCL/LCL), Meniscal Tears, and Compartmental Osteoarthritis.
- **Axial View**: Ideal for Patellofemoral Osteoarthritis, Cartilage Fissuring, and Bone Contusions.

```
       ┌───────────────────────┐
       │   Knee MRI Study      │
       └───────────┬───────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
[ Sagittal ]   [ Coronal ]    [ Axial ]
 (3 Triplets)   (3 Triplets)  (3 Triplets)
    │              │              │
    └──────────────┼──────────────┘
                   ▼
  ┌─────────────────────────────────┐
  │  Physical 130mm FOV Normalizer   │
  └────────────────┬────────────────┘
                   ▼
  ┌─────────────────────────────────┐
  │ DINOv2 Vision Transformer (timm)│
  └────────────────┬────────────────┘
                   ▼
  ┌─────────────────────────────────┐
  │ Cross-View Gated Attention Pool │
  └────────────────┬────────────────┘
                   ▼
  ┌─────────────────────────────────┐
  │   12 Abnormality Predictions    │
  └─────────────────────────────────┘
```

---

## 🛠️ Key Technical Innovations

1. **Multi-Planar MRI Sequence Selector**:
   - Intelligently queries `train_series.csv` / `test_series.csv` to select matching `Sagittal`, `Coronal`, and `Axial` series.
   - Prioritizes `Fluid_Sensitive == 1` and `Fat_Suppression == 1` sequences for highest diagnostic contrast.

2. **Physical Millimeter Geometry Normalization**:
   - Uses `ImageOrientationPatient` and `ImagePositionPatient` vectors to project slices along the anatomical normal.
   - Crops exact **130.0 mm physical Field of View (FOV)** using `PixelSpacing` metadata to ensure scale-invariance across scanner vendors (GE, Siemens, Philips).

3. **Cross-View Gated Attention Transformer**:
   - Embeds each planar triplet through a shared **DINOv2** backbone (`vit_small_patch14_reg4_dinov2.lvd142m` with `dynamic_img_size=True`).
   - Learns attention weights across planes:
     $$\mathbf{z}_{\text{fused}} = \sum_{v \in \{\text{Sag}, \text{Cor}, \text{Ax}\}} \alpha_v \cdot f(X_v)$$

4. **Asymmetric Loss for Class Imbalance**:
   - Implements Asymmetric Loss / Pos-Weighted BCE with soft labels extracted from radiology reports (`report_labels_v2.csv`) to regularize low-prevalence findings (e.g. Fractures, Synovitis).

5. **Robust Offline Inference & Ensembling**:
   - Zero-internet compliance on Kaggle.
   - Dynamic path fallback resolving `test_series/` structures seamlessly.
   - 5-Fold out-of-fold weight averaging with Test-Time Augmentation (TTA).

---

## 📁 Repository Structure

```
├── notebooks/
│   ├── RSNA_Knee_MultiView_DINOv2_Training.ipynb   # SOTA Multi-Planar Training Pipeline
│   ├── RSNA_Knee_MultiView_DINOv2_Inference.ipynb  # 5-Fold Multi-Planar Ensemble Inference
│   ├── RSNA_Knee_DINOv2_Baseline.ipynb             # Single-Plane Baseline Training
│   └── RSNA_Knee_DINOv2_Inference.ipynb            # Single-Plane Ensemble Inference
├── generate_multiview_training_nb.py               # Generates the Multi-Planar Training Notebook
├── generate_multiview_inference_nb.py              # Generates the Multi-Planar Inference Notebook
├── generate_nb.py                                  # Baseline Generator
├── generate_inference_nb.py                        # Baseline Inference Generator
├── rsna-knee-competition.md                        # Official Competition Specification
└── research-report.md                              # Deep-Dive Literature & Methodology Report
```

---

## 🚀 Execution Instructions on Kaggle

### 1. Training (Multi-Planar Model)
1. Create a new Kaggle notebook and import [`notebooks/RSNA_Knee_MultiView_DINOv2_Training.ipynb`](file:///C:/Users/ahmed/projects/rsna/notebooks/RSNA_Knee_MultiView_DINOv2_Training.ipynb).
2. Attach:
   - Official Dataset: `rsna-knee-abnormality-detection`
   - Community Labels: `pilkwang/rsna-knee-llm-labels`
   - Pretrained Weights: `pilkwang/rsna-knee-weights`
3. Run training across folds (or set `CFG.fold_to_train = 0..4`) to save best model checkpoints.

### 2. Inference (Leaderboard Submission)
1. Create a new Kaggle notebook and import [`notebooks/RSNA_Knee_MultiView_DINOv2_Inference.ipynb`](file:///C:/Users/ahmed/projects/rsna/notebooks/RSNA_Knee_MultiView_DINOv2_Inference.ipynb).
2. Attach:
   - Official Dataset: `rsna-knee-abnormality-detection`
   - Your Trained Weights Dataset: `my-rsna-dino-weights` (containing your `.pth` files)
3. Toggle **Internet OFF** in the right-side panel.
4. Click **Save & Run All (Commit)** -> Go to Viewer -> Click **Submit to Competition**!
