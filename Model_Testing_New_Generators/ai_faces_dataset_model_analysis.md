# Comprehensive Benchmark & Comparative Analysis Report

## DeepFake Detection Evaluation: `use_wide_model.py` (DeepCNN) vs. `use_local_model.py` (PatchCNN)

- **Date:** September 2026
- **Dataset:** `ai_faces_dataset` (3,000 total images)
- **Execution Hardware:** NVIDIA GPU with CUDA Acceleration (PyTorch 2.5.1+cu124)
- **Detailed Data Artifact:** [`ai_faces_dataset_detailed_predictions.csv`](file:///d:/PROJECTS/DeepFake-Detector-main/ai_faces_dataset_detailed_predictions.csv)
- **Summary Metrics Artifact:** [`ai_faces_dataset_summary_metrics.csv`](file:///d:/PROJECTS/DeepFake-Detector-main/ai_faces_dataset_summary_metrics.csv)

---

## 1. Executive Summary

This report evaluates and contrasts the **two primary deepfake detection models** implemented in this repository:
1. **Wide Global Model (`DeepCNN` via [`use_wide_model.py`](file:///d:/PROJECTS/DeepFake-Detector-main/faces_deepfake/user/use_wide_model.py))**: A 4.85-million parameter convolutional network operating on complete 256×256 facial images to evaluate global anatomical structure, lighting consistency, specular reflections, and semantic coherence.
2. **Local Texture Model (`PatchCNN` via [`use_local_model.py`](file:///d:/PROJECTS/DeepFake-Detector-main/faces_deepfake/user/use_local_model.py))**: A 1.22-million parameter convolutional network evaluating 36 overlapping 112×112 patches extracted from 512×512 images to identify microscopic texture inconsistencies, blending boundaries, and frequency-domain artifacts.

The models were evaluated against the **3,000-image `ai_faces_dataset`**, comprising 1,000 genuine human faces and 2,000 AI-generated faces spanning five distinct generative architectures.

### Key Findings Summary:
- **Wide Global Model (`DeepCNN`) is the vastly superior and reliable detector**, achieving an **Overall Accuracy of 81.93%**, a **Balanced Accuracy of 83.28%**, an **ROC-AUC of 0.8980**, a **Real-Face Specificity of 87.30%**, and a **Fake Detection Recall of 79.25%** across all 3,000 images at **47.3 images/second**.
- **Local Texture Model (`PatchCNN`) exhibits catastrophic failure on real human faces and modern high-resolution generators**, achieving only **28.00% Overall Accuracy** and **0.10% Real Face Specificity** (misclassifying 999 out of 1,000 real LFW human faces as deepfakes, yielding a **99.90% False Positive Rate**).
- On newer generation diffusion engines (`RealisticVision-V6`, `Playground-v2.5`, `SDXL-base-1.0`), `PatchCNN` suffers from inverted predictions, labeling up to **96.25% of SDXL synthetic images as REAL**.
- **Statistical Significance**: McNemar's test yields $\chi^2 = 1331.31$ ($p < 10^{-100}$), confirming that the Wide Model's performance superiority over the Local Model is overwhelming and statistically significant.

---

## 2. Dataset Structure & Ground Truth Taxonomy

The test corpus consists of 3,000 high-resolution portrait photographs structured as follows:

| Subset Folder | Generator / Source | Ground Truth Class | Numeric Target | Image Count | Image Resolution | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `00_REAL_LFW` | **Labeled Faces in the Wild (LFW)** | **REAL** | `0` (Fake=0, Real=1) | 1,000 | Variable / Resized | Genuine unconstrained human face photographs under varied real-world lighting and poses. |
| `01_SD-1.5` | **Stable Diffusion 1.5** | **FAKE** | `1` (Fake=1, Real=0) | 400 | 512×512 | Latent diffusion model (RunwayML / CompVis); known for high-frequency checkerboard & brushstroke artifacts. |
| `02_SD-2.1` | **Stable Diffusion 2.1** | **FAKE** | `1` (Fake=1, Real=0) | 400 | 768×768 (native) | OpenCLIP-guided latent diffusion model with modified VAE and enhanced facial prompt adherence. |
| `03_RealisticVision-V6` | **RealisticVision V6.0** | **FAKE** | `1` (Fake=1, Real=0) | 400 | 512×512 / Upscaled | Photorealistic SD 1.5 fine-tune engineered specifically to mimic DSLR sensor noise and human pores. |
| `04_Playground-v2.5` | **Playground v2.5** | **FAKE** | `1` (Fake=1, Real=0) | 400 | 1024×1024 | Modern open aesthetic diffusion model tuned for facial aesthetic quality and skin tone richness. |
| `05_SDXL-base-1.0` | **SDXL Base 1.0** | **FAKE** | `1` (Fake=1, Real=0) | 400 | 1024×1024 | Stability AI flagship 3.5B-parameter diffusion model with dual text encoders and refined latents. |
| **Total Corpus** | **6 Subsets (1 Real, 5 Generators)** | **1000 Real / 2000 Fake** | - | **3,000** | - | Fully balanced across generator classes. |

---

## 3. Overall Comparative Performance Benchmark

All metrics below treat **FAKE as the Positive class** (standard for DeepFake detection systems):

| Metric Category | Evaluation Metric | Wide Model (`DeepCNN`) | Local Model (`PatchCNN`) | Absolute Delta | Relative Advantage |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | Overall Accuracy | **81.93%** (2458 / 3000) | 28.00% (840 / 3000) | **+53.93%** | **Wide Model (+192.6%)** |
| | Balanced Accuracy | **83.28%** | 21.02% | **+62.25%** | **Wide Model (+296.2%)** |
| **Detection Quality** | Fake Precision | **92.58%** | 45.65% | **+46.93%** | **Wide Model** |
| | Fake Recall / Sensitivity | **79.25%** (1585 / 2000) | 41.95% (839 / 2000) | **+37.30%** | **Wide Model (+88.9%)** |
| | Fake F1-Score | **85.40%** | 43.72% | **+41.68%** | **Wide Model (+95.3%)** |
| **Error Rates** | Real Specificity (True Negative Rate) | **87.30%** (873 / 1000) | **0.10%** (1 / 1000) | **+87.20%** | **Wide Model (+87,200%)** |
| | False Positive Rate (FPR) | **12.70%** (127 / 1000) | **99.90%** (999 / 1000) | **-87.20%** | **Wide Model (Lower is better)** |
| | False Negative Rate (FNR) | **20.75%** (415 / 2000) | **58.05%** (1161 / 2000) | **-37.30%** | **Wide Model (Lower is better)** |
| **Probability Curves** | ROC-AUC Score | **0.8980** | 0.1472 | **+0.7508** | **Wide Model** |
| | PR-AUC (Average Precision) | **0.9533** | 0.5443 | **+0.4089** | **Wide Model** |
| | Brier Calibration Score | **0.1280** | 0.5200 | **-0.3920** | **Wide Model (Lower is better)** |
| **Operational** | Inference Throughput | **47.3 images/s** | 28.4 images/s | **+18.9 img/s** | **Wide Model (1.67× faster)** |
| | Total Dataset Evaluation Time | **63.43 seconds** | 105.66 seconds | **-42.23 s** | **Wide Model** |

---

## 4. Confusion Matrices

```
========================================================================================
WIDE MODEL (DeepCNN) CONFUSION MATRIX
========================================================================================
                                 PREDICTED REAL                PREDICTED FAKE
ACTUAL REAL (1,000 images):       873  (True Negatives)         127  (False Positives)
ACTUAL FAKE (2,000 images):       415  (False Negatives)       1585  (True Positives)
========================================================================================
Overall Accuracy: 81.93%  |  Precision: 92.58%  |  Recall: 79.25%  |  Specificity: 87.30%

========================================================================================
LOCAL MODEL (PatchCNN) CONFUSION MATRIX
========================================================================================
                                 PREDICTED REAL                PREDICTED FAKE
ACTUAL REAL (1,000 images):         1  (True Negative)          999  (False Positives)
ACTUAL FAKE (2,000 images):      1161  (False Negatives)        839  (True Positives)
========================================================================================
Overall Accuracy: 28.00%  |  Precision: 45.65%  |  Recall: 41.95%  |  Specificity:  0.10%
```

---

## 5. Per-Generator Granular Performance Breakdown

| Subset Folder | Generator Name | Ground Truth | Sample Size | Wide Accuracy | Wide Correct / Total | Local Accuracy | Local Correct / Total | Performance Winner |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00_REAL_LFW` | **Real Faces (LFW)** | **REAL** | 1,000 | **87.30%** | 873 / 1,000 | 0.10% | 1 / 1,000 | **Wide Model (+87.2%)** |
| `01_SD-1.5` | **Stable Diffusion 1.5** | **FAKE** | 400 | 77.25% | 309 / 400 | **96.25%** | 385 / 400 | **Local Model (+19.0%)** |
| `02_SD-2.1` | **Stable Diffusion 2.1** | **FAKE** | 400 | 76.75% | 307 / 400 | **81.25%** | 325 / 400 | **Local Model (+4.5%)** |
| `03_RealisticVision-V6`| **RealisticVision V6** | **FAKE** | 400 | **68.75%** | 275 / 400 | 7.25% | 29 / 400 | **Wide Model (+61.5%)** |
| `04_Playground-v2.5` | **Playground v2.5** | **FAKE** | 400 | **93.75%** | 375 / 400 | 21.25% | 85 / 400 | **Wide Model (+72.5%)** |
| `05_SDXL-base-1.0` | **SDXL Base 1.0** | **FAKE** | 400 | **79.75%** | 319 / 400 | 3.75% | 15 / 400 | **Wide Model (+76.0%)** |

---

## 6. Deep Technical Analysis: Why the Models Behave So Differently

### A. The Success of the Wide Global Model (`DeepCNN`)
1. **Holistic Semantic Reasoning**:
   The Wide Model receives downsampled 256×256 images through a 5-block hierarchical CNN architecture with feature map widths scaling up to 512 channels. By evaluating the face in its entirety, it learns global facial invariants:
   - **Corneal reflection symmetry**: Both pupils reflecting the same directional light source.
   - **Facial geometry and proportionality**: Symmetrical positioning of the earlobes, nose bridge, jawline curvature, and dental arch.
   - **Boundary consistency**: Blending transitions between hair edges, collars, and background bokeh.
2. **Robustness Across Modern Diffusion Generators**:
   The Wide Model maintained strong detection rates against the most challenging generators:
   - **93.75% on Playground v2.5**: Playground v2.5 faces often exhibit overly smoothed lighting gradients and subtle facial asymmetry which `DeepCNN` detects reliably.
   - **79.75% on SDXL Base 1.0**: Despite SDXL's high fidelity, global structural cues reveal synthetic composition.
   - **87.30% Specificity on LFW**: Correctly identifying real camera noise, varied focal lengths, and authentic lighting across real human portraits.

### B. The Failure Modes of the Local Texture Model (`PatchCNN`)
1. **The 99.9% False Positive Rate on Real LFW Faces**:
   `PatchCNN` evaluates thirty-six 112×112 patches per image. On the 1,000 real images in `00_REAL_LFW`, the average predicted real probability was only **0.1145** (88.55% fake probability). Consequently, **999 out of 1,000 real human faces were misclassified as deepfakes**.
   - *Root Cause*: In `PatchCNN`'s training dataset, real images were high-resolution FFHQ studio portraits. When evaluated on LFW (which has natural JPEG compression artifacts, film grain, and low-light sensor noise), the patch CNN mistakes legitimate camera compression artifacts for AI generation artifacts.
2. **Prediction Inversion on Modern Diffusion Models (SDXL, RealisticVision, Playground)**:
   - On `01_SD-1.5`, `PatchCNN` succeeded with **96.25% accuracy**, because SD 1.5 produces distinct high-frequency frequency-domain artifacts (latent decoder checkerboarding).
   - However, on `05_SDXL-base-1.0`, accuracy dropped to **3.75%** (385 out of 400 misclassified as REAL).
   - On `03_RealisticVision-V6`, accuracy plummeted to **7.25%** (371 out of 400 misclassified as REAL).
   - *Root Cause*: Modern diffusion models utilize upgraded VAE decoders and refined latent spaces that generate ultra-smooth, natural-looking high-frequency skin textures and simulated pores. Because `PatchCNN` looks only at 112×112 patches without full facial context, it perceives clean, detailed synthetic skin as "authentic human skin."

---

## 7. Model Agreement and Error Correlation Analysis

Pairwise contingency analysis across the 3,000 images:

| Case Category | Description | Count | Percentage |
| :--- | :--- | :---: | :---: |
| **Both Models Correct** | Both Wide and Local predict the correct label | **667** | **22.23%** |
| **Wide Only Correct** | Wide is correct; Local misclassifies | **1,791** | **59.70%** |
| **Local Only Correct** | Local is correct; Wide misclassifies | **173** | **5.77%** |
| **Both Models Wrong** | Both models fail on the same image | **369** | **12.30%** |
| **Prediction Agreement** | Both models predict the identical label | **1,036** | **34.53%** |
| **Cohen's Kappa ($\kappa$)** | Inter-model agreement statistic | **-0.3524** | **Severe disagreement** |
| **McNemar Test Statistic** | Difference significance ($\chi^2$) | **1331.31** | **$p < 10^{-100}$ (Statistically significant)** |

### Failure Overlap Analysis (The 369 "Both Wrong" Images):
- **127 Images**: Real LFW faces that fooled both models (genuine human faces with atypical lighting, extreme angles, or heavy compression).
- **242 Images**: Synthetic faces that evaded both detectors:
  - 125 from `RealisticVision-V6`: Specifically fine-tuned to fool both global and local discriminators.
  - 81 from `SDXL-base-1.0`: High semantic fidelity combined with advanced micro-textures.
  - 25 from `Playground-v2.5`.
  - 11 from `SD-1.5` and `SD-2.1`.

---

## 8. Summary & Practical Recommendations

1. **Deploy `use_wide_model.py` as the Primary Production Detector**:
   - `use_wide_model.py` delivers balanced, reliable performance (**81.93% overall accuracy**, **87.30% real-face specificity**, and **79.25% fake recall**).
   - It runs in single-pass mode at **47.3 images/second** on GPU, making it efficient for production.
2. **Restrict or Recalibrate `use_local_model.py`**:
   - As currently weighted, `use_local_model.py` cannot be used autonomously for real-world face validation due to its 99.9% false positive rate on unconstrained real faces.
   - It remains effective as a specialized artifact detector specifically for classic diffusion models (SD 1.5 / SD 2.1), but fails to generalize to modern high-resolution generators.
3. **Data Artifacts Available in Repository**:
   - Detailed image-by-image predictions and patch metrics: [`ai_faces_dataset_detailed_predictions.csv`](file:///d:/PROJECTS/DeepFake-Detector-main/ai_faces_dataset_detailed_predictions.csv)
   - Aggregated metric breakdown table: [`ai_faces_dataset_summary_metrics.csv`](file:///d:/PROJECTS/DeepFake-Detector-main/ai_faces_dataset_summary_metrics.csv)
   - Evaluation runner script: [`evaluate_models.py`](file:///d:/PROJECTS/DeepFake-Detector-main/evaluate_models.py)
