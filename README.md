# DeepFake Detector

A local AI-face/deepfake detector built with two custom PyTorch CNNs and a deterministic weighted fusion:

- **Wide CNN (`DeepCNN`)** — analyses the full 256 × 256 face image for global structure, lighting, and semantic inconsistencies.
- **Patch CNN (`PatchCNN`)** — analyses 36 overlapping 112 × 112 patches extracted from a 512 × 512 image to detect local texture artifacts and boundary inconsistencies.

The repository includes trained checkpoints, a dark-mode browser-based local web interface with five XAI explainability methods, command-line inference scripts, and supplementary notebooks documenting dataset creation, training, and evaluation.

> **Disclaimer:** This project is an experimental classifier. Its output is a model prediction, not proof that an image is real or AI-generated. Do not use it as the sole basis for high-stakes, legal, safety, or identity decisions.

---

## Features

- Detects whether an uploaded face image is likely **real** or **AI-generated**.
- Combines global image analysis (Wide CNN) and local patch-level analysis (Patch CNN).
- Local web interface with image upload and live camera capture support.
- Displays fusion verdict, both model verdicts, confidence, patch statistics, and six explainability visualizations:
  - **Grad-CAM**, **Grad-CAM++**, **Score-CAM**, **Integrated Gradients**, **LIME** (wide model)
  - **Patch probability heatmap** with interactive 6 × 6 matrix (local model)
- Color-coded **REAL** / **FAKE** verdict badges with glow indicators.
- Uploaded images are stored only in a temporary file during inference, then deleted immediately.

---

## Model pipeline

```text
Input image
   |
   +──► Wide CNN: resize to 256 × 256 ──► global real probability
   |
   +──► Patch CNN: resize to 512 × 512 ──► 36 overlapping 112 × 112 patches
                                            ──► mean local real probability
   |
   +──► weighted mean ──► REAL / FAKE decision
```

Fusion formula:

```text
real_probability =
  (1.0072312107559 × wide_real_probability + 1.0 × patch_real_probability)
  / (1.0072312107559 + 1.0)
```

An image is labelled **REAL** when the real probability ≥ `0.5`; otherwise **FAKE**.

| Component | Input | Parameters | Purpose |
| --- | --- | ---: | --- |
| Wide CNN / `DeepCNN` | 256 × 256 RGB | 4,847,777 | Global facial structure and composition |
| Patch CNN / `PatchCNN` | 36 × 112 × 112 RGB patches | 1,215,393 | Local texture and artifact detection |
| Weighted fusion | Two real probabilities | 0 | Final deterministic decision |

---

## Project structure

```text
.
├── requirements.txt                        # PyTorch cu124 wheels + Pillow, NumPy, scikit-learn
├── faces_deepfake/
│   ├── models/
│   │   ├── basemodel_wide.py               # DeepCNN architecture definition
│   │   ├── basemodel_local.py              # PatchCNN architecture definition
│   │   ├── best_wide_model.pth             # Trained Wide CNN checkpoint (~55.6 MB)
│   │   └── best_local_model.pth            # Trained Patch CNN checkpoint (~14.0 MB)
│   ├── user/
│   │   ├── use_wide_model.py               # Wide CNN inference (single image and batch)
│   │   ├── use_local_model.py              # Patch CNN inference (single image and batch)
│   │   ├── use_weighted_fusion.py          # Two-model weighted fusion (CLI + importable)
│   │   ├── explainability.py               # 5 XAI engines + patch heatmap generator
│   │   ├── calculate_model_parameters.py   # Architecture/compute report generator
│   │   └── model_parameters_report.md      # Generated model parameters report
│   └── webapp/
│       ├── server.py                       # Python stdlib HTTP server (no framework)
│       ├── index.html                      # Single-page UI
│       ├── app.js                          # Frontend logic (upload, XAI, patch matrix)
│       ├── styles.css                      # Dark-mode CSS
│       └── README.md                       # Webapp-specific documentation
├── ALL_TRAINING_AND_TESTING/
│   ├── Dataset_creation/                   # SDXL generation notebook and dataset provenance
│   ├── Training_and_Testing/               # Training, inference, and evaluation notebooks
│   ├── Wide_model_training_metrics.csv     # Epoch-by-epoch Wide CNN training log
│   └── patch_model_training_metrics.csv    # Epoch-by-epoch Patch CNN training log
├── XAI/
│   ├── xai_visualization.ipynb             # XAI batch comparison notebook
│   ├── xai.py                              # Standalone XAI script
│   ├── xai_comparison_grid.png             # Sample grid of all 5 XAI methods
│   ├── xai_failure_case.png                # Documented failure case analysis
│   └── xai_manifest.csv                   # XAI result index
├── Full_Prediction_original_ds/            # Batch prediction results on training dataset
│   ├── all_predictions_batch.csv
│   ├── model_metrics.csv
│   └── evaluate_models.py
├── Model_Testing_New_Generators/           # Evaluation against new AI generators
│   ├── ai_faces_dataset_detailed_predictions.csv
│   ├── ai_faces_dataset_summary_metrics.csv
│   ├── ai_faces_dataset_model_analysis.md
│   └── evaluate_models.py
├── model_Testing_img_degradation/          # Robustness tests under image degradation
├── ai_faces_dataset/                       # Evaluation image set (new generators)
└── ai_faces_dataset.zip                    # Archive of ai_faces_dataset/
```

---

## Quick start

### Prerequisites

- Python 3.10 or 3.11 recommended (3.12+ should work but is untested).
- An NVIDIA GPU with CUDA 12.4 is recommended. The supplied `requirements.txt` installs CUDA 12.4 PyTorch wheels.
- CPU-only inference works but is significantly slower for patch inference and XAI generation.

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The model checkpoint files must be present at:

```text
faces_deepfake/models/best_wide_model.pth
faces_deepfake/models/best_local_model.pth
```

### Run the local web app

```powershell
python faces_deepfake/webapp/server.py
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.  
The server is bound to localhost only — it is not publicly exposed.

The **Scan Image** tab accepts JPG, JPEG, PNG, WEBP, and BMP images up to **12 MB** and returns:

- Ensemble (weighted fusion) verdict + probabilities
- Wide CNN verdict + probabilities + 5 XAI heatmaps with opacity slider
- Patch CNN verdict + probabilities + interactive 6 × 6 patch matrix + patch heatmap

The **Model Data** tab renders the architecture/compute report from `user/model_parameters_report.md`.

See [`faces_deepfake/webapp/README.md`](faces_deepfake/webapp/README.md) for full webapp API documentation.

---

## Command-line inference

**Weighted two-model detector** (opens a file dialog):

```powershell
python faces_deepfake/user/use_weighted_fusion.py
```

Prints the Wide CNN result, Patch CNN result, and final fused real/fake probability.

**Regenerate the model parameters report:**

```powershell
python faces_deepfake/user/calculate_model_parameters.py --save-report
```

**Generate XAI heatmaps for a single image:**

```powershell
python faces_deepfake/user/explainability.py --image path/to/face.jpg --output xai_results/
```

---

## Dataset and training

The training dataset contains 100,000 face images:

| Class | Count | Sources |
| --- | ---: | --- |
| Real | 50,000 | FFHQ faces |
| AI-generated / fake | 50,000 | 20,000 SDXL · 15,000 StyleGAN2-ADA · 15,000 StyleGAN3 |

Split with seed `42` into 80,000 training and 20,000 validation images, balanced by class.

Further documentation:

- [`ALL_TRAINING_AND_TESTING/Dataset_creation/`](ALL_TRAINING_AND_TESTING/Dataset_creation/) — SDXL generation and dataset provenance
- [`ALL_TRAINING_AND_TESTING/Training_and_Testing/`](ALL_TRAINING_AND_TESTING/Training_and_Testing/) — training, inference, and evaluation notebooks

Key notebooks:

| Notebook | Purpose |
| --- | --- |
| `SDXL_DS_AI_Faces.ipynb` | SDXL AI-face generation |
| `Normal_CNN_Training.ipynb` | Full-image `DeepCNN` training |
| `Simple_Patch_CNN_Training.ipynb` | Patch-based `PatchCNN` training |
| `All_Model__testing.ipynb` | Model loading, inference, and comparative evaluation |

---

## Recorded evaluation results

`Full_Prediction_original_ds/model_metrics.csv` records results on the full training + validation dataset:

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Wide CNN | 100.000% | 100.000% | 100.000% | 100.000% |
| Patch CNN | 99.999% | 99.998% | 100.000% | 99.999% |

> These figures are from an evaluation that scans the training and validation splits used during model development — **they are not independent held-out test-set metrics**. Evaluate on a separately curated, unseen test set before reporting performance externally.

Cross-generator evaluation results (new AI generators not seen during training) are in [`Model_Testing_New_Generators/ai_faces_dataset_model_analysis.md`](Model_Testing_New_Generators/ai_faces_dataset_model_analysis.md).

---

## Notes and limitations

- Checkpoint models were trained on FFHQ (real) and SDXL / StyleGAN-generated (fake) faces. Generalization to other generators, heavy compression, face swaps, video frame captures, or non-face imagery is not guaranteed.
- XAI heatmaps highlight regions that influenced *this model's* output — they do not establish the origin or authenticity of an image.
- The web app is local-only. To deploy externally, update the host/port binding in `server.py` and choose CPU or GPU-compatible dependencies accordingly.
- Use the inference scripts in `faces_deepfake/user/` for the current real/fake probability convention. Earlier notebook versions may use a different label-mapping convention.

---

## License

MIT License.
