# DeepFake Detector

A local AI-face/deepfake detector built with two custom PyTorch CNNs:

- **Wide CNN (`DeepCNN`)** analyses the full 256 x 256 face image for global structure, lighting, and semantic inconsistencies.
- **Patch CNN (`PatchCNN`)** analyses 36 overlapping 112 x 112 patches from a 512 x 512 image to identify local texture artifacts and boundary inconsistencies.

The final verdict is a deterministic weighted fusion of the two models' real-image probabilities. The repository includes the trained checkpoints, a browser-based local web interface, command-line inference scripts, explainability heatmaps, and notebooks documenting dataset creation and training.

> This project is an experimental classifier. Its output is a model prediction, not proof that an image is real or AI-generated. Do not use it as the sole basis for high-stakes, legal, safety, or identity decisions.

## Features

- Detects whether an uploaded face image is likely **real** or **AI-generated**.
- Combines global image analysis and local patch-level analysis.
- Provides a local web interface with image upload and camera capture support.
- Displays the final score, both model scores, confidence, patch statistics, and explainability visualizations.
- Includes Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, LIME, and patch heatmap generation.
- Keeps uploaded web-app images in a temporary file only during analysis, then deletes them.

## Model pipeline

```text
Input image
   |
   +--> Wide CNN: resize to 256 x 256 --> global real probability
   |
   +--> Patch CNN: resize to 512 x 512 --> 36 overlapping 112 x 112 patches
                                             --> mean local real probability
   |
   +--> weighted mean --> REAL / FAKE decision
```

The current fusion formula is:

```text
real_probability =
  (1.0072312107559 * wide_real_probability + 1.0 * patch_real_probability)
  / (1.0072312107559 + 1.0)
```

An image is labelled **REAL** when the resulting real probability is at least `0.5`; otherwise it is labelled **FAKE**.

| Component | Input | Parameters | Purpose |
| --- | --- | ---: | --- |
| Wide CNN / `DeepCNN` | 256 x 256 RGB image | 4,847,777 | Global facial structure and composition |
| Patch CNN / `PatchCNN` | 36 x 112 x 112 RGB patches | 1,215,393 | Local texture and artifact detection |
| Weighted fusion | Two real probabilities | 0 | Final decision |

## Project structure

```text
.
├── requirements.txt
├── faces_deepfake/
│   ├── models/                         # CNN definitions and trained .pth checkpoints
│   ├── user/                           # Inference, fusion, explainability, model report
│   ├── webapp/                         # Local browser UI and Python HTTP server
│   ├── others/                         # Stored evaluation CSVs and utilities
│   └── ALL_TRAINING_AND_TESTING/
│       ├── Dataset_creation/           # SDXL notebook and dataset provenance
│       └── Training_and_Testing/       # Training, inference, and evaluation notebooks
```

## Quick start

### Prerequisites

- Python 3.10 or 3.11 is recommended.
- An NVIDIA GPU with CUDA 12.4 is recommended for best performance. The supplied requirements install the CUDA 12.4 PyTorch wheels.
- CPU inference may work with a CPU-compatible PyTorch installation, but patch inference and explainability generation will be slower.

### Install

From the repository root, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The model checkpoint files must remain at:

```text
faces_deepfake/models/best_wide_model.pth
faces_deepfake/models/best_local_model.pth
```

### Run the local web app

```powershell
python faces_deepfake/webapp/server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The app is intentionally bound to localhost, so it is not publicly exposed by default.

The **Scan image** page accepts JPG, JPEG, PNG, WEBP, and BMP images up to 12 MB. It returns the fusion result, the individual CNN outputs, patch-level statistics, and available explainability maps.

## Command-line inference

Run the weighted two-model detector and select an image in the file dialog:

```powershell
python faces_deepfake/user/use_weighted_fusion.py
```

The script prints the wide-model result, patch-model result, and final fused real/fake probability.

To regenerate the model architecture and checkpoint report:

```powershell
python faces_deepfake/user/calculate_model_parameters.py --save-report
```

## Dataset and training

The training dataset contains 100,000 face images:

| Class | Count | Sources |
| --- | ---: | --- |
| Real | 50,000 | FFHQ faces |
| AI-generated / fake | 50,000 | 20,000 SDXL, 15,000 StyleGAN2-ADA, 15,000 StyleGAN3 |

It was split with seed `42` into 80,000 training images and 20,000 validation images, balanced by class.

Further documentation is available in:

- [Dataset creation](faces_deepfake/ALL_TRAINING_AND_TESTING/Dataset_creation/full_creation.md)
- [Training and testing](faces_deepfake/ALL_TRAINING_AND_TESTING/Training_and_Testing/full_train_and_test.md)

The associated notebooks are:

- `SDXL_DS_AI_Faces.ipynb` — SDXL AI-face generation.
- `Normal_CNN_Training.ipynb` — full-image `DeepCNN` training.
- `Simple_Patch_CNN_Training.ipynb` — patch-based `PatchCNN` training.
- `All_Model__testing.ipynb` — model loading, inference, and comparative evaluation workflow.

## Recorded evaluation results

`faces_deepfake/others/model_metrics.csv` records the following results:

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Wide CNN | 100.000% | 100.000% | 100.000% | 100.000% |
| Patch CNN | 99.999% | 99.998% | 100.000% | 99.999% |

These figures come from an evaluation that scans both the training and validation folders. They are useful for verifying the recorded pipeline, but **they are not independent held-out test-set metrics** and should not be used as a claim of real-world generalization. Evaluate the detector on an unseen, separately curated test set before reporting performance externally.

## Notes and limitations

- The supplied checkpoint models were trained on FFHQ real faces and SDXL/StyleGAN-generated faces. They may not generalize to other generators, compression levels, face manipulations, non-face imagery, or real-world editing pipelines.
- Explainability results show regions that influenced this model's output; they do not establish the origin or authenticity of an image.
- The training/testing documentation notes label-mapping and helper-function consistency points in the original notebooks. Use the maintained inference scripts in `faces_deepfake/user/` for the current real/fake probability convention.
- The web app is intentionally local-only. Deployment requires adapting the host/port binding and choosing CPU or GPU-compatible dependencies.

## License

No license file is currently included in this repository. Add a license before distributing or reusing the project publicly.
