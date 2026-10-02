# Training and Testing Documentation

## Overview

This folder contains the notebooks used to prepare the face-image dataset, train two binary deepfake detectors, and evaluate their predictions:

| Notebook                            | Role                                                                                                       |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `Normal_CNN_Training.ipynb`       | Trains the full-image (global) CNN.                                                                        |
| `Simple_Patch_CNN_Training.ipynb` | Trains the overlapping-patch CNN.                                                                          |
| `All_Model__testing.ipynb`        | Loads both trained models, provides batch-inference utilities, and prepares comparative evaluation output. |

Both models perform binary image classification using the same real-versus-AI-generated face dataset. The notebooks were written to run in Google Colab with CUDA support, with model checkpoints and logs saved to Google Drive.

## Dataset preparation

The training notebooks use the same data-preparation workflow. They download four Kaggle datasets:

- `bhargavkumar1729/upload-folder-sdxl-aiface-dataset` — SDXL-generated face images
- `bhargavkumar1729/stylegan3-pytorch-faces-15k` — StyleGAN3-generated face images
- `bhargavkumar1729/stylegan2-ada-pytorch-faces-15k` — StyleGAN2-ADA-generated face images
- `bhargavkumar1729/gan-real-faces-50k-ffhq` — 50,000 real FFHQ face images

The generated images are gathered from the SDXL, StyleGAN2, and StyleGAN3 directories. The real images are read from the downloaded `real` directory. Only JPG, JPEG, and PNG files are included.

With the random seed fixed to `42`, each class is shuffled and split as follows:

| Split           |       Real faces | AI-generated faces |             Total |
| --------------- | ---------------: | -----------------: | ----------------: |
| Training        |           40,000 |             40,000 |            80,000 |
| Validation      |           10,000 |             10,000 |            20,000 |
| **Total** | **50,000** |   **50,000** | **100,000** |

The final directory structure is:

```text
Faces_Project/
  Dataset/
    train/
      real/
      fake/
    val/
      real/
      fake/
```

The preparation cells move and rename files to `real_XXXXX.<ext>` and `fake_XXXXX.<ext>`, then delete the temporary `kaggle_data` download directory to reduce Colab storage use. If fewer than 50,000 images are found for either class, the code falls back to an 80/20 split of the available files.

## Common training setup

Both training notebooks use PyTorch, `torchvision.datasets.ImageFolder`, `DataLoader`, CUDA when available, `BCEWithLogitsLoss`, AdamW optimisation, cosine-annealing learning-rate scheduling, and automatic mixed precision on CUDA.

The training augmentations are:

- resize to the model's input resolution;
- random horizontal flip with probability 0.5;
- random rotation up to 5 degrees;
- colour jitter with brightness, contrast, and saturation set to 0.1;
- conversion to a tensor and normalization using mean and standard deviation `[0.5, 0.5, 0.5]`.

Validation images are resized, converted to tensors, and normalized, but are not augmented. The notebooks seed Python, NumPy, and PyTorch with `42`; full deterministic cuDNN operation is noted but not enabled in the code.

## Model 1: full-image DeepCNN

`Normal_CNN_Training.ipynb` trains a custom full-image CNN on 256 x 256 face images.

### Architecture

The `DeepCNN` has five convolutional blocks. Each block uses two 3 x 3 convolutions, batch normalization, ReLU activation, and max pooling. The channel widths increase from 32 to 64, 128, 256, and 512. The classification head uses adaptive average pooling, a 512-to-256 fully connected layer with ReLU and 0.4 dropout, and a final one-logit output layer. The notebook identifies this model as approximately 4.8 million parameters.

### Training configuration

| Setting               |                      Value |
| --------------------- | -------------------------: |
| Input size            |                  256 x 256 |
| Batch size            |                        256 |
| Planned epochs        |                         20 |
| Initial learning rate |                      0.001 |
| Weight decay          |                     0.0001 |
| Data-loader workers   |                          4 |
| Decision threshold    | sigmoid probability >= 0.5 |

At every epoch, the notebook calculates training loss and accuracy, then validation accuracy, precision, recall, and F1 score. It saves a resumable `latest_checkpoint.pth`, saves `best_model.pth` when validation accuracy improves, and appends epoch metrics to `training_metrics.csv`.

The model artifacts are configured for Google Drive at:

```text
/content/drive/MyDrive/Faces_Project/checkpoints/
```

## Model 2: PatchCNN

`Simple_Patch_CNN_Training.ipynb` trains a second model designed to identify local image artifacts rather than only global face characteristics.

### Patch workflow

Each image is resized to 512 x 512 pixels and split exhaustively into overlapping 112 x 112 patches. The stride is `112 - 20 = 92` pixels. The extraction routine explicitly includes the final edge position, producing a 6 x 6 grid, or **36 patches per image**.

During training and validation, every patch receives the image-level class label and contributes independently to the binary cross-entropy loss. For image-level inference, the model averages all patch probabilities; this mean is used as the final score and is thresholded at 0.5.

### Architecture and configuration

The `PatchCNN` contains four convolutional blocks with the same convolution, batch-normalization, ReLU, and max-pooling pattern as the global CNN. Its channel widths are 32, 64, 128, and 256. Its classifier head uses adaptive average pooling followed by 256-to-128-to-64 fully connected layers, 0.3 dropout after the first two layers, and one output logit. The notebook describes it as an approximately 1.2-million-parameter model.

| Setting               |     Value |
| --------------------- | --------: |
| Input size            | 512 x 512 |
| Patch size            | 112 x 112 |
| Patch overlap         | 20 pixels |
| Patches per image     |        36 |
| Batch size            |        16 |
| Planned epochs        |        25 |
| Initial learning rate |    0.0001 |
| Weight decay          |    0.0001 |
| Data-loader workers   |         4 |

The notebook saves the latest checkpoint, best checkpoint/model, final model, and a per-epoch CSV log in:

```text
/content/drive/MyDrive/Faces_Project/patch_checkpoints/
```

## Testing and evaluation

`All_Model__testing.ipynb` reconstructs both architectures and loads the saved best models:

- global CNN: `checkpoints/best_model.pth`;
- patch CNN: `patch_checkpoints/best_patch_model.pth`.

It provides batch inference for the global CNN at 256 x 256 (default batch size 64) and for the patch CNN at 512 x 512 (default batch size 32). Both return a predicted class, confidence, and score. The patch inference also records the mean, variance, standard deviation, count, and individual probabilities of its 36 patches.

The evaluation section gathers all images from both the `train` and `val` folders, so it is a **whole-dataset evaluation (100,000 images)** rather than an independent held-out test set. It is designed to write:

- `all_predictions.csv`: true label plus both models' predictions, scores, and confidence values for every image;
- `model_metrics.csv`: accuracy, precision, recall, F1, true positives, true negatives, false positives, and false negatives for each model.

These files are configured to be saved under:

```text
/content/drive/MyDrive/Faces_Project/evaluation_results/
```

## Important implementation notes

The following points reflect the notebooks as currently stored and matter when reproducing or interpreting the evaluation:

1. **Class Index Mapping and Inference Consistency:** By default, PyTorch's `ImageFolder` assigns class indices alphabetically, meaning the `fake` folder is assigned index `0` and the `real` folder is assigned index `1`. During the initial full-dataset evaluation, the raw model outputs reflected this underlying index mapping. To ensure the final reported metrics and deployment scripts align with the standard semantic meaning (where a probability $\ge 0.5$ indicates a "REAL" image), the evaluation outputs were explicitly corrected in the final CSV logs. Furthermore, the current inference scripts (`use_wide_model.py` and `use_local_model.py`) have been explicitly updated to deterministically enforce `label = "REAL" if prob >= 0.5 else "FAKE"`. This guarantees that all future testing, fusion, and web app deployments use the correct, intuitive class mapping.
2. The evaluation cell in the notebook calls `predict_image_wide` and `predict_image_patch`. Ensure compatible single-image wrapper functions are defined, or call the batch functions with one image at a time.
3. Since the evaluation scans both training and validation folders, its reported metrics do not measure generalization to a separate unseen test set. A final performance claim should be based on a separately held-out test dataset (as performed in the external 3,000-image evaluation).
4. Checkpoints include model, optimizer, scheduler, scaler, and random-number-generator states to support resuming. Exact results may still vary slightly because deterministic cuDNN settings are commented out.

## Execution order

1. Run the dataset-download and split cells in either training notebook once.
2. Run `Normal_CNN_Training.ipynb` to train or resume the full-image DeepCNN.
3. Run `Simple_Patch_CNN_Training.ipynb` to train or resume the PatchCNN.
4. Ensure the best checkpoints are present in the configured Google Drive locations.
5. Run `All_Model__testing.ipynb` to load both models and perform batch inference or comparative evaluation after addressing the helper-name and label-mapping notes above.
