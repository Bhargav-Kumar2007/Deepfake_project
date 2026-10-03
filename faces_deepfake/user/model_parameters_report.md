# DeepFake Detector: Two-CNN Model Parameters & Architecture

The active system has two CNNs only: `PatchCNN` for local texture analysis and `DeepCNN` for whole-face analysis. Their real probabilities are combined by a deterministic weighted mean. There is no learned meta-ensemble, third model, ensemble checkpoint, or engineered-feature classifier.

## Active pipeline overview

| Component           | Architecture             | Parameters | FP32 memory | Checkpoint |
| ------------------- | ------------------------ | ---------: | ----------: | ---------: |
| Local texture model | `PatchCNN`             |  1,215,393 |     4.64 MB |   13.98 MB |
| Wide global model   | `DeepCNN`              |  4,847,777 |    18.49 MB |   55.57 MB |
| Weighted fusion     | Deterministic formula    |          0 |        0 MB |       None |
| Complete pipeline   | Dual CNN + weighted mean |  6,063,170 |    23.13 MB |   69.55 MB |

## Local Texture Model: PatchCNN

- **Purpose:** Detects local texture artifacts, boundary seams, and high-frequency inconsistencies.
- **Input:** 112x112 RGB patches extracted from a 512x512 image.
- **Current inference grid:** 36 patches (6x6).
- **Parameters:** 1,215,393 trainable; 1,928 BatchNorm buffer elements.
- **Layer parameters:** Conv2d 1,172,256; Linear 41,217; BatchNorm2d 1,920.
- **Compute:** 646.71 MMACs / 1.293 GFLOPs per patch; approximately 23.28 GMACs / 46.56 GFLOPs per 36-patch image.
- **Checkpoint metadata:** epoch 7; best validation accuracy 99.5565%; validation loss 0.014241.
- **Recorded base-model metrics:** accuracy 0.999990; F1 0.999990.

## Wide Global Model: DeepCNN

- **Purpose:** Detects whole-face structural, lighting, and semantic inconsistencies.
- **Input:** 256x256 RGB image, normalized to [-1, 1].
- **Parameters:** 4,847,777 trainable; 3,978 BatchNorm buffer elements.
- **Layer parameters:** Conv2d 4,712,224; Linear 131,585; BatchNorm2d 3,968.
- **Compute:** 4.28 GMACs / 8.57 GFLOPs per image.
- **Checkpoint metadata:** epoch 10; best validation accuracy 97.1750%; best F1-score 0.971669.
- **Recorded base-model metrics:** accuracy 1.000000; F1 1.000000.

## Weighted two-CNN fusion

The final real probability is:

```text
real_probability = (1.0 * wide_real_probability + 1.0 * patch_real_probability) / (1.0072312107559 + 1.0)
```

- **Wide : Patch weight ratio:** 1 : 1
- **Learned parameters:** 0
- **Additional checkpoint:** none
- **Decision rule:** REAL at a real probability of at least 0.5; otherwise FAKE.

## End-to-end inference budget

- **Total vision parameters:** 6,063,170
- **Total FP32 weight memory:** 23.13 MB
- **CNN checkpoint storage:** 69.55 MB
- **Vision compute:** approximately 27.56 GMACs / 55.13 GFLOPs per image using the current 36-patch local grid.
- **Fusion compute:** one weighted average; negligible compared with CNN inference.

Run `python faces_deepfake/user/calculate_model_parameters.py --save-report` in the model environment to regenerate this report from the installed checkpoints.
