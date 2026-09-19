# AI Face Dataset Creation

## Purpose

This document records how the AI-generated face-image portion of the dataset was created for the deepfake-detection experiments. The dataset contains **50,000 synthetic face images** produced with three generative-model sources.

| Source | Image count | Generation approach |
| --- | ---: | --- |
| AdaGAN 2 | 15,000 | Generated directly with an open-source GAN model |
| GAN 3 | 15,000 | Generated directly with an open-source GAN model |
| Stable Diffusion XL (SDXL) | 20,000 | Generated with the included `SDXL_DS_AI_Faces.ipynb` notebook |
| **Total** | **50,000** | |

All images in this dataset are synthetic images intended to represent AI-generated faces.

## AdaGAN 2 and GAN 3 images

The AdaGAN 2 and GAN 3 subsets contain 15,000 images each. They were created by running the corresponding open-source GAN models directly and collecting their generated image outputs.

The generation scripts or model repositories for these two GANs are **not included in this folder**. Therefore, this repository records the dataset composition and the fact that these images came from directly used open-source models, but it does not provide a runnable local script to reproduce those two subsets. Any exact reproduction requires access to the same upstream model implementations, checkpoints, and generation settings used at the time of creation.

## SDXL images

The 20,000-image SDXL subset was generated with [`SDXL_DS_AI_Faces.ipynb`](SDXL_DS_AI_Faces.ipynb). The notebook is designed for Google Colab with a CUDA-capable GPU and uses the following setup:

- **Base model:** `stabilityai/stable-diffusion-xl-base-1.0`
- **Library:** Hugging Face `diffusers`
- **Precision:** `torch.float16`
- **Image size:** 512 x 512 pixels
- **File format:** PNG
- **Inference steps:** 10
- **Guidance scale:** 7.5
- **Seed selection:** a new random 32-bit seed is selected for every image
- **Safety/style filtering:** a negative prompt excludes cartoon, anime, illustration, painting, drawing, CGI, 3D-render, doll-like, blurry, low-quality, and visibly malformed outputs.

### Prompt design

The notebook defines 50 prompts for photorealistic, DSLR-style face portraits. The prompts intentionally vary age, gender presentation, facial features, hairstyle, expression, clothing, lighting, pose, and background characteristics. This promotes visual diversity in the SDXL subset rather than producing repeated portraits from a single description.

Each prompt begins with a photorealism-oriented description such as `ultra realistic DSLR portrait photograph` and then adds distinctive facial or scene details. A common negative prompt is supplied during generation to discourage non-photographic styles and obvious image defects.

### Batch generation workflow

The notebook organizes generated files under `SDXL_AIface_ds/`, with one `batch_<number>` directory per run. For each configured batch, it:

1. Selects the configured group of prompts.
2. Generates images one at a time on the CUDA device.
3. Saves each output as `img_0000.png`, `img_0001.png`, and so on in that batch directory.
4. Compresses the completed batch into a ZIP archive.
5. Downloads the ZIP archive from Colab for local dataset assembly.

The notebook variables `START_BATCH`, `END_BATCH`, `PROMPTS_PER_BATCH`, and `IMAGES_PER_PROMPT` make it possible to run the process in stages and resume at a later batch. The batch range and/or per-prompt image count were scheduled across runs until the intended **20,000 SDXL images** were collected. The checked-in notebook shows the generation method and can be adjusted through those configuration variables when another run is needed.

## Dataset assembly

After generation, the outputs from the three sources were collected into the AI-face dataset while retaining their source-level counts:

- 15,000 AdaGAN 2 images
- 15,000 GAN 3 images
- 20,000 SDXL images

The resulting 50,000-image collection was then used as the AI-generated-image component for subsequent training and testing work in this project.

## Reproducibility notes

The SDXL procedure is documented in the included notebook. Exact pixel-for-pixel regeneration is not expected because it uses randomly selected seeds unless those seeds are recorded separately. The AdaGAN 2 and GAN 3 subsets are not reproducible from this directory alone because their external model code, checkpoints, and generation scripts are not stored here.

For future dataset runs, record the upstream GAN repository and checkpoint versions, the seed used for every image, package versions, hardware/runtime details, and final per-source file manifests. Doing so will make the dataset provenance more fully reproducible.
