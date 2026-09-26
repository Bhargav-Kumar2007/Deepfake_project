"""Generate reproducible XAI evidence for the Wide DeepCNN model only.

This script deliberately does not load, score, or describe the local PatchCNN.
It reuses the five attribution implementations in
``faces_deepfake/user/explainability.py``: Grad-CAM, Grad-CAM++, Score-CAM,
Integrated Gradients, and LIME.

Run from the repository root:
    python XAI/xai.py

The default input is the ten supplied images under ``XAI/dataset``.  Results
are written to ``XAI/results`` and include individual overlays, raw heatmaps,
contact sheets, an audit CSV, model provenance JSON, and a Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FACES_DIR = REPO_ROOT / "faces_deepfake"
USER_DIR = FACES_DIR / "user"
MODELS_DIR = FACES_DIR / "models"
CHECKPOINT = MODELS_DIR / "best_wide_model.pth"
DATASET_DIR = SCRIPT_DIR / "dataset"
RESULTS_DIR = SCRIPT_DIR / "results"
METHODS = (
    ("gradcam", "Grad-CAM"),
    ("gradcam_plus_plus", "Grad-CAM++"),
    ("score_cam", "Score-CAM"),
    ("integrated_gradients", "Integrated Gradients"),
    ("lime", "LIME"),
)

# Import the canonical five explainers rather than a duplicate implementation.
for path in (USER_DIR, MODELS_DIR, FACES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import explainability as wide_xai_engine  # noqa: E402
from explainability import (  # noqa: E402
    GradCAM,
    GradCAMPlusPlus,
    IntegratedGradients,
    LIMEExplainer,
    ScoreCAM,
    apply_colormap,
    create_heatmap_overlay,
)
from use_wide_model import load_model, transform  # noqa: E402

# The web explainer's module defaults to 512px, whereas the trained wide-model
# inference path explicitly uses 256px.  Its methods share these module globals,
# so align them before any explainer is instantiated (including LIME's internal
# perturbation transform) to avoid mixed-resolution attributions.
wide_xai_engine.IMAGE_SIZE = 256
wide_xai_engine.transform = transform


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_images(dataset: Path) -> Iterable[Tuple[Path, str]]:
    for class_name in ("real", "fake"):
        folder = dataset / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Required input folder is missing: {folder}")
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                yield image_path, class_name.upper()


def last_conv_layer(model: nn.Module) -> nn.Module:
    layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not layers:
        raise RuntimeError("Wide model has no convolutional layer for CAM attribution.")
    return layers[-1]


def normalize(heatmap: np.ndarray) -> np.ndarray:
    lo, hi = float(heatmap.min()), float(heatmap.max())
    return (heatmap - lo) / (hi - lo) if hi > lo else np.zeros_like(heatmap)


def generate_heatmaps(model: nn.Module, tensor: torch.Tensor, image: Image.Image, target: str) -> Dict[str, np.ndarray]:
    """Run precisely the five techniques defined in explainability.py."""
    target_layer = last_conv_layer(model)
    with GradCAM(model, target_layer) as explainer:
        gradcam = explainer.generate(tensor, target)
    with GradCAMPlusPlus(model, target_layer) as explainer:
        gradcam_pp = explainer.generate(tensor, target)
    scorecam = ScoreCAM(model, target_layer, max_channels=28).generate(tensor, target)
    integrated_gradients = IntegratedGradients(model, steps=20).generate(tensor, target)
    lime = LIMEExplainer(model, num_superpixels=48, num_samples=50).generate(image, tensor, target)
    return {
        "gradcam": normalize(gradcam),
        "gradcam_plus_plus": normalize(gradcam_pp),
        "score_cam": normalize(scorecam),
        "integrated_gradients": normalize(integrated_gradients),
        "lime": normalize(lime),
    }


def label_image(image: Image.Image, title: str) -> Image.Image:
    image = image.convert("RGB")
    band = 30
    canvas = Image.new("RGB", (image.width, image.height + band), "white")
    canvas.paste(image, (0, band))
    ImageDraw.Draw(canvas).text((8, 8), title, fill="black", font=ImageFont.load_default())
    return canvas


def contact_sheet(original: Image.Image, overlays: Dict[str, Image.Image], title: str) -> Image.Image:
    tiles = [label_image(original, "Input image")]
    tiles.extend(label_image(overlays[key], name) for key, name in METHODS)
    tile_w, tile_h = tiles[0].size
    sheet = Image.new("RGB", (tile_w * 3, tile_h * 2 + 34), "white")
    ImageDraw.Draw(sheet).text((8, 8), title, fill="black", font=ImageFont.load_default())
    for index, tile in enumerate(tiles):
        x, y = (index % 3) * tile_w, 34 + (index // 3) * tile_h
        sheet.paste(tile, (x, y))
    return sheet


def write_report(rows: List[Dict[str, str]], provenance: Dict[str, object]) -> None:
    total, correct = len(rows), sum(row["correct"] == "yes" for row in rows)
    content = f"""# Wide Model XAI Evidence

## Scope

This artefact explains **only** `DeepCNN` loaded from
`faces_deepfake/models/best_wide_model.pth`. The local PatchCNN is intentionally
excluded because the benchmark reports show weak generalization for it.

## Reproducibility

- Run timestamp (UTC): `{provenance['generated_at_utc']}`
- Checkpoint SHA-256: `{provenance['checkpoint_sha256']}`
- Device: `{provenance['device']}`
- Input size: `256 x 256` (the wide model's inference preprocessing)
- Attribution target: the wide model's own predicted class
- Methods: Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, LIME
- Inputs: `{total}` (`{correct}` predictions match the supplied folder label)

## Output layout

Each image folder contains `input.png`, five `*_overlay.png` files, five
`*_heatmap.png` files, and `contact_sheet.png`. `xai_manifest.csv` is the
machine-readable prediction/attribution audit trail; `provenance.json` records
the exact checkpoint fingerprint and software/runtime details.

## Important interpretation note

Warm colours show regions that positively support the displayed predicted class,
not ground-truth correctness or causal proof. The five maps are complementary;
they should be assessed together with the prediction probabilities.
"""
    (RESULTS_DIR / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    global RESULTS_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASET_DIR, help="Folder containing real/ and fake/ images.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR, help="Destination for XAI evidence.")
    parser.add_argument("--clean", action="store_true", help="Remove the existing output directory before generating results.")
    args = parser.parse_args()

    RESULTS_DIR = args.output.resolve()
    if args.clean and RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Wide-model checkpoint not found: {CHECKPOINT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(CHECKPOINT), device=device)
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False  # required for Grad-CAM backward hooks
    model.eval()

    provenance: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Wide Model (DeepCNN)",
        "checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "input_size": [256, 256],
        "methods_from": "faces_deepfake/user/explainability.py",
        "methods": [name for _, name in METHODS],
        "target_policy": "model predicted class (REAL when real_probability >= 0.5, else FAKE)",
    }
    (RESULTS_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    rows: List[Dict[str, str]] = []
    images = list(find_images(args.input.resolve()))
    print(f"Running five XAI methods for {len(images)} images on {device}.")
    for index, (image_path, ground_truth) in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.name} ({ground_truth})")
        original = Image.open(image_path).convert("RGB")
        resized = original.resize((256, 256), Image.Resampling.BILINEAR)
        tensor = transform(resized).unsqueeze(0).to(device)
        with torch.no_grad():
            real_probability = float(torch.sigmoid(model(tensor)[0, 0]).item())
        fake_probability = 1.0 - real_probability
        prediction = "REAL" if real_probability >= 0.5 else "FAKE"
        heatmaps = generate_heatmaps(model, tensor, resized, prediction)

        sample_dir = RESULTS_DIR / ground_truth.lower() / image_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        resized.save(sample_dir / "input.png")
        overlays: Dict[str, Image.Image] = {}
        for key, _ in METHODS:
            heatmap = heatmaps[key]
            overlay = create_heatmap_overlay(resized, heatmap, alpha=0.52)
            overlays[key] = overlay
            overlay.save(sample_dir / f"{key}_overlay.png")
            Image.fromarray(apply_colormap(heatmap)).save(sample_dir / f"{key}_heatmap.png")
        contact_sheet(resized, overlays, f"Wide DeepCNN | {prediction} | real={real_probability:.4f}, fake={fake_probability:.4f}").save(sample_dir / "contact_sheet.png")

        row = {
            "source_image": str(image_path.relative_to(REPO_ROOT)),
            "ground_truth_folder": ground_truth,
            "wide_prediction": prediction,
            "real_probability": f"{real_probability:.8f}",
            "fake_probability": f"{fake_probability:.8f}",
            "confidence": f"{max(real_probability, fake_probability):.8f}",
            "correct": "yes" if prediction == ground_truth else "no",
            "evidence_folder": str(sample_dir.relative_to(REPO_ROOT)),
        }
        for key, _ in METHODS:
            row[f"{key}_mean"] = f"{float(heatmaps[key].mean()):.8f}"
            row[f"{key}_max"] = f"{float(heatmaps[key].max()):.8f}"
        rows.append(row)

    with (RESULTS_DIR / "xai_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(rows, provenance)
    print(f"Complete. Evidence saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
