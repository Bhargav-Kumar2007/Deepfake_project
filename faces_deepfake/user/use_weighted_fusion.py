import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import torch

# Ensure all relevant directories are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
FACES_DIR = CURRENT_DIR.parent
MODELS_DIR = FACES_DIR / "models"
OTHERS_DIR = FACES_DIR / "others"

for p in [CURRENT_DIR, MODELS_DIR, FACES_DIR, OTHERS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Import models and inference functions
from basemodel_wide import DeepCNN
from basemodel_local import PatchCNN
from use_wide_model import (
    load_model as load_wide_model,
    predict_image as predict_image_wide,
    DEFAULT_CHECKPOINT_PATH as WIDE_CHECKPOINT_PATH
)
from use_local_model import (
    load_model as load_local_model,
    predict_image as predict_image_patch,
    DEFAULT_CHECKPOINT_PATH as LOCAL_CHECKPOINT_PATH
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset-wide mean-confidence ratio supplied for the two base CNNs.
# The scores below are always *real* probabilities.
WIDE_WEIGHT = 1.0072312107559
PATCH_WEIGHT = 1.0

# Model singletons for cached loading
_cached_wide_model = None
_cached_local_model = None
def get_models(device=DEVICE):
    """
    Loads and caches the Wide CNN and Patch CNN.
    """
    global _cached_wide_model, _cached_local_model

    if _cached_wide_model is None:
        print(f"[INIT] Loading Wide CNN on {device}...")
        _cached_wide_model = load_wide_model(WIDE_CHECKPOINT_PATH, device=device)

    if _cached_local_model is None:
        print(f"[INIT] Loading Patch CNN on {device}...")
        _cached_local_model = load_local_model(LOCAL_CHECKPOINT_PATH, device=device)

    return _cached_wide_model, _cached_local_model


def predict_image(image_path, device=DEVICE):
    """
    Predict deepfake probability with a weighted mean of Wide + Patch CNN scores.

    Args:
        image_path: path to image file
        device: torch device (cuda / cpu)

    Returns:
        dict containing weighted-fusion prediction, probabilities, and sub-model details.
    """
    wide_model, patch_model = get_models(device)

    # 1. Run Wide Model (Global view)
    wide_res = predict_image_wide(image_path, model=wide_model, device=device)

    # 2. Run Patch Model (Local texture view)
    local_res = predict_image_patch(image_path, model=patch_model, device=device)

    # 3. Deterministic weighted fusion of Wide + Patch CNNs.
    real_prob = float(
        (WIDE_WEIGHT * wide_res["real_probability"] + PATCH_WEIGHT * local_res["real_probability"])
        / (WIDE_WEIGHT + PATCH_WEIGHT)
    )
    fake_prob = 1.0 - real_prob
    prediction = "REAL" if real_prob >= 0.5 else "FAKE"
    confidence = real_prob if real_prob >= 0.5 else fake_prob

    return {
        "filename": os.path.basename(image_path),
        "image_path": image_path,
        "prediction": prediction,
        "confidence": confidence,
        "real_probability": real_prob,
        "fake_probability": fake_prob,
        "wide_model": {
            "prediction": wide_res["prediction"],
            "confidence": wide_res["confidence"],
            "real_probability": wide_res["real_probability"],
            "fake_probability": wide_res["fake_probability"]
        },
        "local_model": {
            "prediction": local_res["prediction"],
            "confidence": local_res["confidence"],
            "mean": local_res["mean"],
            "real_probability": local_res["real_probability"],
            "fake_probability": local_res["fake_probability"],
            "std_dev": local_res["std_dev"],
            "variance": local_res["variance"],
            "patch_count": local_res["patch_count"],
            "patch_probs": local_res.get("patch_probs", [])
        },
        "fusion": {
            "method": "weighted mean of base-model real probabilities",
            "wide_weight": WIDE_WEIGHT,
            "patch_weight": PATCH_WEIGHT
        }
    }


def predict_batch(image_paths, device=DEVICE):
    """Predict weighted-fusion results for multiple images."""
    return [predict_image(p, device=device) for p in image_paths]


def mainfunc(file_path):
    """
    Compatibility wrapper returning (real_probability, is_real_binary).
    """
    res = predict_image(file_path)
    is_real = 1.0 if res["prediction"] == "REAL" else 0.0
    return res["real_probability"], is_real


# ============================================================
# FILE SELECTION & CLI ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image for DeepFake Weighted Fusion Analysis",
        filetypes=[
            ("Image Files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        print("No file selected.")
        sys.exit()

    print(f"\nAnalyzing: {file_path}")
    print(f"Inference Device: {DEVICE}")

    res = predict_image(file_path)

    print("\n" + "=" * 65)
    print("      DEEPFAKE TWO-CNN WEIGHTED FUSION ANALYSIS REPORT")
    print("=" * 65)
    print(f"File: {res['filename']}")
    print("-" * 65)
    print("1. GLOBAL WIDE MODEL (Full Face Composition):")
    print(f"   Verdict:          {res['wide_model']['prediction']}")
    print(f"   Confidence:       {res['wide_model']['confidence'] * 100:.2f}%")
    print(f"   Real Probability: {res['wide_model']['real_probability']:.4f}")

    print("\n2. LOCAL PATCH MODEL (Micro Texture & Artifacts):")
    print(f"   Verdict:          {res['local_model']['prediction']}")
    print(f"   Confidence:       {res['local_model']['confidence'] * 100:.2f}%")
    print(f"   Mean Score:       {res['local_model']['mean']:.4f}")
    print(f"   Patch Std Dev:    {res['local_model']['std_dev']:.4f}")
    print(f"   Total Patches:    {res['local_model']['patch_count']}")

    print("\n3. FINAL WEIGHTED FUSION DECISION:")
    print(f"   VERDICT:          {res['prediction']}")
    print(f"   CONFIDENCE:       {res['confidence'] * 100:.2f}%")
    print(f"   Real Probability: {res['real_probability']:.4f}")
    print(f"   Fake Probability: {res['fake_probability']:.4f}")
    print("-" * 65)

    if res['prediction'] == "REAL":
        print("   STATUS:           🟢 REAL PHOTO")
    else:
        print("   STATUS:           🔴 AI GENERATED / DEEPFAKE")
    print("=" * 65)
