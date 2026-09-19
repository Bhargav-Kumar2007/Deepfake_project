import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Ensure relevant directories are in sys.path for direct script execution
CURRENT_DIR = Path(__file__).resolve().parent
FACES_DIR = CURRENT_DIR.parent
MODELS_DIR = FACES_DIR / "models"

for p in [CURRENT_DIR, MODELS_DIR, FACES_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from basemodel_wide import DeepCNN
except ImportError:
    from models.basemodel_wide import DeepCNN

# ============================================================
# CONFIGURATION
# ============================================================
candidate_paths = [
    MODELS_DIR / "best_wide_model.pth",
    CURRENT_DIR / "best_wide_model.pth",
    FACES_DIR / "best_wide_model.pth",
]
DEFAULT_CHECKPOINT_PATH = str(
    next((p for p in candidate_paths if p.exists()), MODELS_DIR / "best_wide_model.pth")
)
IMAGE_SIZE = 256
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# TRANSFORMS
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Global model cache for lazy loading
_cached_model = None


# ============================================================
# LOAD MODEL
# ============================================================
def load_model(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=DEVICE):
    """
    Loads and returns the DeepCNN model with trained weights.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = DeepCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def get_default_model(device=DEVICE):
    """Returns a cached instance of the loaded model."""
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model(DEFAULT_CHECKPOINT_PATH, device=device)
    return _cached_model


# ============================================================
# INFERENCE FUNCTIONS
# ============================================================
def predict_batch_wide(image_paths, batch_size=BATCH_SIZE, model=None, device=DEVICE):
    """
    Predict on a batch of images using DeepCNN (Wide Model).

    Args:
        image_paths: list of image file paths
        batch_size: number of images per batch
        model: optional preloaded DeepCNN model instance
        device: torch device

    Returns:
        list of dicts with prediction results
    """
    if model is None:
        model = get_default_model(device)

    results = []
    total_images = len(image_paths)

    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []

        # Load and transform images
        for img_path in batch_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            batch_tensors.append(img_tensor)

        # Stack into a single batch tensor [B, 3, 256, 256]
        batch_tensor = torch.stack(batch_tensors).to(device)

        # Forward pass
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

        # Handle single image case (squeeze may make it 0-d scalar)
        if len(batch_paths) == 1:
            probs = np.atleast_1d(probs)

        # Store results
        for img_path, prob_val in zip(batch_paths, probs):
            prob = float(prob_val)
            label = "REAL" if prob >= 0.5 else "FAKE"
            confidence = prob if prob >= 0.5 else 1.0 - prob

            results.append({
                "filename": os.path.basename(img_path),
                "image_path": img_path,
                "probability": prob,
                "real_probability": prob,
                "fake_probability": 1.0 - prob,
                "prediction": label,
                "confidence": confidence
            })

    return results


# Alias matching predict_batch
predict_batch = predict_batch_wide


def predict_image(image_path, model=None, device=DEVICE):
    """
    Predict deepfake probability on a single image.

    Args:
        image_path: path to image file
        model: optional preloaded DeepCNN model instance
        device: torch device

    Returns:
        dict with prediction results: filename, image_path, probability, prediction, confidence
    """
    results = predict_batch_wide([image_path], batch_size=1, model=model, device=device)
    return results[0]


# Alias for compatibility with colab.txt Cell 12
predict_image_wide = predict_image


def mainfunc(file_path):
    """
    Convenience wrapper returning probability for single image.
    """
    res = predict_image(file_path)
    return res["probability"]


# ============================================================
# FILE SELECTION & CLI ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image for Deep CNN (Wide Model)",
        filetypes=[
            ("Image Files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        print("No file selected.")
        sys.exit()

    print(f"\nAnalyzing: {file_path}")
    print(f"Device: {DEVICE}")
    res = predict_image(file_path)

    print("\n" + "=" * 50)
    print("WIDE MODEL (DeepCNN) PREDICTION RESULT")
    print("=" * 50)
    print(f"Filename:         {res['filename']}")
    print(f"Prediction:       {res['prediction']}")
    print(f"Confidence:       {res['confidence'] * 100:.2f}%")
    print(f"Real Probability: {res['real_probability']:.4f}")
    print(f"Fake Probability: {res['fake_probability']:.4f}")
    if res['prediction'] == "REAL":
        print("Status:           🟢 REAL PHOTO")
    else:
        print("Status:           🔴 AI GENERATED / DEEPFAKE")
    print("=" * 50)
