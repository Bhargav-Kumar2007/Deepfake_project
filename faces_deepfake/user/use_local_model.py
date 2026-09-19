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
    from basemodel_local import PatchCNN
except ImportError:
    from models.basemodel_local import PatchCNN

# ============================================================
# CONFIGURATION
# ============================================================
candidate_paths = [
    MODELS_DIR / "best_local_model.pth",
    CURRENT_DIR / "best_local_model.pth",
    FACES_DIR / "best_local_model.pth",
]
DEFAULT_CHECKPOINT_PATH = str(
    next((p for p in candidate_paths if p.exists()), MODELS_DIR / "best_local_model.pth")
)
IMAGE_SIZE = 512
PATCH_SIZE = 112
OVERLAP = 20
BATCH_SIZE = 32
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
# PATCH EXTRACTION
# ============================================================
def extract_patches_batch(images, patch_size=PATCH_SIZE, overlap=OVERLAP):
    """
    Extract patches from a batch of images.

    Args:
        images: [B, 3, H, W] tensor
        patch_size: size of each patch (default 112)
        overlap: overlap between patches (default 20)

    Returns:
        List of [B, 3, patch_size, patch_size] tensors (one per patch position)
    """
    _, _, H, W = images.shape
    stride = patch_size - overlap

    positions = []
    pos = 0
    while pos + patch_size <= H:
        positions.append(pos)
        if pos + patch_size == H:
            break
        pos += stride
        if pos + patch_size > H:
            positions.append(H - patch_size)
            break

    patches = []
    for y in positions:
        for x in positions:
            patch = images[:, :, y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    return patches


# ============================================================
# LOAD MODEL
# ============================================================
def load_model(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=DEVICE):
    """
    Loads and returns the PatchCNN model with trained weights.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = PatchCNN().to(device)
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
def predict_batch_patch(image_paths, batch_size=BATCH_SIZE, model=None, device=DEVICE):
    """
    Predict on a batch of images using Patch CNN.

    Args:
        image_paths: list of image paths
        batch_size: number of images per batch
        model: optional preloaded PatchCNN model instance
        device: torch device

    Returns:
        list of dicts with prediction results and statistics
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

        # Stack into a single batch tensor [B, 3, 512, 512]
        batch_tensor = torch.stack(batch_tensors).to(device)
        current_batch_size = batch_tensor.size(0)

        # Extract patches: list of [B, 3, 112, 112] tensors
        patches = extract_patches_batch(batch_tensor, patch_size=PATCH_SIZE, overlap=OVERLAP)

        # Process all patches through the model
        patch_probs_list = []
        with torch.no_grad():
            for patch in patches:
                logits = model(patch)  # [B, 1]
                probs = torch.sigmoid(logits).squeeze(1)  # [B]
                patch_probs_list.append(probs.cpu().numpy())

        # Stack patch probabilities: [num_patches, B]
        all_patch_probs = np.stack(patch_probs_list, axis=0)  # [num_patches, B]

        # Calculate statistics per image
        for j in range(current_batch_size):
            img_probs = all_patch_probs[:, j]  # [num_patches]

            mean = float(np.mean(img_probs))
            variance = float(np.var(img_probs))
            std_dev = float(np.std(img_probs))

            prediction = "REAL" if mean >= 0.5 else "FAKE"
            confidence = mean if mean >= 0.5 else 1.0 - mean

            results.append({
                "filename": os.path.basename(batch_paths[j]),
                "image_path": batch_paths[j],
                "prediction": prediction,
                "confidence": confidence,
                "mean": mean,
                "real_probability": mean,
                "fake_probability": 1.0 - mean,
                "variance": variance,
                "std_dev": std_dev,
                "patch_count": len(patches),
                "patch_probs": img_probs.tolist()
            })

    return results


# Alias matching predict_batch
predict_batch = predict_batch_patch


def predict_image(image_path, model=None, device=DEVICE):
    """
    Predict deepfake probability on a single image using Patch CNN.

    Args:
        image_path: path to image file
        model: optional preloaded PatchCNN model instance
        device: torch device

    Returns:
        dict with prediction results: filename, prediction, confidence, mean, variance, std_dev, patch_count
    """
    results = predict_batch_patch([image_path], batch_size=1, model=model, device=device)
    return results[0]


# Alias for compatibility with colab.txt Cell 12
predict_image_patch = predict_image


def mainfunc(file_path):
    """
    Convenience wrapper returning (mean_prob, is_real) for compatibility.
    """
    res = predict_image(file_path)
    is_real = 1.0 if res["prediction"] == "REAL" else 0.0
    return res["mean"], is_real


# ============================================================
# FILE SELECTION & CLI ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image for Patch CNN (Local Model)",
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
    print("LOCAL MODEL (PatchCNN) PREDICTION RESULT")
    print("=" * 50)
    print(f"Filename:               {res['filename']}")
    print(f"Prediction:             {res['prediction']}")
    print(f"Confidence:             {res['confidence'] * 100:.2f}%")
    print(f"Real Probability:       {res['real_probability']:.4f}")
    print(f"Fake Probability:       {res['fake_probability']:.4f}")
    print(f"Standard Deviation:     {res['std_dev']:.4f}")
    print(f"Variance:               {res['variance']:.4f}")
    print(f"Total Patches Evaluated:{res['patch_count']}")
    if res['prediction'] == "REAL":
        print("Status:                 🟢 REAL PHOTO")
    else:
        print("Status:                 🔴 AI GENERATED / DEEPFAKE")
    print("=" * 50)
