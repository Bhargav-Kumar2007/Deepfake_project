"""Explainability (XAI) engine for DeepFake Detector models.

Provides 5 distinct attribution methods:
1. Grad-CAM (Gradient-weighted Class Activation Mapping)
2. Grad-CAM++ (Second/third-order gradient weighting for fine-grained localization)
3. Score-CAM (Gradient-free activation masking with model score perturbation)
4. Integrated Gradients (Path integral of gradients from baseline to input)
5. LIME (Superpixel perturbation with weighted linear surrogate explanation)
"""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Ensure relevant directories are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
FACES_DIR = CURRENT_DIR.parent
MODELS_DIR = FACES_DIR / "models"

for p in [CURRENT_DIR, MODELS_DIR, FACES_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from basemodel_wide import DeepCNN
    from use_wide_model import (
        DEFAULT_CHECKPOINT_PATH as WIDE_CHECKPOINT_PATH,
        load_model as load_wide_model,
    )
except ImportError:
    from models.basemodel_wide import DeepCNN
    from user.use_wide_model import (
        DEFAULT_CHECKPOINT_PATH as WIDE_CHECKPOINT_PATH,
        load_model as load_wide_model,
    )

IMAGE_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard normalization for DeepCNN on 512x512 transformed images
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

_cached_xai_model = None


def get_xai_model(checkpoint_path=WIDE_CHECKPOINT_PATH, device=DEVICE):
    """
    Loads and caches DeepCNN with in-place ReLUs disabled so backward hooks work cleanly.
    """
    global _cached_xai_model
    if _cached_xai_model is None:
        model = load_wide_model(checkpoint_path, device=device)
        # Disable inplace in all ReLU layers to avoid backward hook memory conflicts
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        model.eval()
        _cached_xai_model = model
    return _cached_xai_model


# ============================================================
# COLORMAP & OVERLAY UTILITIES (PURE NUMPY - ZERO MATPLOTLIB)
# ============================================================
def _jet_colormap_lut() -> np.ndarray:
    """Precompute 256x3 JET colormap: 0.0 is Deep Blue (Real), 1.0 is Vivid Red (AI/Fake)."""
    points = [0.0, 0.25, 0.50, 0.75, 1.0]
    r_pts = [0, 0, 60, 245, 235]
    g_pts = [20, 180, 210, 130, 20]
    b_pts = [220, 240, 60, 10, 15]
    x = np.linspace(0.0, 1.0, 256)
    r = np.interp(x, points, r_pts)
    g = np.interp(x, points, g_pts)
    b = np.interp(x, points, b_pts)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


JET_LUT = _jet_colormap_lut()


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """
    Converts a 2D float heatmap in [0.0, 1.0] to an RGB numpy array (uint8) using Jet colormap.
    """
    norm = np.clip(heatmap, 0.0, 1.0)
    indices = (norm * 255).astype(np.uint8)
    return JET_LUT[indices]


def create_heatmap_overlay(
    original_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.55
) -> Image.Image:
    """
    Overlays a 2D float heatmap on a PIL Image.
    """
    orig_resized = original_pil.resize((heatmap.shape[1], heatmap.shape[0]), Image.Resampling.BILINEAR).convert("RGB")
    orig_arr = np.array(orig_resized, dtype=np.float32)
    color_heat = apply_colormap(heatmap).astype(np.float32)

    blended = (1.0 - alpha) * orig_arr + alpha * color_heat
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def pil_to_base64(img: Image.Image, format: str = "JPEG", quality: int = 88) -> str:
    """Encodes a PIL Image to a base64 data URL."""
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


# ============================================================
# 1. GRAD-CAM
# ============================================================
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = []

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __enter__(self):
        self._handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(self, input_tensor: torch.Tensor, target_class: str = "FAKE") -> np.ndarray:
        """
        Generates Grad-CAM heatmap.
        Args:
            input_tensor: [1, 3, H, W]
            target_class: 'FAKE' or 'REAL'
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)  # [1, 1]
        raw_logit = logits[0, 0]

        # In this model: high logit -> REAL (1), low logit -> FAKE (0).
        # To explain FAKE, target is -raw_logit; to explain REAL, target is +raw_logit.
        target_score = -raw_logit if target_class == "FAKE" else raw_logit
        target_score.backward()

        # Global average pooling of gradients: [1, C, 1, 1]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam


# ============================================================
# 2. GRAD-CAM++
# ============================================================
class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = []

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __enter__(self):
        self._handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(self, input_tensor: torch.Tensor, target_class: str = "FAKE") -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        raw_logit = logits[0, 0]
        target_score = -raw_logit if target_class == "FAKE" else raw_logit
        target_score.backward()

        # Gradients: [1, C, H, W], Activations: [1, C, H, W]
        grads = self.gradients
        acts = self.activations

        grads_pow2 = grads.pow(2)
        grads_pow3 = grads.pow(3)
        sum_acts_grads_pow3 = torch.sum(acts * grads_pow3, dim=(2, 3), keepdim=True)

        eps = 1e-7
        denom = 2.0 * grads_pow2 + sum_acts_grads_pow3
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom) * eps)

        aij = grads_pow2 / denom
        aij = torch.where(grads != 0.0, aij, torch.zeros_like(aij))

        weights = torch.sum(aij * F.relu(grads), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam


# ============================================================
# 3. SCORE-CAM (FAST BATCHED GRADIENT-FREE ATTRIBUTION)
# ============================================================
class ScoreCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, max_channels: int = 36):
        self.model = model
        self.target_layer = target_layer
        self.max_channels = max_channels
        self.activations = None
        self._handle = None

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def generate(self, input_tensor: torch.Tensor, target_class: str = "FAKE") -> np.ndarray:
        self._handle = self.target_layer.register_forward_hook(self._save_activation)
        try:
            with torch.no_grad():
                baseline_logits = self.model(input_tensor)
                baseline_logit = baseline_logits[0, 0].item()
                acts = self.activations[0]  # [C, H, W]

                # Select top-K most energetic channels to keep execution under 250ms
                channel_energy = torch.mean(acts, dim=(1, 2))
                num_c = min(self.max_channels, acts.shape[0])
                top_indices = torch.topk(channel_energy, k=num_c).indices

                selected_acts = acts[top_indices]  # [K, H, W]

                # Normalize each activation map to [0, 1]
                mins = selected_acts.amin(dim=(1, 2), keepdim=True)
                maxs = selected_acts.amax(dim=(1, 2), keepdim=True)
                diff = maxs - mins
                diff[diff == 0] = 1.0
                norm_acts = (selected_acts - mins) / diff

                # Upsample masks to input image dimensions [K, 1, 256, 256]
                norm_acts_up = F.interpolate(
                    norm_acts.unsqueeze(1),
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False
                )

                # Mask the input image: [K, 3, 256, 256]
                masked_inputs = input_tensor * norm_acts_up

                # Forward pass all masked images in batches
                logits = []
                batch_size = 18
                for b_start in range(0, num_c, batch_size):
                    batch = masked_inputs[b_start:b_start + batch_size]
                    out = self.model(batch)
                    logits.append(out.squeeze(-1))

                all_logits = torch.cat(logits, dim=0)  # [K]
                if target_class == "FAKE":
                    scores = -all_logits
                else:
                    scores = all_logits

                # Softmax weights over channel scores
                weights = F.softmax(scores, dim=0).view(num_c, 1, 1)

                cam = torch.sum(weights * norm_acts_up.squeeze(1), dim=0)  # [256, 256]
                cam = F.relu(cam).cpu().numpy()

                cam_min, cam_max = cam.min(), cam.max()
                if cam_max > cam_min:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                else:
                    cam = np.zeros_like(cam)
                return cam
        finally:
            if self._handle:
                self._handle.remove()


# ============================================================
# 4. INTEGRATED GRADIENTS
# ============================================================
class IntegratedGradients:
    def __init__(self, model: nn.Module, steps: int = 24):
        self.model = model
        self.steps = steps

    def generate(self, input_tensor: torch.Tensor, target_class: str = "FAKE") -> np.ndarray:
        # Baseline: Black image in normalized coordinates (-1.0 since mean=0.5, std=0.5)
        baseline = torch.full_like(input_tensor, -1.0)

        # Scale inputs along the straight line from baseline to input
        alphas = torch.linspace(1.0 / self.steps, 1.0, self.steps, device=input_tensor.device)
        scaled_inputs = [baseline + alpha * (input_tensor - baseline) for alpha in alphas]
        batch_inputs = torch.cat(scaled_inputs, dim=0).requires_grad_(True)  # [steps, 3, 256, 256]

        self.model.zero_grad()
        logits = self.model(batch_inputs)  # [steps, 1]

        if target_class == "FAKE":
            target_scores = -logits.sum()
        else:
            target_scores = logits.sum()

        target_scores.backward()

        grads = batch_inputs.grad.detach()  # [steps, 3, 256, 256]
        avg_grads = torch.mean(grads, dim=0, keepdim=True)  # [1, 3, 256, 256]

        # Path integral attribution: (x - x0) * avg_grad
        attribution = (input_tensor - baseline) * avg_grads  # [1, 3, 256, 256]

        # Channel-wise magnitude (L2 norm across RGB)
        attribution_map = torch.norm(attribution, dim=1).squeeze().cpu().numpy()

        # Gaussian-like spatial smoothing using separable 1D convolution
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
        kernel /= kernel.sum()
        smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=attribution_map)
        smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=smoothed)

        # Clip high outliers (99th percentile) to enhance perceptual clarity
        p99 = np.percentile(smoothed, 99.0)
        p01 = np.percentile(smoothed, 1.0)
        if p99 > p01:
            smoothed = np.clip((smoothed - p01) / (p99 - p01), 0.0, 1.0)
        else:
            smoothed = np.zeros_like(smoothed)
        return smoothed


# ============================================================
# 5. LIME (LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS)
# ============================================================
class LIMEExplainer:
    def __init__(self, model: nn.Module, num_superpixels: int = 42, num_samples: int = 70):
        self.model = model
        self.num_superpixels = num_superpixels
        self.num_samples = num_samples

    def _segment_image(self, img_pil: Image.Image) -> np.ndarray:
        """
        Segments image into compact spatial superpixels using MiniBatchKMeans on (Y, X, R, G, B).
        Extremely fast (~30ms) and avoids native C-extension compilation issues.
        """
        img_resized = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0  # [256, 256, 3]

        h, w, _ = arr.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        # Normalize spatial coordinates to [0, 1] with weight balancing color vs space
        y_norm = y_coords / float(h)
        x_norm = x_coords / float(w)

        # Feature vector per pixel: [Y*1.5, X*1.5, R, G, B]
        features = np.stack([y_norm * 1.5, x_norm * 1.5, arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]], axis=-1)
        features = features.reshape(-1, 5)

        kmeans = MiniBatchKMeans(
            n_clusters=self.num_superpixels,
            batch_size=2048,
            random_state=42,
            n_init=1,
            max_iter=15
        )
        labels = kmeans.fit_predict(features)
        return labels.reshape(h, w)

    def generate(
        self,
        img_pil: Image.Image,
        input_tensor: torch.Tensor,
        target_class: str = "FAKE"
    ) -> np.ndarray:
        segments = self._segment_image(img_pil)
        num_segments = np.max(segments) + 1

        # Superpixel average colors for replacement
        img_resized = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        img_arr = np.array(img_resized, dtype=np.float32)

        sp_means = {}
        for sp_id in range(num_segments):
            mask = (segments == sp_id)
            if np.any(mask):
                sp_means[sp_id] = img_arr[mask].mean(axis=0)
            else:
                sp_means[sp_id] = np.array([128.0, 128.0, 128.0], dtype=np.float32)

        # Generate binary perturbations Z [num_samples, num_segments]
        # Sample 0 is original image (all 1s)
        np.random.seed(42)
        Z = np.random.binomial(1, 0.5, size=(self.num_samples, num_segments)).astype(np.float32)
        Z[0, :] = 1.0

        # Construct perturbed image tensors
        perturbed_tensors = []
        for i in range(self.num_samples):
            pert_arr = img_arr.copy()
            zeros = np.where(Z[i] == 0)[0]
            for sp_id in zeros:
                pert_arr[segments == sp_id] = sp_means[sp_id]
            # Convert to PIL and apply tensor transform
            pert_pil = Image.fromarray(np.clip(pert_arr, 0, 255).astype(np.uint8))
            perturbed_tensors.append(transform(pert_pil))

        batch_perturbed = torch.stack(perturbed_tensors).to(input_tensor.device)

        # Model inference on perturbed batch
        with torch.no_grad():
            preds = []
            for b_idx in range(0, self.num_samples, 24):
                sub_batch = batch_perturbed[b_idx:b_idx + 24]
                logits = self.model(sub_batch).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
            all_real_probs = np.concatenate(preds)

        if target_class == "FAKE":
            y_target = 1.0 - all_real_probs  # Fake probability
        else:
            y_target = all_real_probs  # Real probability

        # Distances and exponential kernel weights
        # Cosine / Euclidean distance between Z and [1, 1, ..., 1]
        distances = np.linalg.norm(Z - 1.0, axis=1) / np.sqrt(num_segments)
        sigma = 0.5
        kernel_weights = np.exp(-(distances ** 2) / (sigma ** 2))

        # Fit weighted linear surrogate (Ridge regression)
        reg = Ridge(alpha=1.0, fit_intercept=True)
        reg.fit(Z, y_target, sample_weight=kernel_weights)

        sp_weights = reg.coef_  # [num_segments]

        # Map learned weights back to pixel mask
        heatmap = np.zeros_like(segments, dtype=np.float32)
        for sp_id in range(num_segments):
            heatmap[segments == sp_id] = sp_weights[sp_id]

        # Keep positive contributions supporting the verdict
        heatmap = np.maximum(heatmap, 0.0)
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap)
        return heatmap


# ============================================================
# MASTER EXPLAINABILITY PIPELINE
# ============================================================
def generate_all_explanations(
    image_input: Union[str, Image.Image],
    target_class: Optional[str] = None,
    device: torch.device = DEVICE,
    alpha_blend: float = 0.52
) -> Dict:
    """
    Executes all 5 XAI methods on the provided image and returns base64 overlays and descriptions.

    Args:
        image_input: File path or PIL Image instance.
        target_class: 'FAKE' or 'REAL'. If None, automatically set to model's predicted class.
        device: Torch device (cuda/cpu).
        alpha_blend: Alpha transparency for heatmap overlay on original photo.

    Returns:
        Dictionary containing predicted verdict, class probabilities, heatmaps, and descriptions.
    """
    if isinstance(image_input, (str, Path)):
        orig_pil = Image.open(image_input).convert("RGB")
    else:
        orig_pil = image_input.convert("RGB")

    # Resize to exact 512x512 square for both model and heatmap alignment
    transformed_pil = orig_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    transformed_b64 = pil_to_base64(transformed_pil)

    input_tensor = transform(transformed_pil).unsqueeze(0).to(device)
    model = get_xai_model(device=device)

    # 1. Forward pass to determine verdict if target_class is None
    with torch.no_grad():
        initial_logit = model(input_tensor)[0, 0].item()
        real_prob = float(torch.sigmoid(torch.tensor(initial_logit)).item())
        fake_prob = 1.0 - real_prob

    if target_class is None:
        target_class = "REAL" if real_prob >= 0.5 else "FAKE"

    # Target convolutional layer for CAM methods (dynamically find last Conv2d)
    target_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            target_conv = module
    if target_conv is None:
        raise ValueError("Could not find a Conv2d layer in DeepCNN.")

    # Run the 5 explainers
    # 1. Grad-CAM
    with GradCAM(model, target_conv) as gcam:
        heat_gradcam = gcam.generate(input_tensor, target_class=target_class)

    # 2. Grad-CAM++
    with GradCAMPlusPlus(model, target_conv) as gcampp:
        heat_gradcampp = gcampp.generate(input_tensor, target_class=target_class)

    # 3. Score-CAM
    score_cam = ScoreCAM(model, target_conv, max_channels=28)
    heat_scorecam = score_cam.generate(input_tensor, target_class=target_class)

    # 4. Integrated Gradients
    ig = IntegratedGradients(model, steps=20)
    heat_ig = ig.generate(input_tensor, target_class=target_class)

    # 5. LIME
    lime = LIMEExplainer(model, num_superpixels=48, num_samples=50)
    heat_lime = lime.generate(transformed_pil, input_tensor, target_class=target_class)

    # Build response dictionary
    methods_heatmaps = {
        "gradcam": heat_gradcam,
        "gradcam_plus_plus": heat_gradcampp,
        "score_cam": heat_scorecam,
        "integrated_gradients": heat_ig,
        "lime": heat_lime,
    }

    descriptions = {
        "gradcam": {
            "name": "Grad-CAM",
            "concept": "Gradient-weighted Class Activation Mapping",
            "description": "Computes the coarse activation map from the final convolutional layer weighted by first-order gradients. Highlights broad regions (eyes, mouth, contour) driving the verdict.",
        },
        "gradcam_plus_plus": {
            "name": "Grad-CAM++",
            "concept": "Generalized CAM with Higher-Order Gradients",
            "description": "Utilizes second- and third-order positive partial derivatives. Offers sharper, fine-grained localization for subtle deepfake blending seams and micro-artifacts.",
        },
        "score_cam": {
            "name": "Score-CAM",
            "concept": "Gradient-Free Activation Mask Attribution",
            "description": "Bypasses gradient vanishing or saturation by masking the input face with feature activations and measuring output confidence changes. Highly robust against adversarial noise.",
        },
        "integrated_gradients": {
            "name": "Integrated Gradients",
            "concept": "Path Integral Gradient Attribution",
            "description": "Integrates gradients along the interpolation path from a neutral baseline to the input face. Accurately pinpoints pixel-level sensitive facial features and texture boundaries.",
        },
        "lime": {
            "name": "LIME",
            "concept": "Local Interpretable Model-agnostic Explanations",
            "description": "Segments the face into superpixels, creates perturbed samples, and fits a locally weighted surrogate linear model to identify the most decisive facial patches.",
        },
    }

    overlays = {}
    raw_heatmaps = {}

    for key, hmap in methods_heatmaps.items():
        # Overlay blended on 512x512 transformed photo
        overlay_pil = create_heatmap_overlay(transformed_pil, hmap, alpha=alpha_blend)
        overlays[key] = pil_to_base64(overlay_pil)

        # Pure colormap heatmap (512x512)
        pure_heat_pil = Image.fromarray(apply_colormap(hmap))
        raw_heatmaps[key] = pil_to_base64(pure_heat_pil)

    return {
        "target_class": target_class,
        "real_probability": real_prob,
        "fake_probability": fake_prob,
        "transformed_image": transformed_b64,
        "overlays": overlays,
        "raw_heatmaps": raw_heatmaps,
        "descriptions": descriptions,
    }



# ============================================================
# SIMPLE PATCH MODEL HEATMAP (LOCAL MODEL EXPLAINABILITY)
# ============================================================
def generate_patch_heatmap(
    patch_probs: List[float],
    image_input: Optional[Union[str, Image.Image]] = None,
    image_size: int = 512,
    patch_size: int = 112,
    overlap: int = 20,
    target: str = "FAKE",
    alpha_blend: float = 0.55
) -> Dict:
    """
    Constructs a 2D spatial probability heatmap for the Local Patch model.
    Maps all 36 patch scores to their 512x512 coordinates with smooth overlap blending.
    """
    stride = patch_size - overlap
    positions = []
    pos = 0
    while pos + patch_size <= image_size:
        positions.append(pos)
        if pos + patch_size == image_size:
            break
        pos += stride
        if pos + patch_size > image_size:
            positions.append(image_size - patch_size)
            break

    heat_acc = np.zeros((image_size, image_size), dtype=np.float32)
    count_acc = np.zeros((image_size, image_size), dtype=np.float32)

    patch_grid = []
    k = 0
    for r_idx, y in enumerate(positions):
        for c_idx, x in enumerate(positions):
            if k < len(patch_probs):
                real_p = float(patch_probs[k])
                fake_p = float(1.0 - real_p)
                # Exact color coordination: 1.0 = AI/Fake (Red), 0.0 = Real (Blue)
                score = fake_p

                heat_acc[y:y + patch_size, x:x + patch_size] += score
                count_acc[y:y + patch_size, x:x + patch_size] += 1.0

                patch_grid.append({
                    "index": k + 1,
                    "row": r_idx + 1,
                    "col": c_idx + 1,
                    "x": int(x),
                    "y": int(y),
                    "size": int(patch_size),
                    "real_probability": round(real_p, 4),
                    "fake_probability": round(fake_p, 4),
                    "verdict": "REAL" if real_p >= 0.5 else "FAKE",
                })
                k += 1

    count_acc[count_acc == 0] = 1.0
    patch_heatmap = heat_acc / count_acc

    pure_heat_pil = Image.fromarray(apply_colormap(patch_heatmap))
    raw_b64 = pil_to_base64(pure_heat_pil)

    overlay_b64 = None
    if image_input is not None:
        if isinstance(image_input, (str, Path)):
            orig_pil = Image.open(image_input).convert("RGB")
        else:
            orig_pil = image_input.convert("RGB")
        transformed_pil = orig_pil.resize((image_size, image_size), Image.Resampling.BILINEAR)
        overlay_pil = create_heatmap_overlay(transformed_pil, patch_heatmap, alpha=alpha_blend)
        overlay_b64 = pil_to_base64(overlay_pil)

    return {
        "raw_heatmap": raw_b64,
        "overlay": overlay_b64,
        "patch_grid": patch_grid,
        "patch_count": len(patch_grid),
    }


# ============================================================
# CLI ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate XAI heatmaps for DeepFake Detector.")
    parser.add_argument("--image", type=str, required=True, help="Path to input face image.")
    parser.add_argument("--output", type=str, default="xai_results", help="Output directory.")
    parser.add_argument("--target", type=str, default=None, choices=["FAKE", "REAL"], help="Target class to explain.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalyzing: {args.image} on {DEVICE}...")
    results = generate_all_explanations(args.image, target_class=args.target, device=DEVICE)

    print(f"Target Explanation: {results['target_class']}")
    print(f"Real Probability:   {results['real_probability']:.4f}")
    print(f"Fake Probability:   {results['fake_probability']:.4f}")

    orig_pil = Image.open(args.image).convert("RGB")
    for key, desc in results["descriptions"].items():
        overlay_b64 = results["overlays"][key].split(",")[1]
        out_file = out_dir / f"{key}_overlay.jpg"
        with open(out_file, "wb") as f:
            f.write(base64.b64decode(overlay_b64))
        print(f"Saved: {out_file} ({desc['name']})")

    print(f"\nAll 5 heatmaps successfully saved to: {out_dir.resolve()}\n")
