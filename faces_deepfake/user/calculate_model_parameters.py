"""Report parameters and runtime metadata for the two-CNN detector pipeline.

Usage:
    python faces_deepfake/user/calculate_model_parameters.py
    python faces_deepfake/user/calculate_model_parameters.py --json
    python faces_deepfake/user/calculate_model_parameters.py --save-report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

USER_DIR = Path(__file__).resolve().parent
FACES_DIR = USER_DIR.parent
MODELS_DIR = FACES_DIR / "models"
OTHERS_DIR = FACES_DIR / "others"
LOCAL_CHECKPOINT_PATH = MODELS_DIR / "best_local_model.pth"
WIDE_CHECKPOINT_PATH = MODELS_DIR / "best_wide_model.pth"
METRICS_PATH = OTHERS_DIR / "model_metrics.csv"
WIDE_WEIGHT = 1.0072312107559
PATCH_WEIGHT = 1.0

for directory in (MODELS_DIR, FACES_DIR, USER_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def calculate_module_complexity(model, input_tensor):
    """Return MACs and approximate FLOPs for Conv2d and Linear layers."""
    import torch
    import torch.nn as nn

    total_macs = 0
    hooks = []

    def conv_hook(module, _inputs, output):
        nonlocal total_macs
        batch, out_channels, out_height, out_width = output.shape
        kernel_height, kernel_width = module.kernel_size
        total_macs += (
            batch * out_channels * out_height * out_width
            * (module.in_channels // module.groups) * kernel_height * kernel_width
        )

    def linear_hook(module, inputs, _output):
        nonlocal total_macs
        batch = inputs[0].shape[0] if inputs[0].ndim > 1 else 1
        total_macs += batch * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    with torch.no_grad():
        model.eval()
        model(input_tensor)
    for hook in hooks:
        hook.remove()
    return total_macs, total_macs * 2


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    import torch

    metadata: dict[str, Any] = {"size_bytes": path.stat().st_size if path.exists() else 0}
    if not path.exists():
        return metadata
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            for name in ("epoch", "best_val_acc", "best_f1", "last_val_loss"):
                if name in checkpoint:
                    metadata[name] = checkpoint[name]
            group = checkpoint.get("optimizer_state_dict", {}).get("param_groups", [{}])[0]
            for name in ("lr", "initial_lr", "weight_decay"):
                if name in group:
                    metadata[name] = group[name]
    except Exception as error:
        metadata["read_error"] = str(error)
    return metadata


def analyse_cnn(model_class, name: str, checkpoint: Path, input_shape: tuple[int, int, int], patches_per_image=1):
    import torch

    model = model_class()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    layer_parameters = {"Conv2d": 0, "BatchNorm2d": 0, "Linear": 0, "Other": 0}
    for module in model.modules():
        module_name = module.__class__.__name__
        direct_parameters = sum(parameter.numel() for parameter in module.parameters(recurse=False))
        layer_parameters[module_name if module_name in layer_parameters else "Other"] += direct_parameters
    macs, flops = calculate_module_complexity(model, torch.randn(1, *input_shape))
    return {
        "model_name": name,
        "model_class": model_class.__name__,
        "checkpoint_path": str(checkpoint),
        "checkpoint": checkpoint_metadata(checkpoint),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "buffer_elements": sum(buffer.numel() for buffer in model.buffers()),
        "layer_type_breakdown": layer_parameters,
        "memory_fp32_mb": total_parameters * 4 / 1024**2,
        "input_shape": list(input_shape),
        "patches_per_image": patches_per_image,
        "macs_per_forward": macs,
        "flops_per_forward": flops,
        "macs_per_image": macs * patches_per_image,
        "flops_per_image": flops * patches_per_image,
    }


def load_base_metrics() -> dict[str, dict[str, str]]:
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open(encoding="utf-8", newline="") as file:
        return {row["model_type"]: row for row in csv.DictReader(file)}


def get_all_models_summary() -> dict[str, Any]:
    """Build a fresh, two-CNN-only summary."""
    from basemodel_local import PatchCNN
    from basemodel_wide import DeepCNN

    # The current local inference code extracts 6 x 6 patches from its 512px input.
    local = analyse_cnn(PatchCNN, "PatchCNN (Local Texture Model)", LOCAL_CHECKPOINT_PATH, (3, 112, 112), 36)
    wide = analyse_cnn(DeepCNN, "DeepCNN (Wide Global Model)", WIDE_CHECKPOINT_PATH, (3, 256, 256))
    total_parameters = local["total_parameters"] + wide["total_parameters"]
    total_checkpoint_bytes = local["checkpoint"]["size_bytes"] + wide["checkpoint"]["size_bytes"]
    return {
        "local_model": local,
        "wide_model": wide,
        "weighted_fusion": {
            "method": "weighted mean of base-model real probabilities",
            "wide_weight": WIDE_WEIGHT,
            "patch_weight": PATCH_WEIGHT,
            "formula": "(wide_weight * wide_real_probability + patch_weight * patch_real_probability) / (wide_weight + patch_weight)",
            "learned_parameters": 0,
            "checkpoint_required": False,
        },
        "base_model_metrics": load_base_metrics(),
        "pipeline": {
            "total_learnable_parameters": total_parameters,
            "fp32_memory_mb": total_parameters * 4 / 1024**2,
            "checkpoint_storage_mb": total_checkpoint_bytes / 1024**2,
            "vision_macs_per_image": local["macs_per_image"] + wide["macs_per_image"],
            "vision_flops_per_image": local["flops_per_image"] + wide["flops_per_image"],
        },
    }


def generate_markdown_report(summary: dict[str, Any]) -> str:
    local, wide, fusion, pipeline = (summary[key] for key in ("local_model", "wide_model", "weighted_fusion", "pipeline"))
    metrics = summary["base_model_metrics"]
    local_metrics, wide_metrics = metrics.get("patch_cnn", {}), metrics.get("wide_cnn", {})
    checkpoint = lambda model: model["checkpoint"]
    return f"""# DeepFake Detector: Two-CNN Model Parameters & Architecture

This report covers the active detector: the **Wide Global CNN** and **Patch Texture CNN**. Their real probabilities are combined by a deterministic weighted mean. There is no learned meta-ensemble, third model, ensemble checkpoint, or engineered-feature classifier.

## 1. Active Pipeline Overview

| Component | Architecture | Parameters | FP32 memory | Checkpoint |
| --- | --- | ---: | ---: | ---: |
| Local texture model | `PatchCNN` | {local['total_parameters']:,} | {local['memory_fp32_mb']:.2f} MB | {checkpoint(local)['size_bytes'] / 1024**2:.2f} MB |
| Wide global model | `DeepCNN` | {wide['total_parameters']:,} | {wide['memory_fp32_mb']:.2f} MB | {checkpoint(wide)['size_bytes'] / 1024**2:.2f} MB |
| Weighted fusion | Deterministic formula | 0 | 0 MB | None |
| Complete pipeline | Dual CNN + weighted mean | {pipeline['total_learnable_parameters']:,} | {pipeline['fp32_memory_mb']:.2f} MB | {pipeline['checkpoint_storage_mb']:.2f} MB |

## 2. Local Texture Model: PatchCNN

- **Purpose:** Finds local texture artifacts, boundary seams, and high-frequency inconsistencies.
- **Input:** 112x112 RGB patches from a 512x512 image.
- **Current inference grid:** 36 patches (6x6).
- **Parameters:** {local['total_parameters']:,} trainable; {local['buffer_elements']:,} BatchNorm buffer elements.
- **Layer parameters:** Conv2d {local['layer_type_breakdown']['Conv2d']:,}; Linear {local['layer_type_breakdown']['Linear']:,}; BatchNorm2d {local['layer_type_breakdown']['BatchNorm2d']:,}.
- **Compute:** {local['macs_per_forward'] / 1e6:.2f} MMACs / {local['flops_per_forward'] / 1e9:.2f} GFLOPs per patch; {local['macs_per_image'] / 1e9:.2f} GMACs / {local['flops_per_image'] / 1e9:.2f} GFLOPs per 36-patch image.
- **Checkpoint:** epoch {checkpoint(local).get('epoch', 'unknown')}; best validation accuracy {checkpoint(local).get('best_val_acc', 'unknown')}; validation loss {checkpoint(local).get('last_val_loss', 'unknown')}.
- **Recorded base-model metrics:** accuracy {local_metrics.get('accuracy', 'not available')}; F1 {local_metrics.get('f1', 'not available')}.

## 3. Wide Global Model: DeepCNN

- **Purpose:** Evaluates whole-face structure, lighting, and semantic coherence.
- **Input:** 256x256 RGB image, normalized to [-1, 1].
- **Parameters:** {wide['total_parameters']:,} trainable; {wide['buffer_elements']:,} BatchNorm buffer elements.
- **Layer parameters:** Conv2d {wide['layer_type_breakdown']['Conv2d']:,}; Linear {wide['layer_type_breakdown']['Linear']:,}; BatchNorm2d {wide['layer_type_breakdown']['BatchNorm2d']:,}.
- **Compute:** {wide['macs_per_image'] / 1e9:.2f} GMACs / {wide['flops_per_image'] / 1e9:.2f} GFLOPs per image.
- **Checkpoint:** epoch {checkpoint(wide).get('epoch', 'unknown')}; best validation accuracy {checkpoint(wide).get('best_val_acc', 'unknown')}; best F1 {checkpoint(wide).get('best_f1', 'unknown')}.
- **Recorded base-model metrics:** accuracy {wide_metrics.get('accuracy', 'not available')}; F1 {wide_metrics.get('f1', 'not available')}.

## 4. Weighted Two-CNN Fusion

The final real probability is:

```text
real_probability = ({fusion['wide_weight']} * wide_real_probability + {fusion['patch_weight']} * patch_real_probability) / ({fusion['wide_weight']} + {fusion['patch_weight']})
```

- **Wide : Patch weight ratio:** {fusion['wide_weight']} : {fusion['patch_weight']}
- **Learned parameters:** 0
- **Additional checkpoint:** none
- **Decision:** REAL when the weighted real probability is at least 0.5; otherwise FAKE.

## 5. End-to-End Inference Budget

- **Total vision parameters:** {pipeline['total_learnable_parameters']:,}
- **Total FP32 weight memory:** {pipeline['fp32_memory_mb']:.2f} MB
- **CNN checkpoint storage:** {pipeline['checkpoint_storage_mb']:.2f} MB
- **Vision compute:** {pipeline['vision_macs_per_image'] / 1e9:.2f} GMACs / {pipeline['vision_flops_per_image'] / 1e9:.2f} GFLOPs per image.
- **Fusion compute:** one weighted average; negligible compared with CNN inference.
"""


def main():
    parser = argparse.ArgumentParser(description="Calculate two-CNN detector parameters and metadata.")
    parser.add_argument("--json", action="store_true", help="Write the raw summary as JSON.")
    parser.add_argument("--save-report", nargs="?", const=str(USER_DIR / "model_parameters_report.md"), help="Write a Markdown report.")
    args = parser.parse_args()
    summary = get_all_models_summary()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(generate_markdown_report(summary))
    if args.save_report:
        report_path = Path(args.save_report)
        report_path.write_text(generate_markdown_report(summary), encoding="utf-8")
        print(f"[REPORT SAVED] {report_path.resolve()}")


if __name__ == "__main__":
    main()
