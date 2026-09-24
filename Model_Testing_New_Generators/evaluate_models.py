"""
Evaluation Script for DeepFake Detector Models on ai_faces_dataset.
Evaluates:
  1. use_local_model.py (PatchCNN)
  2. use_wide_model.py  (DeepCNN)
Outputs:
  - ai_faces_dataset_detailed_predictions.csv (every image detail)
  - ai_faces_dataset_summary_metrics.csv (aggregated metrics by subset & overall)
  - ai_faces_dataset_analysis_report.md (comprehensive markdown analysis)
"""

import os
import sys
import csv
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    cohen_kappa_score
)

# Ensure paths
WORKSPACE_DIR = Path(__file__).resolve().parent
FACES_DIR = WORKSPACE_DIR / "faces_deepfake"
MODELS_DIR = FACES_DIR / "models"
USER_DIR = FACES_DIR / "user"
DATASET_DIR = WORKSPACE_DIR / "ai_faces_dataset"

for p in [str(USER_DIR), str(MODELS_DIR), str(FACES_DIR), str(WORKSPACE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import only the 2 specified models
from use_wide_model import (
    get_default_model as get_wide_model,
    predict_batch_wide
)
from use_local_model import (
    get_default_model as get_local_model,
    predict_batch_patch
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[EVAL] Execution Device: {DEVICE}")

# Folder and generator mapping
SUBSET_MAP = {
    "00_REAL_LFW": {
        "generator": "Real (LFW)",
        "ground_truth": "REAL",
        "gt_fake_binary": 0,
        "gt_real_binary": 1
    },
    "01_SD-1.5": {
        "generator": "Stable Diffusion 1.5",
        "ground_truth": "FAKE",
        "gt_fake_binary": 1,
        "gt_real_binary": 0
    },
    "02_SD-2.1": {
        "generator": "Stable Diffusion 2.1",
        "ground_truth": "FAKE",
        "gt_fake_binary": 1,
        "gt_real_binary": 0
    },
    "03_RealisticVision-V6": {
        "generator": "RealisticVision V6.0",
        "ground_truth": "FAKE",
        "gt_fake_binary": 1,
        "gt_real_binary": 0
    },
    "04_Playground-v2.5": {
        "generator": "Playground v2.5",
        "ground_truth": "FAKE",
        "gt_fake_binary": 1,
        "gt_real_binary": 0
    },
    "05_SDXL-base-1.0": {
        "generator": "SDXL Base 1.0",
        "ground_truth": "FAKE",
        "gt_fake_binary": 1,
        "gt_real_binary": 0
    }
}


def load_manifest():
    manifest_file = DATASET_DIR / "manifest.csv"
    manifest_data = {}
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest_data[row["filename"]] = {
                    "manifest_model": row.get("model", ""),
                    "prompt": row.get("prompt", ""),
                    "seed": row.get("seed", "")
                }
    return manifest_data


def collect_images():
    images = []
    manifest_data = load_manifest()
    
    for folder_name, meta in sorted(SUBSET_MAP.items()):
        folder_path = DATASET_DIR / folder_name
        if not folder_path.is_dir():
            print(f"[WARN] Folder missing: {folder_path}")
            continue
        
        valid_files = sorted([
            f for f in folder_path.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ])
        
        for f in valid_files:
            m_info = manifest_data.get(f.name, {})
            images.append({
                "file_path": str(f),
                "relative_path": str(f.relative_to(WORKSPACE_DIR)).replace("\\", "/"),
                "filename": f.name,
                "subset_folder": folder_name,
                "generator": meta["generator"],
                "ground_truth": meta["ground_truth"],
                "gt_fake_binary": meta["gt_fake_binary"],
                "gt_real_binary": meta["gt_real_binary"],
                "prompt": m_info.get("prompt", ""),
                "seed": m_info.get("seed", "")
            })
            
    return images


def run_evaluation():
    images = collect_images()
    print(f"[EVAL] Total images collected: {len(images)}")
    
    file_paths = [img["file_path"] for img in images]
    
    # 1. Wide Model (DeepCNN) Inference
    print("\n" + "="*50)
    print("[EVAL] Running Wide Model (DeepCNN)...")
    wide_model = get_wide_model(device=DEVICE)
    t0_wide = time.time()
    wide_results = predict_batch_wide(file_paths, batch_size=64, model=wide_model, device=DEVICE)
    t1_wide = time.time()
    wide_total_time = t1_wide - t0_wide
    wide_fps = len(images) / wide_total_time
    print(f"[EVAL] Wide Model completed in {wide_total_time:.2f}s ({wide_fps:.1f} img/s)")
    
    # 2. Local Model (PatchCNN) Inference
    print("\n" + "="*50)
    print("[EVAL] Running Local Model (PatchCNN)...")
    local_model = get_local_model(device=DEVICE)
    t0_local = time.time()
    local_results = predict_batch_patch(file_paths, batch_size=32, model=local_model, device=DEVICE)
    t1_local = time.time()
    local_total_time = t1_local - t0_local
    local_fps = len(images) / local_total_time
    print(f"[EVAL] Local Model completed in {local_total_time:.2f}s ({local_fps:.1f} img/s)")
    
    # 3. Combine and Build Detailed Records
    print("\n" + "="*50)
    print("[EVAL] Merging predictions and analyzing results...")
    
    detailed_rows = []
    
    for idx, (img_meta, wide_res, local_res) in enumerate(zip(images, wide_results, local_results), start=1):
        gt = img_meta["ground_truth"]  # "REAL" or "FAKE"
        
        # Wide metrics
        w_pred = wide_res["prediction"]  # "REAL" or "FAKE"
        w_prob_real = float(wide_res["real_probability"])
        w_prob_fake = float(wide_res["fake_probability"])
        w_conf = float(wide_res["confidence"])
        w_correct = (w_pred == gt)
        if w_correct:
            w_err = "None"
        elif gt == "FAKE" and w_pred == "REAL":
            w_err = "False Negative (Fake as Real)"
        else:
            w_err = "False Positive (Real as Fake)"
            
        # Local metrics
        l_pred = local_res["prediction"]
        l_prob_real = float(local_res["real_probability"])
        l_prob_fake = float(local_res["fake_probability"])
        l_conf = float(local_res["confidence"])
        l_patch_mean = float(local_res.get("mean", l_prob_real))
        l_patch_std = float(local_res.get("std_dev", 0.0))
        l_patch_var = float(local_res.get("variance", 0.0))
        patch_probs = local_res.get("patch_probs", [])
        l_patch_min = float(np.min(patch_probs)) if len(patch_probs) > 0 else l_prob_real
        l_patch_max = float(np.max(patch_probs)) if len(patch_probs) > 0 else l_prob_real
        
        l_correct = (l_pred == gt)
        if l_correct:
            l_err = "None"
        elif gt == "FAKE" and l_pred == "REAL":
            l_err = "False Negative (Fake as Real)"
        else:
            l_err = "False Positive (Real as Fake)"
            
        agreement = (w_pred == l_pred)
        both_correct = (w_correct and l_correct)
        both_wrong = ((not w_correct) and (not l_correct))
        
        detailed_rows.append({
            "file_id": idx,
            "relative_path": img_meta["relative_path"],
            "filename": img_meta["filename"],
            "subset_folder": img_meta["subset_folder"],
            "generator": img_meta["generator"],
            "ground_truth": gt,
            "gt_fake_binary": img_meta["gt_fake_binary"],
            "gt_real_binary": img_meta["gt_real_binary"],
            # Wide Model Details
            "wide_prediction": w_pred,
            "wide_real_probability": round(w_prob_real, 6),
            "wide_fake_probability": round(w_prob_fake, 6),
            "wide_confidence": round(w_conf, 6),
            "wide_correct": w_correct,
            "wide_error_type": w_err,
            # Local Model Details
            "local_prediction": l_pred,
            "local_real_probability": round(l_prob_real, 6),
            "local_fake_probability": round(l_prob_fake, 6),
            "local_confidence": round(l_conf, 6),
            "local_patch_mean": round(l_patch_mean, 6),
            "local_patch_std": round(l_patch_std, 6),
            "local_patch_var": round(l_patch_var, 6),
            "local_patch_min": round(l_patch_min, 6),
            "local_patch_max": round(l_patch_max, 6),
            "local_correct": l_correct,
            "local_error_type": l_err,
            # Comparison
            "models_agree": agreement,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "wide_only_correct": (w_correct and not l_correct),
            "local_only_correct": (l_correct and not w_correct),
            # Metadata
            "prompt": img_meta["prompt"],
            "seed": img_meta["seed"]
        })
        
    # Write detailed CSV
    csv_out_path = WORKSPACE_DIR / "ai_faces_dataset_detailed_predictions.csv"
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(detailed_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"[EVAL] Detailed predictions saved to: {csv_out_path}")
    
    # 4. Compute Comprehensive Statistics
    stats = compute_statistics(detailed_rows, wide_fps, local_fps, wide_total_time, local_total_time)
    
    # Save summary metrics CSV
    summary_csv_path = WORKSPACE_DIR / "ai_faces_dataset_summary_metrics.csv"
    save_summary_csv(stats, summary_csv_path)
    print(f"[EVAL] Summary metrics saved to: {summary_csv_path}")
    
    # Generate Markdown Report
    md_out_path = WORKSPACE_DIR / "ai_faces_dataset_model_analysis.md"
    generate_markdown_report(stats, md_out_path)
    print(f"[EVAL] Analysis Markdown report saved to: {md_out_path}")
    
    return stats


def calculate_metrics_for_subset(rows, model_prefix):
    """
    Computes classification metrics for a given subset and model ('wide' or 'local').
    Binary convention: Positive = FAKE (1), Negative = REAL (0).
    """
    y_true_fake = np.array([r["gt_fake_binary"] for r in rows])
    y_pred_fake = np.array([1 if r[f"{model_prefix}_prediction"] == "FAKE" else 0 for r in rows])
    y_prob_fake = np.array([r[f"{model_prefix}_fake_probability"] for r in rows])
    y_prob_real = np.array([r[f"{model_prefix}_real_probability"] for r in rows])
    
    n_total = len(rows)
    acc = accuracy_score(y_true_fake, y_pred_fake)
    
    # Confusion matrix:
    # y_true can be single class if subset is pure Real or pure Fake
    has_both_classes = (len(np.unique(y_true_fake)) > 1)
    
    if has_both_classes:
        tn, fp, fn, tp = confusion_matrix(y_true_fake, y_pred_fake, labels=[0, 1]).ravel()
        prec_fake = precision_score(y_true_fake, y_pred_fake, zero_division=0)
        rec_fake = recall_score(y_true_fake, y_pred_fake, zero_division=0)
        f1_fake = f1_score(y_true_fake, y_pred_fake, zero_division=0)
        spec_real = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        bal_acc = balanced_accuracy_score(y_true_fake, y_pred_fake)
        try:
            roc_auc = roc_auc_score(y_true_fake, y_prob_fake)
        except Exception:
            roc_auc = float("nan")
        try:
            pr_auc = average_precision_score(y_true_fake, y_prob_fake)
        except Exception:
            pr_auc = float("nan")
        brier = brier_score_loss(y_true_fake, y_prob_fake)
    else:
        # Single class subset
        is_all_real = (y_true_fake[0] == 0)
        if is_all_real:
            # 00_REAL_LFW
            tn = int(np.sum(y_pred_fake == 0))  # predicted Real
            fp = int(np.sum(y_pred_fake == 1))  # predicted Fake
            fn = 0
            tp = 0
            prec_fake = 0.0 if fp == 0 else 0.0
            rec_fake = 0.0
            f1_fake = 0.0
            spec_real = tn / n_total
            fpr = fp / n_total
            fnr = 0.0
        else:
            # All Fake folder
            tn = 0
            fp = 0
            fn = int(np.sum(y_pred_fake == 0))  # predicted Real
            tp = int(np.sum(y_pred_fake == 1))  # predicted Fake
            prec_fake = 1.0 if tp > 0 else 0.0
            rec_fake = tp / n_total
            f1_fake = 2 * rec_fake / (1 + rec_fake) if (1 + rec_fake) > 0 else 0.0
            spec_real = 0.0
            fpr = 0.0
            fnr = fn / n_total
            
        bal_acc = acc
        roc_auc = float("nan")
        pr_auc = float("nan")
        brier = float(np.mean((y_true_fake - y_prob_fake) ** 2))
        
    return {
        "count": n_total,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "fake_precision": float(prec_fake),
        "fake_recall": float(rec_fake),
        "fake_f1": float(f1_fake),
        "specificity": float(spec_real),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "mean_real_prob": float(np.mean(y_prob_real)),
        "mean_fake_prob": float(np.mean(y_prob_fake)),
        "mean_confidence": float(np.mean([r[f"{model_prefix}_confidence"] for r in rows]))
    }


def compute_statistics(detailed_rows, wide_fps, local_fps, wide_time, local_time):
    # Overall metrics
    overall_wide = calculate_metrics_for_subset(detailed_rows, "wide")
    overall_local = calculate_metrics_for_subset(detailed_rows, "local")
    
    # Subsets metrics
    subsets = sorted(list(set(r["subset_folder"] for r in detailed_rows)))
    subsets_metrics = {}
    for s in subsets:
        s_rows = [r for r in detailed_rows if r["subset_folder"] == s]
        subsets_metrics[s] = {
            "name": s,
            "generator": s_rows[0]["generator"],
            "ground_truth": s_rows[0]["ground_truth"],
            "wide": calculate_metrics_for_subset(s_rows, "wide"),
            "local": calculate_metrics_for_subset(s_rows, "local")
        }
        
    # Model comparison contingency table
    both_correct = sum(1 for r in detailed_rows if r["both_correct"])
    wide_only = sum(1 for r in detailed_rows if r["wide_only_correct"])
    local_only = sum(1 for r in detailed_rows if r["local_only_correct"])
    both_wrong = sum(1 for r in detailed_rows if r["both_wrong"])
    agreement_count = sum(1 for r in detailed_rows if r["models_agree"])
    agreement_rate = agreement_count / len(detailed_rows)
    
    # Cohen's Kappa between Wide and Local predictions
    w_preds = [1 if r["wide_prediction"] == "FAKE" else 0 for r in detailed_rows]
    l_preds = [1 if r["local_prediction"] == "FAKE" else 0 for r in detailed_rows]
    kappa = cohen_kappa_score(w_preds, l_preds)
    
    # McNemar's test calculation
    # b = wide correct, local wrong; c = local correct, wide wrong
    b = wide_only
    c = local_only
    if (b + c) > 0:
        mcnemar_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    else:
        mcnemar_stat = 0.0
        
    return {
        "total_images": len(detailed_rows),
        "wide_fps": wide_fps,
        "local_fps": local_fps,
        "wide_time": wide_time,
        "local_time": local_time,
        "overall_wide": overall_wide,
        "overall_local": overall_local,
        "subsets_metrics": subsets_metrics,
        "comparison": {
            "both_correct": both_correct,
            "wide_only_correct": wide_only,
            "local_only_correct": local_only,
            "both_wrong": both_wrong,
            "agreement_count": agreement_count,
            "agreement_rate": agreement_rate,
            "cohens_kappa": kappa,
            "mcnemar_stat": mcnemar_stat
        }
    }


def save_summary_csv(stats, csv_path):
    rows = []
    # Overall rows
    for model_name, m_stats in [("Wide Model (DeepCNN)", stats["overall_wide"]), ("Local Model (PatchCNN)", stats["overall_local"])]:
        rows.append({
            "Subset": "OVERALL DATASET",
            "Generator": "All (LFW + 5 AI Generators)",
            "Ground_Truth": "Mixed (1000 Real / 2000 Fake)",
            "Model": model_name,
            "Images_Count": m_stats["count"],
            "Accuracy": f"{m_stats['accuracy'] * 100:.2f}%",
            "Balanced_Accuracy": f"{m_stats['balanced_accuracy'] * 100:.2f}%",
            "Fake_Precision": f"{m_stats['fake_precision'] * 100:.2f}%",
            "Fake_Recall_Sensitivity": f"{m_stats['fake_recall'] * 100:.2f}%",
            "Fake_F1_Score": f"{m_stats['fake_f1'] * 100:.2f}%",
            "Real_Specificity": f"{m_stats['specificity'] * 100:.2f}%",
            "False_Positive_Rate": f"{m_stats['fpr'] * 100:.2f}%",
            "False_Negative_Rate": f"{m_stats['fnr'] * 100:.2f}%",
            "ROC_AUC": f"{m_stats['roc_auc']:.4f}" if not np.isnan(m_stats['roc_auc']) else "N/A",
            "PR_AUC": f"{m_stats['pr_auc']:.4f}" if not np.isnan(m_stats['pr_auc']) else "N/A",
            "Brier_Score": f"{m_stats['brier_score']:.4f}",
            "Mean_Confidence": f"{m_stats['mean_confidence'] * 100:.2f}%",
            "TP": m_stats["tp"],
            "FP": m_stats["fp"],
            "TN": m_stats["tn"],
            "FN": m_stats["fn"]
        })
        
    # Per-subset rows
    for s_name, s_data in stats["subsets_metrics"].items():
        for model_name, m_key in [("Wide Model (DeepCNN)", "wide"), ("Local Model (PatchCNN)", "local")]:
            m_stats = s_data[m_key]
            rows.append({
                "Subset": s_name,
                "Generator": s_data["generator"],
                "Ground_Truth": s_data["ground_truth"],
                "Model": model_name,
                "Images_Count": m_stats["count"],
                "Accuracy": f"{m_stats['accuracy'] * 100:.2f}%",
                "Balanced_Accuracy": f"{m_stats['balanced_accuracy'] * 100:.2f}%",
                "Fake_Precision": f"{m_stats['fake_precision'] * 100:.2f}%",
                "Fake_Recall_Sensitivity": f"{m_stats['fake_recall'] * 100:.2f}%",
                "Fake_F1_Score": f"{m_stats['fake_f1'] * 100:.2f}%",
                "Real_Specificity": f"{m_stats['specificity'] * 100:.2f}%",
                "False_Positive_Rate": f"{m_stats['fpr'] * 100:.2f}%",
                "False_Negative_Rate": f"{m_stats['fnr'] * 100:.2f}%",
                "ROC_AUC": f"{m_stats['roc_auc']:.4f}" if not np.isnan(m_stats['roc_auc']) else "N/A",
                "PR_AUC": f"{m_stats['pr_auc']:.4f}" if not np.isnan(m_stats['pr_auc']) else "N/A",
                "Brier_Score": f"{m_stats['brier_score']:.4f}",
                "Mean_Confidence": f"{m_stats['mean_confidence'] * 100:.2f}%",
                "TP": m_stats["tp"],
                "FP": m_stats["fp"],
                "TN": m_stats["tn"],
                "FN": m_stats["fn"]
            })
            
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_markdown_report(stats, md_path):
    ow = stats["overall_wide"]
    ol = stats["overall_local"]
    comp = stats["comparison"]
    sub = stats["subsets_metrics"]
    
    content = f"""# DeepFake Detection Models Benchmark Analysis: `use_wide_model.py` vs `use_local_model.py`

**Dataset:** `ai_faces_dataset` (3,000 images total)  
- **Real Faces (`00_REAL_LFW`):** 1,000 images (Labeled as REAL)  
- **AI-Generated Faces:** 2,000 images (Labeled as FAKE across 5 generators, 400 images each)  
  - `01_SD-1.5` (Stable Diffusion 1.5)
  - `02_SD-2.1` (Stable Diffusion 2.1)
  - `03_RealisticVision-V6` (RealisticVision V6.0)
  - `04_Playground-v2.5` (Playground v2.5)
  - `05_SDXL-base-1.0` (SDXL Base 1.0)  
**Evaluated Models:**
1. **Wide Global Model (`DeepCNN` via `use_wide_model.py`):** 4.85M parameters, analyzes full 256x256 image context.
2. **Local Texture Model (`PatchCNN` via `use_local_model.py`):** 1.22M parameters, analyzes 36 overlapping 112x112 patches over a 512x512 image.

---

## 1. Executive Summary & Overall Performance

| Metric | Wide Model (`DeepCNN`) | Local Model (`PatchCNN`) | Delta (Wide - Local) | Superior Model |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **{ow['accuracy']*100:.2f}%** | **{ol['accuracy']*100:.2f}%** | { (ow['accuracy'] - ol['accuracy'])*100:+.2f}% | {'Wide Model' if ow['accuracy'] > ol['accuracy'] else 'Local Model'} |
| **Balanced Accuracy** | **{ow['balanced_accuracy']*100:.2f}%** | **{ol['balanced_accuracy']*100:.2f}%** | { (ow['balanced_accuracy'] - ol['balanced_accuracy'])*100:+.2f}% | {'Wide Model' if ow['balanced_accuracy'] > ol['balanced_accuracy'] else 'Local Model'} |
| **Fake Detection Recall (Sensitivity)** | **{ow['fake_recall']*100:.2f}%** | **{ol['fake_recall']*100:.2f}%** | { (ow['fake_recall'] - ol['fake_recall'])*100:+.2f}% | {'Wide Model' if ow['fake_recall'] > ol['fake_recall'] else 'Local Model'} |
| **Real Face Specificity** | **{ow['specificity']*100:.2f}%** | **{ol['specificity']*100:.2f}%** | { (ow['specificity'] - ol['specificity'])*100:+.2f}% | {'Wide Model' if ow['specificity'] > ol['specificity'] else 'Local Model'} |
| **Fake Precision** | **{ow['fake_precision']*100:.2f}%** | **{ol['fake_precision']*100:.2f}%** | { (ow['fake_precision'] - ol['fake_precision'])*100:+.2f}% | {'Wide Model' if ow['fake_precision'] > ol['fake_precision'] else 'Local Model'} |
| **Fake F1-Score** | **{ow['fake_f1']*100:.2f}%** | **{ol['fake_f1']*100:.2f}%** | { (ow['fake_f1'] - ol['fake_f1'])*100:+.2f}% | {'Wide Model' if ow['fake_f1'] > ol['fake_f1'] else 'Local Model'} |
| **False Positive Rate (Real as Fake)** | **{ow['fpr']*100:.2f}%** | **{ol['fpr']*100:.2f}%** | { (ow['fpr'] - ol['fpr'])*100:+.2f}% | {'Wide Model (Lower)' if ow['fpr'] < ol['fpr'] else 'Local Model (Lower)'} |
| **False Negative Rate (Fake as Real)** | **{ow['fnr']*100:.2f}%** | **{ol['fnr']*100:.2f}%** | { (ow['fnr'] - ol['fnr'])*100:+.2f}% | {'Wide Model (Lower)' if ow['fnr'] < ol['fnr'] else 'Local Model (Lower)'} |
| **ROC-AUC Score** | **{ow['roc_auc']:.4f}** | **{ol['roc_auc']:.4f}** | { (ow['roc_auc'] - ol['roc_auc']):+.4f} | {'Wide Model' if ow['roc_auc'] > ol['roc_auc'] else 'Local Model'} |
| **PR-AUC Score** | **{ow['pr_auc']:.4f}** | **{ol['pr_auc']:.4f}** | { (ow['pr_auc'] - ol['pr_auc']):+.4f} | {'Wide Model' if ow['pr_auc'] > ol['pr_auc'] else 'Local Model'} |
| **Brier Score (Calibration error)** | **{ow['brier_score']:.4f}** | **{ol['brier_score']:.4f}** | { (ow['brier_score'] - ol['brier_score']):+.4f} | {'Wide Model' if ow['brier_score'] < ol['brier_score'] else 'Local Model'} |
| **Mean Model Confidence** | **{ow['mean_confidence']*100:.2f}%** | **{ol['mean_confidence']*100:.2f}%** | { (ow['mean_confidence'] - ol['mean_confidence'])*100:+.2f}% | - |
| **Throughput (Inference Speed)** | **{stats['wide_fps']:.1f} images/s** | **{stats['local_fps']:.1f} images/s** | {stats['wide_fps'] / stats['local_fps']:.1f}x speedup | Wide Model |

---

## 2. Overall Confusion Matrices

### Wide Model (`DeepCNN`)
```
Total Samples: 3000
-------------------------------------------------------
                      Predicted REAL      Predicted FAKE
Actual REAL (1000):        {ow['tn']:<6} (TN)          {ow['fp']:<6} (FP)
Actual FAKE (2000):        {ow['fn']:<6} (FN)          {ow['tp']:<6} (TP)
-------------------------------------------------------
```

### Local Model (`PatchCNN`)
```
Total Samples: 3000
-------------------------------------------------------
                      Predicted REAL      Predicted FAKE
Actual REAL (1000):        {ol['tn']:<6} (TN)          {ol['fp']:<6} (FP)
Actual FAKE (2000):        {ol['fn']:<6} (FN)          {ol['tp']:<6} (TP)
-------------------------------------------------------
```

---

## 3. Detailed Breakdown by Subset & Generator

This breakdown evaluates how effectively each model generalizes to real faces and diverse AI generator architectures (ranging from classic Stable Diffusion 1.5 to modern SDXL 1.0 and Playground v2.5).

| Subset Folder | Generator Type | Ground Truth | Total Images | Wide Model Acc (%) | Wide Correct / Total | Local Model Acc (%) | Local Correct / Total | Best Model |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

    for s_name, s_data in sub.items():
        w_acc = s_data["wide"]["accuracy"] * 100
        l_acc = s_data["local"]["accuracy"] * 100
        w_corr = s_data["wide"]["tp"] if s_data["ground_truth"] == "FAKE" else s_data["wide"]["tn"]
        l_corr = s_data["local"]["tp"] if s_data["ground_truth"] == "FAKE" else s_data["local"]["tn"]
        tot = s_data["wide"]["count"]
        best = "Tie" if abs(w_acc - l_acc) < 0.01 else ("Wide Model" if w_acc > l_acc else "Local Model")
        content += f"| `{s_name}` | {s_data['generator']} | **{s_data['ground_truth']}** | {tot} | **{w_acc:.2f}%** | {w_corr}/{tot} | **{l_acc:.2f}%** | {l_corr}/{tot} | {best} |\n"

    content += f"""
---

## 4. Per-Subset Deep Dive & Error Rates

"""
    for s_name, s_data in sub.items():
        w = s_data["wide"]
        l = s_data["local"]
        gt = s_data["ground_truth"]
        content += f"""### `{s_name}` — {s_data['generator']} ({gt})
- **Sample Count:** {w['count']}
- **Wide Model Performance:**
  - Accuracy / Detection Rate: **{w['accuracy']*100:.2f}%** ({w['tp'] if gt == 'FAKE' else w['tn']}/{w['count']})
  - Mean Fake Probability: **{w['mean_fake_prob']:.4f}** | Mean Real Probability: **{w['mean_real_prob']:.4f}**
  - Errors: **{w['fn'] if gt == 'FAKE' else w['fp']}** ({'False Negatives: Fake misclassified as Real' if gt == 'FAKE' else 'False Positives: Real misclassified as Fake'})
- **Local Model Performance:**
  - Accuracy / Detection Rate: **{l['accuracy']*100:.2f}%** ({l['tp'] if gt == 'FAKE' else l['tn']}/{l['count']})
  - Mean Fake Probability: **{l['mean_fake_prob']:.4f}** | Mean Real Probability: **{l['mean_real_prob']:.4f}**
  - Errors: **{l['fn'] if gt == 'FAKE' else l['fp']}** ({'False Negatives: Fake misclassified as Real' if gt == 'FAKE' else 'False Positives: Real misclassified as Fake'})

"""

    content += f"""---

## 5. Model Agreement & Disagreement Analysis

To assess whether the models have complementary discriminative features or correlated errors, we examine the pairwise contingency:

| Outcome Category | Count | Percentage of Dataset |
| :--- | :---: | :---: |
| **Both Models Correct** | {comp['both_correct']} | {comp['both_correct']/stats['total_images']*100:.2f}% |
| **Wide Model Correct, Local Model Wrong** | {comp['wide_only_correct']} | {comp['wide_only_correct']/stats['total_images']*100:.2f}% |
| **Local Model Correct, Wide Model Wrong** | {comp['local_only_correct']} | {comp['local_only_correct']/stats['total_images']*100:.2f}% |
| **Both Models Wrong** | {comp['both_wrong']} | {comp['both_wrong']/stats['total_images']*100:.2f}% |
| **Prediction Agreement Rate** | {comp['agreement_count']} | {comp['agreement_rate']*100:.2f}% |
| **Cohen's Kappa ($\kappa$)** | {comp['cohens_kappa']:.4f} | {'High agreement' if comp['cohens_kappa'] > 0.6 else ('Moderate agreement' if comp['cohens_kappa'] > 0.4 else 'Low agreement')} |
| **McNemar Chi-Square Statistic** | {comp['mcnemar_stat']:.4f} | {'Statistically significant difference (p < 0.05)' if comp['mcnemar_stat'] > 3.841 else 'Not statistically significant'} |

---

## 6. Architectural Insights & Behavioral Comparison

### Wide Global Model (`DeepCNN` - 4,847,777 parameters)
1. **Full-Image Semantic Perspective:** Operates on the entire 256x256 facial image. Captures global facial symmetry, illumination consistency, eye reflections (corneal specular highlights), jawline proportions, and perspective distortions.
2. **Computational Efficiency:** Direct single-pass forward calculation without patch decomposition yields high throughput ({stats['wide_fps']:.1f} images/s).
3. **Generalization Profile:**
   - Evaluates structural and geometric coherence across modern diffusion models (SDXL, Playground v2.5).

### Local Texture Model (`PatchCNN` - 1,215,393 parameters)
1. **High-Frequency Patch Perspective:** Extracts 36 overlapping 112x112 patches from a 512x512 image. Focuses on microscopic frequency domain artifacts, blending boundaries, noise patterns, hair strands, and skin pore regularity.
2. **Computational Overhead:** Requires processing 36 forward passes per image, resulting in higher MAC count and lower throughput ({stats['local_fps']:.1f} images/s).
3. **Generalization Profile:**
   - Highly sensitive to fine-grained patch-level variations, but prone to false positives on real images if natural skin textures or lighting variations resemble synthetic artifacts.

---

## 7. Artifacts Generated

1. `ai_faces_dataset_detailed_predictions.csv`: Contains per-image predictions, exact confidence scores, probabilities, patch variance and extremes, ground truth, error categories, and generation prompts/seeds.
2. `ai_faces_dataset_summary_metrics.csv`: Tabulated precision, recall, specificity, AUC, and accuracy across all subsets.
3. `ai_faces_dataset_model_analysis.md`: This comprehensive evaluation and comparative analysis report.
"""
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run_evaluation()
