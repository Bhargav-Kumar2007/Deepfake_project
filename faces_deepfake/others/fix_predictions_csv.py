import os
import sys
import csv
import shutil
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
FACES_DIR = CURRENT_DIR.parent
OTHERS_DIR = FACES_DIR / "others"

candidate_csv_paths = [
    CURRENT_DIR / "all_predictions_batch.csv",
    OTHERS_DIR / "all_predictions_batch.csv",
    FACES_DIR / "all_predictions_batch.csv"
]
DEFAULT_CSV_PATH = next((p for p in candidate_csv_paths if p.exists()), CURRENT_DIR / "all_predictions_batch.csv")
BACKUP_CSV_PATH = DEFAULT_CSV_PATH.with_name("all_predictions_batch_original.csv.bak")
METRICS_CSV_PATH = DEFAULT_CSV_PATH.with_name("model_metrics.csv")


def calculate_metrics(predictions, ground_truth):
    """
    Calculate accuracy, precision, recall, f1 as specified in colab.txt Cell 12.
    Binary definition: fake=1, real=0.
    """
    pred_binary = [1 if p == 'fake' else 0 for p in predictions]
    true_binary = [1 if gt == 'fake' else 0 for gt in ground_truth]

    TP = sum(1 for p, t in zip(pred_binary, true_binary) if p == 1 and t == 1)
    TN = sum(1 for p, t in zip(pred_binary, true_binary) if p == 0 and t == 0)
    FP = sum(1 for p, t in zip(pred_binary, true_binary) if p == 1 and t == 0)
    FN = sum(1 for p, t in zip(pred_binary, true_binary) if p == 0 and t == 1)

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': TP,
        'tn': TN,
        'fp': FP,
        'fn': FN
    }


def fix_predictions_csv(input_csv_path=DEFAULT_CSV_PATH, make_backup=True):
    input_path = Path(input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found at: {input_path}")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Make backup of original file if needed
    if make_backup and not BACKUP_CSV_PATH.exists():
        print(f"[BACKUP] Creating backup at: {BACKUP_CSV_PATH}")
        shutil.copy2(input_path, BACKUP_CSV_PATH)

    temp_output_path = input_path.with_suffix(".tmp")
    
    true_labels = []
    wide_preds = []
    patch_preds = []

    print(f"[PROCESSING] Reversing polarity for: {input_path}")
    count = 0

    with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
         open(temp_output_path, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = [
            'image_path', 'true_label',
            'wide_pred', 'wide_prob', 'wide_conf',
            'patch_pred', 'patch_mean', 'patch_std', 'patch_var', 'patch_conf'
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            count += 1
            true_label = row['true_label'].strip().lower()

            wide_prob = float(row['wide_prob'])
            patch_mean = float(row['patch_mean'])
            patch_std = float(row['patch_std'])
            patch_var = float(row['patch_var'])

            # Reversed polarity: >= 0.5 is REAL, < 0.5 is FAKE
            wide_pred = "real" if wide_prob >= 0.5 else "fake"
            wide_conf = wide_prob if wide_prob >= 0.5 else 1.0 - wide_prob

            patch_pred = "real" if patch_mean >= 0.5 else "fake"
            patch_conf = patch_mean if patch_mean >= 0.5 else 1.0 - patch_mean

            true_labels.append(true_label)
            wide_preds.append(wide_pred)
            patch_preds.append(patch_pred)

            writer.writerow({
                'image_path': row['image_path'],
                'true_label': true_label,
                'wide_pred': wide_pred,
                'wide_prob': f"{wide_prob:.6f}",
                'wide_conf': f"{wide_conf:.6f}",
                'patch_pred': patch_pred,
                'patch_mean': f"{patch_mean:.6f}",
                'patch_std': f"{patch_std:.6f}",
                'patch_var': f"{patch_var:.6f}",
                'patch_conf': f"{patch_conf:.6f}"
            })

    # Replace original file atomically with fixed content
    temp_output_path.replace(input_path)
    print(f"[SUCCESS] Successfully updated {count:,} records in: {input_path}")

    # Compute metrics exactly as in colab.txt Cell 12
    wide_metrics = calculate_metrics(wide_preds, true_labels)
    patch_metrics = calculate_metrics(patch_preds, true_labels)

    print("\n" + "=" * 65)
    print("RECALCULATED MODEL PERFORMANCE (REVERSED POLARITY)")
    print("=" * 65)
    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 65)
    print(f"{'Wide CNN':<15} {wide_metrics['accuracy']*100:<11.2f}% {wide_metrics['precision']*100:<11.2f}% {wide_metrics['recall']*100:<11.2f}% {wide_metrics['f1']*100:<11.2f}%")
    print(f"{'Patch CNN':<15} {patch_metrics['accuracy']*100:<11.2f}% {patch_metrics['precision']*100:<11.2f}% {patch_metrics['recall']*100:<11.2f}% {patch_metrics['f1']*100:<11.2f}%")
    print("=" * 65)

    # Save metrics CSV (matching colab.txt Cell 12)
    with open(METRICS_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        mwriter = csv.writer(f)
        mwriter.writerow(['model_type', 'accuracy', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn'])
        mwriter.writerow([
            'wide_cnn',
            f"{wide_metrics['accuracy']:.6f}",
            f"{wide_metrics['precision']:.6f}",
            f"{wide_metrics['recall']:.6f}",
            f"{wide_metrics['f1']:.6f}",
            wide_metrics['tp'], wide_metrics['tn'], wide_metrics['fp'], wide_metrics['fn']
        ])
        mwriter.writerow([
            'patch_cnn',
            f"{patch_metrics['accuracy']:.6f}",
            f"{patch_metrics['precision']:.6f}",
            f"{patch_metrics['recall']:.6f}",
            f"{patch_metrics['f1']:.6f}",
            patch_metrics['tp'], patch_metrics['tn'], patch_metrics['fp'], patch_metrics['fn']
        ])
    print(f"[SAVED] Saved updated metrics to: {METRICS_CSV_PATH}")


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    fix_predictions_csv(csv_file)
