# Wide Model XAI Evidence

## Scope

This artefact explains **only** `DeepCNN` loaded from
`faces_deepfake/models/best_wide_model.pth`. The local PatchCNN is intentionally
excluded because the benchmark reports show weak generalization for it.

## Reproducibility

- Run timestamp (UTC): `2026-09-25T17:17:36.646675+00:00`
- Checkpoint SHA-256: `965bc5e7660a26cf5903c8826140bcd0a44d268ac8d8cbad122f823177dfbaa5`
- Device: `cuda`
- Input size: `256 x 256` (the wide model's inference preprocessing)
- Attribution target: the wide model's own predicted class
- Methods: Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, LIME
- Inputs: `10` (`9` predictions match the supplied folder label)

## Output layout

Each image folder contains `input.png`, five `*_overlay.png` files, five
`*_heatmap.png` files, and `contact_sheet.png`. `xai_manifest.csv` is the
machine-readable prediction/attribution audit trail; `provenance.json` records
the exact checkpoint fingerprint and software/runtime details.

## Important interpretation note

Warm colours show regions that positively support the displayed predicted class,
not ground-truth correctness or causal proof. The five maps are complementary;
they should be assessed together with the prediction probabilities.
