# DeepFake Detector — Web App

A dark-themed, browser-based local UI for the DeepFake Detector pipeline.  
Runs a Python HTTP server on `http://127.0.0.1:8000` — no external framework required.

---

## Starting the server

Run from the **repository root** using the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe faces_deepfake/webapp/server.py
```

Then open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.  
Press `Ctrl+C` to stop.

The server binds to localhost only and is **not publicly exposed**.

---

## Files

| File | Purpose |
| --- | --- |
| `server.py` | Python `ThreadingHTTPServer` — serves static files and exposes two API routes |
| `index.html` | Single-page UI — two tabs: **Scan Image** and **Model Data** |
| `app.js` | All frontend logic — upload handling, analysis, XAI rendering, patch matrix |
| `styles.css` | Dark-mode design — glassmorphism cards, color-coded verdicts, animated spinner |

---

## API routes

| Method | Route | Description |
| --- | --- | --- |
| `POST` | `/api/analyze` | Accepts a `multipart/form-data` image upload (field name `image`), runs inference, returns JSON |
| `GET` | `/api/model-report` | Returns the model parameters report from `user/model_parameters_report.md` as JSON |
| `GET` | `/api/health` | Returns `{"status": "ok"}` |

### `/api/analyze` response shape

```json
{
  "filename": "photo.jpg",
  "prediction": "FAKE",
  "confidence": 0.9312,
  "real_probability": 0.0688,
  "fake_probability": 0.9312,
  "wide_model": {
    "prediction": "FAKE",
    "confidence": 0.9601,
    "real_probability": 0.0399,
    "fake_probability": 0.9601
  },
  "local_model": {
    "prediction": "FAKE",
    "confidence": 0.8981,
    "real_probability": 0.1019,
    "fake_probability": 0.8981,
    "mean": 0.10193,
    "std_dev": 0.04221,
    "variance": 0.00178,
    "patch_count": 36,
    "patch_probs": [...],
    "patch_heatmap_overlay": "data:image/jpeg;base64,...",
    "patch_raw_heatmap":    "data:image/jpeg;base64,...",
    "patch_grid": [
      { "index": 1, "row": 1, "col": 1, "x": 0, "y": 0,
        "size": 112, "real_probability": 0.1019, "fake_probability": 0.8981, "verdict": "FAKE" },
      ...
    ]
  },
  "fusion": {
    "method": "weighted mean of base-model real probabilities",
    "wide_weight": 1.0072312107559,
    "patch_weight": 1.0
  },
  "transformed_image": "data:image/jpeg;base64,...",
  "heatmaps": {
    "gradcam":               "data:image/jpeg;base64,...",
    "gradcam_plus_plus":     "data:image/jpeg;base64,...",
    "score_cam":             "data:image/jpeg;base64,...",
    "integrated_gradients":  "data:image/jpeg;base64,...",
    "lime":                  "data:image/jpeg;base64,..."
  },
  "raw_heatmaps": { ... },
  "descriptions": {
    "gradcam": { "name": "Grad-CAM", "concept": "...", "description": "..." },
    ...
  },
  "target_explanation": "FAKE"
}
```

All heatmap and image values are base64-encoded JPEG data URLs. If XAI generation fails (e.g. missing dependency), `heatmaps` and `raw_heatmaps` will be empty objects `{}` and the UI shows an unavailability notice — predictions are still returned.

---

## UI features

### Scan Image tab

- **Upload Photo** — accepts JPG, JPEG, PNG, WEBP, BMP up to **12 MB**.
- **Use Camera** — captures a still from the user's webcam.
- **Rescan** — re-runs analysis on the currently loaded image using cached models.

After analysis, three result sections are shown:

#### 1. Wide Model Prediction & Explainability
Displays verdict (color-coded **REAL** / **FAKE**), confidence, and real/fake probabilities.  
Shows five XAI heatmaps overlaid on the 512 × 512 transformed face:

| Method | Key |
| --- | --- |
| Grad-CAM | `gradcam` |
| Grad-CAM++ | `gradcam_plus_plus` |
| Score-CAM | `score_cam` |
| Integrated Gradients | `integrated_gradients` |
| LIME | `lime` |

Use the opacity slider to blend between the face and the heatmap layer.  
The **Compare All (5)** button shows all heatmaps in a grid.  
If XAI is unavailable, a notice is shown and the prediction tables still display.

#### 2. Patch Mean (Local Patch CNN) Prediction
Displays verdict, confidence, mean patch score, standard deviation, and variance across 36 patches.  
An interactive **6 × 6 patch matrix** lets you hover any cell to see:
- Per-patch fake probability
- Row / column position on the 512 × 512 image
- A bounding-box highlight on the heatmap overlay

The **Top Suspicious Patches** list shows the 5 patches with the highest fake probability.

#### 3. Ensemble (Weighted Fusion) Outcome
Final verdict from the deterministic weighted mean, with both model weights shown.

### Model Data tab
Renders `user/model_parameters_report.md` as formatted HTML — architecture summary, layer counts, parameter/memory/compute budgets, checkpoint metadata, and recorded base-model metrics.

---

## Upload constraints

| Constraint | Value |
| --- | --- |
| Max file size | 12 MB |
| Accepted formats | JPG, JPEG, PNG, WEBP, BMP |
| Image validation | PIL `verify()` run before inference |
| Temp file lifecycle | Written → inference → deleted (always, even on error) |

---

## Dependencies (resolved from project venv)

| Package | Used for |
| --- | --- |
| `torch` / `torchvision` | Model inference, transforms |
| `Pillow` | Image open, resize, verify, base64 encode |
| `numpy` | Heatmap arithmetic |
| `scikit-learn` | LIME superpixel segmentation (MiniBatchKMeans, Ridge) |

No web framework (Flask, FastAPI, etc.) is used — only Python stdlib `http.server`.
