# DeepFake Detector web app

This is a small, dependency-free local UI for the existing Wide CNN and Patch CNN.

From the repository root, use the same Python environment where the project's Torch dependencies are installed:

```powershell
python faces_deepfake/webapp/server.py
```

Open `http://127.0.0.1:8000`. The **Scan image** tab accepts uploads or a browser camera capture. It displays the weighted-fusion decision, probabilities, both CNN decisions, patch count, and local-texture variation. **Rescan** runs the selected image through the cached models again.

The final verdict is a weighted mean of the base CNN real probabilities (`wide:patch = 1.0072312107559:1`). The **Model data** tab renders the active two-CNN report, including architecture and layer counts, parameter/memory/compute budgets, checkpoint metadata, and base-model metrics. The source data is produced by `user/calculate_model_parameters.py`.

Uploaded images are checked, stored only in a temporary file during inference, and deleted immediately afterwards. The server is bound to localhost only.
