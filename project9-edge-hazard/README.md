# TinySmoke+Siren

Early hazard detection using a cheap gas sensor array (drift-prone) fused with audio (sirens, crackles). Designs TinyML-friendly features and domain adaptation to handle sensor drift.

Innovation: <$25 bill-of-materials with optional microcontroller sketch and quantized model.

## Why this matters
- Real-world impact with small, concrete wins.
- CPU-first baselines; scale up only when needed.
- Clear provenance and respectful handling of data.

## Key techniques
- feature fusion, domain adaptation (drift), quantized classifier (optional)

## Datasets (public)
- UCI Gas Sensor Array Drift: https://archive.ics.uci.edu/dataset/95/gas+sensor+array+drift
- UrbanSound8K: https://urbansounddataset.weebly.com/urbansound8k.html

## Step-by-step
1. Create a fresh environment and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. (Optional) Place or symlink datasets under `data/` (see links above).
3. Train a tiny baseline:
   ```bash
   python src/train.py --epochs 3 --out runs/base
   ```
4. Run inference/demo:
   ```bash
   python src/infer.py --ckpt runs/base/model.pt
   ```

## Code map
```
src/
  datasets.py        # loads real data if present; otherwise uses synthetic samples
  models/
    core.py          # minimal torch/sklearn models per project flavor
  train.py           # training loop with helpful prints
  infer.py           # simple inference/demo script
```

## Extensions
- Swap the baseline model for an advanced variant aligned with the project's theme.
- Add uncertainty/fairness/energy tracking where appropriate.
- Publish a small demo (Streamlit/Gradio) and screenshots.
