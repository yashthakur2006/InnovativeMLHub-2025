# Sleep Guardian FL

Federated learning for sleep staging + stress spillover indicators from wearables (simulated clients). Clients train locally and send **privacy-preserving** updates (optional DP-SGD), aggregated with FedAvg.

Innovation: end-to-end privacy budget tracking and model explainability via band-power and time-in-stage summaries.

## Why this matters
- Real-world impact with small, concrete wins.
- CPU-first baselines; scale up only when needed.
- Clear provenance and respectful handling of data.

## Key techniques
- FedAvg/FedProx, DP-SGD (optional), secure aggregation (stub)

## Datasets (public)
- Sleep-EDF: https://physionet.org/content/sleep-edfx/1.0.0/
- WESAD: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

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
