# Leaf+Sky: Urban Micro-Farm Co-Pilot

A balcony/rooftop-farming co-pilot that fuses **leaf images** with **local weather reanalysis** and optional **satellite context** to recommend watering, shading, and pest vigilance schedules. Unlike row-crop farms, micro-farms face heat islands and planters that dry at different rates.

Innovation: multimodal fusion that learns from plant leaf patches (stress cues), tabular weather features (hourly temperature, humidity proxies), and optional planter graph relations. Ships with a CPU-first baseline and hooks to add Sentinel-2 or phone EXIF-based sunlight estimates.

## Why this matters
- Real-world impact with small, concrete wins.
- CPU-first baselines; scale up only when needed.
- Clear provenance and respectful handling of data.

## Key techniques
- CNN-lite + tabular fusion, attention pooling, planter graph (optional)

## Datasets (public)
- PlantVillage (leaf disease images): https://data.mendeley.com/datasets/tywbtsjrjv/1
- ERA5 hourly (reanalysis via CDS API): https://cds.climate.copernicus.eu/
- Sentinel-2 (Copernicus Open Access Hub): https://scihub.copernicus.eu/

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
