# Sundai Hackathon — AlphaEarth-Conditioned Gap Filling (GCS-ready)

This repo predicts **daily 30 m** reflectance from **daily MODIS (coarse)** + **AlphaEarth embeddings (high-res priors)**.
It reads **NetCDF** directly from **Google Cloud Storage (gs://)** via `gcsfs/fsspec`.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Authenticate to GCP for access to gs:// (choose ONE):
gcloud auth application-default login          # uses your user account (recommended for hackathon)
# or
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

## Train (example)
```bash
python train.py   --landsat_paths gs://sundai-satellite-data/landsat/*.nc   --modis_paths  gs://sundai-satellite-data/modis/*.nc   --embed_paths  gs://sundai-satellite-data/embeddings/*.nc   --landsat_var landsat_reflectance   --modis_var   modis_reflectance   --embed_var   embeddings   --bands red nir green blue swir1 swir2   --out_dir runs/exp1   --epochs 10 --batch 8 --patch 128   --gcs_auth google_default
```

## Inference (daily 30 m per site/year)
```bash
python infer.py   --ckpt runs/exp1/best.pt   --landsat_paths gs://sundai-satellite-data/landsat/*.nc   --modis_paths  gs://sundai-satellite-data/modis/*.nc   --embed_paths  gs://sundai-satellite-data/embeddings/*.nc   --site TX03 --year 2021   --out_dir predictions/TX03_2021   --gcs_auth google_default
```

## Baseline (STARFM-lite)
```bash
python baselines/starfm_lite.py   --landsat_paths gs://sundai-satellite-data/landsat/*.nc   --modis_paths  gs://sundai-satellite-data/modis/*.nc   --site TX03 --year 2021   --out_dir predictions/starfm_TX03_2021   --gcs_auth google_default
```

### Auth options
- `--gcs_auth google_default` → uses ADC from `gcloud auth application-default login`
- `--gcs_auth anon` → public buckets
- `--gcs_auth /path/key.json` → service-account JSON key

### Why these steps
- **Train** learns how MODIS daily patterns + AlphaEarth priors map to 30 m detail.
- **Infer** produces a filled 30 m stack for every day, even when Landsat is cloudy/missing.
- **Baseline** gives a sanity floor using classic fusion (no embeddings).

See README in the archive for more detail.
