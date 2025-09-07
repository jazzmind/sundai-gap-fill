How to use it (step-by-step, minimal jargon)

0) One-time setup
	1.	Install tools:

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

	2.	Let the code read your bucket (choose one):

	•	Easiest for hackathon:

gcloud auth application-default login

	•	Or a service-account key:

export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json


1) Train the model

This teaches a small UNet to turn (MODIS daily @ t + AlphaEarth embeddings for the year + day-of-year) into Landsat-like 30 m reflectance @ t.

python train.py \
  --landsat_paths gs://sundai-satellite-data/landsat/*.nc \
  --modis_paths  gs://sundai-satellite-data/modis/*.nc \
  --embed_paths  gs://sundai-satellite-data/embeddings/*.nc \
  --landsat_var landsat_reflectance \
  --modis_var   modis_reflectance \
  --embed_var   embeddings \
  --bands red nir green blue swir1 swir2 \
  --out_dir runs/exp1 \
  --epochs 10 --batch 8 --patch 128 \
  --gcs_auth google_default

What happens under the hood (plain English):
	•	We read your NetCDFs directly from GCS (no local copies needed).
	•	For each date with Landsat, we:
	•	Upsample daily MODIS (coarse) to the Landsat grid.
	•	Fetch AlphaEarth embeddings for the same year (your screenshot shows files like alpha_earth_US-*.nc; the dataset treats these as high-res “context maps” that describe the land).
	•	Add day-of-year (sine/cosine) so the model knows where we are in the seasonal cycle.
	•	Teach a UNet to output the Landsat reflectance.
	•	We also simulate MODIS cloud blocks during training so the model learns to cope when MODIS is missing locally.

2) Inference (make your daily 30 m stack)

For a site + year, create a filled daily series—even on days Landsat is cloudy.

python infer.py \
  --ckpt runs/exp1/best.pt \
  --landsat_paths gs://sundai-satellite-data/landsat/*.nc \
  --modis_paths  gs://sundai-satellite-data/modis/*.nc \
  --embed_paths  gs://sundai-satellite-data/embeddings/*.nc \
  --site TX03 --year 2021 \
  --out_dir predictions/TX03_2021 \
  --gcs_auth google_default

What it does:
	•	For each day in that year:
	•	Pull MODIS@day t and embeddings@year for your site.
	•	Predict a 30 m reflectance image for that day.
	•	Save a NetCDF (one file per day) in predictions/TX03_2021.

3) Baseline (for comparison)

Classic fusion without embeddings (sanity floor):

python baselines/starfm_lite.py \
  --landsat_paths gs://sundai-satellite-data/landsat/*.nc \
  --modis_paths  gs://sundai-satellite-data/modis/*.nc \
  --site TX03 --year 2021 \
  --out_dir predictions/starfm_TX03_2021 \
  --gcs_auth google_default

4) How to validate (keep it honest)
	•	Hold out sites and/or years during training, then evaluate on them:
	•	Report RMSE/MAE per band, SSIM on RGB, and spectral-angle (SAM) for fidelity.
	•	Plot NDVI/EVI time series for important pixels/fields and compare to Landsat on available days.
	•	Expect the learned model to beat STARFM-lite on sharpness and small-scale structure, especially when embeddings capture land class/texture.

⸻

Why each data source is in the loop (simple mental model)
	•	MODIS = your daily heartbeat (coarse but up to date).
	•	AlphaEarth embeddings = your prior knowledge about what each 30 m pixel is like (crop/grassland/soil/phenology patterns), so the model can “paint” believable fine detail.
	•	Day-of-year = season context (spring vs late summer).
	•	Landsat (labels) = the truth on clear days to learn from; not used as input at inference time.

⸻

Gotchas & tips
	•	Auth: If you see permission errors, ensure your account has Storage Object Viewer on the project and that you ran gcloud auth application-default login.
	•	Variable names: If your NetCDF vars are named slightly differently, tweak --landsat_var, --modis_var, --embed_var.
	•	Memory: Remote NetCDF reads are chunked; if you hit memory walls, reduce --batch, or add --patch 96 (smaller training crops).
	•	Sites: The example CLI filters by --site at inference. If your NetCDF structure uses another coordinate name, call that out and I’ll adjust the loader in data/dataset.py.

