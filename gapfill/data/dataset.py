# data/dataset.py
import glob
import math
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


def _expand_urls(paths, storage_options=None):
    """
    Expand user patterns into concrete URLs/paths.

    Supports:
      - '@file.txt' -> read newline-separated paths (each may be gs://, https://, or local)
      - 'gs://bucket/prefix/*.nc' -> glob via gcsfs (requires LIST permission)
      - 'gs://bucket/path/file.nc' (no wildcard) -> return as-is (no LIST needed)
      - 'https://storage.googleapis.com/.../file.nc' -> return as-is (no globbing)
      - local globs -> Python glob
    """
    import fsspec

    out = []
    for p in paths:
        if not p:
            continue

        # Filelist indirection
        if p.startswith("@"):
            with open(p[1:], "r") as f:
                more = [line.strip() for line in f if line.strip()]
            out.extend(_expand_urls(more, storage_options=storage_options))
            continue

        # GCS
        if p.startswith("gs://"):
            # Only glob if there is an explicit wildcard
            if any(ch in p for ch in ["*", "?", "["]):
                fs = fsspec.filesystem("gcs", **(storage_options or {}))
                # strip scheme for fs.glob, then re-add scheme to matches
                bucket_key = p[5:]
                matches = fs.glob(bucket_key)
                out.extend(
                    [m if m.startswith("gs://") else f"gs://{m}" for m in matches]
                )
            else:
                out.append(p)
            continue

        # HTTPS (public objects) – no globbing
        if p.startswith("https://") or p.startswith("http://"):
            out.append(p)
            continue

        # Local filesystem glob
        out.extend(glob.glob(p))

    return sorted(out)


def _open_mf(paths, storage_options=None, engine="h5netcdf"):
    files = _expand_urls(paths, storage_options=storage_options)
    if not files:
        raise FileNotFoundError(f"No files matched: {paths}")

    # Only pass storage_options to backend when actually reading gs://
    backend_kwargs = {}
    if any(f.startswith("gs://") for f in files) and (storage_options is not None):
        backend_kwargs["storage_options"] = storage_options

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        engine=engine,
        backend_kwargs=backend_kwargs,
        chunks={},
    )
    return ds


def _rescale_reflectance(arr, scale=10000.0):
    """Scale reflectance-like arrays from 0..scale to ~0..1.2 (minor headroom)."""
    a = np.asarray(arr, dtype=np.float32) / float(scale)
    return np.clip(a, 0.0, 1.2)


def _stack_bands(ds, wanted_names):
    """
    Build a [band, y, x] DataArray by stacking per-band variables from ds.

    Handles common synonyms:
      nir -> nir08
      swir1 -> swir16
      swir2 -> swir22
    """
    synonyms = {
        "nir": ["nir", "nir08", "nir8", "nir_08"],
        "swir1": ["swir1", "swir16", "swir_1", "swir_16"],
        "swir2": ["swir2", "swir22", "swir_2", "swir_22"],
        "blue": ["blue", "B02", "band2"],
        "green": ["green", "B03", "band3"],
        "red": ["red", "B04", "band4"],
    }
    stacked = []
    names_out = []
    for name in wanted_names:
        var_name = None
        for cand in synonyms.get(name, [name]):
            if cand in ds.data_vars:
                var_name = cand
                break
        if var_name is None:
            # Skip missing bands gracefully
            continue
        da = ds[var_name]
        if not set(["y", "x"]).issubset(da.dims):
            # Expect 2D or 3D with time; time selection should be done by caller.
            continue
        stacked.append(da)
        names_out.append(name)
    if not stacked:
        raise ValueError(
            f"None of the requested bands {wanted_names} found in dataset vars: {list(ds.data_vars)}"
        )
    arr = xr.concat(stacked, dim="band").assign_coords(band=names_out)
    return arr


class AlphaEarthGapfillDataset(Dataset):
    """
    Dataset for AlphaEarth-conditioned gap filling.

    Expects three NetCDF groups / file patterns:
      - Landsat-like high-res per-band variables (e.g., red, nir08, blue, ...)
      - MODIS daily per-band variables on coarse grid (will be upsampled)
      - Embeddings variable (default name 'embedding') with dims [band, y, x, time] or [band, y, x, year]

    Notes:
      - Reflectance is assumed scaled to ~0..10000 and will be divided by 10000.
      - Embeddings time is interpreted as yearly; we pick the nearest year to each sample date.
      - No 'site' dimension is required.
    """

    def __init__(
        self,
        landsat_paths,
        modis_paths,
        embed_paths,
        landsat_var=None,  # unused (compat placeholder)
        modis_var=None,  # unused (compat placeholder)
        embed_var="embedding",
        bands=("red", "nir", "green", "blue", "swir1", "swir2"),
        patch=128,
        simulate_modis_gaps=True,
        modis_drop_prob=0.2,
        seed=42,
        gcs_auth="google_default",
    ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        # Storage options for GCS auth
        if gcs_auth == "google_default":
            storage_options = {"token": "google_default"}
        elif gcs_auth == "anon":
            storage_options = {"token": None}
        else:
            # assume path to a service-account JSON
            storage_options = {"token": gcs_auth}

        self.ls_ds = _open_mf(
            landsat_paths, storage_options=storage_options, engine="h5netcdf"
        )
        self.md_ds = _open_mf(
            modis_paths, storage_options=storage_options, engine="h5netcdf"
        )
        self.eb_ds = _open_mf(
            embed_paths, storage_options=storage_options, engine="h5netcdf"
        )

        self.embed_var = embed_var
        self.bands = list(bands)
        self.patch = patch
        self.sim_modis = simulate_modis_gaps
        self.modis_drop_prob = modis_drop_prob

        # Build index of valid timestamps based on Landsat availability
        self.samples = []
        self.time_index = self.ls_ds["time"].values
        for i, t in enumerate(self.time_index):
            try:
                ls_slice = self.ls_ds.sel(time=t)
                ls_stack = _stack_bands(ls_slice, self.bands)
                if np.all(np.isnan(ls_stack.values)):
                    continue
                self.samples.append(i)
            except Exception:
                # If stacking fails (e.g., no matching band vars), skip
                continue

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _choose_patch(H, W, ps):
        y0 = 0 if H <= ps else np.random.randint(0, H - ps)
        x0 = 0 if W <= ps else np.random.randint(0, W - ps)
        return slice(y0, y0 + ps), slice(x0, x0 + ps)

    def __getitem__(self, idx):
        import pandas as pd

        i = self.samples[idx]
        t = self.time_index[i]

        # --- Landsat label (high-res), stacked to [B,H,W]
        ls_slice = self.ls_ds.sel(time=t)
        ls = _stack_bands(ls_slice, self.bands)  # [B,H,W]
        ls_np = _rescale_reflectance(ls.values)

        # --- MODIS input (upsampled to Landsat grid)
        md_slice = self.md_ds.sel(time=t)
        md = _stack_bands(md_slice, self.bands)
        md_up = md.interp_like(ls, method="linear")
        md_np = _rescale_reflectance(md_up.values)

        # --- Embeddings (yearly), align to Landsat grid
        eb_name = self.embed_var if self.embed_var in self.eb_ds.data_vars else "embedding"
        eb_all = self.eb_ds[eb_name]
        yr = pd.to_datetime(str(t)).year
        if "time" in eb_all.coords:
            # time likely has yearly stamps like 2017-01-01; select nearest
            eb_year = eb_all.sel(time=str(yr), method="nearest")
        elif "year" in eb_all.coords:
            eb_year = eb_all.sel(year=yr)
        else:
            eb_year = eb_all
        eb_al = eb_year.interp_like(ls, method="nearest")
        eb_np = np.asarray(eb_al.values, dtype=np.float32)  # [E,H,W]

        # Simulate MODIS cloud/dropout blocks
        if self.sim_modis and random.random() < self.modis_drop_prob:
            h, w = md_np.shape[-2:]
            bh = max(16, h // 6)
            bw = max(16, w // 6)
            y0 = np.random.randint(0, max(1, h - bh))
            x0 = np.random.randint(0, max(1, w - bw))
            md_np[:, y0 : y0 + bh, x0 : x0 + bw] = np.nan

        # Choose random training patch
        H, W = ls_np.shape[-2:]
        ys, xs = self._choose_patch(H, W, self.patch)
        ls_np = ls_np[:, ys, xs]
        md_np = md_np[:, ys, xs]
        eb_np = eb_np[:, ys, xs]

        # Replace remaining NaNs in inputs with local medians
        def _nan_to_med(a):
            if np.isnan(a).any():
                med = np.nanmedian(a, axis=(-2, -1), keepdims=True)
                a = np.where(np.isnan(a), med, a)
            return a

        md_np = _nan_to_med(md_np)
        eb_np = _nan_to_med(eb_np)

        # Time encodings (DOY sin/cos) as two scalar channels
        dt = pd.to_datetime(str(t))
        doy = dt.timetuple().tm_yday
        sin = np.full((1, ls_np.shape[1], ls_np.shape[2]), np.sin(2 * np.pi * doy / 365.0), dtype=np.float32)
        cos = np.full((1, ls_np.shape[1], ls_np.shape[2]), np.cos(2 * np.pi * doy / 365.0), dtype=np.float32)

        x = np.concatenate([md_np, eb_np, sin, cos], axis=0)  # [C_in,H,W]
        y = ls_np  # [C_out,H,W]

        return torch.from_numpy(x), torch.from_numpy(y)
