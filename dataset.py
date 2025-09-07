import os, glob, math, random, warnings
from pathlib import Path
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


def _expand_urls(paths, storage_options=None):
    """Expand user patterns into concrete URLs/paths.
    - If 'gs://...' with wildcards, use gcsfs.glob and return *gs://bucket/key* URLs.
    - If 'https://storage.googleapis.com/...' allow direct files (no glob).
    - Else, use local glob.
    """
    import glob as _glob
    out = []
    for p in paths:
        if p.startswith('gs://'):
            import fsspec
            fs = fsspec.filesystem('gcs', **(storage_options or {}))
            # Strip scheme for fs.glob, then re-add 'gs://'
            bucket_key = p[5:]
            matches = fs.glob(bucket_key)
            out.extend([f'gs://{m}' if not m.startswith('gs://') else m for m in matches])
        elif p.startswith('https://'):
            # No globbing over https; just pass through
            out.append(p)
        else:
            out.extend(_glob.glob(p))
    return sorted(out)

def _open_mf(paths, storage_options=None, engine='h5netcdf'):
    files = _expand_urls(paths, storage_options=storage_options)
    if not files:
        raise FileNotFoundError(f"No files matched: {paths}")
    backend_kwargs = {}
    if storage_options is not None:
        backend_kwargs['storage_options'] = storage_options
    ds = xr.open_mfdataset(files, combine='by_coords', engine=engine, backend_kwargs=backend_kwargs, chunks={})
    return ds

def _rescale_reflectance(arr, scale=10000.0):
    a = np.asarray(arr, dtype=np.float32) / float(scale)
    return np.clip(a, 0.0, 1.2)


def _stack_bands(ds, wanted_names):
    """Return DataArray shaped [band,y,x] by stacking variables from ds.
    Supports synonyms like nir->nir08, swir1->swir16, swir2->swir22.
    """
    synonyms = {
        'nir': ['nir','nir08','nir8','nir_08'],
        'swir1': ['swir1','swir16','swir_1','swir_16'],
        'swir2': ['swir2','swir22','swir_2','swir_22'],
        'blue': ['blue','B02','band2'],
        'green':['green','B03','band3'],
        'red':  ['red','B04','band4'],
    }
    stacked = []
    names_out = []
    for name in wanted_names:
        candidates = synonyms.get(name, [name])
        var_name = None
        for c in candidates:
            if c in ds.data_vars:
                var_name = c; break
        if var_name is None:
            # Skip missing bands gracefully
            continue
        da = ds[var_name]
        # Accept either [time,y,x] or already [y,x]; leave time to outer code
        if set(['y','x']).issubset(da.dims):
            stacked.append(da); names_out.append(name)
    if not stacked:
        raise ValueError(f"None of the requested bands {wanted_names} found in dataset vars: {list(ds.data_vars)}")
    # If time exists, the caller will select time before calling this helper.
    arr = xr.concat(stacked, dim='band').assign_coords(band=names_out)
    return arr

class AlphaEarthGapfillDataset(Dataset):
    def __init__(self,
                 landsat_paths,
                 modis_paths,
                 embed_paths,
                 landsat_var='landsat_reflectance',
                 modis_var='modis_reflectance',
                 embed_var='embeddings',
                 bands=('red','nir','green','blue'),
                 patch=128,
                 simulate_modis_gaps=True,
                 modis_drop_prob=0.2,
                 seed=42,
                 gcs_auth='google_default'):
        super().__init__()
        random.seed(seed); np.random.seed(seed)

        if gcs_auth == 'google_default':
            storage_options = {'token': 'google_default'}
        elif gcs_auth == 'anon':
            storage_options = {'token': None}
        else:
            storage_options = {'token': gcs_auth}

        self.ls_ds = _open_mf(landsat_paths, storage_options=storage_options, engine='h5netcdf')
        self.md_ds = _open_mf(modis_paths,  storage_options=storage_options, engine='h5netcdf')
        self.eb_ds = _open_mf(embed_paths,  storage_options=storage_options, engine='h5netcdf')

        self.ls_var = landsat_var
        self.md_var = modis_var
        self.eb_var = embed_var
        self.bands = list(bands)
        self.patch = patch
        self.sim_modis = simulate_modis_gaps
        self.modis_drop_prob = modis_drop_prob

        self.samples = []
        self.time_index = self.ls_ds['time'].values

        for i, t in enumerate(self.time_index):
            ls_t = self.ls_ds[self.ls_var].sel(time=t)
            if np.all(np.isnan(ls_t)):
                continue
            self.samples.append(i)

    def __len__(self):
        return len(self.samples)

    def _choose_patch(self, H, W):
        ps = self.patch
        y0 = 0 if H<=ps else np.random.randint(0, H-ps)
        x0 = 0 if W<=ps else np.random.randint(0, W-ps)
        return slice(y0, y0+ps), slice(x0, x0+ps)

    def __getitem__(self, idx):
        i = self.samples[idx]
        t = self.time_index[i]

        ls_slice = self.ls_ds.sel(time=t)
        ls = _stack_bands(ls_slice, self.bands)
        ls_np = _rescale_reflectance(ls.values)

        md_slice = self.md_ds.sel(time=t)
        md = _stack_bands(md_slice, self.bands)
        md_up = md.interp_like(ls, method='linear')
        md_np = _rescale_reflectance(md_up.values)

        eb_var = self.eb_var if self.eb_var in self.eb_ds.data_vars else 'embedding'
        eb_all = self.eb_ds[eb_var]
        import pandas as pd
        yr = pd.to_datetime(str(t)).year
        # If embeddings have time coord of yearly stamps, select nearest year
        if 'time' in eb_all.coords:
            eb_year = eb_all.sel(time=str(yr), method='nearest')
        elif 'year' in eb_all.coords:
            eb_year = eb_all.sel(year=yr)
        else:
            eb_year = eb_all
        # Ensure dims are [band,y,x]
        if 'band' in eb_year.dims and set(['y','x']).issubset(eb_year.dims):
            eb_da = eb_year
        else:
            raise ValueError('Embeddings must have dims including band,y,x')
        eb_al = eb_da.interp_like(ls, method='nearest')
        eb_np = np.asarray(eb_al.values, dtype=np.float32)

        if self.sim_modis and random.random() < self.modis_drop_prob:
            h, w = md_np.shape[-2:]
            bh = max(16, h//6); bw = max(16, w//6)
            y0 = np.random.randint(0, h-bh); x0 = np.random.randint(0, w-bw)
            md_np[:, y0:y0+bh, x0:x0+bw] = np.nan

        H, W = ls_np.shape[-2:]
        ys, xs = self._choose_patch(H, W)
        ls_np = ls_np[:, ys, xs]; md_np = md_np[:, ys, xs]; eb_np = eb_np[:, ys, xs]

        def nan_to_med(a):
            if np.isnan(a).any():
                med = np.nanmedian(a, axis=(-2,-1), keepdims=True)
                a = np.where(np.isnan(a), med, a)
            return a
        md_np = nan_to_med(md_np); eb_np = nan_to_med(eb_np)

        import pandas as pd, numpy as np
        dt = pd.to_datetime(str(t))
        doy = dt.timetuple().tm_yday
        sin = np.full((1, ls_np.shape[1], ls_np.shape[2]), np.sin(2*np.pi*doy/365.0), dtype=np.float32)
        cos = np.full((1, ls_np.shape[1], ls_np.shape[2]), np.cos(2*np.pi*doy/365.0), dtype=np.float32)

        x = np.concatenate([md_np, eb_np, sin, cos], axis=0)
        y = ls_np
        return torch.from_numpy(x), torch.from_numpy(y)
