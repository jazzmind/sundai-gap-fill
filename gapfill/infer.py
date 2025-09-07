import argparse, os, numpy as np, torch, xarray as xr
from tqdm import tqdm
from models.unet import UNetSmall

import xarray as xr, numpy as np, pandas as pd, torch, os
from models.unet import UNetSmall

def _stack_bands(ds, wanted_names):
    synonyms = {'nir':['nir','nir08','nir8','nir_08'],'swir1':['swir1','swir16','swir_1','swir_16'],'swir2':['swir2','swir22','swir_2','swir_22'],'blue':['blue','B02'],'green':['green','B03'],'red':['red','B04']}
    out = []; names=[]
    for name in wanted_names:
        var=None
        for cand in synonyms.get(name,[name]):
            if cand in ds.data_vars: var=cand; break
        if var is None: continue
        out.append(ds[var]); names.append(name)
    da = xr.concat(out, dim='band').assign_coords(band=names)
    return da


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--landsat_paths', nargs='+', required=True)
    ap.add_argument('--modis_paths',  nargs='+', required=True)
    ap.add_argument('--embed_paths',  nargs='+', required=True)
    ap.add_argument('--landsat_var', default=None)
    ap.add_argument('--modis_var',   default=None)
    ap.add_argument('--embed_var',   default='embedding')
    ap.add_argument('--site', required=False)
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--gcs_auth', default='google_default')
    return ap.parse_args()

def main():
    args = parse(); os.makedirs(args.out_dir, exist_ok=True)
    ck = torch.load(args.ckpt, map_location='cpu')
    model = UNetSmall(ck['in_ch'], ck['out_ch']); model.load_state_dict(ck['model']); model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'; model.to(device)

    if args.gcs_auth == 'google_default':
        backend_kwargs = {'storage_options': {'token': 'google_default'}}
    elif args.gcs_auth == 'anon':
        backend_kwargs = {'storage_options': {'token': None}}
    else:
        backend_kwargs = {'storage_options': {'token': args.gcs_auth}}

    ls_ds = xr.open_mfdataset(args.landsat_paths, combine='by_coords', engine='h5netcdf', backend_kwargs=backend_kwargs)
    md_ds = xr.open_mfdataset(args.modis_paths,  combine='by_coords', engine='h5netcdf', backend_kwargs=backend_kwargs)
    eb_ds = xr.open_mfdataset(args.embed_paths,  combine='by_coords', engine='h5netcdf', backend_kwargs=backend_kwargs)

    # Select times for the requested year based on Landsat timeline
    times = [t for t in ls_ds['time'].values if str(t).startswith(str(args.year))]

    for t in tqdm(times, desc='Predict daily 30 m'):
        ls_slice = ls_ds.sel(time=t)
        md_slice = md_ds.sel(time=t)
        ls_b = _stack_bands(ls_slice, ['red','nir','green','blue','swir1','swir2'])
        md_b = _stack_bands(md_slice, ['red','nir','green','blue','swir1','swir2'])
        mod  = md_b.interp_like(ls_b, method='linear')
        # Embeddings: pick nearest year in the embedding time coord
        import pandas as pd, numpy as np
        yr = pd.to_datetime(str(t)).year
        eb_var = args.embed_var if args.embed_var in eb_ds.data_vars else 'embedding'
        if 'time' in eb_ds[eb_var].coords:
            eb_year = eb_ds[eb_var].sel(time=str(yr), method='nearest')
        else:
            eb_year = eb_ds[eb_var]
        eb_year = eb_year.interp_like(ls_b, method='nearest')
        # Day-of-year encodings
        dt = pd.to_datetime(str(t)); sin_d = np.sin(2*np.pi*dt.timetuple().tm_yday/365.0); cos_d = np.cos(2*np.pi*dt.timetuple().tm_yday/365.0)
        x = np.concatenate([
            np.nan_to_num(mod.values/10000.0, nan=np.nanmedian(mod.values)/10000.0),
            np.nan_to_num(eb_year.values, nan=np.nanmedian(eb_year.values)),
            np.full((1,)+ls_b.shape[-2:], sin_d, dtype=np.float32),
            np.full((1,)+ls_b.shape[-2:], cos_d, dtype=np.float32)
        ], axis=0)
        xb = torch.from_numpy(x).unsqueeze(0).to(device, dtype=torch.float32)
        with torch.no_grad():
            yhat = model(xb)[0].cpu().numpy()
        da = xr.DataArray(yhat, dims=('band','y','x'), coords={'band': ls_b['band'].values, 'y': ls_b['y'], 'x': ls_b['x']})
        tag = f"{args.year}_{str(t)[:10].replace('-', '')}"
        da.to_netcdf(os.path.join(args.out_dir, f'pred_{(args.site or 'site')}_{tag}.nc'))

if __name__ == '__main__':
    main()
