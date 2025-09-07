import argparse, os, numpy as np, xarray as xr

def _stack_bands(ds, wanted_names):
    import xarray as xr
    synonyms = {'nir':['nir','nir08'],'swir1':['swir1','swir16'],'swir2':['swir2','swir22'],'blue':['blue'],'green':['green'],'red':['red']}
    out=[]; names=[]
    for name in wanted_names:
        var=None
        for c in synonyms.get(name,[name]):
            if c in ds.data_vars: var=c; break
        if var is None: continue
        out.append(ds[var]); names.append(name)
    return xr.concat(out, dim='band').assign_coords(band=names)


def fit_linear(x, y):
    x = x.reshape(-1,1); y = y.reshape(-1,1)
    mask = np.isfinite(x[:,0]) & np.isfinite(y[:,0])
    if mask.sum() < 1000: return 0.0, 1.0
    X = np.hstack([np.ones((mask.sum(),1)), x[mask]])
    beta = np.linalg.pinv(X) @ y[mask]; a,b = beta[0,0], beta[1,0]; return a,b
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--landsat_paths', nargs='+', required=True)
    ap.add_argument('--modis_paths',  nargs='+', required=True)
    ap.add_argument('--site', required=False)
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--gcs_auth', default='google_default')
    args = ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)

    if args.gcs_auth == 'google_default': backend_kwargs = {'storage_options': {'token': 'google_default'}}
    elif args.gcs_auth == 'anon':        backend_kwargs = {'storage_options': {'token': None}}
    else:                                backend_kwargs = {'storage_options': {'token': args.gcs_auth}}

    ls_ds = xr.open_mfdataset(args.landsat_paths, combine='by_coords', engine='h5netcdf', backend_kwargs=backend_kwargs)
    md_ds = xr.open_mfdataset(args.modis_paths,  combine='by_coords', engine='h5netcdf', backend_kwargs=backend_kwargs)


    times = [t for t in md_ds['time'].values if str(t).startswith(str(args.year))]
    # Build a single Landsat stack across year to find anchors
    ls_full = xr.concat([_stack_bands(ls_ds.sel(time=t), ['red','nir','green','blue','swir1','swir2']) for t in ls_ds['time'].values], dim='time')
    for t in times:
        before = [tt for tt in ls_ds['time'].values if tt<=t]
        after  = [tt for tt in ls_ds['time'].values if tt>=t]
        if not before or not after: continue
        t0 = before[-1]; t1 = after[0]
        ls0 = _stack_bands(ls_ds.sel(time=t0), ['red','nir','green','blue','swir1','swir2'])
        ls1 = _stack_bands(ls_ds.sel(time=t1), ['red','nir','green','blue','swir1','swir2'])
        md0 = _stack_bands(md_ds.sel(time=t0), ['red','nir','green','blue','swir1','swir2']).interp_like(ls0, method='linear')
        md1 = _stack_bands(md_ds.sel(time=t1), ['red','nir','green','blue','swir1','swir2']).interp_like(ls1, method='linear')
        mdt = _stack_bands(md_ds.sel(time=t),  ['red','nir','green','blue','swir1','swir2']).interp_like(ls0, method='linear')

        pred_bands = []
        for bi in range(ls0.shape[0]):
            a0,b0 = fit_linear(md0[bi].values, ls0[bi].values)
            a1,b1 = fit_linear(md1[bi].values, ls1[bi].values)
            w1 = (np.datetime64(t) - np.datetime64(t0)) / (np.datetime64(t1) - np.datetime64(t0) + np.timedelta64(1,'D'))
            w0 = 1.0 - float(w1)
            p0 = a0 + b0 * mdt[bi].values
            p1 = a1 + b1 * mdt[bi].values
            pred = w0*p0 + (1.0-w0)*p1
            pred_bands.append(pred)
        import xarray as xr
        pred = np.stack(pred_bands, axis=0)
        da = xr.DataArray(pred, dims=('band','y','x'), coords={'band': ls0['band'].values, 'y': ls0['y'], 'x': ls0['x']})
        tag = f"{args.year}_{str(t)[:10].replace('-', '')}"
        da.to_netcdf(os.path.join(args.out_dir, f'starfm_{(args.site or 'site')}_{tag}.nc'))

if __name__ == '__main__':
    main()
