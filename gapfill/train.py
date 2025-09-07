import argparse, os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset import AlphaEarthGapfillDataset
from models.unet import UNetSmall
from losses import L1SAMLoss
from metrics import rmse, mae, psnr, ssim_simple

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--landsat_paths', nargs='+', required=True)
    ap.add_argument('--modis_paths',  nargs='+', required=True)
    ap.add_argument('--embed_paths',  nargs='+', required=True)
    ap.add_argument('--landsat_var', default=None)
    ap.add_argument('--modis_var',   default=None)
    ap.add_argument('--embed_var',   default='embedding')
    ap.add_argument('--bands', nargs='+', default=['red','nir','green','blue'])
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--patch', type=int, default=128)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--gcs_auth', default='google_default', help='google_default|anon|/path/key.json')
    return ap.parse_args()

def main():
    args = parse(); os.makedirs(args.out_dir, exist_ok=True)
    ds = AlphaEarthGapfillDataset(
        landsat_paths=args.landsat_paths,
        modis_paths=args.modis_paths,
        embed_paths=args.embed_paths,
        landsat_var=args.landsat_var,
        modis_var=args.modis_var,
        embed_var=args.embed_var,
        bands=tuple(args.bands),
        patch=args.patch,
        gcs_auth=args.gcs_auth,
    )
    n = len(ds); n_val = max(100, int(0.1*n)); n_train = max(1, n - n_val)
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.workers)

    x0, y0 = ds[0]; in_ch = x0.shape[0]; out_ch = y0.shape[0]
    model = UNetSmall(in_ch, out_ch)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'; model.to(device)

    opt = AdamW(model.parameters(), lr=args.lr)
    loss_fn = L1SAMLoss()
    scaler = GradScaler()
    best_val = float('inf'); best_path = os.path.join(args.out_dir, 'best.pt')

    for epoch in range(1, args.epochs+1):
        model.train(); tr_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for xb, yb in pbar:
            xb, yb = xb.to(device, dtype=torch.float32), yb.to(device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            with autocast():
                pred = model(xb); loss = loss_fn(pred, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tr_loss += loss.item(); pbar.set_postfix(loss=tr_loss/max(1,len(pbar)))

        model.eval(); v_rmse=v_mae=v_psnr=v_ssim=0.0; n_batches=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, dtype=torch.float32), yb.to(device, dtype=torch.float32)
                pred = model(xb)
                v_rmse += rmse(pred, yb); v_mae += mae(pred, yb); v_psnr += psnr(pred, yb); v_ssim += ssim_simple(pred, yb)
                n_batches += 1
        v_rmse/=n_batches; v_mae/=n_batches; v_psnr/=n_batches; v_ssim/=n_batches
        log = f'val_rmse={v_rmse:.4f} val_mae={v_mae:.4f} val_psnr={v_psnr:.2f} val_ssim={v_ssim:.3f}'
        print(log)
        with open(os.path.join(args.out_dir, 'log.txt'), 'a') as f: f.write(log + '\n')
        if v_rmse < best_val: best_val = v_rmse; 
        if v_rmse <= best_val:
            best_val = v_rmse
            import torch as _t; _t.save({'model':model.state_dict(),'in_ch':in_ch,'out_ch':out_ch}, best_path)
    print('Best checkpoint:', best_path)

if __name__ == '__main__':
    main()
