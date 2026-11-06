"""Write per-recording embeddings (one HDF5 per recording, dataset `embedding`, one row per
5-second patch) for a pre-trained encoder or a baseline. From the repository root:

    python -m sleepbench.embed --method cl_pairwise            # checkpoint used in the paper
    python -m sleepbench.embed --method mae --checkpoint X.pt  # a specific checkpoint
    python -m sleepbench.embed --method base_fft               # no pre-training

Output: <output_root>/embeddings/<method>/<study_id>.hdf5
"""
import argparse
import os
import re
from pathlib import Path

import h5py
import torch
from einops import rearrange
from scipy import signal
from tqdm import tqdm

from . import paths
from .pretrain.dataset import SetTransformerDataset, collate_fn_MAE
from .pretrain.model import CL, MAE
from .pretrain.run import build_config
from .pretrain.utils import load_data

BASELINES = ('base_time', 'base_fft')
SPLITS = ['pretrain', 'train', 'validation', 'test']


def fft_embedder(x):
    """Baseline (Freq): log-amplitude and phase spectra of each 5-s patch, averaged over the
    channels of each modality (channel indices as used for the published baseline)."""
    x = rearrange(x, 'b c (t w) -> b c t w', t=60, w=640)
    f = torch.fft.rfft(x, dim=-1)
    amplitude = torch.log10(torch.abs(f)[..., :-1] / 640 + 1e-6) + 6
    phase = torch.angle(f)[..., :-1]
    amplitude, phase = amplitude[..., ::5], phase[..., ::5]
    groups = [slice(0, 7), slice(7, 13), slice(14, 15), slice(15, 16)]
    amplitude = torch.cat([amplitude[:, g].mean(1, keepdim=True) for g in groups], dim=1)
    phase = torch.cat([phase[:, g].mean(1, keepdim=True) for g in groups], dim=1)
    return rearrange(torch.cat((amplitude, phase), dim=3), 'b c t w -> b t (c w)')


def time_embedder(x):
    """Baseline (Time): one channel per modality, decimated 5x, one row per 5-s patch."""
    x = signal.decimate(x[:, [0, 11, 13, 15], :].cpu().numpy(), q=5, axis=-1)
    x = rearrange(torch.tensor(x.copy()), 'b c (t w) -> b c t w', t=60, w=128)
    return rearrange(x, 'b c t w -> b t (c w)')


def pick_checkpoint(method_dir, pattern=None):
    """The checkpoint matching `pattern` (CL: fixed training step), otherwise the one with the
    lowest validation loss (MAE/DAE: early stopping)."""
    if pattern:
        ckpts = sorted(Path(method_dir).rglob(pattern))
        if len(ckpts) != 1:
            raise SystemExit(f'expected one checkpoint {pattern} under {method_dir}, found {len(ckpts)}')
        return ckpts[0]
    ckpts = list(Path(method_dir).rglob('*_val_loss_*.pt'))
    if not ckpts:
        raise SystemExit(f'no best-validation checkpoint (*_val_loss_*.pt) under {method_dir}')
    return min(ckpts, key=lambda f: float(re.search(r'_val_loss_([0-9.eE+-]+?)\.pt$', f.name).group(1)))


def load_encoder(method, cfg, checkpoint, device):
    if cfg['model'] == 'CL':
        model = CL(input_size=cfg['input_size'], output_size=cfg['output_size'],
                   num_heads=cfg['num_heads'], num_layers=cfg['num_layers'],
                   max_seq_len=cfg['max_seq_len'], dropout=cfg['dropout'])
    else:  # MAE and DAE variants share the MAE encoder; no masking when embedding
        model = MAE(input_size=640, output_size=128, mask_ratio=0.0)
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state.get('model_state_dict', state))
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--checkpoint', help='default: as in the paper (see configs.yaml)')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--splits', nargs='*', default=SPLITS)
    ap.add_argument('--max_files', type=int, help='embed only this many (random) recordings per split, for testing')
    args = ap.parse_args()
    p = paths.load()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, cfg = build_config('mae' if args.method in BASELINES else args.method)
    cfg.update(max_files=args.max_files, val_size=None, num_workers=args.num_workers)
    if args.method == 'base_fft':
        embed = fft_embedder
    elif args.method == 'base_time':
        embed = time_embedder
    else:
        ckpt = args.checkpoint or pick_checkpoint(p.models / args.method, cfg.get('embed_checkpoint'))
        print(f'checkpoint: {ckpt}', flush=True)
        model = load_encoder(args.method, cfg, ckpt, device)
        embed = lambda x: model(x.to(device))[1]

    out_dir = p.embeddings / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    channel_groups = load_data(cfg['channel_groups_path'])
    for split in args.splits:
        ds = SetTransformerDataset(cfg, channel_groups, split=split)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, collate_fn=collate_fn_MAE)
        for x, _, files, _, starts in tqdm(dl, desc=split):
            with torch.no_grad():
                emb = embed(x)
            for e, f, start in zip(emb, files, starts):
                out = out_dir / f"{os.path.basename(f).split('.')[0]}.hdf5"
                row0 = start // (128 * 5)          # 5-second patches since recording start
                with h5py.File(out, 'a') as h:
                    if 'embedding' in h:
                        d = h['embedding']
                        if d.shape[0] < row0 + e.shape[0]:
                            d.resize((row0 + e.shape[0],) + e.shape[1:])
                        d[row0:row0 + e.shape[0]] = e.cpu().numpy()
                    else:
                        h.create_dataset('embedding', data=e.cpu().numpy(), chunks=(128,) + e.shape[1:],
                                         maxshape=(None,) + e.shape[1:])


if __name__ == '__main__':
    main()
