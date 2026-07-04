#!/usr/bin/env python3
"""
download_weights.py — Download pretrained RandLA-Net SemanticKITTI weights.

Usage:
    python3 download_weights.py [--out /path/to/weights.pth]

The script downloads the tsunghan-wu/RandLA-Net-pytorch checkpoint
(PyTorch native, SemanticKITTI, ~52.9% mIoU) and saves it as a .pth file
compatible with RandLANet.from_pretrained().

After downloading, set model_path in config/semantic.yaml to the output path.
"""

import argparse
import os
import sys
import urllib.request

## Known PyTorch checkpoints for RandLA-Net SemanticKITTI
#CHECKPOINTS = {
#    'tsunghan': {
#        'url':    'https://github.com/tsunghan-wu/RandLA-Net-pytorch/releases/download/v1.0/checkpoint.tar',
#        'mIoU':   '52.9%',
#        'format': 'tar',   # contains model_state_dict key
#    },
#}

# Known PyTorch checkpoints for RandLA-Net SemanticKITTI
CHECKPOINTS = {
    'tsunghan': {
        'url':    'https://raw.githubusercontent.com/tsunghan-wu/RandLA-Net-pytorch/master/pretrain_model/checkpoint.tar',
        'mIoU':   '52.9%',
        'format': 'tar',   # contains model_state_dict key
    },
}

def download(url: str, dest: str):
    print(f'Downloading {url}')
    print(f'Saving to   {dest}')

    def _progress(count, block_size, total_size):
        pct = min(100, int(count * block_size * 100 / max(1, total_size)))
        bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
        print(f'\r  [{bar}] {pct}%', end='', flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def main():
    ap = argparse.ArgumentParser(description='Download RandLA-Net pretrained weights')
    ap.add_argument('--source', default='tsunghan',
                    choices=list(CHECKPOINTS.keys()),
                    help='Which checkpoint to download')
    ap.add_argument('--out', default=None,
                    help='Output path (default: weights/<source>.pth alongside this script)')
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(os.path.dirname(script_dir), 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    ck   = CHECKPOINTS[args.source]
    dest = args.out or os.path.join(weights_dir, f'semantickitti_{args.source}.tar')

    print(f'Source : {args.source}')
    print(f'mIoU   : {ck["mIoU"]} on SemanticKITTI val')
    print()

    download(ck['url'], dest)

    print(f'\nDone. Set model_path in config/semantic.yaml to:\n  {dest}')
    print()
    print('Note: this checkpoint was trained on outdoor driving scenes (SemanticKITTI).')
    print('For indoor/corridor use, fine-tuning on your own data is recommended.')


if __name__ == '__main__':
    main()
