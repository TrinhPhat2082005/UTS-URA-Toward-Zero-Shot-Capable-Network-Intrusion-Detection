"""Checkpoint save/load — copy nguyen tu UTS_Layer1/clad/util/checkpoint.py."""

import os
import torch


def save_checkpoint(path: str, model, optimizer, epoch: int, metrics: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metrics': metrics,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt.get('epoch', 0), ckpt.get('metrics', {})
