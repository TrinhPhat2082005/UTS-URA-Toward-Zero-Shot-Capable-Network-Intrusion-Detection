"""Warmup + cosine-annealing LR schedule — copy nguyen tu UTS_Layer1/clad/util/schedules.py."""

import math


class WarmupCosineSchedule:
    """
    Linear warmup for `warmup_steps`, then cosine decay to `min_lr`.

    Usage:
        schedule = WarmupCosineSchedule(optimizer, warmup_steps, total_steps,
                                        init_lr, peak_lr, min_lr)
        for step in range(total_steps):
            schedule.step()
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 init_lr: float = 1e-6, peak_lr: float = 4.4e-6,
                 min_lr: float = 1e-6):
        self.optimizer     = optimizer
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self.init_lr       = init_lr
        self.peak_lr       = peak_lr
        self.min_lr        = min_lr
        self._step         = 0
        self._set_lr(init_lr)

    def step(self):
        self._step += 1
        lr = self._get_lr()
        self._set_lr(lr)
        return lr

    def _get_lr(self):
        s = self._step
        if s <= self.warmup_steps:
            # Linear warmup
            return self.init_lr + (self.peak_lr - self.init_lr) * s / self.warmup_steps
        # Cosine decay
        progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
