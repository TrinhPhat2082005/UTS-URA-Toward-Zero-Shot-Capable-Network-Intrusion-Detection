"""
Do latency tung chang cua MOT sample (time.perf_counter, don vi giay).

    t = StageTimer()
    with t("layer1"):
        ...
    t.as_dict()   # {"layer1": 0.0123, ...} theo thu tu chay; goi lai cung ten thi cong don

GPU: chang co CUDA (VLM) phai goi torch.cuda.synchronize() truoc khi thoat khoi `with` de do dung;
vlm/runner.py da do rieng thoi gian generate nen pipeline dung so do.
"""
from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self.times = {}

    @contextmanager
    def __call__(self, stage):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[stage] = self.times.get(stage, 0.0) + (time.perf_counter() - t0)

    def add(self, stage, seconds):
        self.times[stage] = self.times.get(stage, 0.0) + float(seconds)

    def as_dict(self):
        return dict(self.times)

    def total(self):
        return sum(self.times.values())
