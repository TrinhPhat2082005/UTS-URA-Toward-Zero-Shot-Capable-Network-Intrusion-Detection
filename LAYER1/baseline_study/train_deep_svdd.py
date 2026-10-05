"""GPU entry point for Mixture Deep SVDD with benign-quantile calibration."""

import os
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault(
    "SVDD_ARTIFACTS_ROOT",
    os.environ.get("DEEP_SVDD_ARTIFACTS_ROOT", str(BASE_DIR / "artifacts" / "deep_svdd")),
)

from train_svdd_only import main  # noqa: E402


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. In Colab choose Runtime > Change runtime type > GPU, "
            "then rerun this script."
        )
    main()
