"""Paths and project-wide constants."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SHOPREC_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TOP_K = 12  # final list length (matches the Kaggle MAP@12 metric)
RETRIEVAL_K = 100  # candidate-set size used for retrieval recall
HORIZON_DAYS = 7  # length of each validation / test window


def dataset_dir(name: str = "full") -> Path:
    """Directory of a prepared dataset, e.g. ``full`` or ``sample``."""
    return PROCESSED_DIR / name
