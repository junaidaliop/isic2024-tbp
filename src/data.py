"""Data access for SLICE-3D: metadata CSV + lesion crops stored in an HDF5 file.

The Kaggle release ships images inside `train-image.hdf5` keyed by `isic_id`
(JPEG bytes). We decode lazily so the 400k crops never all sit in RAM.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

ID = "isic_id"
GROUP = "patient_id"
TARGET = "target"


def load_metadata(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    assert ID in df.columns, f"metadata missing {ID!r}"
    return df


class HDF5Images:
    """Lazy reader for `train-image.hdf5` / `test-image.hdf5`.

    Usage:
        with HDF5Images('data/train-image.hdf5') as imgs:
            arr = imgs['ISIC_0015670']      # -> HxWx3 uint8 RGB
    """

    def __init__(self, path: str | Path):
        import h5py  # imported here so the module loads without h5py present

        self.path = str(path)
        self._h5 = h5py.File(self.path, "r")

    def __getitem__(self, isic_id: str) -> np.ndarray:
        from PIL import Image

        raw = self._h5[isic_id][()]
        buf = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
        return np.array(Image.open(io.BytesIO(buf)).convert("RGB"))

    def keys(self):
        return list(self._h5.keys())

    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "HDF5Images":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
