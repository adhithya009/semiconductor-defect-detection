"""Convert the raw WM-811K pickle (data/LSWMD.pkl) into data/full_dataset.npz (32x32 maps)."""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import interpolate

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 32
SOURCE, OUTPUT = "data/LSWMD.pkl", "data/full_dataset.npz" if SIZE == 32 else f"data/full_dataset_{SIZE}.npz"

# Compatibility aliases required to unpickle this legacy pandas object.
import pandas.core.indexes as _idx
import pandas.core.indexes.base as _base
sys.modules["pandas.indexes"] = _idx
sys.modules["pandas.indexes.base"] = _base
sys.modules["pandas.indexes.numeric"] = _base
if not hasattr(_base, "Int64Index"):
    _base.Int64Index = _base.Index
    _base.Float64Index = _base.Index

with open(SOURCE, "rb") as file:
    df = pickle.load(file, encoding="latin1")
df["label"] = df.failureType.apply(lambda v: v[0][0] if len(v) else None)
df = df.dropna(subset=["label"]).reset_index(drop=True)
classes = sorted(df.label.unique())
labels = df.label.map({name: i for i, name in enumerate(classes)}).to_numpy(dtype=np.int64)
lot_ids, _ = pd.factorize(df.lotName, sort=True)
print(f"Preparing {len(df):,} labeled maps at {SIZE} x {SIZE}")

maps = np.empty((len(df), SIZE, SIZE), dtype=np.float32 if SIZE == 32 else np.uint8)  # larger sizes: raw 0/1/2 values
for i, wafer_map in enumerate(df.waferMap):
    tensor = torch.as_tensor(wafer_map, dtype=torch.float32)[None, None]
    resized = interpolate(tensor, size=(SIZE, SIZE), mode="nearest")[0, 0].numpy()
    maps[i] = resized / 2.0 if SIZE == 32 else resized
np.savez(OUTPUT + ".partial.npz", X=maps, y=labels, groups=lot_ids.astype(np.int32), classes=np.asarray(classes))
os.replace(OUTPUT + ".partial.npz", OUTPUT)
print(f"Saved {OUTPUT}")
