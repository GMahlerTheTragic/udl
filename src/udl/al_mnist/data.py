from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MnistSplits:
    init_labeled_idx: np.ndarray
    val_idx: np.ndarray
    pool_idx: np.ndarray


class HFDatasetsMnist(Dataset):

    def __init__(self, hf_split, indices: np.ndarray | None = None):
        self._ds = hf_split
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices) if self._indices is not None else len(self._ds)

    def __getitem__(self, i: int):
        idx = int(self._indices[i]) if self._indices is not None else i
        ex = self._ds[idx]
        img = ex["image"]
        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        # We are standardizing with the published metrics.
        x = (x - 0.1307) / 0.3081
        y = int(ex["label"])
        return x, y, idx


def load_mnist_hf() -> tuple[Any, Any]:
    ds = load_dataset("mnist")
    return ds["train"], ds["test"]


# For weight decay slection
def _balanced_sample_indices(labels: np.ndarray, total: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per = total // len(classes)
    if per * len(classes) != total:
        raise ValueError(
            f"total={total} must be divisible by num_classes={len(classes)}"
        )
    picked: list[int] = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) < per:
            raise ValueError(
                f"Not enough examples for class {c}: need {per}, have {len(idx)}"
            )
        chosen = rng.choice(idx, size=per, replace=False)
        picked.extend(chosen.tolist())
    picked = np.array(picked, dtype=np.int64)
    rng.shuffle(picked)
    return picked


def make_splits(
    train_split,
    init_labeled: int,
    val_size: int,
    num_classes: int,
    seed: int,
) -> MnistSplits:
    labels = np.array(train_split["label"], dtype=np.int64)
    classes = np.unique(labels)
    if len(classes) != num_classes:
        raise ValueError(f"Expected {num_classes} classes, got {len(classes)}")

    init_idx = _balanced_sample_indices(labels, total=init_labeled, seed=seed)

    mask = np.ones(len(labels), dtype=bool)
    mask[init_idx] = False
    remaining = np.where(mask)[0]
    rem_labels = labels[remaining]
    val_local = _balanced_sample_indices(rem_labels, total=val_size, seed=seed + 1)
    val_idx = remaining[val_local]

    mask[val_idx] = False
    pool_idx = np.where(mask)[0]
    return MnistSplits(init_labeled_idx=init_idx, val_idx=val_idx, pool_idx=pool_idx)
