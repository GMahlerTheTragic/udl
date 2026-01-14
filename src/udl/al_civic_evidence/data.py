from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk

from udl.al_civic_evidence.legacy_civic import (
    download_all_evidence_items,
    write_processed_splits,
)


LEVELS = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class CivicDataPaths:
    base_dir: Path
    raw_dir: Path
    processed_dir: Path
    hf_dir: Path


def _paths(
    artifacts_dir: Path,
    dataset_name: str = "civic-evidence",
    dataset_subdir: str = "datasets",
):
    base = artifacts_dir / dataset_subdir / dataset_name
    return CivicDataPaths(
        base_dir=base,
        raw_dir=base / "01_raw",
        processed_dir=base / "02_processed",
        hf_dir=base / "hf",
    )


def _df_to_hf_multilabel(df: pd.DataFrame) -> Dataset:
    keep = ["sourceId", "sourceAbstract", *LEVELS]
    df = df.loc[:, keep].reset_index(drop=True)
    for L in LEVELS:
        df[L] = df[L].astype(np.float32)
    return Dataset.from_pandas(df, preserve_index=False)


def build_and_save_dataset(
    artifacts_dir: Path,
    dataset_name: str = "civic-evidence",
    dataset_subdir: str = "datasets",
    rebuild: bool = False,
) -> Path:
    p = _paths(
        artifacts_dir=artifacts_dir,
        dataset_name=str(dataset_name),
        dataset_subdir=str(dataset_subdir),
    )
    p.raw_dir.mkdir(parents=True, exist_ok=True)
    p.processed_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = p.raw_dir / "civic_evidence.csv"

    if rebuild or not raw_csv.exists():
        df = download_all_evidence_items(page_size=50)
        df.to_csv(raw_csv, index=False)

    out_files = write_processed_splits(
        raw_csv=str(raw_csv), processed_dir=str(p.processed_dir)
    )

    train_mc = pd.read_csv(out_files["train_mc_csv"])
    val_mc = pd.read_csv(out_files["val_mc_csv"])
    test_mc = pd.read_csv(out_files["test_mc_csv"])

    for L in LEVELS:
        for s in (train_mc, val_mc, test_mc):
            if L not in s.columns:
                s[L] = 0
    train_mc = train_mc[["sourceId", "sourceAbstract", *LEVELS]]
    val_mc = val_mc[["sourceId", "sourceAbstract", *LEVELS]]
    test_mc = test_mc[["sourceId", "sourceAbstract", *LEVELS]]

    ds = DatasetDict(
        train=_df_to_hf_multilabel(train_mc),
        validation=_df_to_hf_multilabel(val_mc),
        test=_df_to_hf_multilabel(test_mc),
    )
    p.hf_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(p.hf_dir))
    return p.hf_dir


def load_or_doinload_dataset(
    artifacts_dir: Path,
    dataset_name: str = "civic-evidence",
    dataset_subdir: str = "datasets",
    rebuild: bool = False,
) -> DatasetDict:
    p = _paths(
        artifacts_dir=artifacts_dir,
        dataset_name=str(dataset_name),
        dataset_subdir=str(dataset_subdir),
    )
    if rebuild or not p.hf_dir.exists():
        build_and_save_dataset(
            artifacts_dir=artifacts_dir,
            dataset_name=str(dataset_name),
            dataset_subdir=str(dataset_subdir),
            rebuild=rebuild,
        )
    return load_from_disk(str(p.hf_dir))
