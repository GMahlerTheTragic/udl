import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import dataclasses
import logging


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def initialize_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def dir_atifacts() -> Path:
    return get_repo_root() / "artifacts"


def setup_huggingface_stuff(artifacts: Path) -> None:
    hf_home = artifacts / "hf_cache"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))


@dataclass(frozen=True)
class RunPathsConfig:
    run_dir: Path
    csv_output_path: Path


def make_run_paths_config() -> RunPathsConfig:
    run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = dir_atifacts() / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPathsConfig(run_dir=run_dir, csv_output_path=run_dir / "raw_metrics.csv")


def step_rows_to_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    dicts = []
    for it in rows:
        dicts.append(dataclasses.asdict(it))
    columns = list(dicts[0].keys())
    with path.open(mode="w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)
