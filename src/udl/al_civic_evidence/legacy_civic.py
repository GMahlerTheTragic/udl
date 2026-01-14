# Took this from here: https://github.com/GMahlerTheTragic/civic
import os
import logging
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

CIVIC_GRAPH_QL_ENDPOINT = "https://civicdb.org/api/graphql"

GRAPH_QL_QUERY = """
    {{
        evidenceItems(first: {}, after: {}) {{
            totalCount
            pageCount
            nodes {{
                id
                status
                molecularProfile {{
                    id
                    name
                }}
                evidenceType
                evidenceLevel
                evidenceRating
                evidenceDirection
                description
                disease {{
                    id
                    name
                }}
                therapies {{
                    id
                    name
                }}
                source {{
                    abstract
                    id
                }}
                variantOrigin
                significance
            }}
            pageInfo {{
                endCursor
                hasNextPage
            }}
        }}
    }}
"""

log = logging.getLogger(__name__)


def download_civic_evidence(first: int, after: str):
    log.info("CIViC download: requesting page after=%s (first=%s)", after, first)
    response = requests.post(
        CIVIC_GRAPH_QL_ENDPOINT, json={"query": GRAPH_QL_QUERY.format(first, after)}
    )
    return response.json().get("data")


def _expand_column(
    json: dict[str, Any], name: str, replacements: dict[str, str], is_list: bool = False
):
    if json[name] is not None:
        for k, v in replacements.items():
            if is_list:
                json[v] = list(map(lambda x, y=k: x[y], json[name]))
            else:
                json[v] = json[name][k]
    json.pop(name)


def clean_columns(json: dict[str, Any]):
    _expand_column(
        json,
        "molecularProfile",
        {"id": "molecularProfileId", "name": "molecularProfileName"},
    )
    _expand_column(json, "disease", {"id": "diseaseId", "name": "diseaseName"})
    _expand_column(json, "source", {"abstract": "sourceAbstract", "id": "sourceId"})
    _expand_column(
        json,
        "therapies",
        {"id": "therapyIds", "name": "therapyNames"},
        is_list=True,
    )
    return json


def download_all_evidence_items(page_size: int = 50) -> pd.DataFrame:
    log.info("CIViC download: start (page_size=%s)", page_size)
    items: list[dict[str, Any]] = []
    data = download_civic_evidence(page_size, '""')
    items.extend(data["evidenceItems"]["nodes"])
    page = 1
    page_count = data["evidenceItems"]["pageCount"]
    total_count = data["evidenceItems"]["totalCount"]
    log.info(
        "CIViC download: page %s/%s (totalCount=%s)", page, page_count, total_count
    )
    while data["evidenceItems"]["pageInfo"]["hasNextPage"]:
        page += 1
        log.info(
            "CIViC download: page %s/%s (endCursor=%s)",
            page,
            page_count,
            data["evidenceItems"]["pageInfo"]["endCursor"],
        )
        data = download_civic_evidence(
            page_size, f'"{data["evidenceItems"]["pageInfo"]["endCursor"]}"'
        )
        items.extend(data["evidenceItems"]["nodes"])
    df = pd.DataFrame(list(map(clean_columns, items)))
    df = df.sort_index(axis=1)
    log.info("CIViC download: done (rows=%s)", len(df))
    return df


KEEP_COLS = [
    "id",
    "status",
    "prependString",
    "sourceAbstract",
    "sourceId",
    "evidenceLevel",
]


def remove_duplicates(data: pd.DataFrame, column_combination: list[str]):
    duplicated = data.duplicated(
        subset=column_combination,
        keep=False,
    )
    data = data.loc[~duplicated]
    return data


def do_data_cleaning(data: pd.DataFrame):
    data = data.dropna(subset=["sourceAbstract", "evidenceLevel"], how="any")
    data = remove_duplicates(
        data,
        [
            "diseaseId",
            "significance",
            "molecularProfileId",
            "therapyIds",
            "sourceId",
        ],
    )

    data["therapyNames"] = data.therapyNames.map(
        lambda x: x.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace(" ", "")
        .replace(",", "-")
    )
    data["therapyNames"] = data.therapyNames.map(lambda x: np.nan if x == "" else x)
    data = data.dropna(
        subset=["diseaseName", "significance", "molecularProfileName", "therapyNames"],
        how="all",
    )
    data = data.fillna("Unknown")
    assert data.isna().sum().sum() == 0

    data["prependString"] = (
        "DiseaseName: "
        + data.diseaseName
        + "\n"
        + "Molecular Profile Name: "
        + data.molecularProfileName
        + "\n"
        + "Therapies: "
        + data.therapyNames
        + "\n"
        + "Significance: "
        + data.significance
    )
    n_duplicates = data.duplicated(
        subset=["prependString", "sourceId"], keep=False
    ).sum()
    assert n_duplicates == 0

    data = data.loc[
        :,
        [
            "id",
            "status",
            "prependString",
            "sourceAbstract",
            "sourceId",
            "evidenceLevel",
        ],
    ]
    return data


def _compile_multi_class_data_set(data: pd.DataFrame):
    data_multi_class = (
        data.pivot_table(
            columns="evidenceLevel",
            index="sourceId",
            values="id",
            aggfunc=lambda x: 1 if len(x) >= 1 else 0,
        )
        .fillna(0)
        .reset_index()
    )
    data_multi_class = data_multi_class.merge(
        data[["sourceId", "sourceAbstract"]].drop_duplicates(),
        on="sourceId",
        how="left",
    )
    assert data_multi_class["sourceAbstract"].isna().sum() == 0
    return data_multi_class


def filter_for_unique_abstracts(data: pd.DataFrame):
    return data.drop_duplicates(["sourceId"], keep=False).reset_index()


def get_stratified_train_test_split(data: pd.DataFrame, test_size: float, class_col):
    train_data, test_data = train_test_split(
        data, test_size=test_size, stratify=data[class_col], random_state=42
    )
    return train_data, test_data


def write_processed_splits(raw_csv: str, processed_dir: str) -> dict[str, str]:
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(raw_csv)
    df = do_data_cleaning(df)
    df_unique_abstracts = filter_for_unique_abstracts(df)
    df_multi_class = _compile_multi_class_data_set(df)

    train_full, test_full = get_stratified_train_test_split(df, 0.2, "evidenceLevel")
    val_full, test_full = get_stratified_train_test_split(
        test_full, 0.5, "evidenceLevel"
    )

    train_ua, test_ua = get_stratified_train_test_split(
        df_unique_abstracts, 0.2, "evidenceLevel"
    )
    val_ua, test_ua = get_stratified_train_test_split(test_ua, 0.5, "evidenceLevel")

    def _mc_from_split(split_df: pd.DataFrame) -> pd.DataFrame:
        src_ids = split_df["sourceId"].drop_duplicates()
        return df_multi_class.loc[df_multi_class["sourceId"].isin(src_ids)].reset_index(
            drop=True
        )

    train_mc = _mc_from_split(train_full)
    val_mc = _mc_from_split(val_full)
    test_mc = _mc_from_split(test_full)

    for L in ["A", "B", "C", "D", "E"]:
        for s in (train_mc, val_mc, test_mc):
            if L not in s.columns:
                s[L] = 0

    out = {}
    out["train_csv"] = os.path.join(processed_dir, "civic_evidence_train.csv")
    out["val_csv"] = os.path.join(processed_dir, "civic_evidence_val.csv")
    out["test_csv"] = os.path.join(processed_dir, "civic_evidence_test.csv")
    train_full.to_csv(out["train_csv"], index=False, columns=KEEP_COLS)
    val_full.to_csv(out["val_csv"], index=False, columns=KEEP_COLS)
    test_full.to_csv(out["test_csv"], index=False, columns=KEEP_COLS)

    out["train_mc_csv"] = os.path.join(processed_dir, "civic_evidence_train_mc.csv")
    out["val_mc_csv"] = os.path.join(processed_dir, "civic_evidence_val_mc.csv")
    out["test_mc_csv"] = os.path.join(processed_dir, "civic_evidence_test_mc.csv")
    train_mc.to_csv(out["train_mc_csv"], index=False)
    val_mc.to_csv(out["val_mc_csv"], index=False)
    test_mc.to_csv(out["test_mc_csv"], index=False)

    return out
