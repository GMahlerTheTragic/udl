from dataclasses import dataclass
import tempfile
from typing import Any
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import inspect
from torch.utils.data import DataLoader
import torch
from transformers import DataCollatorWithPadding


@dataclass(frozen=True)
class TrainOutputs:
    val_thresholds: list[float] | None
    val_weighted_f1: float


@dataclass(frozen=True)
class TrainConfig:
    pretrained: str
    text_field: str
    max_length: int | None
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    num_train_epochs: float
    warmup_ratio: float
    logging_steps: int
    max_steps: int | None


def _prepare_multilabel_dataset(ds, tokenizer, text_field: str, max_length: int | None):
    def tok(batch):
        encodin = tokenizer(batch[text_field], truncation=True, max_length=max_length)
        labels = []
        n = len(batch["A"])
        for i in range(n):
            labels.append(
                [
                    float(batch["A"][i]),
                    float(batch["B"][i]),
                    float(batch["C"][i]),
                    float(batch["D"][i]),
                    float(batch["E"][i]),
                ]
            )
        encodin["labels"] = labels
        return encodin

    remove_cols = [
        c for c in ds.column_names if c not in (text_field, "A", "B", "C", "D", "E")
    ]
    ds = ds.remove_columns(remove_cols) if remove_cols else ds
    ds = ds.map(tok, batched=True, load_from_cache_file=False, keep_in_memory=True)
    keep = {"input_ids", "attention_mask", "labels"}
    drop = [c for c in ds.column_names if c not in keep]
    return ds.remove_columns(drop) if drop else ds


def _compute_multilabel_metrics(logits, labels, thresholds: list[float] | None):

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    probs = torch.sigmoid(torch.tensor(logits)).cpu().numpy()
    y_true = np.asarray(labels)

    if thresholds is None:
        thr = np.full((probs.shape[1],), 0.5, dtype=np.float32)
    else:
        thr = np.asarray(thresholds, dtype=np.float32)

    y_pred = (probs >= thr[None, :]).astype(np.int64)
    # We only report weighted f1. See paper for https://github.com/GMahlerTheTragic/civic for a report of the individual f1s.
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_label_f1": [
            float(x) for x in f1_score(y_true, y_pred, average=None, zero_division=0)
        ],
    }


# Optimising thresholds
def _calibrate_thresholds_prc(logits, labels) -> list[float]:
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    probs = torch.sigmoid(torch.tensor(logits)).cpu().numpy()
    y_true = np.asarray(labels)

    thresholds: list[float] = []
    K = probs.shape[1]
    for k in range(K):
        precision, recall, thr = precision_recall_curve(y_true[:, k], probs[:, k])
        f1 = (2 * precision * recall) / (precision + recall + 1e-12)
        if len(thr) == 0:
            thresholds.append(0.5)
            continue
        best = int(np.nanargmax(f1[:-1]))
        thresholds.append(float(thr[best]))
    return thresholds


def train_multilabel_distilbert(cfg: TrainConfig, train_ds, val_ds):

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained, use_fast=True)
    train_ds = _prepare_multilabel_dataset(
        train_ds, tokenizer, text_field=cfg.text_field, max_length=cfg.max_length
    )
    val_ds = _prepare_multilabel_dataset(
        val_ds, tokenizer, text_field=cfg.text_field, max_length=cfg.max_length
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    sig = inspect.signature(TrainingArguments.__init__)

    def compute_metrics(eval_pred):
        logits = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)
        if logits is None or labels is None:
            logits, labels = eval_pred
        return _compute_multilabel_metrics(
            logits=logits, labels=labels, thresholds=None
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.pretrained,
        num_labels=5,
        problem_type="multi_label_classification",
    )

    with tempfile.TemporaryDirectory() as tmp:
        ta_kwargs: dict[str, Any] = dict(
            output_dir=str(tmp),
            learning_rate=float(cfg.learning_rate),
            per_device_train_batch_size=int(cfg.per_device_train_batch_size),
            per_device_eval_batch_size=int(cfg.per_device_eval_batch_size),
            num_train_epochs=float(cfg.num_train_epochs),
            warmup_ratio=float(cfg.warmup_ratio),
            weight_decay=0.0,
            disable_tqdm=True,
            evaluation_strategy="no",
            save_strategy="no",
            report_to=[],
            load_best_model_at_end=False,
        )
        if cfg.max_steps is not None:
            ta_kwargs["max_steps"] = int(cfg.max_steps)
        if "optim" in sig.parameters:
            ta_kwargs["optim"] = "adamw_torch"
        if (
            "evaluation_strategy" not in sig.parameters
            and "eval_strategy" in sig.parameters
        ):
            ta_kwargs["eval_strategy"] = ta_kwargs.pop("evaluation_strategy")
        args = TrainingArguments(
            **{k: v for k, v in ta_kwargs.items() if k in sig.parameters}
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        train_metrics = trainer.train().metrics
        eval_out = trainer.predict(val_ds)

    thresholds = _calibrate_thresholds_prc(
        logits=eval_out.predictions, labels=eval_out.label_ids
    )
    val_metrics = _compute_multilabel_metrics(
        logits=eval_out.predictions, labels=eval_out.label_ids, thresholds=thresholds
    )
    score = float(val_metrics["weighted_f1"])
    return (
        model,
        tokenizer,
        TrainOutputs(val_thresholds=thresholds, val_weighted_f1=float(score)),
        {"train": train_metrics, "val": val_metrics},
    )


def eval_multilabel(
    model,
    tokenizer,
    ds,
    device,
    thresholds: list[float] | None,
    text_field: str,
    max_length: int | None,
    batch_size: int,
) -> dict[str, Any]:

    def tokenzi(batch):
        enc = tokenizer(batch[text_field], truncation=True, max_length=max_length)
        labels = []
        n = len(batch["A"])
        for i in range(n):
            labels.append(
                [
                    float(batch["A"][i]),
                    float(batch["B"][i]),
                    float(batch["C"][i]),
                    float(batch["D"][i]),
                    float(batch["E"][i]),
                ]
            )
        enc["labels"] = labels
        return enc

    ds_tok = ds.map(
        tokenzi, batched=True, load_from_cache_file=False, keep_in_memory=True
    )
    keep = {"input_ids", "attention_mask", "labels"}
    drop = [c for c in ds_tok.column_names if c not in keep]
    ds_tok = ds_tok.remove_columns(drop) if drop else ds_tok

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collate_fn(examples: list[dict[str, Any]]):
        lbl = torch.tensor([ex["labels"] for ex in examples], dtype=torch.float32)
        ex_inputs = [
            {"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"]}
            for ex in examples
        ]
        out = collator(ex_inputs)
        out["labels"] = lbl
        return out

    loader = DataLoader(
        ds_tok,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    model = model.to(device)
    # switches dropout off (i.e. uses the scaling variant of it)
    model.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(out.logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return _compute_multilabel_metrics(
        logits=logits, labels=labels, thresholds=thresholds
    )
