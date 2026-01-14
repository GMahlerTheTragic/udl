from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


ACQUISITION_FUNTIONS = Literal["random", "entropy", "bald", "var_ratios", "mean_std"]


def H_For_Bernoulli_Random(
    p,
):
    p = torch.clamp(p, 1e-8, 1 - 1e-8)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def tokenize_pool_texts(ds, tokenizer, text_field, max_length):
    ds_tokenized = ds.map(
        lambda x: tokenizer(x[text_field], truncation=True, max_length=max_length),
        batched=True,
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    keep_cols = {"input_ids", "attention_mask"}
    droping = [c for c in ds_tokenized.column_names if c not in keep_cols]
    return ds_tokenized.remove_columns(droping) if droping else ds_tokenized


@dataclass(frozen=True)
class MCDropoutScores:
    scores: np.ndarray
    mean_probs: np.ndarray


def score_pool_with_mc_dropout(
    model,
    pool_ds_tok,
    tokenizer,
    device: torch.device,
    acq: ACQUISITION_FUNTIONS,
    mc_samples: int,
    batch_size: int,
    seed: int,
    label_weights: np.ndarray | None = None,
) -> MCDropoutScores:
    if acq == "random":
        rng = np.random.RandomState(int(seed))
        scores = rng.rand(len(pool_ds_tok)).astype(np.float32)
        mean_probs = np.zeros((len(pool_ds_tok), 5), dtype=np.float32)
        return MCDropoutScores(scores=scores, mean_probs=mean_probs)

    if tokenizer is None:
        loader = DataLoader(
            pool_ds_tok, batch_size=batch_size, shuffle=False, num_workers=0
        )
    else:
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        loader = DataLoader(
            pool_ds_tok,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
    model = model.to(device)
    model.train()

    w: torch.Tensor | None = None
    if label_weights is not None:
        w_np = np.asarray(label_weights, dtype=np.float32).reshape(-1)
        w = torch.tensor(w_np / float(w_np.sum()), dtype=torch.float32, device=device)

    all_scores: list[np.ndarray] = []
    all_mean_probs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            probs_t: list[torch.Tensor] = []
            for _ in range(int(mc_samples)):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                probs_t.append(torch.sigmoid(out.logits))

            probs = torch.stack(probs_t, dim=0)
            mean_p = probs.mean(dim=0)

            def _agg(per_label: torch.Tensor) -> torch.Tensor:
                return (
                    per_label.mean(dim=1)
                    if w is None
                    else (per_label * w[None, :]).sum(dim=1)
                )

            if acq == "entropy":
                s = _agg(H_For_Bernoulli_Random(mean_p))
            elif acq == "bald":
                # Note this is different tha for al_mnist
                s = _agg(
                    H_For_Bernoulli_Random(mean_p)
                    - H_For_Bernoulli_Random(probs).mean(dim=0)
                )
            elif acq == "mean_std":
                s = _agg(probs.std(dim=0))
            elif acq == "var_ratios":
                classified = (probs >= 0.5).float()
                m = classified.mean(dim=0)
                vr = 1.0 - torch.maximum(m, 1.0 - m)
                s = _agg(vr)
            else:
                raise ValueError(f"Unknown acquisition: {acq}")

            all_scores.append(s.detach().cpu().numpy().astype(np.float32))
            all_mean_probs.append(mean_p.detach().cpu().numpy().astype(np.float32))

    scores = (
        np.concatenate(all_scores, axis=0)
        if all_scores
        else np.zeros((0,), dtype=np.float32)
    )
    mean_probs = (
        np.concatenate(all_mean_probs, axis=0)
        if all_mean_probs
        else np.zeros((0, 5), dtype=np.float32)
    )
    return MCDropoutScores(scores=scores, mean_probs=mean_probs)
