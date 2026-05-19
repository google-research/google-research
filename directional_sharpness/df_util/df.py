# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sam_train import (  # noqa: E402
    SAM,
    disable_running_stats,
    enable_running_stats,
    get_datetime,
    set_device,
)


# ----------------------------- Batch helpers -----------------------------


def split_batch(batch):
    """Split a batch into (inputs, labels)."""
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Tuple/list batch must contain at least 2 elements (inputs, labels).")
        return batch[0], batch[1]

    if isinstance(batch, dict):
        if "labels" in batch:
            label_key = "labels"
        elif "label" in batch:
            label_key = "label"
        else:
            raise KeyError("Dict batch must include 'label' or 'labels'.")
        labels = batch[label_key]
        inputs = {k: v for k, v in batch.items() if k != label_key and torch.is_tensor(v)}
        return inputs, labels

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _to_device_inputs(inputs, device):
    if isinstance(inputs, dict):
        return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    return inputs.to(device, non_blocking=True)


def forward_from_inputs(model, inputs):
    out = model(**inputs) if isinstance(inputs, dict) else model(inputs)
    return out.logits if hasattr(out, "logits") else out


def infer_batch_size(batch):
    inputs, _ = split_batch(batch)
    if isinstance(inputs, torch.Tensor):
        return int(inputs.shape[0])
    if isinstance(inputs, dict) and inputs:
        first = next(iter(inputs.values()))
        return int(first.shape[0])
    return 0


def build_loader(
    dataset,
    batch_size,
    workers,
    shuffle,
    collate_fn = None,
):
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        num_workers=max(0, int(workers)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        persistent_workers=(int(workers) > 0),
    )


def _next_batch(state):
    try:
        return next(state["iter"])
    except StopIteration:
        state["iter"] = iter(state["loader"])
        return next(state["iter"])


# ----------------------------- Numeric helpers -----------------------------


def _to_float_or_nan(value):
    try:
        x = float(value)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


def _safe_gap(lhs, rhs):
    a = _to_float_or_nan(lhs)
    b = _to_float_or_nan(rhs)
    if np.isfinite(a) and np.isfinite(b):
        return float(a - b)
    return np.nan


def _safe_divide(numer, denom, eps = 1e-12):
    return numer / np.maximum(denom, eps)


def _sanitize_name(text):
    out = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip().lower()).strip("_")
    return out or "value"


# ----------------------------- Param helpers -----------------------------


def _dedup_params(params):
    seen: set = set()
    out: List[nn.Parameter] = []
    for p in params:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def _grad_norms_from_params(
    params,
    *,
    adaptive,
):
    """
    Compute L1/L2 norms of gradient vector (or elementwise w*grad for adaptive)
    without flattening all tensors into a single giant vector.
    """
    sum_l1: Optional[torch.Tensor] = None
    sum_l2_sq: Optional[torch.Tensor] = None

    for p in params:
        g = p.grad
        if g is None:
            continue

        g_detached = g.detach()
        term = (p.detach() * g_detached) if adaptive else g_detached

        if sum_l1 is None:
            sum_l1 = torch.zeros((), dtype=term.dtype, device=term.device)
            sum_l2_sq = torch.zeros((), dtype=term.dtype, device=term.device)

        sum_l1 = sum_l1 + term.abs().sum()
        sum_l2_sq = sum_l2_sq + term.square().sum()

    if sum_l1 is None or sum_l2_sq is None:
        return 0.0, 0.0

    grad_norm_l1 = float(sum_l1.item())
    grad_norm_l2 = float(torch.sqrt(sum_l2_sq).item())
    return grad_norm_l1, grad_norm_l2


class _MaybeEnableGrads:
    """Temporarily enable gradients for a parameter list."""

    def __init__(self, params, enable_grads):
        self.params = list(params)
        self.enable_grads = bool(enable_grads)
        self.previous: List[bool] = []

    def __enter__(self):
        if not self.enable_grads:
            return
        self.previous = [bool(p.requires_grad) for p in self.params]
        for p in self.params:
            if not p.requires_grad:
                p.requires_grad_(True)

    def __exit__(self, exc_type, exc, tb):
        if not self.enable_grads:
            return False
        for p, old in zip(self.params, self.previous):
            if p.requires_grad != old:
                p.requires_grad_(old)
        return False


# ----------------------------- Probe primitives -----------------------------


DEFAULT_AUTO_MICROBATCH_SIZE = 64


def _batch_size_from_target(target):
    if target.ndim == 0:
        return 1
    if target.shape[0] <= 0:
        raise ValueError(f"Invalid target batch shape: {tuple(target.shape)}")
    return int(target.shape[0])


def _slice_inputs(inputs, start, end):
    if isinstance(inputs, dict):
        return {k: v[start:end] for k, v in inputs.items()}
    return inputs[start:end]


def _iter_microbatches(inputs, target, microbatch_size):
    total = _batch_size_from_target(target)
    cap = max(1, int(microbatch_size))

    if target.ndim == 0 or total <= cap:
        yield inputs, target, total
        return

    for start in range(0, total, cap):
        end = min(start + cap, total)
        yield _slice_inputs(inputs, start, end), target[start:end], int(end - start)


def minibatch_loss_acc(
    model,
    batch,
    device,
    *,
    microbatch_size = DEFAULT_AUTO_MICROBATCH_SIZE,
):
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        inputs, target = split_batch(batch)
        inputs = _to_device_inputs(inputs, device)
        target = target.to(device, non_blocking=True)
        total = _batch_size_from_target(target)
        total_loss = 0.0
        total_correct = 0.0
        for mb_inputs, mb_target, mb_n in _iter_microbatches(inputs, target, microbatch_size):
            logits = forward_from_inputs(model, mb_inputs)
            loss = F.cross_entropy(logits, mb_target, reduction="mean")
            total_loss += float(loss.detach().item()) * float(mb_n)
            total_correct += float((logits.argmax(dim=-1) == mb_target).float().sum().item())
        loss = total_loss / float(total)
        acc = total_correct / float(total)
    if was_training:
        model.train()
    return {"loss": float(loss), "acc": float(acc)}


def flatness_on_batch(
    model,
    batch,
    device,
    *,
    norm,
    adaptive,
    flat_r,
    flat_delta,
    freeze_bn,
    params = None,
    enable_grads = False,
    microbatch_size = DEFAULT_AUTO_MICROBATCH_SIZE,
):
    params = _dedup_params(params) if params else [p for p in model.parameters() if p.requires_grad]
    if not params:
        params = _dedup_params([p for _, p in model.named_parameters()])

    for p in params:
        if p.grad is not None:
            p.grad = None

    was_training = bool(model.training)
    model.eval() if freeze_bn else model.train()

    with _MaybeEnableGrads(params, enable_grads=enable_grads):
        inputs, target = split_batch(batch)
        inputs = _to_device_inputs(inputs, device)
        target = target.to(device, non_blocking=True)
        total = _batch_size_from_target(target)
        total_loss = 0.0
        total_correct = 0.0

        for mb_inputs, mb_target, mb_n in _iter_microbatches(inputs, target, microbatch_size):
            logits = forward_from_inputs(model, mb_inputs)
            loss = F.cross_entropy(logits, mb_target, reduction="mean")
            loss_scale = float(mb_n) / float(total)
            (loss * loss_scale).backward()
            total_loss += float(loss.detach().item()) * float(mb_n)
            with torch.no_grad():
                total_correct += float((logits.argmax(dim=-1) == mb_target).float().sum().item())

        grad_norm_l1, grad_norm_l2 = _grad_norms_from_params(params, adaptive=adaptive)
        acc = float(total_correct / float(total))

    if was_training:
        model.train()

    mb_loss = float(total_loss / float(total))
    by_norm: Dict[str, Dict[str, float]] = {}
    for norm_key, grad_norm in (("l1", grad_norm_l1), ("l2", grad_norm_l2)):
        by_norm[norm_key] = {
            "grad_norm": grad_norm,
            "flat_primal": float(flat_r * grad_norm),
            "flat_dual": float((float(flat_delta) - mb_loss) / (grad_norm + 1e-12)),
        }

    chosen_norm = str(norm or "l2").strip().lower()
    if chosen_norm not in by_norm:
        raise ValueError(f"Unsupported flatness norm '{norm}'. Use 'l1' or 'l2'.")
    chosen = by_norm[chosen_norm]
    return {
        "mb_loss": mb_loss,
        "mb_acc": acc,
        "grad_norm": float(chosen["grad_norm"]),
        "flat_primal": float(chosen["flat_primal"]),
        "flat_dual": float(chosen["flat_dual"]),
        "by_norm": by_norm,
    }


def sam_update_on_batch(
    model,
    sam_opt,
    batch,
    device,
    *,
    microbatch_size = DEFAULT_AUTO_MICROBATCH_SIZE,
):
    inputs, target = split_batch(batch)
    inputs = _to_device_inputs(inputs, device)
    target = target.to(device, non_blocking=True)
    total = _batch_size_from_target(target)

    enable_running_stats(model)
    sam_opt.zero_grad(set_to_none=True)
    for mb_inputs, mb_target, mb_n in _iter_microbatches(inputs, target, microbatch_size):
        logits = forward_from_inputs(model, mb_inputs)
        loss = F.cross_entropy(logits, mb_target, reduction="mean")
        loss_scale = float(mb_n) / float(total)
        (loss * loss_scale).backward()
    sam_opt.first_step(zero_grad=True)

    disable_running_stats(model)
    for mb_inputs, mb_target, mb_n in _iter_microbatches(inputs, target, microbatch_size):
        logits2 = forward_from_inputs(model, mb_inputs)
        loss2 = F.cross_entropy(logits2, mb_target, reduction="mean")
        loss_scale = float(mb_n) / float(total)
        (loss2 * loss_scale).backward()
    sam_opt.second_step(zero_grad=True)


# ----------------------------- Probe config -----------------------------


@dataclass
class ProbeConfig:
    out_dir: str
    probe_steps: int
    flat_norm: str
    flat_r: float
    flat_delta: float
    sam_rho: float
    sam_adaptive: bool
    sam_momentum: float
    sam_lr: float
    sam_weight_decay: float
    include_full_batch_step: bool
    num_workers: int
    device_index: int
    include_unfreeze: bool
    show_progress: bool

    probe_step_batch_size: int
    step_train_eval_size: int
    sam_step_batch_size: int

    # If probe_epochs > 0 or probe_extra_steps > 0, total SAM updates become:
    #   probe_epochs * ceil(len(train_ds) / sam_step_batch_size) + probe_extra_steps
    probe_epochs: int = 0
    probe_extra_steps: int = 0

    record_every_steps: int = 1
    record_points: int = 0

    sam_base_opt: str = "sgd"  # "sgd" or "adamw"
    num_gpus: int = 4
    max_concurrent_probes: Optional[int] = None
    auto_microbatch_size: int = DEFAULT_AUTO_MICROBATCH_SIZE


PROBE_COLUMNS = [
    "run_id",
    "dataset",
    "model",
    "seed",
    "optimizer",
    "probe_param_source",
    "lr",
    "wd",
    "batch_size",
    "label_smoothing",
    "dropout",
    "ckpt_path",
    "probe_step_batch_size",
    "step_train_eval_size",
    "sam_step_batch_size",
    "probe_steps",
    "probe_epochs",
    "probe_extra_steps",
    "steps_per_epoch",
    "probe_total_steps",
    "record_every_steps",
    "record_points",
    "record_count_planned",
    "sam_rho",
    "sam_adaptive",
    "flat_norm",
    "freeze_bn",
    "step",
    "subset_size",
    "mb_loss",
    "train_loss",
    "train_acc",
    "test_loss",
    "test_acc",
    "flat_primal",
    "flat_dual",
    "flat_primal_freeze",
    "flat_dual_freeze",
    "gen_gap_acc",
    "gen_gap_loss",
    "timestamp",
]


def _resolve_train_params_source(train_params_source):
    source = str(train_params_source).strip().lower()
    if source not in {"specified", "initial", "continuing"}:
        raise ValueError(
            f"Unsupported train_params_source '{source}'. "
            "Use one of: specified, initial, continuing."
        )
    return source


def _resolve_probe_batch_settings(cfg):
    probe_bs = int(cfg.probe_step_batch_size)
    eval_bs = int(cfg.step_train_eval_size)
    sam_bs = int(cfg.sam_step_batch_size)

    if probe_bs <= 0:
        raise ValueError("ProbeConfig.probe_step_batch_size must be > 0.")
    if eval_bs <= 0:
        raise ValueError("ProbeConfig.step_train_eval_size must be > 0.")
    if sam_bs <= 0:
        raise ValueError("ProbeConfig requires sam_step_batch_size > 0.")

    return {
        "probe_step_batch_size": probe_bs,
        "step_train_eval_size": eval_bs,
        "sam_step_batch_size": sam_bs,
    }


def _resolve_probe_duration(
    cfg,
    *,
    train_ds,
    sam_step_batch_size,
):
    legacy_steps = int(cfg.probe_steps)
    probe_epochs = int(getattr(cfg, "probe_epochs", 0) or 0)
    probe_extra_steps = int(getattr(cfg, "probe_extra_steps", 0) or 0)

    if probe_epochs < 0:
        raise ValueError("ProbeConfig.probe_epochs must be >= 0.")
    if probe_extra_steps < 0:
        raise ValueError("ProbeConfig.probe_extra_steps must be >= 0.")

    use_epoch_mode = (probe_epochs > 0) or (probe_extra_steps > 0)
    if not use_epoch_mode:
        if legacy_steps <= 0:
            raise ValueError(
                "ProbeConfig.probe_steps must be > 0 when probe_epochs/probe_extra_steps are both 0."
            )
        return {
            "probe_epochs": 0,
            "probe_extra_steps": 0,
            "steps_per_epoch": 0,
            "total_steps": int(legacy_steps),
            "duration_mode": "steps",
        }

    try:
        train_size = int(len(train_ds))
    except Exception as exc:
        raise ValueError(
            "Epoch-based probing requires train_ds to define __len__ for step scheduling."
        ) from exc
    if train_size <= 0:
        raise ValueError(f"Invalid train dataset size for epoch-based probing: {train_size}")

    steps_per_epoch = int((train_size + int(sam_step_batch_size) - 1) // int(sam_step_batch_size))
    total_steps = int(probe_epochs * steps_per_epoch + probe_extra_steps)
    if total_steps <= 0:
        raise ValueError(
            "Epoch-based probing resolved to 0 total steps; increase probe_epochs/probe_extra_steps."
        )
    return {
        "probe_epochs": int(probe_epochs),
        "probe_extra_steps": int(probe_extra_steps),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "duration_mode": "epochs_plus_steps",
    }


def _evenly_spaced_record_steps(total_steps, record_points):
    n_steps = int(total_steps)
    n_points = int(record_points)
    if n_steps <= 0 or n_points <= 0:
        return []
    if n_points >= n_steps:
        return list(range(n_steps))
    if n_points == 1:
        return [n_steps - 1]

    last = n_steps - 1
    den = n_points - 1
    return [int((k * last) // den) for k in range(n_points)]


def _resolve_record_steps(cfg, *, total_steps):
    record_every_steps = int(getattr(cfg, "record_every_steps", 1) or 0)
    record_points = int(getattr(cfg, "record_points", 0) or 0)
    if record_every_steps < 0:
        raise ValueError("ProbeConfig.record_every_steps must be >= 0.")
    if record_points < 0:
        raise ValueError("ProbeConfig.record_points must be >= 0.")
    if record_every_steps > 0 and record_points > 0:
        raise ValueError(
            "ProbeConfig.record_every_steps and ProbeConfig.record_points cannot both be > 0."
        )

    # Preserve legacy behavior when neither sparse control is set.
    if record_every_steps == 0 and record_points == 0:
        record_every_steps = 1

    if record_every_steps > 0:
        steps = list(range(0, int(total_steps), int(record_every_steps)))
        if not steps or steps[-1] != int(total_steps) - 1:
            steps.append(int(total_steps) - 1)
        return {
            "steps": steps,
            "record_every_steps": int(record_every_steps),
            "record_points": 0,
            "record_mode": "interval",
        }

    steps = _evenly_spaced_record_steps(total_steps=int(total_steps), record_points=int(record_points))
    if not steps:
        steps = [int(total_steps) - 1]
    return {
        "steps": steps,
        "record_every_steps": 0,
        "record_points": int(record_points),
        "record_mode": "points",
    }


def _unique_probe_csv_path(out_dir, run_id):
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_run = _sanitize_name(run_id)
    return root / f"probe_rows_{safe_run}_{time.time_ns()}.csv"


def _scalar_meta(meta):
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool, np.integer, np.floating)) or v is None:
            out[k] = v
        else:
            out[k] = json.dumps(v, default=str)
    return out


def _build_probe_row(
    *,
    base_meta,
    run_id,
    step,
    probe_steps,
    probe_step_batch_size,
    step_train_eval_size,
    sam_step_batch_size,
    sam_rho,
    sam_adaptive,
    flat_norm,
    subset_size,
    mb_loss,
    train_loss,
    train_acc,
    test_loss,
    test_acc,
    flat_primal,
    flat_dual,
    flat_primal_freeze,
    flat_dual_freeze,
):
    row = dict(base_meta)
    row.update(
        {
            "run_id": run_id,
            "probe_step_batch_size": probe_step_batch_size,
            "step_train_eval_size": step_train_eval_size,
            "sam_step_batch_size": sam_step_batch_size,
            "probe_steps": probe_steps,
            "sam_rho": sam_rho,
            "sam_adaptive": bool(sam_adaptive),
            "flat_norm": flat_norm,
            "freeze_bn": 1,
            "step": int(step),
            "subset_size": int(subset_size),
            "mb_loss": mb_loss,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "flat_primal": flat_primal,
            "flat_dual": flat_dual,
            "flat_primal_freeze": flat_primal_freeze,
            "flat_dual_freeze": flat_dual_freeze,
            "gen_gap_acc": _safe_gap(train_acc, test_acc),
            "gen_gap_loss": _safe_gap(train_loss, test_loss),
            "timestamp": get_datetime(),
        }
    )
    return row


def run_probe(
    *,
    model,
    train_ds,
    val_ds,
    cfg,
    run_id,
    meta,
    sam_params,
    flat_params,
    enable_grads_for_sam,
    enable_grads_for_flatness,
    train_params_source,
    training_lr,
    training_wd,
    continuing_lr,
    continuing_wd,
    collate_fn,
):
    """
    Probe a model by repeating:
      (1) optional flatness/train metrics on selected steps,
      (2) SAM update on a fresh SAM batch every step.

    Measurements are recorded before the SAM update at each selected step.
    """
    device = set_device(cfg.device_index)
    model = model.to(device)

    source = _resolve_train_params_source(train_params_source)

    def _opt_float(v, *, allow_zero):
        x = _to_float_or_nan(v)
        if not np.isfinite(x):
            return None
        if (not allow_zero and x <= 0) or (allow_zero and x < 0):
            return None
        return float(x)

    train_lr_v = _opt_float(training_lr, allow_zero=False)
    train_wd_v = _opt_float(training_wd, allow_zero=True)
    cont_lr_v = _opt_float(continuing_lr, allow_zero=False)
    cont_wd_v = _opt_float(continuing_wd, allow_zero=True)

    if source == "specified":
        sam_lr = float(cfg.sam_lr)
        sam_wd = float(cfg.sam_weight_decay)
    elif source == "initial":
        if train_lr_v is None or train_wd_v is None:
            source = "specified"
            sam_lr = float(cfg.sam_lr)
            sam_wd = float(cfg.sam_weight_decay)
        else:
            sam_lr = float(train_lr_v)
            sam_wd = float(train_wd_v)
    else:
        lr_pick = cont_lr_v if cont_lr_v is not None else train_lr_v
        wd_pick = cont_wd_v if cont_wd_v is not None else train_wd_v
        if lr_pick is None or wd_pick is None:
            source = "specified"
            sam_lr = float(cfg.sam_lr)
            sam_wd = float(cfg.sam_weight_decay)
        else:
            sam_lr = float(lr_pick)
            sam_wd = float(wd_pick)

    batch_cfg = _resolve_probe_batch_settings(cfg)
    probe_bs = batch_cfg["probe_step_batch_size"]
    eval_bs = batch_cfg["step_train_eval_size"]
    sam_bs = batch_cfg["sam_step_batch_size"]
    duration_cfg = _resolve_probe_duration(
        cfg,
        train_ds=train_ds,
        sam_step_batch_size=sam_bs,
    )
    total_probe_steps = int(duration_cfg["total_steps"])
    record_cfg = _resolve_record_steps(cfg, total_steps=total_probe_steps)
    record_steps = [int(s) for s in record_cfg["steps"]]
    record_steps_set = set(record_steps)
    auto_microbatch_size = int(getattr(cfg, "auto_microbatch_size", DEFAULT_AUTO_MICROBATCH_SIZE))
    if auto_microbatch_size <= 0:
        raise ValueError("ProbeConfig.auto_microbatch_size must be > 0.")

    base_meta = _scalar_meta(dict(meta))
    base_meta["probe_param_source"] = source
    base_meta["lr"] = sam_lr
    base_meta["wd"] = sam_wd
    base_meta["auto_microbatch_size"] = int(auto_microbatch_size)
    base_meta["probe_epochs"] = int(duration_cfg["probe_epochs"])
    base_meta["probe_extra_steps"] = int(duration_cfg["probe_extra_steps"])
    base_meta["steps_per_epoch"] = int(duration_cfg["steps_per_epoch"])
    base_meta["probe_total_steps"] = int(total_probe_steps)
    base_meta["record_every_steps"] = int(record_cfg["record_every_steps"])
    base_meta["record_points"] = int(record_cfg["record_points"])
    base_meta["record_count_planned"] = int(len(record_steps))
    base_meta["record_mode"] = str(record_cfg["record_mode"])
    base_meta["probe_duration_mode"] = str(duration_cfg["duration_mode"])

    cached_test_loss = _to_float_or_nan(base_meta.get("test_loss"))
    cached_test_acc = _to_float_or_nan(base_meta.get("test_acc"))

    probe_loader = build_loader(train_ds, batch_size=probe_bs, workers=cfg.num_workers, shuffle=True, collate_fn=collate_fn)
    eval_loader = build_loader(train_ds, batch_size=eval_bs, workers=cfg.num_workers, shuffle=True, collate_fn=collate_fn)
    sam_loader = build_loader(train_ds, batch_size=sam_bs, workers=cfg.num_workers, shuffle=True, collate_fn=collate_fn)

    probe_state = {"loader": probe_loader, "iter": iter(probe_loader)}
    eval_state = {"loader": eval_loader, "iter": iter(eval_loader)}
    sam_state = {"loader": sam_loader, "iter": iter(sam_loader)}

    base_opt = str(getattr(cfg, "sam_base_opt", "sgd") or "sgd").strip().lower()
    if base_opt == "adam":
        base_opt = "adamw"
    if base_opt not in {"sgd", "adamw"}:
        raise ValueError(f"Unsupported sam_base_opt '{base_opt}'. Use 'sgd' or 'adamw'.")

    base_opt_cls = torch.optim.AdamW if base_opt == "adamw" else torch.optim.SGD
    base_kwargs: Dict[str, Any] = {"lr": sam_lr, "weight_decay": sam_wd}
    if base_opt_cls is torch.optim.SGD:
        base_kwargs["momentum"] = float(cfg.sam_momentum)

    sam_params = _dedup_params(sam_params)
    flat_params = _dedup_params(flat_params)

    rows: List[Dict[str, Any]] = []
    norm_keys = ("l1", "l2")

    with _MaybeEnableGrads(sam_params, enable_grads=enable_grads_for_sam):
        sam_opt = SAM(
            sam_params,
            base_opt_cls,
            rho=float(cfg.sam_rho),
            adaptive=bool(cfg.sam_adaptive),
            **base_kwargs,
        )

        # Optional baseline row before probing steps.
        if bool(cfg.include_full_batch_step):
            probe_batch = _next_batch(probe_state)
            eval_batch = _next_batch(eval_state)

            stats_freeze = flatness_on_batch(
                model,
                probe_batch,
                device,
                norm="l2",
                adaptive=cfg.sam_adaptive,
                flat_r=cfg.flat_r,
                flat_delta=cfg.flat_delta,
                freeze_bn=True,
                params=flat_params,
                enable_grads=enable_grads_for_flatness,
                microbatch_size=auto_microbatch_size,
            )
            stats_unfreeze = None
            if cfg.include_unfreeze:
                stats_unfreeze = flatness_on_batch(
                    model,
                    probe_batch,
                    device,
                    norm="l2",
                    adaptive=cfg.sam_adaptive,
                    flat_r=cfg.flat_r,
                    flat_delta=cfg.flat_delta,
                    freeze_bn=False,
                    params=flat_params,
                    enable_grads=enable_grads_for_flatness,
                    microbatch_size=auto_microbatch_size,
                )
            eval_stats = minibatch_loss_acc(
                model,
                eval_batch,
                device,
                microbatch_size=auto_microbatch_size,
            )
            chosen = stats_unfreeze if stats_unfreeze is not None else stats_freeze
            for norm_key in norm_keys:
                chosen_norm = chosen["by_norm"][norm_key]
                freeze_norm = stats_freeze["by_norm"][norm_key]
                rows.append(
                    _build_probe_row(
                        base_meta=base_meta,
                        run_id=run_id,
                        step=-1,
                        probe_steps=int(total_probe_steps),
                        probe_step_batch_size=probe_bs,
                        step_train_eval_size=eval_bs,
                        sam_step_batch_size=sam_bs,
                        sam_rho=float(cfg.sam_rho),
                        sam_adaptive=bool(cfg.sam_adaptive),
                        flat_norm=norm_key,
                        subset_size=infer_batch_size(probe_batch),
                        mb_loss=float(chosen["mb_loss"]),
                        train_loss=float(eval_stats["loss"]),
                        train_acc=float(eval_stats["acc"]),
                        test_loss=float(cached_test_loss),
                        test_acc=float(cached_test_acc),
                        flat_primal=float(chosen_norm["flat_primal"] if stats_unfreeze is not None else np.nan),
                        flat_dual=float(chosen_norm["flat_dual"] if stats_unfreeze is not None else np.nan),
                        flat_primal_freeze=float(freeze_norm["flat_primal"]),
                        flat_dual_freeze=float(freeze_norm["flat_dual"]),
                    )
                )

        step_iter = range(int(total_probe_steps))
        if cfg.show_progress:
            step_iter = tqdm(step_iter, desc=f"probe steps (record {len(record_steps)})", total=int(total_probe_steps))

        for step in step_iter:
            if int(step) in record_steps_set:
                # 1) flatness on fresh probe batch.
                probe_batch = _next_batch(probe_state)
                stats_freeze = flatness_on_batch(
                    model,
                    probe_batch,
                    device,
                    norm="l2",
                    adaptive=cfg.sam_adaptive,
                    flat_r=cfg.flat_r,
                    flat_delta=cfg.flat_delta,
                    freeze_bn=True,
                    params=flat_params,
                    enable_grads=enable_grads_for_flatness,
                    microbatch_size=auto_microbatch_size,
                )
                stats_unfreeze = None
                if cfg.include_unfreeze:
                    stats_unfreeze = flatness_on_batch(
                        model,
                        probe_batch,
                        device,
                        norm="l2",
                        adaptive=cfg.sam_adaptive,
                        flat_r=cfg.flat_r,
                        flat_delta=cfg.flat_delta,
                        freeze_bn=False,
                        params=flat_params,
                        enable_grads=enable_grads_for_flatness,
                        microbatch_size=auto_microbatch_size,
                    )

                # 2) train loss/acc on fresh eval batch.
                eval_batch = _next_batch(eval_state)
                eval_stats = minibatch_loss_acc(
                    model,
                    eval_batch,
                    device,
                    microbatch_size=auto_microbatch_size,
                )

                chosen = stats_unfreeze if stats_unfreeze is not None else stats_freeze
                for norm_key in norm_keys:
                    chosen_norm = chosen["by_norm"][norm_key]
                    freeze_norm = stats_freeze["by_norm"][norm_key]
                    rows.append(
                        _build_probe_row(
                            base_meta=base_meta,
                            run_id=run_id,
                            step=int(step),
                            probe_steps=int(total_probe_steps),
                            probe_step_batch_size=probe_bs,
                            step_train_eval_size=eval_bs,
                            sam_step_batch_size=sam_bs,
                            sam_rho=float(cfg.sam_rho),
                            sam_adaptive=bool(cfg.sam_adaptive),
                            flat_norm=norm_key,
                            subset_size=infer_batch_size(probe_batch),
                            mb_loss=float(chosen["mb_loss"]),
                            train_loss=float(eval_stats["loss"]),
                            train_acc=float(eval_stats["acc"]),
                            test_loss=float(cached_test_loss),
                            test_acc=float(cached_test_acc),
                            flat_primal=float(chosen_norm["flat_primal"] if stats_unfreeze is not None else np.nan),
                            flat_dual=float(chosen_norm["flat_dual"] if stats_unfreeze is not None else np.nan),
                            flat_primal_freeze=float(freeze_norm["flat_primal"]),
                            flat_dual_freeze=float(freeze_norm["flat_dual"]),
                        )
                    )

            # 3) SAM update on fresh SAM batch.
            sam_batch = _next_batch(sam_state)
            sam_update_on_batch(
                model,
                sam_opt,
                sam_batch,
                device,
                microbatch_size=auto_microbatch_size,
            )

    probe_csv = _unique_probe_csv_path(cfg.out_dir, run_id)
    out_df = pd.DataFrame(rows)

    # Keep the canonical columns first, then append any metadata/target columns.
    extra_cols = [c for c in out_df.columns if c not in PROBE_COLUMNS]
    ordered_cols = list(PROBE_COLUMNS) + sorted(extra_cols)
    for col in ordered_cols:
        if col not in out_df.columns:
            out_df[col] = np.nan
    out_df.to_csv(probe_csv, index=False, columns=ordered_cols)
    return str(probe_csv)


# ----------------------------- DF scoring -----------------------------


SCORE_COLUMNS = [
    "FLAT_STD",
    "FLAT_VAR",
    "FLAT_RANGE",
    "FLAT_OVER_LOSS_STD",
    "FLAT_OVER_LOSS_VAR",
    "FLAT_OVER_LOSS_RANGE",
    "FLAT2_OVER_LOSS_STD",
    "FLAT2_OVER_LOSS_VAR",
    "FLAT2_OVER_LOSS_RANGE",
]


def compute_scores_for_series(flat, loss):
    flat = np.asarray(flat, dtype=float)
    loss = np.asarray(loss, dtype=float)
    loss_eps = np.maximum(loss, 1e-12)

    ratio = _safe_divide(flat, loss_eps)
    ratio2 = _safe_divide(flat * flat, loss_eps)

    return {
        "FLAT_STD": float(np.nanstd(flat)),
        "FLAT_VAR": float(np.nanvar(flat)),
        "FLAT_RANGE": float(np.nanmax(flat) - np.nanmin(flat)),
        "FLAT_OVER_LOSS_STD": float(np.nanstd(ratio)),
        "FLAT_OVER_LOSS_VAR": float(np.nanvar(ratio)),
        "FLAT_OVER_LOSS_RANGE": float(np.nanmax(ratio) - np.nanmin(ratio)),
        "FLAT2_OVER_LOSS_STD": float(np.nanstd(ratio2)),
        "FLAT2_OVER_LOSS_VAR": float(np.nanvar(ratio2)),
        "FLAT2_OVER_LOSS_RANGE": float(np.nanmax(ratio2) - np.nanmin(ratio2)),
    }


@dataclass
class DFConfig:
    probe_csv: str
    target_columns: List[str]
    out_dir: Optional[str] = None
    min_steps: int = 2
    min_step_index: int = 0


def _load_and_dedup_probe_csv(path):
    df = pd.read_csv(path)
    if "run_id" not in df.columns or "step" not in df.columns:
        raise KeyError("probe_csv must include at least 'run_id' and 'step' columns.")

    df = df.copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"])
    df["step"] = df["step"].astype(int)

    dedup_cols = ["run_id", "step"]
    if "flat_norm" in df.columns:
        flat_norm = df["flat_norm"].astype(str).str.strip().str.lower()
        flat_norm = flat_norm.where(flat_norm != "nan", "")
        df["flat_norm"] = flat_norm
        dedup_cols = ["run_id", "flat_norm", "step"]

    df["_row_order"] = np.arange(len(df))
    sort_cols = dedup_cols + ["_row_order"]
    if "timestamp" in df.columns:
        sort_cols = dedup_cols + ["timestamp", "_row_order"]
    df = df.sort_values(sort_cols)

    dup_mask = df.duplicated(dedup_cols, keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        dedup_name = "+".join(dedup_cols)
        print(
            f"[compute_df] Found {n_dup} duplicate {dedup_name} rows; "
            f"keeping the last row per ({dedup_name})."
        )

    df = df.drop_duplicates(dedup_cols, keep="last")
    df = df.sort_values(dedup_cols).reset_index(drop=True)
    return df.drop(columns=["_row_order"], errors="ignore")


def _resolve_target_columns(scores_df, cfg):
    targets = [str(c) for c in cfg.target_columns]
    if not targets:
        raise ValueError("DFConfig.target_columns must contain at least one target column.")
    missing = [c for c in targets if c not in scores_df.columns]
    if missing:
        raise KeyError(f"Missing target column(s) in scores dataframe: {missing}")
    return targets


def compute_df(cfg):
    df = _load_and_dedup_probe_csv(cfg.probe_csv)

    for col in ["train_loss", "test_loss", "test_acc", "mb_loss", "flat_primal", "flat_primal_freeze"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flatness columns to score.
    flat_cols: List[str] = []
    for c in ["flat_primal_freeze", "flat_primal"]:
        if c in df.columns and np.isfinite(pd.to_numeric(df[c], errors="coerce")).any():
            flat_cols.append(c)

    if not flat_cols:
        raise ValueError("probe_csv has no usable flatness columns (expected flat_primal_freeze and/or flat_primal).")

    has_flat_norm = "flat_norm" in df.columns and df["flat_norm"].astype(str).str.strip().ne("").any()
    group_cols = ["run_id"] + (["flat_norm"] if has_flat_norm else [])

    rows: List[Dict[str, Any]] = []
    for group_key, g_all in df.groupby(group_cols, sort=False):
        if has_flat_norm:
            run_id = group_key[0]
            flat_norm = group_key[1]
        else:
            run_id = group_key
            flat_norm = None
        g_all = g_all.sort_values("step")
        g = g_all[g_all["step"] >= int(cfg.min_step_index)]
        if len(g) < int(cfg.min_steps):
            continue

        last = g_all.iloc[-1]
        base_run = {
            "run_id": run_id,
            "final_test_acc": _to_float_or_nan(last.get("test_acc", np.nan)),
            "final_test_loss": _to_float_or_nan(last.get("test_loss", np.nan)),
        }
        if flat_norm is not None:
            base_run["flat_norm"] = flat_norm

        # Keep any run-level columns that are constant-like (copied from runs.csv).
        passthrough_cols = [
            c
            for c in g_all.columns
            if c
            not in {
                "run_id",
                "flat_norm",
                "step",
                "subset_size",
                "mb_loss",
                "train_loss",
                "train_acc",
                "test_loss",
                "test_acc",
                "flat_primal",
                "flat_dual",
                "flat_primal_freeze",
                "flat_dual_freeze",
                "gen_gap_acc",
                "gen_gap_loss",
                "timestamp",
            }
        ]

        for col in passthrough_cols:
            if col in base_run:
                continue
            value = g_all[col].dropna()
            if len(value) == 0:
                continue
            base_run[col] = value.iloc[-1]

        for flat_col in flat_cols:
            s = pd.to_numeric(g[flat_col], errors="coerce").to_numpy(dtype=float)
            l = pd.to_numeric(g["train_loss"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(s) & np.isfinite(l)
            if int(valid.sum()) < int(cfg.min_steps):
                continue

            score_row = {
                **base_run,
                "flat_col": flat_col,
                "t_len": int(valid.sum()),
                "WB_value": _to_float_or_nan(
                    g_all.loc[g_all["step"] == 0, flat_col].iloc[-1] if (g_all["step"] == 0).any() else np.nan
                ),
            }
            score_row.update(compute_scores_for_series(s[valid], l[valid]))
            rows.append(score_row)

    scores_df = pd.DataFrame(rows)

    if scores_df.empty:
        corr_df = pd.DataFrame(
            columns=["flat_col", "score_name", "target", "spearman", "kendall", "pearson", "n"]
        )
        out_dir = Path(cfg.out_dir) if cfg.out_dir else Path(cfg.probe_csv).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(out_dir / "df_scores.csv", index=False)
        corr_df.to_csv(out_dir / "df_correlations.csv", index=False)
        print(f"Saved scores -> {out_dir / 'df_scores.csv'}\\nSaved correlations -> {out_dir / 'df_correlations.csv'}")
        return scores_df, corr_df

    target_cols = _resolve_target_columns(scores_df, cfg)

    meta_cols = {
        "run_id",
        "flat_col",
        "t_len",
        "WB_value",
        "final_test_acc",
        "final_test_loss",
    }
    meta_cols.update(target_cols)

    numeric_cols: List[str] = []
    for c in SCORE_COLUMNS:
        if c not in scores_df.columns:
            continue
        vals = pd.to_numeric(scores_df[c], errors="coerce")
        if int(np.isfinite(vals).sum()) >= 2:
            numeric_cols.append(c)

    corr_rows: List[Dict[str, Any]] = []
    corr_group_cols = ["flat_col"] + (["flat_norm"] if "flat_norm" in scores_df.columns else [])
    for group_key, g_flat in scores_df.groupby(corr_group_cols, sort=False):
        if "flat_norm" in scores_df.columns:
            flat_col = group_key[0]
            corr_flat_norm = group_key[1]
        else:
            flat_col = group_key
            corr_flat_norm = None
        for score_name in numeric_cols:
            x = pd.to_numeric(g_flat[score_name], errors="coerce")
            for target in target_cols:
                y = pd.to_numeric(g_flat[target], errors="coerce")
                valid = x.notna() & y.notna()
                n_valid = int(valid.sum())
                if n_valid < 2:
                    continue
                row = {
                    "flat_col": flat_col,
                    "score_name": score_name,
                    "target": target,
                    "spearman": float(x[valid].corr(y[valid], method="spearman")),
                    "kendall": float(x[valid].corr(y[valid], method="kendall")),
                    "pearson": float(x[valid].corr(y[valid], method="pearson")),
                    "n": n_valid,
                }
                if corr_flat_norm is not None:
                    row["flat_norm"] = corr_flat_norm
                corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows)

    out_dir = Path(cfg.out_dir) if cfg.out_dir else Path(cfg.probe_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / "df_scores.csv"
    corr_path = out_dir / "df_correlations.csv"
    scores_df.to_csv(score_path, index=False)
    corr_df.to_csv(corr_path, index=False)
    print(f"Saved scores -> {score_path}\\nSaved correlations -> {corr_path}")
    return scores_df, corr_df
