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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# 4-GPU usage: set NNGridConfig.num_gpus (default 4) or pass a smaller value.
# run_grid will split grid runs across GPUs and merge results into a single runs.csv.
# run_probe_for_runs also parallelizes probing across visible GPUs and writes one merged CSV.

from train import append_csv_row, init_grid_paths, write_json

# Training helpers live in df_util/sam_train.py.
DF_UTIL = Path(__file__).resolve().parents[1] / "df_util"
if str(DF_UTIL) not in sys.path:
    sys.path.insert(0, str(DF_UTIL))

from sam_train import (  # noqa: E402
    SAM,
    disable_running_stats,
    enable_running_stats,
    get_datetime,
    prepare_dataset,
    prepare_model,
    set_device,
    set_seed,
    smooth_crossentropy,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


@dataclass
class NNGridConfig:
    """Configuration for CIFAR-style grid training runs."""

    out_root: str
    experiment_name: str
    append_csv_path: Optional[str] = None
    force_append: bool = False

    datasets: List[str] = field(default_factory=lambda: ["cifar10"])
    models: List[str] = field(default_factory=lambda: ["vgg13_bn"])
    seeds: List[int] = field(default_factory=lambda: [0])
    optimizers: List[str] = field(default_factory=lambda: ["sgd"])
    lrs: List[float] = field(default_factory=lambda: [0.1])
    wds: List[float] = field(default_factory=lambda: [1e-4])
    batch_sizes: List[int] = field(default_factory=lambda: [128])
    dropouts: List[float] = field(default_factory=lambda: [0.0])
    label_smoothing_list: List[float] = field(default_factory=lambda: [0.0])

    max_epochs: int = 200
    train_loss_threshold: Optional[float] = 0.02
    num_workers: int = 2
    momentum: float = 0.9

    sam_rho: float = 2.0
    sam_adaptive: bool = False

    eval_microbatch: int = 256
    compute_test_metrics: bool = True

    stop_no_improve_interval: int = 0
    stop_min_improve: float = 0.01

    lr_milestones: Tuple[int, ...] = (100, 150)
    lr_gamma: float = 0.1

    compile_model: bool = False
    use_amp: bool = True
    device_index: int = 0
    num_gpus: int = 4
    max_concurrent: Optional[int] = None
    show_progress: bool = False


CSV_COLUMNS = [
    "run_id",
    "ckpt_dir",
    "ckpt_path",
    "dataset",
    "model",
    "seed",
    "optimizer",
    "lr",
    "lr_final",
    "wd",
    "batch_size",
    "dropout",
    "label_smoothing",
    "epochs_trained",
    "final_train_loss",
    "final_train_acc",
    "test_loss",
    "test_acc",
]


def set_dropout_p(model: nn.Module, p: float) -> None:
    """
    Set dropout probability for all Dropout-like modules in a model.

    Args:
        model: Model whose dropout layers should be updated.
        p: Dropout probability to apply.
    """
    for module in model.modules():
        if isinstance(
            module,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.AlphaDropout),
        ):
            module.p = float(p)


def build_train_loader(dataset, batch_size: int, workers: int) -> DataLoader:
    """
    Construct a shuffled training DataLoader.

    Args:
        dataset: Training dataset instance.
        batch_size: Batch size per iteration.
        workers: Number of DataLoader workers.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
    )


def build_eval_loader(dataset, micro_batch: int, workers: int) -> DataLoader:
    """
    Construct an evaluation DataLoader with a capped micro-batch size.

    Args:
        dataset: Dataset to evaluate.
        micro_batch: Max batch size per forward pass.
        workers: Number of DataLoader workers.
    """
    bs = min(max(1, micro_batch), len(dataset)) if micro_batch > 0 else len(dataset)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
    )


@torch.no_grad()
def loss_acc_over_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Compute average loss and accuracy over a DataLoader.

    Args:
        model: Model to evaluate.
        loader: DataLoader to iterate over.
        device: Device to use for computation.
    """
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            total_loss += ce(logits, yb).item()
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_n += yb.numel()
    return total_loss / max(1, total_n), total_correct / max(1, total_n)


def train_one_run(
    *,
    device: torch.device,
    dataset_name: str,
    model_name: str,
    seed: int,
    optimizer_name: str,
    lr: float,
    wd: float,
    bs: int,
    dropout_p: float,
    label_smoothing: float,
    max_epochs: int,
    train_loss_threshold: Optional[float],
    num_workers: int,
    momentum: float,
    sam_rho: float,
    sam_adaptive: bool,
    eval_microbatch: int,
    compute_test_metrics: bool,
    stop_no_improve_interval: int,
    stop_min_improve: float,
    lr_milestones: Tuple[int, ...],
    lr_gamma: float,
    compile_model: bool,
    use_amp: bool,
    show_progress: bool = False,
    early_stop_writer: Optional[Callable[[nn.Module, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Train one model configuration and return metrics plus artifacts.

    Args:
        device: Torch device used for training.
        dataset_name: Dataset identifier (e.g., cifar10).
        model_name: Model identifier (see df_util/sam_train.prepare_model).
        seed: Random seed used for init and dataloaders.
        optimizer_name: Optimizer name (sgd/adam/sam).
        lr: Learning rate.
        wd: Weight decay.
        bs: Training batch size.
        dropout_p: Dropout probability to set on model modules.
        label_smoothing: Label smoothing value for training loss.
        max_epochs: Max number of epochs to run.
        train_loss_threshold: Early stop when train loss <= threshold.
        num_workers: DataLoader worker count.
        momentum: SGD/SAM momentum.
        sam_rho: SAM rho parameter.
        sam_adaptive: Whether SAM is adaptive.
        eval_microbatch: Micro-batch size for evaluation loaders.
        compute_test_metrics: Whether to compute test metrics after training.
        stop_no_improve_interval: Window size for plateau early stop.
        stop_min_improve: Plateau threshold for max-min loss range.
        lr_milestones: Epoch milestones for MultiStepLR.
        lr_gamma: Gamma for MultiStepLR.
        compile_model: Whether to call torch.compile on the model.
        use_amp: Whether to use AMP for non-SAM optimizers.
        early_stop_writer: Optional callback for immediate checkpoint/CSV write.
    """
    if dataset_name.startswith("cifar") and model_name.startswith("WideResNet"):
        trainset, testset = prepare_dataset(dataset_name, cutout=True)
    else:
        trainset, testset = prepare_dataset(dataset_name)

    train_loader = build_train_loader(trainset, bs, num_workers)
    train_eval_loader = build_eval_loader(trainset, eval_microbatch, num_workers)
    test_eval_loader = build_eval_loader(testset, eval_microbatch, num_workers)

    model = prepare_model(model_name, dataset_name, seed)
    set_dropout_p(model, dropout_p)
    model.to(device)
    if compile_model:
        model = torch.compile(model)

    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name.lower() == "sam":
        base_opt = torch.optim.SGD
        opt = SAM(
            params,
            base_opt,
            rho=sam_rho,
            adaptive=sam_adaptive,
            lr=lr,
            weight_decay=wd,
            momentum=momentum,
        )
        use_sam = True
    elif optimizer_name.lower() == "sgd":
        opt = torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
        use_sam = False
    elif optimizer_name.lower() == "adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        use_sam = False
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=list(lr_milestones),
        gamma=lr_gamma,
    )
    last_lr_used = float(opt.param_groups[0].get("lr", lr))

    amp_enabled = use_amp and device.type == "cuda" and not use_sam
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    loss_fn = lambda logits, y: smooth_crossentropy(logits, y, smoothing=label_smoothing).mean()
    best_train_loss = float("inf")
    epochs_trained = 0
    early_stop_reason = ""
    early_row_written = False

    window_losses: List[float] = []

    for ep in range(1, max_epochs + 1):
        ep_t0 = time.time()
        model.train()
        last_lr_used = float(opt.param_groups[0].get("lr", last_lr_used))
        iterator = train_loader
        if show_progress:
            iterator = tqdm(train_loader, desc=f"ep{ep:03d}", leave=False)
        for xb, yb in iterator:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if use_sam:
                enable_running_stats(model)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.first_step(zero_grad=True)

                disable_running_stats(model)
                loss2 = loss_fn(model(xb), yb)
                loss2.backward()
                opt.second_step(zero_grad=True)
            else:
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    loss = loss_fn(model(xb), yb)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

        train_loss, train_acc = loss_acc_over_loader(model, train_eval_loader, device)
        epoch_time = time.time() - ep_t0
        best_train_loss = min(best_train_loss, train_loss)
        epochs_trained = ep
        print(
            f"[{dataset_name} | {model_name} | {optimizer_name}] "
            f"ep {ep:03d} ({epoch_time:.1f}s) train_loss={train_loss:.6f} train_acc={train_acc:.4f}",
            flush=True,
        )

        if train_loss_threshold is not None and train_loss <= train_loss_threshold:
            early_stop_reason = (
                f"train_loss <= threshold ({train_loss:.6f} <= {train_loss_threshold})"
            )
            break

        if stop_no_improve_interval > 0:
            window_losses.append(float(train_loss))
            if len(window_losses) > stop_no_improve_interval:
                window_losses = window_losses[-stop_no_improve_interval :]
            if len(window_losses) == stop_no_improve_interval:
                w_range = max(window_losses) - min(window_losses)
                if w_range <= stop_min_improve:
                    early_stop_reason = (
                        f"plateau: window max-min={w_range:.6f} "
                        f"(<= {stop_min_improve}) over last {stop_no_improve_interval} epochs"
                    )
                    print(f"[Early stop] {early_stop_reason}")
                    if early_stop_writer is not None:
                        try:
                            early_stop_writer(
                                model,
                                {
                                    "epoch": ep,
                                    "train_loss": float(train_loss),
                                    "train_acc": float(train_acc),
                                    "best_train_loss": float(best_train_loss),
                                    "lr_final": float(last_lr_used),
                                    "wd_final": float(opt.param_groups[0].get("weight_decay", wd)),
                                    "stop_reason": early_stop_reason,
                                },
                            )
                            early_row_written = True
                        except Exception:
                            print("[Warning] Early-stop writer failed; will proceed normally.")
                    break

        scheduler.step()

    test_loss = test_acc = None
    if compute_test_metrics:
        test_loss, test_acc = loss_acc_over_loader(model, test_eval_loader, device)
    wd_final = float(opt.param_groups[0].get("weight_decay", wd))

    return {
        "model": model,
        "trainset": trainset,
        "testset": testset,
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "best_train_loss": float(best_train_loss),
        "epochs_trained": int(epochs_trained),
        "test_loss": None if test_loss is None else float(test_loss),
        "test_acc": None if test_acc is None else float(test_acc),
        "lr_final": float(last_lr_used),
        "wd_final": float(wd_final),
        "stop_reason": early_stop_reason,
        "early_row_written": early_row_written,
    }


def _split_evenly(seq, n: int):
    buckets = [[] for _ in range(max(1, n))]
    for i, item in enumerate(seq):
        buckets[i % len(buckets)].append(item)
    return buckets


def _ensure_unbuffered_output():
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass


def _merge_csvs(csv_paths: List[str], final_path: Path, columns: Optional[List[str]] = None) -> None:
    import pandas as pd

    frames = []

    for p in csv_paths:
        if not p:
            continue
        path_obj = Path(p)
        if path_obj.exists() and path_obj.stat().st_size > 0:
            frames.append(pd.read_csv(path_obj))
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if columns:
        for col in columns:
            if col not in merged.columns:
                merged[col] = np.nan
        merged.to_csv(final_path, index=False, columns=columns)
    else:
        merged.to_csv(final_path, index=False)


def _run_one_combo_nn(
    *,
    device: torch.device,
    idx: int,
    total_runs: int,
    combo: Tuple[str, ...],
    cfg: NNGridConfig,
    paths,
    csv_path: Path,
) -> None:
    (
        dataset_name,
        model_name,
        seed,
        optimizer_name,
        lr,
        wd,
        bs,
        dropout_p,
        label_smoothing,
    ) = combo

    run_id = (
        f"{dataset_name}__{model_name}__seed{seed}__opt{optimizer_name}"
        f"__lr{lr:g}__wd{wd:g}__bs{bs}__do{dropout_p:g}__ls{label_smoothing:g}"
    )
    run_dir = paths.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==== [{idx}/{total_runs}] {run_id} ====", flush=True)
    set_seed(seed)

    def early_stop_writer(model_obj: nn.Module, info: Dict[str, Any]) -> None:
        ckpt_path_immediate = run_dir / f"model_earlystop_ep{info['epoch']:03d}.pt"
        torch.save(
            {
                "state_dict": model_obj.state_dict(),
                "meta": {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed,
                    "optimizer": optimizer_name,
                    "lr": lr,
                    "lr_final": float(info.get("lr_final", lr)),
                    "wd": wd,
                    "wd_final": float(info.get("wd_final", wd)),
                    "batch_size": bs,
                    "dropout": dropout_p,
                    "label_smoothing": label_smoothing,
                    "epochs_trained": int(info["epoch"]),
                    "final_train_loss": float(info["train_loss"]),
                    "final_train_acc": float(info["train_acc"]),
                    "stop_reason": info.get("stop_reason", "plateau"),
                    "timestamp": get_datetime(),
                },
            },
            ckpt_path_immediate,
        )

        row_now = {
            "run_id": run_id,
            "ckpt_dir": str(run_dir),
            "ckpt_path": str(ckpt_path_immediate),
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "optimizer": optimizer_name,
            "lr": lr,
            "lr_final": float(info.get("lr_final", lr)),
            "wd": wd,
            "batch_size": bs,
            "dropout": dropout_p,
            "label_smoothing": label_smoothing,
            "epochs_trained": int(info["epoch"]),
            "final_train_loss": float(info["train_loss"]),
            "final_train_acc": float(info["train_acc"]),
            "test_loss": "",
            "test_acc": "",
        }
        append_csv_row(csv_path, row_now, CSV_COLUMNS)
        print(f"[Early stop] Appended row to {csv_path}", flush=True)
        print(f"[Early stop] Saved checkpoint to {ckpt_path_immediate}", flush=True)

    try:
        res = train_one_run(
            device=device,
            dataset_name=dataset_name,
            model_name=model_name,
            seed=seed,
            optimizer_name=optimizer_name,
            lr=lr,
            wd=wd,
            bs=bs,
            dropout_p=dropout_p,
            label_smoothing=label_smoothing,
            max_epochs=cfg.max_epochs,
            train_loss_threshold=cfg.train_loss_threshold,
            num_workers=cfg.num_workers,
            momentum=cfg.momentum,
            sam_rho=cfg.sam_rho,
            sam_adaptive=cfg.sam_adaptive,
            eval_microbatch=cfg.eval_microbatch,
            compute_test_metrics=cfg.compute_test_metrics,
            stop_no_improve_interval=cfg.stop_no_improve_interval,
            stop_min_improve=cfg.stop_min_improve,
            lr_milestones=cfg.lr_milestones,
            lr_gamma=cfg.lr_gamma,
            compile_model=cfg.compile_model,
            use_amp=cfg.use_amp,
            show_progress=cfg.show_progress,
            early_stop_writer=early_stop_writer if cfg.stop_no_improve_interval > 0 else None,
        )
    except Exception as exc:
        print(f"[Error] Run failed with exception: {exc}", flush=True)
        print("Run failed; skipping failure and continuing.", flush=True)
        row = {
            "run_id": run_id,
            "ckpt_dir": str(run_dir),
            "ckpt_path": "",
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "optimizer": optimizer_name,
            "lr": lr,
            "lr_final": float("nan"),
            "wd": wd,
            "batch_size": bs,
            "dropout": dropout_p,
            "label_smoothing": label_smoothing,
            "epochs_trained": -1,
            "final_train_loss": float("nan"),
            "final_train_acc": float("nan"),
            "test_loss": float("nan"),
            "test_acc": float("nan"),
        }
        # append_csv_row(csv_path, row, CSV_COLUMNS)
        return

    if res.get("early_row_written", False):
        print("[Info] Early-stop CSV already written; skipping final append.", flush=True)
        return

    ckpt_path = run_dir / "model.pt"
    torch.save(
        {
            "state_dict": res["model"].state_dict(),
            "meta": {
                "dataset": dataset_name,
                "model": model_name,
                "seed": seed,
                "optimizer": optimizer_name,
                "lr": lr,
                "lr_final": float(res.get("lr_final", lr)),
                "wd": wd,
                "wd_final": float(res.get("wd_final", wd)),
                "batch_size": bs,
                "dropout": dropout_p,
                "label_smoothing": label_smoothing,
                "epochs_trained": res["epochs_trained"],
                "final_train_loss": res["train_loss"],
                "final_train_acc": res["train_acc"],
                "timestamp": get_datetime(),
                "stop_reason": res.get("stop_reason", ""),
            },
        },
        ckpt_path,
    )

    row = {
        "run_id": run_id,
        "ckpt_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "optimizer": optimizer_name,
        "lr": lr,
        "lr_final": float(res.get("lr_final", lr)),
        "wd": wd,
        "batch_size": bs,
        "dropout": dropout_p,
        "label_smoothing": label_smoothing,
        "epochs_trained": res["epochs_trained"],
        "final_train_loss": res["train_loss"],
        "final_train_acc": res["train_acc"],
        "test_loss": "" if res["test_loss"] is None else res["test_loss"],
        "test_acc": "" if res["test_acc"] is None else res["test_acc"],
    }
    append_csv_row(csv_path, row, CSV_COLUMNS)

    print(f"Saved checkpoint to: {ckpt_path}", flush=True)
    print(f"Appended row to:     {csv_path}", flush=True)


def _nn_worker_entry(chunk, cfg: NNGridConfig, paths, device_id: int, total_runs: int) -> str:
    if not chunk:
        return ""
    _ensure_unbuffered_output()
    cfg_local = copy.deepcopy(cfg)
    cfg_local.device_index = device_id
    device = set_device(cfg_local.device_index)
    worker_csv = paths.out_dir / f"{paths.csv_path.stem}_gpu{device_id}.csv"
    for idx, combo in chunk:
        _run_one_combo_nn(
            device=device,
            idx=idx,
            total_runs=total_runs,
            combo=combo,
            cfg=cfg_local,
            paths=paths,
            csv_path=worker_csv,
        )
    return str(worker_csv)


def _nn_probe_worker(
    chunk: List[Dict[str, Any]],
    probe_cfg,
    device_id: int,
    probe_columns: Optional[List[str]],
    train_params_source: Optional[str],
    use_original_train_params: Optional[bool],
    train_lr_column: str,
    train_lr_final_column: str,
) -> str:
    if not chunk:
        return ""
    import pandas as pd

    cfg_local = copy.deepcopy(probe_cfg)
    cfg_local.device_index = device_id
    frames = []
    for row in chunk:
        ckpt_path = str(row.get("ckpt_path", row.get("checkpoint_path", "")))
        if not ckpt_path or not Path(ckpt_path).exists():
            print(f"[Probe worker {device_id}] missing ckpt: {ckpt_path}")
            continue
        out_path = run_probe(
            cfg_local,
            run_row=row,
            ckpt_path=ckpt_path,
            train_params_source=train_params_source,
            use_original_train_params=use_original_train_params,
            train_lr_column=train_lr_column,
            train_lr_final_column=train_lr_final_column,
        )
        if out_path and Path(out_path).exists():
            frames.append(pd.read_csv(out_path))
    if not frames:
        return ""
    worker_csv = Path(cfg_local.out_dir) / f"probe_gpu{device_id}.csv"
    merged = pd.concat(frames, ignore_index=True)
    if probe_columns:
        for col in probe_columns:
            if col not in merged.columns:
                merged[col] = np.nan
        merged.to_csv(worker_csv, index=False, columns=probe_columns)
    else:
        merged.to_csv(worker_csv, index=False)
    return str(worker_csv)


def run_grid(cfg: NNGridConfig, start_id: int = 0) -> str:
    """
    Run a grid of CIFAR-style training jobs and write a CSV summary.

    Args:
        cfg: Grid configuration.
        start_id: Skip all runs with index < start_id.
    """
    paths = init_grid_paths(
        out_root=cfg.out_root,
        experiment_name=cfg.experiment_name,
        append_csv_path=cfg.append_csv_path,
        force_append=cfg.force_append,
    )

    write_json(
        paths.out_dir / "config.json",
        {
            **cfg.__dict__,
            "datasets": list(cfg.datasets),
            "models": list(cfg.models),
            "seeds": list(cfg.seeds),
            "optimizers": list(cfg.optimizers),
            "lrs": list(cfg.lrs),
            "wds": list(cfg.wds),
            "batch_sizes": list(cfg.batch_sizes),
            "dropouts": list(cfg.dropouts),
            "label_smoothing_list": list(cfg.label_smoothing_list),
        },
    )

    combos_all = list(
        itertools.product(
            cfg.datasets,
            cfg.models,
            cfg.seeds,
            cfg.optimizers,
            cfg.lrs,
            cfg.wds,
            cfg.batch_sizes,
            cfg.dropouts,
            cfg.label_smoothing_list,
        )
    )
    total_runs = len(combos_all)
    combos = [(i, combo) for i, combo in enumerate(combos_all, 1) if i >= start_id]
    print(f"Total runs: {total_runs}")

    if not combos:
        print("No runs to execute (start_id beyond grid).")
        return str(paths.csv_path)

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested_gpus = max(1, getattr(cfg, "num_gpus", 4) or 1)
    usable_gpus = min(requested_gpus, available_gpus) if available_gpus > 0 else 1
    max_concurrency = getattr(cfg, "max_concurrent", None) or usable_gpus
    max_concurrency = max(1, min(max_concurrency, len(combos)))

    if usable_gpus <= 1 or max_concurrency == 1:
        device = set_device(cfg.device_index)
        for idx, combo in combos:
            _run_one_combo_nn(
                device=device,
                idx=idx,
                total_runs=total_runs,
                combo=combo,
                cfg=cfg,
                paths=paths,
                csv_path=paths.csv_path,
            )
        print("\nAll done")
        print(f"CSV at: {paths.csv_path}")
        print(f"Outputs in: {paths.out_dir}")
        return str(paths.csv_path)

    chunks = _split_evenly(combos, max_concurrency)
    ctx = mp.get_context("spawn")
    worker_csvs: List[str] = []
    with ProcessPoolExecutor(max_workers=max_concurrency, mp_context=ctx) as ex:
        futures = []
        for worker_idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            device_id = worker_idx % max(usable_gpus, 1)
            futures.append(
                ex.submit(
                    _nn_worker_entry,
                    chunk,
                    cfg,
                    paths,
                    device_id,
                    total_runs,
                )
            )
        for fut in as_completed(futures):
            res = fut.result()
            worker_csvs.append(res)
            print(f"[worker finished] {res}", flush=True)

    _merge_csvs(worker_csvs, paths.csv_path, CSV_COLUMNS)

    print("\nAll done")
    print(f"CSV at: {paths.csv_path}")
    print(f"Outputs in: {paths.out_dir}")
    return str(paths.csv_path)


def _resolve_train_params_source(
    train_params_source: Optional[str],
    use_original_train_params: Optional[bool],
    *,
    default_source: str = "continuing",
) -> str:
    """Resolve probe optimizer-param source mode."""
    if train_params_source is None:
        if use_original_train_params is None:
            source = default_source
        else:
            source = "initial" if bool(use_original_train_params) else "specified"
    else:
        source = str(train_params_source).strip().lower()

    aliases = {
        "probe": "specified",
        "manual": "specified",
        "train": "initial",
        "original": "initial",
        "final": "continuing",
        "continued": "continuing",
    }
    source = aliases.get(source, source)
    if source not in {"specified", "initial", "continuing"}:
        raise ValueError(
            f"Unsupported train_params_source '{source}'. "
            "Use one of: specified, initial, continuing."
        )
    return source

def build_probe_datasets(row: Dict[str, Any]):
    """Return CIFAR train/test datasets for probing."""
    dataset_name = row.get("dataset", row.get("dataset_name", ""))
    model_name = row.get("model", row.get("model_name", ""))
    if dataset_name.startswith("cifar") and model_name.startswith("WideResNet"):
        return prepare_dataset(dataset_name, cutout=True)
    return prepare_dataset(dataset_name)


def build_probe_model(row: Dict[str, Any]) -> nn.Module:
    """Build a CIFAR model from a runs.csv row."""
    seed = int(row.get("seed", 0) or 0)
    dataset_name = row.get("dataset", row.get("dataset_name", ""))
    model_name = row.get("model", row.get("model_name", ""))
    dropout_p = float(row.get("dropout", 0.0) or 0.0)
    set_seed(seed)
    model = prepare_model(model_name, dataset_name, seed)
    set_dropout_p(model, dropout_p)
    return model


def run_probe(
    probe_cfg,
    run_row: Dict[str, Any],
    *,
    ckpt_path: str,
    train_params_source: Optional[str] = None,
    use_original_train_params: Optional[bool] = None,
    train_lr_column: str = "lr",
    train_lr_final_column: str = "lr_final",
) -> str:
    """Probe a single CIFAR checkpoint described by run_row."""
    df_root = Path(__file__).resolve().parents[1] / "df_util"
    if str(df_root) not in sys.path:
        sys.path.insert(0, str(df_root))
    import df as df_util

    dataset_name = run_row.get("dataset", run_row.get("dataset_name", ""))
    model_name = run_row.get("model", run_row.get("model_name", ""))
    seed = int(run_row.get("seed", 0) or 0)

    train_ds, val_ds = prepare_dataset(dataset_name)
    set_seed(seed)
    model = prepare_model(model_name, dataset_name, seed)
    set_dropout_p(model, float(run_row.get("dropout", 0.0) or 0.0))

    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("state_dict", payload)
    ckpt_meta = payload.get("meta", {}) or {}
    model.load_state_dict(state)

    source = _resolve_train_params_source(
        train_params_source=train_params_source,
        use_original_train_params=use_original_train_params,
        default_source="continuing",
    )
    training_lr = float(
        run_row.get(train_lr_column, run_row.get("learning_rate", ckpt_meta.get("lr", 0.0))) or 0.0
    )
    continuing_lr = float(run_row.get(train_lr_final_column, ckpt_meta.get("lr_final", training_lr)) or training_lr)
    training_wd = float(run_row.get("wd", run_row.get("weight_decay", ckpt_meta.get("wd", 0.0))) or 0.0)
    continuing_wd = float(run_row.get("wd_final", ckpt_meta.get("wd_final", training_wd)) or training_wd)

    sam_params = [p for p in model.parameters() if p.requires_grad]
    flat_params = [p for p in model.parameters() if p.requires_grad]

    if source == "specified":
        chosen_lr = float(probe_cfg.sam_lr)
        chosen_wd = float(probe_cfg.sam_weight_decay)
    elif source == "initial":
        chosen_lr = training_lr
        chosen_wd = training_wd
    else:
        chosen_lr = continuing_lr
        chosen_wd = continuing_wd

    meta = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "optimizer": run_row.get("optimizer", ckpt_meta.get("optimizer", "adamw")),
        "probe_param_source": source,
        "lr": chosen_lr,
        "wd": chosen_wd,
        "batch_size": run_row.get("batch_size", run_row.get("train_batch_size", ckpt_meta.get("batch_size", ""))),
        "label_smoothing": run_row.get("label_smoothing", ckpt_meta.get("label_smoothing", 0.0)),
        "dropout": run_row.get("dropout", ckpt_meta.get("dropout", 0.0)),
        "ckpt_path": ckpt_path,
    }
    for key, value in run_row.items():
        k = str(key)
        if k not in meta:
            meta[k] = value

    return df_util.run_probe(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=probe_cfg,
        run_id=str(run_row.get("run_id", f"run_{seed}")),
        meta=meta,
        sam_params=sam_params,
        flat_params=flat_params,
        enable_grads_for_sam=True,
        enable_grads_for_flatness=True,
        train_params_source=source,
        training_lr=training_lr,
        training_wd=training_wd,
        continuing_lr=continuing_lr,
        continuing_wd=continuing_wd,
        collate_fn=None,
    )


def run_probe_for_runs(
    probe_cfg,
    runs_csv: str,
    *,
    train_params_source: Optional[str] = None,
    use_original_train_params: Optional[bool] = None,
    train_lr_column: str = "lr",
    train_lr_final_column: str = "lr_final",
) -> str:
    """Load checkpoints from runs.csv and probe each one using df_util.run_probe."""
    import pandas as pd

    runs = pd.read_csv(runs_csv).fillna("")
    if runs.empty:
        return ""

    probe_columns: Optional[List[str]] = None

    rows = [dict(row) for _, row in runs.iterrows()]
    source = _resolve_train_params_source(
        train_params_source=train_params_source,
        use_original_train_params=use_original_train_params,
        default_source="continuing",
    )
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested_gpus = max(1, getattr(probe_cfg, "num_gpus", 4) or 1)
    usable_gpus = min(requested_gpus, available_gpus) if available_gpus > 0 else 1
    max_workers = getattr(probe_cfg, "max_concurrent_probes", None) or usable_gpus
    max_workers = max(1, min(max_workers, len(rows)))

    merged_probe_path = Path(probe_cfg.out_dir) / "probe_merged.csv"
    if merged_probe_path.exists():
        merged_probe_path.unlink()

    if usable_gpus <= 1 or max_workers == 1:
        out_paths = []
        for row in rows:
            ckpt_path = str(row.get("ckpt_path", row.get("checkpoint_path", "")))
            if not ckpt_path or not Path(ckpt_path).exists():
                raise FileNotFoundError(f"missing ckpt for run {row.get('run_id','')} : {ckpt_path}")
            out_paths.append(
                run_probe(
                    probe_cfg,
                    run_row=row,
                    ckpt_path=ckpt_path,
                    train_params_source=source,
                    use_original_train_params=use_original_train_params,
                    train_lr_column=train_lr_column,
                    train_lr_final_column=train_lr_final_column,
                )
            )
        _merge_csvs(out_paths, merged_probe_path, probe_columns)
        return str(merged_probe_path)

    chunks = _split_evenly(rows, max_workers)
    ctx = mp.get_context("spawn")
    worker_csvs: List[str] = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = []
        for worker_idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            device_id = worker_idx % max(usable_gpus, 1)
            futures.append(
                ex.submit(
                    _nn_probe_worker,
                    chunk,
                    probe_cfg,
                    device_id,
                    probe_columns,
                    source,
                    use_original_train_params,
                    train_lr_column,
                    train_lr_final_column,
                )
            )
        for fut in as_completed(futures):
            worker_csvs.append(fut.result())

    _merge_csvs(worker_csvs, merged_probe_path, probe_columns)
    return str(merged_probe_path)
