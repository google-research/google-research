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

#!/usr/bin/env python3
"""Very lightweight end-to-end smoke test.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DFLAT_ROOT = Path(__file__).resolve().parents[1]
DF_UTIL = DFLAT_ROOT / "df_util"
if str(DF_UTIL) not in sys.path:
    sys.path.insert(0, str(DF_UTIL))

import df
import sam_train


def resolve_device():
    try:
        device = sam_train.set_device(0)
    except RuntimeError as exc:
        print(f"[light_test] no accelerator ({exc}); falling back to CPU.")
        return torch.device("cpu")
    print(f"[light_test] using device: {device}")
    return device


def build_model(seed, device):
    # prepare_model seeds internally, so both optimizers start from the same init.
    return sam_train.prepare_model("vgg16_bn", "cifar10", seed).to(device)


def train_one(*, kind, subset, n, device, args):
    """Train a fresh VGG16-BN for `args.epochs` epochs. kind in {sam, sgd}."""
    model = build_model(args.seed, device)
    train_loader = df.build_loader(
        subset, batch_size=args.train_batch_size, workers=0, shuffle=True,
    )
    if kind == "sam":
        opt = sam_train.SAM(
            model.parameters(), torch.optim.SGD,
            rho=float(args.sam_rho), adaptive=False,
            lr=float(args.lr), momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
        )
    elif kind == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=float(args.lr),
            momentum=float(args.momentum), weight_decay=float(args.weight_decay),
        )
    else:
        raise ValueError(kind)

    model.train()
    for ep in range(1, int(args.epochs) + 1):
        n_steps = 0
        for batch in train_loader:
            if kind == "sam":
                df.sam_update_on_batch(model, opt, batch, device)
            else:
                xb, yb = df.split_batch(batch)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(df.forward_from_inputs(model, xb), yb).backward()
                opt.step()
            n_steps += 1
        eval_batch = next(iter(
            df.build_loader(subset, batch_size=min(64, n), workers=0, shuffle=False)
        ))
        m = df.minibatch_loss_acc(model, eval_batch, device)
        print(f"[light_test] [{kind}] epoch {ep}/{args.epochs}: {n_steps} steps, "
              f"train_loss={m['loss']:.4f} train_acc={m['acc']:.4f}")
    return model


def probe_model(*, kind, model, subset, device, args):
    """d-flatness probe: probe_steps batches @ probe_batch_size.

    Returns (flat_primal_array, mb_loss_array) over the probe steps.
    """
    probe_loader = df.build_loader(
        subset, batch_size=args.probe_batch_size, workers=0, shuffle=True,
    )
    print(f"[light_test] [{kind}] d-flatness probe: {args.probe_steps} steps "
          f"@ batch_size={args.probe_batch_size}, norm={args.flat_norm}")
    it = iter(probe_loader)
    primals, losses, duals = [], [], []
    for step in range(1, int(args.probe_steps) + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(probe_loader)
            batch = next(it)
        res = df.flatness_on_batch(
            model, batch, device,
            norm=args.flat_norm, adaptive=False,
            flat_r=float(args.flat_r), flat_delta=float(args.flat_delta),
            freeze_bn=True, enable_grads=True,
        )
        primals.append(res["flat_primal"])
        duals.append(res["flat_dual"])
        losses.append(res["mb_loss"])
        print(f"[light_test] [{kind}]   step {step}: mb_loss={res['mb_loss']:.4f} "
              f"mb_acc={res['mb_acc']:.3f} grad_norm={res['grad_norm']:.4f} "
              f"flat_primal={res['flat_primal']:.6f} flat_dual={res['flat_dual']:.6f}")
    return np.asarray(primals, float), np.asarray(losses, float), np.asarray(duals, float)


def report(kind, primals, losses):
    """Print FLAT_STD (= STD of the probe score) for this model."""
    scores = df.compute_scores_for_series(primals, losses)
    print(f"[light_test] [{kind}] FLAT_STD (probe-score STD) = {scores['FLAT_STD']:.6g}")
    return scores


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(DFLAT_ROOT / "data"),
                   help="CIFAR-10 location; torchvision downloads here if absent")
    p.add_argument("--train-samples", type=int, default=2048,
                   help="CIFAR-10 subset size for the 1-epoch training")
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--probe-steps", type=int, default=10)
    p.add_argument("--probe-batch-size", type=int, default=8)
    p.add_argument("--sam-rho", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--flat-r", type=float, default=0.05)
    p.add_argument("--flat-delta", type=float, default=0.1)
    p.add_argument("--flat-norm", default="l2", choices=["l1", "l2"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    t0 = time.time()
    sam_train.set_seed(args.seed)
    device = resolve_device()

    # --- data: small CIFAR-10 subset (no augmentation for a stable smoke) ---
    trainset, _ = sam_train.prepare_dataset(
        "cifar10", root=str(args.data_root),
        normalize=True, randomize=False, cutout=False,
    )
    n = min(int(args.train_samples), len(trainset))
    subset = torch.utils.data.Subset(trainset, list(range(n)))
    print(f"[light_test] CIFAR-10 subset: {n} samples")

    results = {}
    all_finite = True
    for kind in ("sam", "sgd"):
        sam_train.set_seed(args.seed)  # identical data order / init per model
        model = train_one(kind=kind, subset=subset, n=n, device=device, args=args)
        primals, losses, duals = probe_model(
            kind=kind, model=model, subset=subset, device=device, args=args)
        results[kind] = report(kind, primals, losses)
        all_finite &= bool(np.all(np.isfinite(primals)) and np.all(np.isfinite(duals)))


    print("[light_test] === SAM vs SGD (FLAT_STD) ===")
    print(f"[light_test]   FLAT_STD  sam={results['sam']['FLAT_STD']:.6g}  "
          f"sgd={results['sgd']['FLAT_STD']:.6g}")
    print(f"[light_test] total time {time.time() - t0:.1f}s")

    if all_finite:
        print("[light_test] PASS")
        return 0
    print("[light_test] FAIL: NaN in flatness metrics")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
