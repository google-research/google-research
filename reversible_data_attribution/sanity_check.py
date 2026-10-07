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

r"""Sanity check pipeline for evaluating model convergence on MNIST and CIFAR.

This module provides a fast, lightweight sanity check pipeline to determine the
number of training epochs needed for a model (DNN, CNN, ViT, ResNet, etc.) to
converge under the Adam optimizer. It supports:
1. Datasets: MNIST and CIFAR (CIFAR-10, CIFAR-100), with train/val/test splits.
2. Data Corruption: Choice of label flipping ('label_flip'), Gaussian noise
   ('gaussian'), 'both', or 'none'.
3. Models: DNN, Small DNN, CNN, ViT (Vision Transformer), ResNet-18, Linear/LogReg.
4. Adam Optimizer: Configurable learning rate, betas, eps, and weight decay.
5. Fast Execution: No counterfactual estimation or second-order Hessian computation.
6. Convergence Monitoring: Tracks train/val/test loss & accuracy, gradient norms,
   plateau detection, target thresholds, and logs detailed epoch history.

Usage Example:
  python -m reversible_data_attribution.sanity_check \
    --model_type=CNN \
    --dataset=mnist \
    --num_epochs=15 \
    --noise_type=label_flip \
    --noise_rate=0.1 \
    --learning_rate=0.001
"""

from collections.abc import Sequence
import csv
import json
import os
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torch import nn
from torch.utils import data as torch_data
import tqdm

from reversible_data_attribution import models
from reversible_data_attribution import pipeline
from reversible_data_attribution import utils

if 'gfile' not in globals():
  gfile = None

FLAGS = flags.FLAGS

if 'model_type' not in flags.FLAGS:
  flags.DEFINE_enum(
      'model_type',
      'dnn',
      [
          'dnn',
          'small_dnn',
          'cnn',
          'vit',
          'vision_transformer',
          'resnet',
          'resnet18',
          'linear',
          'logreg',
      ],
      'Model architecture choice for the sanity check.',
  )
if 'dataset' not in flags.FLAGS:
  flags.DEFINE_enum(
      'dataset',
      'mnist',
      [
          'mnist',
          'cifar',
          'cifar10',
          'cifar100',
          'fashion_mnist',
          'imdb',
          'mnist_corrupted',
          'synthetic',
      ],
      'Dataset to use for evaluation ("mnist", "cifar", "cifar10", "cifar100",'
      ' "fashion_mnist", "imdb", "mnist_corrupted", "synthetic").',
  )
if 'data_path' not in flags.FLAGS:
  flags.DEFINE_string(
      'data_path',
      None,
      'Path to dataset npz file. If None, uses default path for the chosen'
      ' dataset.',
  )
if 'num_train' not in flags.FLAGS:
  flags.DEFINE_integer('num_train', 1000, 'Number of training samples.')
if 'num_val' not in flags.FLAGS:
  flags.DEFINE_integer('num_val', 200, 'Number of validation samples.')
if 'num_test' not in flags.FLAGS:
  flags.DEFINE_integer('num_test', 200, 'Number of test samples.')
if 'num_epochs' not in flags.FLAGS:
  flags.DEFINE_integer('num_epochs', 20, 'Maximum number of training epochs.')
if 'batch_size' not in flags.FLAGS:
  flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')
if 'learning_rate' not in flags.FLAGS:
  flags.DEFINE_float(
      'learning_rate', 0.001, 'Learning rate for Adam optimizer.'
  )
if 'beta_1' not in flags.FLAGS:
  flags.DEFINE_float('beta_1', 0.9, 'Beta 1 parameter for Adam optimizer.')
if 'beta_2' not in flags.FLAGS:
  flags.DEFINE_float('beta_2', 0.999, 'Beta 2 parameter for Adam optimizer.')
if 'eps' not in flags.FLAGS:
  flags.DEFINE_float('eps', 1e-8, 'Epsilon parameter for Adam optimizer.')
if 'weight_decay' not in flags.FLAGS:
  flags.DEFINE_float('weight_decay', 0.0, 'Weight decay (L2 penalty) for Adam.')
if 'noise_type' not in flags.FLAGS:
  flags.DEFINE_enum(
      'noise_type',
      'none',
      [
          'none',
          'label_flip',
          'gaussian',
          'both',
          'lowpass',
          'highpass',
          'low_pass',
          'high_pass',
      ],
      'Type of noise to apply to the training dataset.',
  )
if 'noise_rate' not in flags.FLAGS:
  flags.DEFINE_float(
      'noise_rate',
      0.1,
      'Fraction of training dataset to corrupt with noise (between 0.0 and'
      ' 1.0).',
  )
if 'gaussian_std' not in flags.FLAGS:
  flags.DEFINE_float(
      'gaussian_std',
      0.1,
      'Standard deviation of Gaussian noise added to inputs.',
  )
if 'cutoff_freq' not in flags.FLAGS:
  flags.DEFINE_float(
      'cutoff_freq',
      0.5,
      'Cutoff frequency for lowpass / highpass filtering.',
  )
if 'filter_order' not in flags.FLAGS:
  flags.DEFINE_integer(
      'filter_order',
      2,
      'Filter order for lowpass / highpass filtering.',
  )
if 'sampling_rate' not in flags.FLAGS:
  flags.DEFINE_float(
      'sampling_rate',
      60.0,
      'Sampling rate for lowpass / highpass filtering.',
  )
if 'noise_seed' not in flags.FLAGS:
  flags.DEFINE_integer('noise_seed', 42, 'Random seed for noise generation.')
if 'seed' not in flags.FLAGS:
  flags.DEFINE_integer(
      'seed', 42, 'Random seed for model training and initialization.'
  )
if 'device' not in flags.FLAGS:
  flags.DEFINE_string(
      'device', 'cpu', 'Device for computation ("cpu", "cuda", or "auto").'
  )
if 'target_loss' not in flags.FLAGS:
  flags.DEFINE_float(
      'target_loss',
      None,
      'Optional target training loss threshold to determine convergence early.',
  )
if 'target_acc' not in flags.FLAGS:
  flags.DEFINE_float(
      'target_acc',
      None,
      'Optional target training accuracy threshold to determine convergence'
      ' early.',
  )
if 'loss_tolerance' not in flags.FLAGS:
  flags.DEFINE_float(
      'loss_tolerance',
      1e-3,
      'Relative loss decrease tolerance over patience epochs to detect plateau'
      ' convergence.',
  )
if 'patience' not in flags.FLAGS:
  flags.DEFINE_integer(
      'patience',
      3,
      'Number of consecutive epochs with relative loss decrease <'
      ' loss_tolerance to declare plateau.',
  )
if 'early_stopping' not in flags.FLAGS:
  flags.DEFINE_boolean(
      'early_stopping',
      False,
      'Whether to stop training as soon as convergence criteria are met.',
  )
if 'output_dir' not in flags.FLAGS:
  flags.DEFINE_string(
      'output_dir',
      None,
      'Directory to save convergence CSV results and summaries. If None, does'
      ' not save to disk.',
  )


def resolve_device(device_str):
  """Resolves device string to a torch.device instance."""
  if device_str.lower() == 'auto':
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  return torch.device(device_str)


def load_sanity_dataset(
    dataset_name,
    data_path = None,
    num_train = 1000,
    num_val = 200,
    num_test = 200,
    batch_size = 64,
    noise_type = 'none',
    noise_rate = 0.0,
    gaussian_std = 0.1,
    cutoff_freq = 0.5,
    filter_order = 2,
    sampling_rate = 60.0,
    seed = 42,
    shuffle_train = True,
):
  """Loads dataset splits, applies data corruption to training split, and returns DataLoaders.

  Args:
    dataset_name: Name of dataset ('mnist', 'cifar', 'cifar10', 'cifar100',
      'mnist_corrupted', 'synthetic').
    data_path: Optional explicit path to npz dataset file.
    num_train: Number of training samples.
    num_val: Number of validation samples.
    num_test: Number of test samples.
    batch_size: Batch size for DataLoader.
    noise_type: Type of corruption ('none', 'label_flip', 'gaussian', 'both').
    noise_rate: Fraction of training samples to corrupt.
    gaussian_std: Standard deviation of Gaussian noise added to inputs.
    seed: Random seed for noise generation.
    shuffle_train: Whether to shuffle training batches.

  Returns:
    Tuple of:
      train_loader: PyTorch DataLoader for training data (corrupted if
      requested).
      val_loader: PyTorch DataLoader for clean validation data.
      test_loader: PyTorch DataLoader for clean test data.
      input_dim: Feature dimension per sample (e.g. 784 for MNIST, 3072 for
      CIFAR).
      num_classes: Number of target classes.
      corrupted_indices: 1D numpy array of sample indices in the training set
      that were corrupted.
  """
  dataset_name = dataset_name.lower()

  if dataset_name == 'synthetic':
    num_classes = 10
    input_dim = 64
    rng = np.random.default_rng(seed)
    total_samples = num_train + num_val + num_test
    x_all = rng.normal(0.0, 1.0, size=(total_samples, input_dim)).astype(
        np.float32
    )
    true_w = rng.normal(0.0, 1.0, size=(input_dim, num_classes)).astype(
        np.float32
    )
    logits = x_all @ true_w
    y_all = np.argmax(logits, axis=1).astype(np.int64)

    x_tr = x_all[:num_train]
    y_tr = y_all[:num_train]
    x_val = x_all[num_train : num_train + num_val]
    y_val = y_all[num_train : num_train + num_val]
    x_te = x_all[num_train + num_val : total_samples]
    y_te = y_all[num_train + num_val : total_samples]

  elif dataset_name in ('mnist', 'mnist_corrupted'):
    default_path = (
        'third_party/google_research/google_research/reversible_data_attribution/mnist/mnist_local.npz'
        if dataset_name == 'mnist'
        else 'third_party/google_research/google_research/reversible_data_attribution/mnist_corrupted_local.npz'
    )
    data_file = data_path or default_path
    num_classes = 10
    x_all_train, y_all_train, x_te, y_te = pipeline.load_mnist_data(
        data_file,
        num_train=num_train + num_val,
        num_test=num_test,
    )
    x_tr = x_all_train[:num_train]
    y_tr = y_all_train[:num_train]
    x_val = x_all_train[num_train : num_train + num_val]
    y_val = y_all_train[num_train : num_train + num_val]
    input_dim = int(x_tr.shape[1])

  elif dataset_name in ('cifar', 'cifar10', 'cifar100'):
    default_path = 'third_party/google_research/google_research/reversible_data_attribution/cifar/cifar_local.npz'
    data_file = data_path or default_path
    num_classes = 100 if dataset_name == 'cifar100' else 10
    x_all_train, y_all_train, x_te, y_te = pipeline.load_cifar_data(
        data_file,
        num_train=num_train + num_val,
        num_test=num_test,
        flatten=True,
        num_classes=num_classes,
    )
    x_tr = x_all_train[:num_train]
    y_tr = y_all_train[:num_train]
    x_val = x_all_train[num_train : num_train + num_val]
    y_val = y_all_train[num_train : num_train + num_val]
    input_dim = int(x_tr.shape[1])

  elif dataset_name == 'fashion_mnist':
    default_path = 'third_party/google_research/google_research/reversible_data_attribution/fashion_mnist/fashion_mnist.npz'
    data_file = data_path or default_path
    num_classes = 10
    x_all_train, y_all_train, x_te, y_te = pipeline.load_fashion_mnist_data(
        data_file,
        num_train=num_train + num_val,
        num_test=num_test,
    )
    x_tr = x_all_train[:num_train]
    y_tr = y_all_train[:num_train]
    x_val = x_all_train[num_train : num_train + num_val]
    y_val = y_all_train[num_train : num_train + num_val]
    input_dim = int(x_tr.shape[1])

  elif dataset_name == 'imdb':
    default_path = 'third_party/google_research/google_research/reversible_data_attribution/imdb/imdb_local.npz'
    data_file = data_path or default_path
    num_classes = 2
    x_all_train, y_all_train, x_te, y_te = pipeline.load_imdb_data(
        data_file,
        num_train=num_train + num_val,
        num_test=num_test,
    )
    x_tr = x_all_train[:num_train]
    y_tr = y_all_train[:num_train]
    x_val = x_all_train[num_train : num_train + num_val]
    y_val = y_all_train[num_train : num_train + num_val]
    input_dim = int(x_tr.shape[1])

  else:
    raise ValueError(
        f'Unsupported dataset_name: {dataset_name}. Expected one of ["mnist",'
        ' "cifar", "cifar10", "cifar100", "fashion_mnist", "imdb",'
        ' "mnist_corrupted", "synthetic"].'
    )

  # Apply data corruption to training set only
  corrupted_indices = np.array([], dtype=int)
  if noise_type.lower() != 'none' and noise_rate > 0.0:
    logging.info(
        'Applying %s corruption to %d training samples (rate=%.2f, seed=%d)...',
        noise_type,
        len(x_tr),
        noise_rate,
        seed,
    )
    x_tr, y_tr, corrupted_indices = utils.apply_dataset_noise(
        x_train=x_tr,
        y_train=y_tr,
        noise_type=noise_type,
        noise_rate=noise_rate,
        num_classes=num_classes,
        gaussian_std=gaussian_std,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        sampling_rate=sampling_rate,
        seed=seed,
    )

  # Construct PyTorch datasets & dataloaders
  train_dataset = torch_data.TensorDataset(
      torch.from_numpy(x_tr).float(), torch.from_numpy(y_tr).long()
  )
  val_dataset = torch_data.TensorDataset(
      torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()
  )
  test_dataset = torch_data.TensorDataset(
      torch.from_numpy(x_te).float(), torch.from_numpy(y_te).long()
  )

  train_loader = torch_data.DataLoader(
      train_dataset, batch_size=batch_size, shuffle=shuffle_train
  )
  val_loader = torch_data.DataLoader(
      val_dataset, batch_size=batch_size, shuffle=False
  )
  test_loader = torch_data.DataLoader(
      test_dataset, batch_size=batch_size, shuffle=False
  )

  return (
      train_loader,
      val_loader,
      test_loader,
      input_dim,
      num_classes,
      corrupted_indices,
  )


def evaluate_model(
    model,
    data_loader,
    criterion,
    device,
):
  """Evaluates model loss and accuracy over a DataLoader.

  Args:
    model: PyTorch model.
    data_loader: DataLoader to evaluate over.
    criterion: Loss function.
    device: Torch computation device.

  Returns:
    Tuple of (average_loss, accuracy_percentage).
  """
  model.eval()
  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  with torch.no_grad():
    for x_batch, y_batch in data_loader:
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)

      total_loss += float(loss.item()) * len(x_batch)
      preds = outputs.argmax(dim=-1)
      total_correct += int((preds == y_batch).sum().item())
      total_samples += len(x_batch)

  avg_loss = total_loss / max(1, total_samples)
  accuracy = (total_correct / max(1, total_samples)) * 100.0
  return avg_loss, accuracy


def evaluate_clean_subset_accuracy(
    model,
    train_dataset,
    corrupted_indices,
    device,
):
  """Evaluates model accuracy on the clean (uncorrupted) subset of training data."""
  if len(corrupted_indices) == 0:
    return 0.0

  if not hasattr(train_dataset, 'tensors'):
    return 0.0

  all_indices = np.arange(len(train_dataset))
  clean_mask = np.ones(len(train_dataset), dtype=bool)
  clean_mask[corrupted_indices] = False
  clean_indices = all_indices[clean_mask]

  if len(clean_indices) == 0:
    return 0.0

  x_clean = train_dataset.tensors[0][clean_indices].to(device)
  y_clean = train_dataset.tensors[1][clean_indices].to(device)

  model.eval()
  with torch.no_grad():
    outputs = model(x_clean)
    preds = outputs.argmax(dim=-1)
    acc = float((preds == y_clean).float().mean().item()) * 100.0
  return acc


def check_convergence_criteria(
    epoch,
    loss_history,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    target_loss = None,
    target_acc = None,
    loss_tolerance = 1e-3,
    patience = 3,
):
  """Checks whether the model has converged based on thresholds and plateau detection.

  Args:
    epoch: Current epoch index (1-indexed).
    loss_history: List of train losses up to current epoch.
    train_loss: Current training loss.
    train_acc: Current training accuracy.
    val_loss: Current validation loss.
    val_acc: Current validation accuracy.
    target_loss: Target training loss to declare convergence.
    target_acc: Target training accuracy to declare convergence.
    loss_tolerance: Relative loss reduction tolerance.
    patience: Window of epochs to monitor for plateau.

  Returns:
    Tuple of (is_converged, reason_string_or_None).
  """
  _ = epoch, val_loss, val_acc
  # 1. Check explicit target loss
  if target_loss is not None and train_loss <= target_loss:
    return (
        True,
        (
            f'Target train loss reached (loss={train_loss:.4f} <='
            f' target={target_loss:.4f})'
        ),
    )

  # 2. Check explicit target accuracy
  if target_acc is not None and train_acc >= target_acc:
    return (
        True,
        (
            f'Target train accuracy reached (acc={train_acc:.2f}% >='
            f' target={target_acc:.2f}%)'
        ),
    )

  # 3. Check loss plateau over patience epochs
  if len(loss_history) > patience:
    window_loss_past = loss_history[-patience - 1]
    rel_change = abs(window_loss_past - train_loss) / max(
        1e-6, abs(window_loss_past)
    )
    if rel_change < loss_tolerance:
      return True, (
          f'Loss plateau detected over {patience} epochs'
          f' (rel_change={rel_change:.6f} < tol={loss_tolerance:.6f})'
      )

  return False, None


def run_convergence_sanity_check(
    model_type = 'dnn',
    dataset_name = 'mnist',
    data_path = None,
    num_train = 1000,
    num_val = 200,
    num_test = 200,
    num_epochs = 20,
    batch_size = 64,
    learning_rate = 0.001,
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    weight_decay = 0.0,
    noise_type = 'none',
    noise_rate = 0.1,
    gaussian_std = 0.1,
    cutoff_freq = 0.5,
    filter_order = 2,
    sampling_rate = 60.0,
    noise_seed = 42,
    seed = 42,
    device = 'cpu',
    target_loss = None,
    target_acc = None,
    loss_tolerance = 1e-3,
    patience = 3,
    early_stopping = False,
    output_dir = None,
    save_csv = True,
):
  """Executes the full convergence sanity check pipeline.

  Args:
    model_type: Neural network architecture name ('dnn', 'cnn', 'vit', 'resnet',
      'linear', etc.).
    dataset_name: Dataset name ('mnist', 'cifar', 'cifar10', 'cifar100',
      'mnist_corrupted', 'synthetic').
    data_path: Optional path to npz dataset file.
    num_train: Number of training samples.
    num_val: Number of validation samples.
    num_test: Number of test samples.
    num_epochs: Maximum epochs to train for.
    batch_size: Batch size for DataLoader.
    learning_rate: Learning rate for Adam optimizer.
    beta_1: Adam beta1.
    beta_2: Adam beta2.
    eps: Adam epsilon.
    weight_decay: Adam weight decay.
    noise_type: Noise corruption type ('none', 'label_flip', 'gaussian', 'both',
      'lowpass', 'highpass').
    noise_rate: Fraction of training samples to corrupt.
    gaussian_std: Standard deviation for Gaussian noise.
    cutoff_freq: Cutoff frequency for lowpass / highpass filtering.
    filter_order: Filter order for lowpass / highpass filtering.
    sampling_rate: Sampling rate for lowpass / highpass filtering.
    noise_seed: Seed for noise generation.
    seed: Seed for training reproducibility.
    device: Computation device ('cpu', 'cuda', or torch.device).
    target_loss: Optional target train loss for convergence.
    target_acc: Optional target train accuracy for convergence.
    loss_tolerance: Relative loss decrease tolerance for plateau.
    patience: Number of consecutive epochs for plateau detection.
    early_stopping: Stop training immediately once converged.
    output_dir: Optional directory to save output CSV files.
    save_csv: Whether to write CSV artifacts if output_dir is given.

  Returns:
    Dictionary containing:
      - 'convergence_summary': High-level convergence diagnostics and epoch
      needed.
      - 'history': List of per-epoch metrics.
      - 'config': Run configuration and hyperparameters.
      - 'model_summary': Architecture details and parameter count.
  """
  device_obj = resolve_device(device) if isinstance(device, str) else device
  torch.manual_seed(seed)
  np.random.seed(seed)

  logging.info(
      '=== Running Convergence Sanity Check: Model=%s, Dataset=%s,'
      ' Device=%s ===',
      model_type,
      dataset_name,
      device_obj,
  )

  # 1. Load data
  (
      train_loader,
      val_loader,
      test_loader,
      input_dim,
      num_classes,
      corrupted_indices,
  ) = load_sanity_dataset(
      dataset_name=dataset_name,
      data_path=data_path,
      num_train=num_train,
      num_val=num_val,
      num_test=num_test,
      batch_size=batch_size,
      noise_type=noise_type,
      noise_rate=noise_rate,
      gaussian_std=gaussian_std,
      cutoff_freq=cutoff_freq,
      filter_order=filter_order,
      sampling_rate=sampling_rate,
      seed=noise_seed,
      shuffle_train=True,
  )

  # 2. Build model and Adam optimizer
  model = models.get_model(
      model_type=model_type, input_dim=input_dim, output_dim=num_classes
  ).to(device_obj)
  param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

  optimizer = torch.optim.Adam(
      model.parameters(),
      lr=learning_rate,
      betas=(beta_1, beta_2),
      eps=eps,
      weight_decay=weight_decay,
  )
  criterion = nn.CrossEntropyLoss()

  logging.info(
      'Model initialized (%s, trainable parameters: %d, input_dim: %d,'
      ' num_classes: %d)',
      model_type,
      param_count,
      input_dim,
      num_classes,
  )

  # 3. Training & Convergence Loop
  history = []
  loss_history = []
  converged = False
  convergence_epoch: int | None = None
  convergence_reason: str | None = None

  best_val_loss = float('inf')
  best_val_loss_epoch = 0
  best_val_acc = 0.0
  best_val_acc_epoch = 0

  best_test_loss = float('inf')
  best_test_loss_epoch = 0
  best_test_acc = 0.0
  best_test_acc_epoch = 0

  total_training_start = time.time()
  epoch_pbar = tqdm.tqdm(
      range(1, num_epochs + 1),
      desc=f'Training {model_type.upper()} on {dataset_name}',
      unit='epoch',
  )

  for epoch in epoch_pbar:
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_samples = 0
    total_grad_norm = 0.0
    num_batches = 0

    batch_pbar = tqdm.tqdm(
        train_loader,
        desc=f'Epoch {epoch:02d}/{num_epochs:02d}',
        leave=False,
        unit='batch',
    )
    for x_batch, y_batch in batch_pbar:
      x_batch = x_batch.to(device_obj)
      y_batch = y_batch.to(device_obj)

      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()

      # Track gradient norm
      grad_norm = torch.sqrt(
          sum(
              p.grad.norm() ** 2
              for p in model.parameters()
              if p.grad is not None
          )
      ).item()
      total_grad_norm += grad_norm
      num_batches += 1

      optimizer.step()

      running_loss += float(loss.item()) * len(x_batch)
      preds = outputs.argmax(dim=-1)
      running_correct += int((preds == y_batch).sum().item())
      running_samples += len(x_batch)

      batch_pbar.set_postfix({
          'loss': f'{running_loss / max(1, running_samples):.4f}',
          'acc': f'{(running_correct / max(1, running_samples)) * 100.0:.1f}%',
      })

    epoch_time = time.time() - epoch_start_time
    train_loss = running_loss / max(1, running_samples)
    train_acc = (running_correct / max(1, running_samples)) * 100.0
    avg_grad_norm = total_grad_norm / max(1, num_batches)
    loss_history.append(train_loss)

    # Validation and Test Evaluation
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device_obj)
    test_loss, test_acc = evaluate_model(
        model, test_loader, criterion, device_obj
    )

    # Evaluate accuracy on clean subset if noise is active
    clean_train_acc = (
        evaluate_clean_subset_accuracy(
            model, train_loader.dataset, corrupted_indices, device_obj
        )
        if len(corrupted_indices) > 0
        else train_acc
    )

    # Track best metrics
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_val_loss_epoch = epoch
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      best_val_acc_epoch = epoch

    if test_loss < best_test_loss:
      best_test_loss = test_loss
      best_test_loss_epoch = epoch
    if test_acc > best_test_acc:
      best_test_acc = test_acc
      best_test_acc_epoch = epoch

    # Convergence check
    is_epoch_converged, reason = check_convergence_criteria(
        epoch=epoch,
        loss_history=loss_history,
        train_loss=train_loss,
        train_acc=train_acc,
        val_loss=val_loss,
        val_acc=val_acc,
        target_loss=target_loss,
        target_acc=target_acc,
        loss_tolerance=loss_tolerance,
        patience=patience,
    )

    if is_epoch_converged and not converged:
      converged = True
      convergence_epoch = epoch
      convergence_reason = reason
      logging.info('--> Convergence DETECTED at Epoch %d: %s', epoch, reason)

    epoch_metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'clean_train_acc': clean_train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'grad_norm': avg_grad_norm,
        'epoch_time_sec': epoch_time,
        'converged_so_far': converged,
    }
    history.append(epoch_metrics)

    epoch_pbar.set_postfix({
        'tr_loss': f'{train_loss:.4f}',
        'tr_acc': f'{train_acc:.1f}%',
        'val_acc': f'{val_acc:.1f}%',
        'test_acc': f'{test_acc:.1f}%',
    })

    noise_msg = (
        f' (clean_train_acc: {clean_train_acc:.2f}%)'
        if len(corrupted_indices) > 0
        else ''
    )
    logging.info(
        'Epoch %02d/%02d | Train Loss: %.4f, Train Acc: %.2f%%%s | Val Loss:'
        ' %.4f, Val Acc: %.2f%% | Test Acc: %.2f%% | Time: %.2fs',
        epoch,
        num_epochs,
        train_loss,
        train_acc,
        noise_msg,
        val_loss,
        val_acc,
        test_acc,
        epoch_time,
    )

    if early_stopping and converged:
      logging.info(
          'Early stopping triggered at Epoch %d due to convergence.', epoch
      )
      epoch_pbar.close()
      break

  total_duration = time.time() - total_training_start

  # Final convergence status compilation
  if not converged:
    convergence_epoch = best_val_loss_epoch
    convergence_reason = (
        f'Full training completed ({len(history)} epochs); best validation loss'
        f' achieved at epoch {best_val_loss_epoch}'
    )

  final_epoch_metrics = history[-1]
  convergence_summary = {
      'converged': converged,
      'convergence_epoch': convergence_epoch,
      'convergence_reason': convergence_reason,
      'total_epochs_trained': len(history),
      'total_training_duration_sec': total_duration,
      'best_val_loss': best_val_loss,
      'best_val_loss_epoch': best_val_loss_epoch,
      'best_val_acc': best_val_acc,
      'best_val_acc_epoch': best_val_acc_epoch,
      'best_test_loss': best_test_loss,
      'best_test_loss_epoch': best_test_loss_epoch,
      'best_test_acc': best_test_acc,
      'best_test_acc_epoch': best_test_acc_epoch,
      'final_train_loss': final_epoch_metrics['train_loss'],
      'final_train_acc': final_epoch_metrics['train_acc'],
      'final_clean_train_acc': final_epoch_metrics['clean_train_acc'],
      'final_val_loss': final_epoch_metrics['val_loss'],
      'final_val_acc': final_epoch_metrics['val_acc'],
      'final_test_loss': final_epoch_metrics['test_loss'],
      'final_test_acc': final_epoch_metrics['test_acc'],
      'generalization_gap_acc': (
          final_epoch_metrics['train_acc'] - final_epoch_metrics['test_acc']
      ),
  }

  config_summary = {
      'model_type': model_type,
      'dataset': dataset_name,
      'data_path': data_path,
      'num_train': num_train,
      'num_val': num_val,
      'num_test': num_test,
      'num_epochs': num_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate,
      'beta_1': beta_1,
      'beta_2': beta_2,
      'eps': eps,
      'weight_decay': weight_decay,
      'noise_type': noise_type,
      'noise_rate': noise_rate,
      'gaussian_std': gaussian_std,
      'noise_seed': noise_seed,
      'num_corrupted_samples': len(corrupted_indices),
      'seed': seed,
      'device': str(device_obj),
      'target_loss': target_loss,
      'target_acc': target_acc,
      'loss_tolerance': loss_tolerance,
      'patience': patience,
      'early_stopping': early_stopping,
  }

  model_summary = {
      'model_type': model_type,
      'trainable_parameters': param_count,
      'input_dim': input_dim,
      'num_classes': num_classes,
  }

  results = {
      'convergence_summary': convergence_summary,
      'history': history,
      'config': config_summary,
      'model_summary': model_summary,
  }

  if output_dir and save_csv:
    save_sanity_check_results(results, output_dir)

  return results


def _make_dirs(dir_path):
  """Creates directory path supporting local, CNS, and GCS paths."""
  if not dir_path:
    return
  try:
    if hasattr(gfile, 'MakeDirs'):
      gfile.MakeDirs(dir_path)
      return
    elif hasattr(gfile, 'makedirs'):
      gfile.makedirs(dir_path)
      return
  except Exception:
    pass
  os.makedirs(dir_path, exist_ok=True)


def _open_file(file_path, mode = 'r'):
  """Opens file supporting local, CNS, and GCS paths."""
  try:
    if hasattr(gfile, 'GFile'):
      return gfile.GFile(file_path, mode)
    elif hasattr(gfile, 'Open'):
      return gfile.Open(file_path, mode)
  except Exception:
    pass
  return open(file_path, mode)


def save_sanity_check_results(
    results, output_dir
):
  """Saves convergence history and summary metrics to CSV and JSON in output_dir."""
  _make_dirs(output_dir)

  # 1. Save epoch progression history CSV
  history_csv_path = os.path.join(output_dir, 'convergence_history.csv')
  history = results['history']
  if history:
    fieldnames = list(history[0].keys())
    with _open_file(history_csv_path, 'w') as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(history)
    logging.info('Saved convergence history CSV to %s', history_csv_path)

  # 2. Save convergence summary CSV
  summary_csv_path = os.path.join(output_dir, 'convergence_summary.csv')
  summary_data = {
      **results['config'],
      **results['model_summary'],
      **results['convergence_summary'],
  }
  with _open_file(summary_csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric / Parameter', 'Value'])
    for k, v in summary_data.items():
      writer.writerow([k, v])
  logging.info('Saved convergence summary CSV to %s', summary_csv_path)

  # 3. Save JSON summary for programmatic use
  summary_json_path = os.path.join(output_dir, 'sanity_check_summary.json')
  with _open_file(summary_json_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
  logging.info('Saved sanity check JSON summary to %s', summary_json_path)

  return history_csv_path, summary_csv_path


def print_sanity_check_summary(results):
  """Prints an informative, human-readable summary table to standard output."""
  summary = results['convergence_summary']
  config = results['config']
  model_info = results['model_summary']
  history = results['history']

  print('\n' + '=' * 80)
  print('          MODEL CONVERGENCE SANITY CHECK SUMMARY          ')
  print('=' * 80)
  print(
      f'Model Architecture:    {model_info["model_type"].upper()}'
      f' ({model_info["trainable_parameters"]:,} trainable params)'
  )
  print(
      f'Dataset:               {config["dataset"].upper()} (Train:'
      f' {config["num_train"]}, Val: {config["num_val"]}, Test:'
      f' {config["num_test"]})'
  )
  print(
      f'Optimizer:             Adam (lr={config["learning_rate"]},'
      f' beta1={config["beta_1"]}, beta2={config["beta_2"]})'
  )
  print(
      f'Data Corruption:       {config["noise_type"]}'
      f' (Rate={config["noise_rate"]:.2f},'
      f' Corrupted={config["num_corrupted_samples"]}/{config["num_train"]})'
  )
  print(f'Device:                {config["device"]}')
  print('-' * 80)
  print(
      'Converged:            '
      f' {"YES" if summary["converged"] else "NO (Reached max epochs)"}'
  )
  print(
      f'Convergence Epoch:     Epoch {summary["convergence_epoch"]} /'
      f' {summary["total_epochs_trained"]}'
  )
  print(f'Convergence Reason:    {summary["convergence_reason"]}')
  print(
      f'Total Duration:        {summary["total_training_duration_sec"]:.2f}s'
      f' ({summary["total_training_duration_sec"]/max(1, summary["total_epochs_trained"]):.2f}s/epoch)'
  )
  print('-' * 80)
  print('Key Metrics:')
  print(
      f'  Final Train Loss:    {summary["final_train_loss"]:.4f} (Accuracy:'
      f' {summary["final_train_acc"]:.2f}%)'
  )
  if config['num_corrupted_samples'] > 0:
    print(
        f'  Clean Train Acc:     {summary["final_clean_train_acc"]:.2f}% (on'
        f' {config["num_train"] - config["num_corrupted_samples"]} clean train'
        ' samples)'
    )
  print(
      f'  Best Val Loss:       {summary["best_val_loss"]:.4f} (at Epoch'
      f' {summary["best_val_loss_epoch"]}, Acc: {summary["best_val_acc"]:.2f}%)'
  )
  print(
      f'  Best Test Acc:       {summary["best_test_acc"]:.2f}% (at Epoch'
      f' {summary["best_test_acc_epoch"]}, Loss:'
      f' {summary["best_test_loss"]:.4f})'
  )
  print(
      f'  Generalization Gap:  {summary["generalization_gap_acc"]:+.2f}% (Train'
      ' Acc - Test Acc)'
  )
  print('-' * 80)
  print('Epoch History Snippet:')
  print(
      'Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Test Acc | Grad'
      ' Norm'
  )
  for h in history:
    print(
        f'{h["epoch"]:5d} | {h["train_loss"]:10.4f} | {h["train_acc"]:8.2f}% |'
        f' {h["val_loss"]:8.4f} | {h["val_acc"]:6.2f}% | {h["test_acc"]:7.2f}%'
        f' | {h["grad_norm"]:9.4f}'
    )
  print('=' * 80 + '\n')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  results = run_convergence_sanity_check(
      model_type=FLAGS.model_type,
      dataset_name=FLAGS.dataset,
      data_path=FLAGS.data_path,
      num_train=FLAGS.num_train,
      num_val=FLAGS.num_val,
      num_test=FLAGS.num_test,
      num_epochs=FLAGS.num_epochs,
      batch_size=FLAGS.batch_size,
      learning_rate=FLAGS.learning_rate,
      beta_1=FLAGS.beta_1,
      beta_2=FLAGS.beta_2,
      eps=FLAGS.eps,
      weight_decay=FLAGS.weight_decay,
      noise_type=FLAGS.noise_type,
      noise_rate=FLAGS.noise_rate,
      gaussian_std=FLAGS.gaussian_std,
      cutoff_freq=FLAGS.cutoff_freq,
      filter_order=FLAGS.filter_order,
      sampling_rate=FLAGS.sampling_rate,
      noise_seed=FLAGS.noise_seed,
      seed=FLAGS.seed,
      device=FLAGS.device,
      target_loss=FLAGS.target_loss,
      target_acc=FLAGS.target_acc,
      loss_tolerance=FLAGS.loss_tolerance,
      patience=FLAGS.patience,
      early_stopping=FLAGS.early_stopping,
      output_dir=FLAGS.output_dir,
      save_csv=True,
  )

  print_sanity_check_summary(results)


if __name__ == '__main__':
  app.run(main)
