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

"""Diagnostic script to test forward vs backward Adam reconstruction.

Measures floating point error accumulation, moment divergence, negative
variance, and NaN emergence in backward Adam unrolling using AttributionBaseModel
and GPUReversibleTransform.
"""

from collections.abc import Sequence
import copy
import math

from absl import app
from absl import flags
import numpy as np
import torch
from torch import nn

from reversible_data_attribution import base_model
from reversible_data_attribution import models
from reversible_data_attribution import reversible
from reversible_data_attribution import utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_type',
    'small_dnn',
    'Model architecture: "small_dnn", "dnn", "cnn", "resnet18", "linear".',
)
flags.DEFINE_integer('num_steps', 500, 'Total forward/backward steps.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_float('beta_1', 0.9, 'Adam beta1.')
flags.DEFINE_float('beta_2', 0.999, 'Adam beta2.')
flags.DEFINE_float('eps', 1e-8, 'Adam epsilon.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('dim', 64, 'Input dimension (defaults to 784 for CNN).')
flags.DEFINE_integer('num_classes', 10, 'Number of classes.')
flags.DEFINE_string('device', 'cpu', 'Device to use (cpu or cuda).')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_bool(
    'use_reversible_transform',
    False,
    'Whether to use the reversible transform.',
)
flags.DEFINE_integer(
    'reversible_scale',
    1000000000000,
    'Scale for the reversible transform (default 10^12).',
)
flags.DEFINE_integer(
    'checkpoint_freq',
    1,
    'Checkpointing frequency (1 for every step, or e.g. 50).',
)
flags.DEFINE_bool(
    'micro_diagnostic',
    False,
    'Whether to run detailed 5-step sub-operation micro diagnostic.',
)
flags.DEFINE_string(
    'norm_type',
    'batch_norm',
    'Normalization type for ResNet ("batch_norm", "batch_norm_no_stats", "group_norm", "layer_norm", "none").',
)


def run_experiment(
    model_type = 'small_dnn',
    num_steps = 500,
    lr = 0.001,
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    batch_size = 128,
    dim = 64,
    num_classes = 10,
    device = 'cpu',
    seed = 42,
    use_reversible_transform = False,
    reversible_scale = 1000000000000,
    checkpoint_freq = 1,
    norm_type = 'batch_norm',
):
  """Runs forward training and backward inversion diagnostic."""
  device_obj = torch.device(
      device if torch.cuda.is_available() and device == 'cuda' else device
  )
  torch.manual_seed(seed)
  np.random.seed(seed)

  if model_type.lower() == 'cnn' and dim == 64:
    dim = 784

  print('=' * 80)
  print(
      f'Starting Adam Forward/Backward Diagnostic ({num_steps} steps)'
  )
  print(f'Architecture: {model_type} (dim={dim}, num_classes={num_classes})')
  print(
      f'Hyperparameters: lr={lr}, beta_1={beta_1}, beta_2={beta_2}, eps={eps},'
      f' batch_size={batch_size}'
  )
  print(
      f'Reversible transform: {use_reversible_transform}'
      f' (scale={reversible_scale})'
  )
  print(f'Device: {device_obj}')
  print('=' * 80)

  # 1. Create synthetic dataset and DataLoader
  num_samples = batch_size * 20
  x = torch.randn(num_samples, dim, dtype=torch.float32)
  y = torch.randint(0, num_classes, (num_samples,))
  dataset = torch.utils.data.TensorDataset(x, y)
  train_loader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=False
  )

  num_batches = len(train_loader)
  num_epochs = max(1, math.ceil(num_steps / num_batches))
  total_steps = num_epochs * num_batches

  # 2. Instantiate Model and AttributionBaseModel
  model = models.get_model(
      model_type, input_dim=dim, output_dim=num_classes, norm_type=norm_type
  )
  criterion = nn.CrossEntropyLoss()

  base = base_model.AttributionBaseModel(
      model=model,
      loss_fn=criterion,
      lr=lr,
      beta_1=beta_1,
      beta_2=beta_2,
      eps=eps,
      seed=seed,
  )

  # 3. Standard forward training pass
  print(
      f'\n>>> Running standard FORWARD training pass for {total_steps} steps...'
  )
  base.train_model(
      train_loader=train_loader,
      num_epochs=num_epochs,
      device=device_obj,
      save_all_ckpts=True,
      checkpoint_freq=1,
      save_gradients=True,
  )
  unquant_checkpoints = base.checkpoints()

  # 4. Quantized forward training pass (if requested)
  quant_checkpoints = None
  momentum_buffer = None
  variance_buffer = None
  quant_model = None

  if use_reversible_transform:
    print(
        '\n>>> Running quantized FORWARD training pass via'
        ' retrain_with_quantize...'
    )
    quant_model, quant_checkpoints, momentum_buffer, variance_buffer = (
        base.retrain_with_quantize(
            train_loader=train_loader,
            num_epochs=num_epochs,
            device=device_obj,
            save_all_ckpts=True,
            checkpoint_freq=checkpoint_freq,
            quantization_scale=reversible_scale,
        )
    )
    print(
        'The data type of quantized model is: ',
        next(quant_model.parameters()).dtype,
    )

  # 5. Initialize Backward Reconstruction Pass
  print(
      f'\n>>> Running BACKWARD reconstruction pass from step {total_steps}'
      ' down to 0...'
  )

  if (
      use_reversible_transform
      and quant_model is not None
      and quant_checkpoints is not None
  ):
    curr_model = copy.deepcopy(quant_model).to(device_obj)
    curr_momentum = [
        m.clone().to(device_obj)
        for m in quant_checkpoints[total_steps].momentum
    ]
    curr_variance = [
        v.clone().to(device_obj)
        for v in quant_checkpoints[total_steps].variance
    ]
  else:
    curr_model = copy.deepcopy(base.model()).to(device_obj)
    curr_momentum = [
        m.clone().to(device_obj)
        for m in unquant_checkpoints[total_steps].momentum
    ]
    curr_variance = [
        v.clone().to(device_obj)
        for v in unquant_checkpoints[total_steps].variance
    ]

  batches = utils.prepare_batches(train_loader)

  first_divergence_step = None
  first_negative_var_step = None
  first_nan_step = None
  first_inf_step = None

  log_milestones = [
      0, 1, 2, 5, 10, 20, 50, 75, 100, 105, 110, 115, 118, 119, 120, 150,
      200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000,
  ]

  if use_reversible_transform:
    print('\n' + '-' * 155)
    print(
        f'{"Back Steps":<11} | {"Step t":<7} | {"Theta RelErr":<14} |'
        f' {"Mom RelErr":<14} | {"Var RelErr":<14} | {"Q-Theta RelErr":<15} |'
        f' {"Q-Mom RelErr":<14} | {"Q-Var RelErr":<14} | {"Status"}'
    )
    print('-' * 155)
  else:
    print('\n' + '-' * 115)
    print(
        f'{"Back Steps":<11} | {"Step t":<7} | {"Theta MaxErr":<14} |'
        f' {"Theta RelErr":<14} | {"Mom RelErr":<14} | {"Var RelErr":<14} |'
        f' {"Min Var":<14} | {"Status"}'
    )
    print('-' * 115)

  quant_theta_rel_err = 0.0
  quant_m_rel_err = 0.0
  quant_v_rel_err = 0.0
  max_quant_theta_rel_err = 0.0
  max_quant_m_rel_err = 0.0
  max_quant_v_rel_err = 0.0

  for i in range(total_steps):
    t_current = total_steps - i
    t_prev = t_current - 1
    step = t_current

    x_batch, y_batch, _ = batches[(t_current - 1) % len(batches)]
    x_batch = x_batch.to(device_obj)
    y_batch = y_batch.to(device_obj)

    try:
      # Compute one backward step entirely on device
      model_prev, momentum_prev, variance_prev = (
          utils.compute_metadata_adam_backward(
              x_batch=x_batch,
              y_batch=y_batch,
              model=curr_model,
              loss_fn=criterion,
              momentum=curr_momentum,
              variance=curr_variance,
              lr=lr,
              beta_1=beta_1,
              beta_2=beta_2,
              eps=eps,
              step=step,
              device=device_obj,
              momentum_buffer=(
                  momentum_buffer if use_reversible_transform else None
              ),
              variance_buffer=(
                  variance_buffer if use_reversible_transform else None
              ),
              scale=reversible_scale if use_reversible_transform else 1,
          )
      )
    except ValueError as err:
      print(f'\n[!] Error at backward step {i+1} (t={t_prev}): {err}')
      first_nan_step = (i + 1, t_prev)
      break

    # Ground truth comparisons on device
    true_theta = [
        p.to(device_obj)
        for p in unquant_checkpoints[t_prev].model.parameters()
    ]
    true_m = [m.to(device_obj) for m in unquant_checkpoints[t_prev].momentum]
    true_v = [v.to(device_obj) for v in unquant_checkpoints[t_prev].variance]

    recon_theta = list(model_prev.parameters())
    recon_m = momentum_prev
    recon_v = variance_prev

    # Relative L2 errors vs unquantized float ground truth
    theta_diffs = [
        torch.max(torch.abs(r - g)).item()
        for r, g in zip(recon_theta, true_theta)
    ]
    theta_max_err = max(theta_diffs)

    theta_l2_diff = (
        sum(
            torch.sum((r - g) ** 2).item()
            for r, g in zip(recon_theta, true_theta)
        )
        ** 0.5
    )
    theta_l2_true = sum(torch.sum(g**2).item() for g in true_theta) ** 0.5
    theta_rel_err = theta_l2_diff / max(1e-6, theta_l2_true)

    m_l2_diff = (
        sum(torch.sum((r - g) ** 2).item() for r, g in zip(recon_m, true_m))
        ** 0.5
    )
    m_l2_true = sum(torch.sum(g**2).item() for g in true_m) ** 0.5
    m_rel_err = m_l2_diff / max(1e-6, m_l2_true)

    v_l2_diff = (
        sum(torch.sum((r - g) ** 2).item() for r, g in zip(recon_v, true_v))
        ** 0.5
    )
    v_l2_true = sum(torch.sum(g**2).item() for g in true_v) ** 0.5
    v_rel_err = v_l2_diff / max(1e-6, v_l2_true)

    # Relative errors vs quantized forward reference
    if (
        use_reversible_transform
        and quant_checkpoints is not None
        and t_prev in quant_checkpoints
    ):
      q_true_theta = [
          p.to(device_obj)
          for p in quant_checkpoints[t_prev].model.parameters()
      ]
      q_true_m = [
          m.to(device_obj) for m in quant_checkpoints[t_prev].momentum
      ]
      q_true_v = [
          v.to(device_obj) for v in quant_checkpoints[t_prev].variance
      ]

      q_theta_diff = (
          sum(
              torch.sum((r - g) ** 2).item()
              for r, g in zip(recon_theta, q_true_theta)
          )
          ** 0.5
      )
      q_theta_norm = sum(torch.sum(g**2).item() for g in q_true_theta) ** 0.5
      quant_theta_rel_err = q_theta_diff / max(1e-6, q_theta_norm)

      q_m_diff = (
          sum(torch.sum((r - g) ** 2).item() for r, g in zip(recon_m, q_true_m))
          ** 0.5
      )
      q_m_norm = sum(torch.sum(g**2).item() for g in q_true_m) ** 0.5
      quant_m_rel_err = q_m_diff / max(1e-6, q_m_norm)

      q_v_diff = (
          sum(torch.sum((r - g) ** 2).item() for r, g in zip(recon_v, q_true_v))
          ** 0.5
      )
      q_v_norm = sum(torch.sum(g**2).item() for g in q_true_v) ** 0.5
      quant_v_rel_err = q_v_diff / max(1e-6, q_v_norm)

    max_quant_theta_rel_err = max(max_quant_theta_rel_err, quant_theta_rel_err)
    max_quant_m_rel_err = max(max_quant_m_rel_err, quant_m_rel_err)
    max_quant_v_rel_err = max(max_quant_v_rel_err, quant_v_rel_err)

    min_v = min(torch.min(v).item() for v in recon_v)
    has_nan = any(
        torch.isnan(p).any().item()
        for p in recon_theta + recon_m + recon_v
    )
    has_inf = any(
        torch.isinf(p).any().item()
        for p in recon_theta + recon_m + recon_v
    )

    if first_divergence_step is None and (
        theta_rel_err > 0.01 or m_rel_err > 0.01
    ):
      first_divergence_step = (i + 1, t_prev, theta_rel_err, m_rel_err)

    if first_negative_var_step is None and min_v < 0:
      first_negative_var_step = (i + 1, t_prev, min_v)

    if first_nan_step is None and has_nan:
      first_nan_step = (i + 1, t_prev)

    if first_inf_step is None and has_inf:
      first_inf_step = (i + 1, t_prev)

    status_str = 'OK'
    if has_nan:
      status_str = 'NaN DETECTED!'
    elif has_inf:
      status_str = 'Inf DETECTED!'
    elif min_v < 0:
      status_str = 'NEGATIVE VAR'
    elif theta_rel_err > 1.0:
      status_str = 'DIVERGED (>100%)'

    num_back_steps = i + 1
    if (
        num_back_steps in log_milestones
        or (num_back_steps % 100 == 0)
        or has_nan
        or (num_back_steps == total_steps)
    ):
      if use_reversible_transform:
        print(
            f'{num_back_steps:<11} | {t_prev:<7} | {theta_rel_err:<14.4e} |'
            f' {m_rel_err:<14.4e} | {v_rel_err:<14.4e} |'
            f' {quant_theta_rel_err:<15.4e} | {quant_m_rel_err:<14.4e} |'
            f' {quant_v_rel_err:<14.4e} | {status_str}'
        )
      else:
        print(
            f'{num_back_steps:<11} | {t_prev:<7} | {theta_max_err:<14.4e} |'
            f' {theta_rel_err:<14.4e} | {m_rel_err:<14.4e} |'
            f' {v_rel_err:<14.4e} | {min_v:<14.4e} | {status_str}'
        )

    if has_nan:
      print(
          f'\n[!] Terminating early at backward step {num_back_steps}'
          f' (t={t_prev}) because NaN corrupted weights.'
      )
      break

    # Periodic checkpoint reset / snapping: only reset at periodic timestamps
    is_periodic_reset = (
        checkpoint_freq is not None
        and checkpoint_freq > 0
        and (t_prev % checkpoint_freq == 0)
        and (t_prev > 0)
    )
    if is_periodic_reset:
      if (
          use_reversible_transform
          and quant_checkpoints is not None
          and t_prev in quant_checkpoints
      ):
        ckpt_anchor = quant_checkpoints[t_prev]
        curr_model = copy.deepcopy(ckpt_anchor.model).to(device_obj)
        curr_momentum = [
            m.clone().to(device_obj) for m in ckpt_anchor.momentum
        ]
        curr_variance = [
            v.clone().to(device_obj) for v in ckpt_anchor.variance
        ]
      elif not use_reversible_transform and t_prev in unquant_checkpoints:
        ckpt_anchor = unquant_checkpoints[t_prev]
        curr_model = copy.deepcopy(ckpt_anchor.model).to(device_obj)
        curr_momentum = [
            m.clone().to(device_obj) for m in ckpt_anchor.momentum
        ]
        curr_variance = [
            v.clone().to(device_obj) for v in ckpt_anchor.variance
        ]
      else:
        curr_model = model_prev
        curr_momentum = momentum_prev
        curr_variance = variance_prev
    else:
      curr_model = model_prev
      curr_momentum = momentum_prev
      curr_variance = variance_prev

  if use_reversible_transform:
    print('-' * 155)
  else:
    print('-' * 115)

  print('\n=== SUMMARY OF RESULTS ===')
  if first_divergence_step:
    print(
        f'1. First significant divergence (>1% relative error): at backward'
        f' step {first_divergence_step[0]} (t={first_divergence_step[1]}),'
        f' Theta RelErr={first_divergence_step[2]:.4e}, Mom'
        f' RelErr={first_divergence_step[3]:.4e}'
    )
  if first_negative_var_step:
    print(
        f'2. First negative variance encountered (v < 0): at backward step'
        f' {first_negative_var_step[0]} (t={first_negative_var_step[1]}), Min'
        f' Var={first_negative_var_step[2]:.4e}'
    )
  if first_inf_step:
    print(
        f'3. First Inf encountered: at backward step {first_inf_step[0]}'
        f' (t={first_inf_step[1]})'
    )
  if first_nan_step:
    print(
        f'4. First NaN encountered: at backward step {first_nan_step[0]}'
        f' (t={first_nan_step[1]})'
    )
  else:
    print('4. No NaN encountered within steps executed.')

  if use_reversible_transform:
    print(
        f'5. Final Distance to Quantized Forward State at t=0:'
        f' Theta RelErr={quant_theta_rel_err:.4e},'
        f' Mom RelErr={quant_m_rel_err:.4e},'
        f' Var RelErr={quant_v_rel_err:.4e}'
    )
    print(
        f'6. Max Distance to Quantized Forward State across all steps:'
        f' Theta RelErr={max_quant_theta_rel_err:.4e},'
        f' Mom RelErr={max_quant_m_rel_err:.4e},'
        f' Var RelErr={max_quant_v_rel_err:.4e}'
    )
  print('=' * 80)


def run_micro_diagnostic(
    model_type = 'small_dnn',
    num_steps = 50,
    lr = 0.001,
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    batch_size = 128,
    dim = 64,
    num_classes = 10,
    device = 'cpu',
    seed = 42,
    reversible_scale = 1000000000000,
    checkpoint_freq = None,
    norm_type = 'batch_norm',
):
  """Runs detailed sub-operation micro diagnostic across all backward steps.

  Pinpoints the exact step and exact sub-operation where reconstructed states
  first diverge from forward training.
  """
  device_obj = torch.device(
      device if torch.cuda.is_available() and device == 'cuda' else device
  )
  torch.manual_seed(seed)
  np.random.seed(seed)

  if model_type.lower() == 'cnn' and dim == 64:
    dim = 784

  print('=' * 120)
  print(
      f'STARTING MULTI-STEP MICRO DIAGNOSTIC ({num_steps} steps, freq={checkpoint_freq}, norm={norm_type})'
  )
  print(f'Architecture: {model_type} (dim={dim}, num_classes={num_classes})')
  print(
      f'Hyperparams: lr={lr}, beta_1={beta_1}, beta_2={beta_2}, eps={eps},'
      f' scale={reversible_scale}'
  )
  print('=' * 120)

  num_samples = max(1000, num_steps * batch_size)
  x = torch.randn(num_samples, dim, dtype=torch.float32)
  y = torch.randint(0, num_classes, (num_samples,))
  dataset = torch.utils.data.TensorDataset(x, y)
  train_loader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=False
  )
  batches = utils.prepare_batches(train_loader)
  criterion = nn.CrossEntropyLoss()

  model = models.get_model(
      model_type, input_dim=dim, output_dim=num_classes, norm_type=norm_type
  ).to(device=device_obj, dtype=torch.float64)
  split_sizes = [p.numel() for p in model.parameters()]
  param_dtype = torch.float64
  num_params = sum(split_sizes)

  momentum_buffer = reversible.GPUReversibleTransform(
      num_params, gamma=beta_1, max_steps=num_steps, device=device_obj
  )
  variance_buffer = reversible.GPUReversibleTransform(
      num_params,
      gamma=beta_2,
      max_steps=num_steps,
      device=device_obj,
      mode='ceil',
  )

  # Caches for forward pass
  fwd_weights_int = {}
  fwd_weights_float = {}
  fwd_grads = {}
  fwd_grad_scaled_m = {}
  fwd_grad_scaled_v = {}
  fwd_m_decayed = {}
  fwd_v_decayed = {}
  fwd_scaled_m = {}
  fwd_scaled_v = {}
  fwd_unscaled_m = {}
  fwd_unscaled_v = {}
  fwd_m_hat = {}
  fwd_v_hat = {}
  fwd_update_vec_int = {}
  quant_checkpoints = {}

  scaled_m = torch.zeros(num_params, dtype=torch.int64, device=device_obj)
  scaled_v = torch.ones(num_params, dtype=torch.int64, device=device_obj)

  # Initialize model weights as integers
  w_int = [
      torch.round(w.data.to(torch.float64) * reversible_scale).to(torch.int64)
      for w in model.parameters()
  ]
  w_float = [w.to(param_dtype) / reversible_scale for w in w_int]
  utils.update_model_parameters(model, w_float)

  fwd_weights_int[0] = [w.clone() for w in w_int]
  fwd_weights_float[0] = [w.clone() for w in w_float]
  fwd_scaled_m[0] = scaled_m.clone()
  fwd_scaled_v[0] = scaled_v.clone()
  fwd_unscaled_m[0] = scaled_m.to(torch.float64) / reversible_scale
  fwd_unscaled_v[0] = scaled_v.to(torch.float64) / reversible_scale

  print(f'\n>>> Running {num_steps} FORWARD steps and caching all sub-steps...')
  for t in range(1, num_steps + 1):
    x_b, y_b, _ = batches[(t - 1) % len(batches)]
    x_b = x_b.to(device=device_obj, dtype=param_dtype)
    y_b = y_b.to(device_obj)

    # 1. Gradient
    model.zero_grad(set_to_none=True)
    y_hat = model(x_b)
    loss = criterion(y_hat, y_b)
    grads = torch.autograd.grad(
        loss, list(model.parameters()), create_graph=False
    )
    grad_flat = torch.cat([g.detach().flatten() for g in grads]).to(
        torch.float64
    )
    fwd_grads[t - 1] = [g.detach().clone() for g in grads]

    # 2. Scaled gradients
    gsm = torch.round((1.0 - beta_1) * grad_flat * reversible_scale).to(
        torch.int64
    )
    gsv = torch.ceil(
        (1.0 - beta_2) * (grad_flat**2) * reversible_scale
    ).to(torch.int64)
    fwd_grad_scaled_m[t - 1] = gsm.clone()
    fwd_grad_scaled_v[t - 1] = gsv.clone()

    # 3. Buffer decay
    m_dec = momentum_buffer.forward_decay(scaled_m)
    v_dec = variance_buffer.forward_decay(scaled_v)
    fwd_m_decayed[t - 1] = m_dec.clone()
    fwd_v_decayed[t - 1] = v_dec.clone()

    scaled_m = m_dec + gsm
    scaled_v = v_dec + gsv
    fwd_scaled_m[t] = scaled_m.clone()
    fwd_scaled_v[t] = scaled_v.clone()

    # 4. Unscaled and bias corrected
    unscaled_m = scaled_m.to(torch.float64) / reversible_scale
    unscaled_v = scaled_v.to(torch.float64) / reversible_scale
    fwd_unscaled_m[t] = unscaled_m.clone()
    fwd_unscaled_v[t] = unscaled_v.clone()

    bc1 = 1.0 - (beta_1**t)
    bc2 = 1.0 - (beta_2**t)
    m_hat = unscaled_m / bc1
    v_hat = unscaled_v / bc2
    fwd_m_hat[t] = m_hat.clone()
    fwd_v_hat[t] = v_hat.clone()

    # 5. Update vector with Cauchy-Schwarz limit
    up_limit = utils.compute_update_limit(beta_1, beta_2, t)
    r_unscaled = m_hat / (eps + torch.sqrt(v_hat))
    r_clamped = torch.clamp(r_unscaled, -up_limit, up_limit)
    up_vec = lr * r_clamped
    up_vec_int = torch.round(up_vec * reversible_scale).to(torch.int64)
    fwd_update_vec_int[t] = up_vec_int.clone()

    # 6. Apply weights
    up_int_splits = torch.split(up_vec_int, split_sizes)
    w_int = [w - u.view(w.shape) for w, u in zip(w_int, up_int_splits)]
    w_float = [w.to(param_dtype) / reversible_scale for w in w_int]
    utils.update_model_parameters(model, w_float)
    fwd_weights_int[t] = [w.clone() for w in w_int]
    fwd_weights_float[t] = [w.clone() for w in w_float]

    is_ckpt = (
        (checkpoint_freq is not None and checkpoint_freq > 0 and t % checkpoint_freq == 0)
        or t == num_steps
        or t == 0
    )
    if is_ckpt:
      cur_m_splits = [
          t_m.view(p.shape).to(torch.float64).cpu()
          for t_m, p in zip(
              torch.split(unscaled_m, split_sizes), model.parameters()
          )
      ]
      cur_v_splits = [
          t_v.view(p.shape).to(torch.float64).cpu()
          for t_v, p in zip(
              torch.split(unscaled_v, split_sizes), model.parameters()
          )
      ]
      quant_checkpoints[t] = base_model.CheckpointState(
          step=t,
          model=copy.deepcopy(model).cpu(),
          momentum=cur_m_splits,
          variance=cur_v_splits,
          gradients=None,
      )

  print('Forward pass completed.')
  print('\n>>> Running BACKWARD UNROLLING with Sub-Operation Verification...')
  print(
      '-------------------------------------------------------------------------------------------------------------------------------------------------------------'
  )
  print(
      f'{"Back Step":<10} | {"Step t":<8} | {"W RelErr":<12} | {"G RelErr":<12} | {"GSM Diff":<10} | {"GSV Diff":<10} | {"TM Diff":<10} | {"TV Diff":<10} | {"M IntDiff":<10} | {"V IntDiff":<10} | Status'
  )
  print(
      '-------------------------------------------------------------------------------------------------------------------------------------------------------------'
  )

  curr_model = copy.deepcopy(model)
  curr_m = [
      t.view(p.shape).to(device=device_obj, dtype=p.dtype)
      for t, p in zip(
          torch.split(fwd_unscaled_m[num_steps], split_sizes), model.parameters()
      )
  ]
  curr_v = [
      t.view(p.shape).to(device=device_obj, dtype=p.dtype)
      for t, p in zip(
          torch.split(fwd_unscaled_v[num_steps], split_sizes), model.parameters()
      )
  ]

  first_div_step = None
  first_div_info = {}

  for t in range(num_steps, 0, -1):
    t_prev = t - 1
    x_b, y_b, _ = batches[t_prev % len(batches)]
    x_b = x_b.to(device=device_obj, dtype=param_dtype)
    y_b = y_b.to(device_obj)

    # Optional checkpoint snapping
    if (
        checkpoint_freq is not None
        and checkpoint_freq > 0
        and t in quant_checkpoints
        and t != num_steps
    ):
      ckpt = quant_checkpoints[t]
      curr_model = copy.deepcopy(ckpt.model).to(device=device_obj, dtype=param_dtype)
      curr_m = [m.clone().to(device=device_obj, dtype=param_dtype) for m in ckpt.momentum]
      curr_v = [v.clone().to(device=device_obj, dtype=param_dtype) for v in ckpt.variance]

    recon_model, recon_m, recon_v = utils.compute_metadata_adam_backward(
        x_batch=x_b,
        y_batch=y_b,
        model=curr_model,
        loss_fn=criterion,
        momentum=curr_m,
        variance=curr_v,
        lr=lr,
        beta_1=beta_1,
        beta_2=beta_2,
        eps=eps,
        step=t,
        device=device_obj,
        momentum_buffer=momentum_buffer,
        variance_buffer=variance_buffer,
        scale=reversible_scale,
    )

    # Check 1: Model weights theta_{t-1}
    true_w = fwd_weights_float[t_prev]
    recon_w = list(recon_model.parameters())
    w_max_err = max(torch.max(torch.abs(r - g)).item() for r, g in zip(recon_w, true_w))
    w_l2_diff = (sum(torch.sum((r - g)**2).item() for r, g in zip(recon_w, true_w)))**0.5
    w_l2_true = max(1e-12, sum(torch.sum(g**2).item() for g in true_w)**0.5)
    w_rel_err = w_l2_diff / w_l2_true

    # Check 2: Recomputed Loss Gradient g_{t-1}
    true_g = fwd_grads[t_prev]
    recon_g = utils.compute_gradient(
        x_b, y_b, recon_model, criterion, create_graph=False, retain_graph=False
    )
    g_max_err = max(torch.max(torch.abs(r - g)).item() for r, g in zip(recon_g, true_g))
    g_l2_diff = (sum(torch.sum((r - g)**2).item() for r, g in zip(recon_g, true_g)))**0.5
    g_l2_true = max(1e-12, sum(torch.sum(g**2).item() for g in true_g)**0.5)
    g_rel_err = g_l2_diff / g_l2_true

    # Check 3: Scaled Gradient Integers
    recon_g_flat = torch.cat([g.detach().flatten() for g in recon_g]).to(torch.float64)
    recon_gsm = torch.round((1.0 - beta_1) * recon_g_flat * reversible_scale).to(torch.int64)
    recon_gsv = torch.ceil((1.0 - beta_2) * (recon_g_flat**2) * reversible_scale).to(torch.int64)
    gsm_int_diff = (recon_gsm - fwd_grad_scaled_m[t_prev]).abs().max().item()
    gsv_int_diff = (recon_gsv - fwd_grad_scaled_v[t_prev]).abs().max().item()

    # Check 4: Decayed Target Integers
    curr_scaled_m = torch.round(
        torch.cat([m.flatten() for m in curr_m]).to(torch.float64) * reversible_scale
    ).to(torch.int64)
    curr_scaled_v = torch.round(
        torch.cat([v.flatten() for v in curr_v]).to(torch.float64) * reversible_scale
    ).to(torch.int64)
    target_m_int = curr_scaled_m - recon_gsm
    target_v_int = torch.clamp_min(curr_scaled_v - recon_gsv, 0)
    tm_diff = (target_m_int - fwd_m_decayed[t_prev]).abs().max().item()
    tv_diff = (target_v_int - fwd_v_decayed[t_prev]).abs().max().item()

    # Check 5: Reconstructed Momentum & Variance Integers
    recon_m_flat = torch.round(
        torch.cat([m.flatten() for m in recon_m]).to(torch.float64) * reversible_scale
    ).to(torch.int64)
    recon_v_flat = torch.round(
        torch.cat([v.flatten() for v in recon_v]).to(torch.float64) * reversible_scale
    ).to(torch.int64)
    m_int_diff = (recon_m_flat - fwd_scaled_m[t_prev]).abs().max().item()
    v_int_diff = (recon_v_flat - fwd_scaled_v[t_prev]).abs().max().item()

    has_div = (
        w_rel_err > 1e-6
        or g_rel_err > 1e-6
        or gsm_int_diff > 0
        or gsv_int_diff > 0
        or tm_diff > 0
        or tv_diff > 0
        or m_int_diff > 0
        or v_int_diff > 0
    )

    status_str = "DIVERGED" if has_div else "OK"
    back_idx = num_steps - t + 1
    if back_idx <= 10 or back_idx % 10 == 0 or back_idx == num_steps or has_div:
      print(
          f'{back_idx:<10} | {t_prev:<8} | {w_rel_err:<12.4e} | {g_rel_err:<12.4e} | {gsm_int_diff:<10} | {gsv_int_diff:<10} | {tm_diff:<10} | {tv_diff:<10} | {m_int_diff:<10} | {v_int_diff:<10} | {status_str}'
      )

    if has_div and first_div_step is None:
      first_div_step = t_prev
      print(f'\n--- DEBUG STEP t={t} -> {t_prev} ---')
      print(f'target_m_int[:5]:       {target_m_int[:5].tolist()}')
      print(f'fwd_m_decayed[t_prev][:5]: {fwd_m_decayed[t_prev][:5].tolist()}')
      print(f'target_m_int diff:      {(target_m_int - fwd_m_decayed[t_prev]).abs().max().item()}')
      print(f'recon_m_flat[:5]:       {recon_m_flat[:5].tolist()}')
      print(f'fwd_scaled_m[t_prev][:5]: {fwd_scaled_m[t_prev][:5].tolist()}')
      print(f'recon_m_flat diff:      {(recon_m_flat - fwd_scaled_m[t_prev]).abs().max().item()}')
      print(f'recon_v_flat diff:      {(recon_v_flat - fwd_scaled_v[t_prev]).abs().max().item()}')
      print(f'momentum_buffer active_buffer[:5]: {momentum_buffer.active_buffer[:5].tolist()}')
      print('-----------------------------------\n')
      first_div_info = {
          'back_idx': back_idx,
          'step_t': t,
          'step_prev': t_prev,
          'w_max_err': w_max_err,
          'w_rel_err': w_rel_err,
          'g_max_err': g_max_err,
          'g_rel_err': g_rel_err,
          'gsm_int_diff': gsm_int_diff,
          'gsv_int_diff': gsv_int_diff,
          'tm_diff': tm_diff,
          'tv_diff': tv_diff,
          'm_int_diff': m_int_diff,
          'v_int_diff': v_int_diff,
      }

    curr_model = recon_model
    curr_m = recon_m
    curr_v = recon_v

  print(
      '-------------------------------------------------------------------------------------------------------------------------------------------------------------'
  )
  print('\n=== MICRO DIAGNOSTIC SUMMARY ===')
  if first_div_step is not None:
    print(
        f'🚨 FIRST DIVERGENCE OCCURRED AT BACKWARD STEP {first_div_info["back_idx"]}'
        f' (t={first_div_info["step_t"]} -> {first_div_info["step_prev"]}):'
    )
    print(
        f'   1. Weight Rel Error:      {first_div_info["w_rel_err"]:.6e}'
        f' (Max Abs = {first_div_info["w_max_err"]:.6e})'
    )
    print(
        f'   2. Gradient Rel Error:    {first_div_info["g_rel_err"]:.6e}'
        f' (Max Abs = {first_div_info["g_max_err"]:.6e})'
    )
    print(
        f'   3. Scaled Grad Int Diffs: gsm={first_div_info["gsm_int_diff"]},'
        f' gsv={first_div_info["gsv_int_diff"]}'
    )
    print(
        f'   4. Decayed Target Diffs:  tm={first_div_info["tm_diff"]},'
        f' tv={first_div_info["tv_diff"]}'
    )
    print(
        f'   5. State Integer Diffs:   m={first_div_info["m_int_diff"]},'
        f' v={first_div_info["v_int_diff"]}'
    )

    if first_div_info['w_rel_err'] > 1e-6 and first_div_info['gsm_int_diff'] == 0:
      print('   👉 PRIMARY ROOT CAUSE: Weight update step theta_{t-1} calculation drifted first.')
    elif first_div_info['g_rel_err'] > 1e-6 and first_div_info['w_rel_err'] < 1e-10:
      print('   👉 PRIMARY ROOT CAUSE: Gradient evaluation g_{t-1} on theta_{t-1} drifted first.')
    elif first_div_info['gsm_int_diff'] > 0 or first_div_info['gsv_int_diff'] > 0:
      print('   👉 PRIMARY ROOT CAUSE: Integer quantization rounding of gradients (gsm/gsv) drifted first.')
    elif first_div_info['tm_diff'] > 0 or first_div_info['tv_diff'] > 0:
      print('   👉 PRIMARY ROOT CAUSE: Target integer subtraction (scaled_t - grad_scaled) drifted first.')
    elif first_div_info['m_int_diff'] > 0 or first_div_info['v_int_diff'] > 0:
      print('   👉 PRIMARY ROOT CAUSE: Reversible GPU transform backward_decay operator drifted first.')
  else:
    print('✅ ALL SUB-OPERATIONS ACROSS ALL BACKWARD STEPS MATCHED FORWARD PASS WITH ZERO DIVERGENCE!')
  print('=' * 120 + '\n')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.micro_diagnostic:
    run_micro_diagnostic(
        model_type=FLAGS.model_type,
        num_steps=FLAGS.num_steps,
        lr=FLAGS.lr,
        beta_1=FLAGS.beta_1,
        beta_2=FLAGS.beta_2,
        eps=FLAGS.eps,
        batch_size=FLAGS.batch_size,
        dim=FLAGS.dim,
        num_classes=FLAGS.num_classes,
        device=FLAGS.device,
        seed=FLAGS.seed,
        reversible_scale=FLAGS.reversible_scale,
        checkpoint_freq=FLAGS.checkpoint_freq,
        norm_type=FLAGS.norm_type,
    )
  else:
    run_experiment(
        model_type=FLAGS.model_type,
        num_steps=FLAGS.num_steps,
        lr=FLAGS.lr,
        beta_1=FLAGS.beta_1,
        beta_2=FLAGS.beta_2,
        eps=FLAGS.eps,
        batch_size=FLAGS.batch_size,
        dim=FLAGS.dim,
        num_classes=FLAGS.num_classes,
        device=FLAGS.device,
        seed=FLAGS.seed,
        use_reversible_transform=FLAGS.use_reversible_transform,
        reversible_scale=FLAGS.reversible_scale,
        checkpoint_freq=FLAGS.checkpoint_freq,
        norm_type=FLAGS.norm_type,
    )


if __name__ == '__main__':
  app.run(main)
