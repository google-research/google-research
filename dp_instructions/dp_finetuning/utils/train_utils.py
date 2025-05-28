# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Training and evaluation utilities."""

import sys
import time
import numpy as np
import torch
from utils import dp_utils


def eval_epoch(
    model,
    accelerator,
    eval_dataloader,
    epoch,
    args,
    description='end',
    eval_only=False,
):
  """Evaluate model on eval dataset."""
  model.eval()
  eval_total_loss = torch.zeros(1, device=accelerator.device)
  eval_total_num_tokens = 0

  loss_denom = args.block_size

  start_time = time.time()

  if eval_only:
    eval_loss_list = []

  with torch.no_grad():
    for step, batch in enumerate(eval_dataloader):
      with accelerator.no_sync(model):
        outputs = model(**batch)
        # get number of non-zero loss elements
        num_tokens = torch.sum(batch['labels'] != -100).detach()
        eval_total_num_tokens += num_tokens
        if eval_only:
          eval_loss_list.append(outputs.loss.detach().item())
        loss = outputs.loss * (num_tokens / loss_denom)
        eval_total_loss += loss.detach()
        progress = step * 100 / len(eval_dataloader)
        if step % 100 == 0:
          elasp_time = time.time() - start_time
          accelerator.print(
              f'epoch {epoch}, eval progress {progress:.2f}%, eval time'
              f' {elasp_time:.2f}s'
          )
          sys.stdout.flush()

  eval_epoch_loss = (eval_total_loss * loss_denom) / eval_total_num_tokens
  eval_epoch_loss = torch.mean(accelerator.gather(eval_epoch_loss)).item()
  accelerator.print(
      f'At epoch {epoch} {description}, eval loss {eval_epoch_loss:.4f}'
  )

  if eval_only:
    eval_loss_list = np.array(eval_loss_list)
    np.save(f'{args.output_dir}/eval_loss_list.npy', eval_loss_list)

  return eval_epoch_loss


def train_epoch(
    model,
    tokenizer,
    accelerator,
    optimizer,
    lr_scheduler,
    train_dataloader,
    epoch,
    args,
    completed_steps,
    progress_bar,
):
  """Train model on train dataset."""
  del tokenizer
  model.train()
  require_grad_params = [p for p in model.parameters() if p.requires_grad]

  recent_loss = torch.zeros(1, device=accelerator.device)
  total_loss = torch.zeros(1, device=accelerator.device)
  recent_num_tokens = 0
  total_num_tokens = 0

  # prevent loss explosion
  loss_denom = args.block_size

  # initialize variables for tracking training states
  accumulated_steps = 0
  for step, batch in enumerate(train_dataloader):
    # sync gradients by ourself, instead of using accelerator.accummulate
    with accelerator.no_sync(model):

      # get number of non-zero loss elements
      num_tokens = torch.sum(batch['labels'] != -100).detach()
      # track number of recent/total non-zero loss elements,
      # used for loss normalization
      recent_num_tokens += num_tokens
      total_num_tokens += num_tokens

      outputs = model(**batch)
      loss = outputs.loss * (num_tokens / loss_denom)
      accelerator.backward(loss)

      # track of the loss of the most recent updates
      recent_loss += loss.detach()
      # track of the loss of the entire epoch
      total_loss += loss.detach()

      if args.clip_norm >= 0:
        # clip per-example gradients, and sum to .accumulated_grad
        dp_utils.clip_and_accumulate_perexample_grads(
            require_grad_params, accumulated_steps, args.clip_norm, accelerator
        )

      accumulated_steps += 1
      if (
          accumulated_steps == args.gradient_accumulation_steps
          or step == len(train_dataloader) - 1
      ):
        # sync gradients
        if not args.clip_norm >= 0:
          # undo mixed precision scaling.
          accelerator.unscale_gradients(optimizer=optimizer)
          # when lip_norm > 0, the above two steps are done in
          # `clip_and_accumulate_perexample_grads`
          grads_to_sync = [p.grad for p in require_grad_params]
        else:
          grads_to_sync = [p.accumulated_grad for p in require_grad_params]
        synced_grads = accelerator.gather(grads_to_sync)
        synced_grads = [
            g.view(accelerator.num_processes, *p.shape)
            for g, p in zip(synced_grads, require_grad_params)
        ]
        synced_grads = [torch.sum(g, dim=0) for g in synced_grads]

        if args.noise_multiplier > 0:
          if accelerator.is_main_process:
            noises = []
            for g in synced_grads:
              noises.append(
                  torch.normal(
                      0,
                      args.clip_norm * args.noise_multiplier,
                      size=g.shape,
                      device=g.device,
                      dtype=g.dtype,
                  )
              )
          else:
            noises = [torch.zeros_like(g) for g in synced_grads]
          # synchronize noise
          noises = accelerator.reduce(noises, reduction='sum', scale=1.0)
          # add noise to gradients
          synced_grads = [g + n for g, n in zip(synced_grads, noises)]

        # average over whole batch
        synced_grads = [g / args.logical_batch_size for g in synced_grads]
        # assign synced grads to model, overwrite the default grads
        for g, p in zip(synced_grads, require_grad_params):
          p.grad = g.to(p.dtype)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        performed_optimizer_step = True
        accumulated_steps = 0
      else:
        performed_optimizer_step = False

      if performed_optimizer_step:
        progress_bar.update(1)
        completed_steps += 1
        sys.stdout.flush()
        if (completed_steps) % args.log_freq == 0:
          current_loss = (recent_loss / recent_num_tokens) * loss_denom
          accelerator.print(
              f'epoch {epoch}, completed steps {completed_steps}, train loss'
              f' {current_loss.item()}, current lr'
              f' {lr_scheduler.get_last_lr()[0]}'
          )
          sys.stdout.flush()
          recent_loss = 0
          recent_num_tokens = 0

        if completed_steps >= args.max_train_steps:
          break

  epoch_loss = (total_loss / total_num_tokens) * loss_denom
  # synchronize epoch loss
  epoch_loss = torch.mean(accelerator.gather(epoch_loss)).item()

  return epoch_loss, completed_steps
