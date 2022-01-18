# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Evaluation function handler for sequential or cascaded."""
import torch
import torch.nn.functional as F


class SequentialEvalLoop:
  """Evaluation loop for sequential model."""

  def __init__(self, num_classes, flags):
    self.num_classes = num_classes
    self.flags = flags

  def __call__(self, net, loader, criterion, device):
    net.eval()

    batch_losses = []
    batch_accs = []
    batch_logits = []
    batch_embeddings = []
    ys = []

    # Embedding hook
    def embedding_hook_fn(module, x, output):  # pylint: disable=unused-argument
      global embedding  # pylint: disable=global-variable-undefined
      embedding = x[0]
    _ = net.fc.register_forward_hook(embedding_hook_fn)

    for data, targets in loader:
      # One-hot-ify targets
      y = torch.eye(self.num_classes)[targets]

      # Determine device placement
      data = data.to(device, non_blocking=True)

      # Forward pass
      with torch.no_grad():
        logits = net(data, t=0)

      if self.flags.get('keep_logits', False):
        batch_logits.append(logits)
        batch_embeddings.append(embedding.cpu())
        ys.append(targets)

      # Determine device placement
      targets = targets.to(logits.device, non_blocking=True)
      y = y.to(logits.device, non_blocking=True)

      # Compute loss
      loss = criterion(logits, y)
      batch_losses.append(loss.item())

      # Predictions
      softmax = F.softmax(logits, dim=1)
      y_pred = torch.argmax(softmax, dim=1)

      # Updates running statistics
      n_correct = torch.eq(targets, y_pred).sum()
      acc_i = n_correct / float(targets.shape[0])
      batch_accs.append(acc_i.item())

    # Compute loss and accuracy
    # loss = np.mean(batch_losses)
    # accuracy = np.mean(batch_accs)

    if self.flags.get('keep_logits', False):
      logged_data = {
          'logits': batch_logits,
          'embeddings': batch_embeddings,
          'y': ys,
      }
      return batch_losses, batch_accs, logged_data
    else:
      return batch_losses, batch_accs


class CascadedEvalLoop(object):
  """Evaluation loop for cascaded model."""

  def __init__(self, n_timesteps, num_classes, flags):
    self.n_timesteps = n_timesteps
    self.num_classes = num_classes
    self.flags = flags

  def __call__(self, net, loader, criterion, device):
    net.eval()

    batch_logits = []
    batch_embeddings = []
    ys = []

    # Embedding hook
    def embedding_hook_fn(module, x, output):  # pylint: disable=unused-argument
      global embedding  # pylint: disable=global-variable-undefined
      embedding = x[0]
    _ = net.fc.register_forward_hook(embedding_hook_fn)

    for batch_i, (x, targets) in enumerate(loader):
      # One-hot-ify targets
      y = torch.eye(self.num_classes)[targets]

      if self.flags.get('keep_logits', False):
        ys.append(targets)

      # Determine device placement
      x = x.to(device, non_blocking=True)

      timestep_accs = torch.zeros(self.n_timesteps)
      timestep_losses = torch.zeros(self.n_timesteps)
      timestep_logits = []
      timestep_embeddings = []
      for t in range(self.n_timesteps):
        # Forward pass
        with torch.no_grad():
          logits_t = net(x, t)

        if self.flags.get('keep_logits', False):
          timestep_logits.append(logits_t)
          timestep_embeddings.append(embedding)

        # Determine device placement
        targets = targets.to(logits_t.device, non_blocking=True)
        y = y.to(logits_t.device, non_blocking=True)

        # Compute loss
        loss_i = criterion(logits_t, y)

        # Log loss
        timestep_losses[t] = loss_i.item()

        # Predictions
        softmax_t = F.softmax(logits_t, dim=1)
        y_pred = torch.argmax(softmax_t, dim=1)

        # Updates running accuracy statistics
        n_correct = torch.eq(targets, y_pred).sum()
        acc_i = n_correct / float(targets.shape[0])
        timestep_accs[t] = acc_i

      # Update batch loss and compute average
      timestep_losses = timestep_losses.unsqueeze(dim=0)
      timestep_accs = timestep_accs.unsqueeze(dim=0)
      if batch_i == 0:
        batch_losses = timestep_losses
        batch_accs = timestep_accs
      else:
        batch_losses = torch.cat([batch_losses, timestep_losses], axis=0)
        batch_accs = torch.cat([batch_accs, timestep_accs], axis=0)

      if self.flags.get('keep_logits', False):
        # stack into shape=(time, batch, n_classes)
        timestep_logits = torch.stack(timestep_logits)
        batch_logits.append(timestep_logits)

        timestep_embeddings = torch.stack(timestep_embeddings)
        batch_embeddings.append(timestep_embeddings)

    # Average over the batches per timestep
    batch_losses = batch_losses.detach().numpy()
    batch_accs = batch_accs.detach().numpy()

    # Compute loss and accuracy
    if self.flags.get('keep_logits', False):
      # concat over batch dim into shape=(time, batch, n_classes)
      batch_logits = torch.cat(batch_logits, dim=1)
      batch_embeddings = torch.cat(batch_embeddings, dim=1)
      ys = torch.cat(ys)
      logged_data = {
          'logits': batch_logits,
          'embeddings': batch_embeddings,
          'y': ys,
      }
      return batch_losses, batch_accs, logged_data
    else:
      return batch_losses, batch_accs


def get_eval_loop(n_timesteps, num_classes, flags):
  """Retrieve sequential or cascaded eval function."""
  if flags.cascaded:
    eval_fxn = CascadedEvalLoop(n_timesteps, num_classes, flags)
  else:
    eval_fxn = SequentialEvalLoop(num_classes, flags)
  return eval_fxn
