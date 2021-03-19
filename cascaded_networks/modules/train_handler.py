# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Training function handler for sequential or cascaded."""
import torch
import torch.nn.functional as F
from cascaded_networks.models import model_utils


class SequentialTrainingScheme:
  """Sequential Training Scheme."""

  def __init__(self, num_classes, flags):
    """Initialize sequential training handler."""
    self.num_classes = num_classes
    self.flags = flags

  def __call__(self, net, loader, criterion, optimizer, device):
    # Flag model for training
    net.train()

    batch_losses = []
    batch_accs = []
    for data, targets in loader:
      # One-hot-ify targets
      y = torch.eye(self.num_classes)[targets]

      # Determine device placement
      data = data.to(device, non_blocking=True)

      # Zero gradients
      optimizer.zero_grad()

      # Run forward pass
      logits = net(data, t=0)

      # Determine device placement
      targets = targets.to(logits.device, non_blocking=True)
      y = y.to(logits.device, non_blocking=True)

      # Compute loss
      loss = criterion(logits, y)

      # Compute gradients
      loss.backward()

      # Weight decay
      model_utils.apply_weight_decay(net, self.flags.weight_decay)

      # Take optimization step
      optimizer.step()

      # Predictions
      softmax_output = F.softmax(logits, dim=1)
      y_pred = torch.argmax(softmax_output, dim=1)

      # Updates batch accs
      n_correct = torch.eq(targets, y_pred).sum()
      acc_i = n_correct / float(targets.shape[0])
      batch_accs.append(acc_i.item())

      # Update batch loss
      batch_losses.append(loss.item())

    # Compute loss and accuracy across nodes
    # loss = np.mean(batch_losses)
    # accuracy = np.mean(batch_accs)

    return batch_losses, batch_accs


class CascadedTrainingSchemes(object):
  """Cascaded training schemes."""

  def __init__(self, n_timesteps, num_classes, flags):
    """Initialize cascaded training handler."""
    self.n_timesteps = n_timesteps
    self.num_classes = num_classes
    self.flags = flags

  def __call__(self, net, loader, criterion, optimizer, device):
    # Flag model for training
    net.train()

    for batch_i, (data, targets) in enumerate(loader):
      # Send to device
      data = data.to(device)
      targets = targets.to(device)

      # Zero out grads
      optimizer.zero_grad()

      predicted_logits = []
      for t in range(self.n_timesteps):
        # Run forward pass
        logit_t = net(data, t)
        predicted_logits.append(logit_t)

      # One-hot-ify targets and send to output device
      targets = targets.to(logit_t.device, non_blocking=True)
      y = torch.eye(self.num_classes)[targets]
      y = y.to(targets.device, non_blocking=True)

      loss = 0
      timestep_losses = torch.zeros(self.n_timesteps)
      timestep_accs = torch.zeros(self.n_timesteps)

      for t in range(len(predicted_logits)):
        logit_i = predicted_logits[t]

        # First term
        sum_term = torch.zeros_like(logit_i)
        t_timesteps = list(range(t+1, self.n_timesteps))
        for i, n in enumerate(t_timesteps, 1):
          logit_k = predicted_logits[n].detach().clone()
          softmax_i = F.softmax(logit_k, dim=1)
          sum_term = sum_term + self.flags.lambda_val**(i - 1) * softmax_i

        # Final terms
        term_1 = (1 - self.flags.lambda_val) * sum_term
        term_2 = self.flags.lambda_val**(self.n_timesteps - t - 1) * y
        softmax_j = term_1 + term_2

        # Compute loss
        loss_i = criterion(pred_logits=logit_i, y_true_softmax=softmax_j)

        # Aggregate loss
        if self.flags.tdl_mode == 'alpha_weighted':
          loss = loss + loss_i
        else:
          # Ignore first timestep loss (all 0's output)
          if t > 0:
            loss = loss + loss_i

        # Log loss item
        timestep_losses[t] = loss_i.item()

        # Predictions
        softmax_i = F.softmax(logit_i, dim=1)
        y_pred = torch.argmax(softmax_i, dim=1)

        # Updates running accuracy statistics
        n_correct = torch.eq(targets, y_pred).sum()
        acc_i = n_correct / float(targets.shape[0])
        timestep_accs[t] = acc_i

      # Normalize loss
      if self.flags.normalize_loss:
        loss = loss / float(self.n_timesteps)

      # Compute gradients
      loss.backward()

      # Weight decay
      model_utils.apply_weight_decay(net, self.flags.weight_decay)

      # Take optimization step
      optimizer.step()

      # Update batch loss and compute average
      timestep_losses = timestep_losses.unsqueeze(dim=0)
      timestep_accs = timestep_accs.unsqueeze(dim=0)
      if batch_i == 0:
        batch_losses = timestep_losses
        batch_accs = timestep_accs
      else:
        batch_losses = torch.cat([batch_losses, timestep_losses], axis=0)
        batch_accs = torch.cat([batch_accs, timestep_accs], axis=0)

    # Average over the batches per timestep
    batch_losses = batch_losses.detach().numpy()
    batch_accs = batch_accs.detach().numpy()

    return batch_losses, batch_accs


def get_train_loop(n_timesteps, num_classes, flags):  # pylint: disable=invalid-name
  """Retrieve sequential or cascaded training function."""
  if flags.cascaded:
    train_fxn = CascadedTrainingSchemes(n_timesteps, num_classes, flags)
  else:
    train_fxn = SequentialTrainingScheme(num_classes, flags)

  return train_fxn
