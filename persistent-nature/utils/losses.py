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

# pylint: disable=invalid-name
"""wrappers for loss functions."""
import lpips
import torch
from torch import nn
import torch.nn.functional as F


def adaptive_downsample256(img, mode='bilinear'):
  img = img.clamp(-1, 1)
  if img.shape[-1] > 256:
    return F.interpolate(img, size=(256, 256), mode=mode)
  else:
    return img


class LPIPS_Loss(nn.Module):
  """Wrapper for LPIPS loss."""

  def __init__(self, model='net-lin', net='vgg', use_gpu=True, spatial=False):
    super(LPIPS_Loss, self).__init__()
    self.model = lpips.LPIPS(net=net, spatial=spatial).eval()

  def forward(self, pred, ref):
    dist = self.model.forward(pred, ref)
    assert dist.shape[2] == 1 and dist.shape[3] == 1
    return dist[:, :, 0, 0]  # squeeze spatial dimensions


def check_loss_input(im0, im1, w):
  """im0 is out and im1 is target and w is mask."""
  assert list(im0.size())[2:] == list(im1.size())[2:], 'spatial dim mismatch'
  if w is not None:
    assert list(im0.size())[2:] == list(w.size())[2:], 'spatial dim mismatch'

  if im1.size(0) != 1:
    assert im0.size(0) == im1.size(0)

  if w is not None and w.size(0) != 1:
    assert im0.size(0) == w.size(0)
  return


# masked lpips
class Masked_LPIPS_Loss(nn.Module):
  """LPIPS loss with spatial weighting."""

  def __init__(self, net='vgg', device='cuda', precision='float'):
    super(Masked_LPIPS_Loss, self).__init__()
    self.lpips = lpips.LPIPS(net=net, spatial=True).eval()
    self.lpips = self.lpips.to(device)
    if precision == 'half':
      self.lpips.half()
    elif precision == 'float':
      self.lpips.float()
    elif precision == 'double':
      self.lpips.double()
    return

  def forward(self, im0, im1, w=None):
    """ims have dimension BCHW while mask is B1HW."""
    check_loss_input(im0, im1, w)
    # lpips takes the sum of each spatial map
    loss = self.lpips(im0, im1)
    if w is not None:
      n = torch.sum(loss * w, [1, 2, 3])
      d = torch.sum(w, [1, 2, 3])
      loss = n / d
    return loss

  def __call__(self, im0, im1, w=None):
    return self.forward(im0, im1, w)


class Masked_L1_Loss(nn.Module):
  """L1 loss with mask."""

  def __init__(self):
    super(Masked_L1_Loss, self).__init__()
    self.loss = nn.L1Loss(reduction='none')

  def forward(self, pred, ref, w=None):
    """ims have dimension BCHW while mask is B1HW."""
    check_loss_input(pred, ref, w)
    loss = self.loss(pred, ref)
    assert pred.shape[1] == ref.shape[1]
    channels = pred.shape[1]
    if w is not None:
      w = w.repeat(1, channels, 1, 1)  # repeat on channel wise dim
      n = torch.sum(loss * w, [1, 2, 3])
      d = torch.sum(w, [1, 2, 3])
      loss = n / d
    return loss


class L1_Loss(nn.Module):
  """Standard L1 loss, for each item in batch."""

  def __init__(self):
    super(L1_Loss, self).__init__()
    self.loss = nn.L1Loss(reduction='none')

  def forward(self, pred, ref):
    """ims have dimension BCHW."""
    # output = N x 1
    loss = self.loss(pred, ref)
    assert pred.shape[1] == ref.shape[1]
    loss = torch.mean(loss, dim=[1, 2, 3])[:, None]
    return loss


class Masked_MSE_Loss(nn.Module):
  """MSE loss with masking."""

  def __init__(self):
    super(Masked_MSE_Loss, self).__init__()
    self.loss = nn.MSELoss(reduction='none')

  def forward(self, pred, ref, w=None):
    """ims have dimension BCHW while mask is B1HW."""
    check_loss_input(pred, ref, w)
    loss = self.loss(pred, ref)
    assert pred.shape[1] == ref.shape[1]
    channels = pred.shape[1]
    if w is not None:
      w = w.repeat(1, channels, 1, 1)  # repeat on channel wise dim
      n = torch.sum(loss * w, [1, 2, 3])
      d = torch.sum(w, [1, 2, 3])
      loss = n / d
    return loss
