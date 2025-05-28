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

"""Supervised Contrastive Loss [1] and ResNet model.

[1] Khosla, Prannay, et al. "Supervised contrastive learning." Advances in
neural information processing systems 33 (2020): 18661-18673.
"""

from typing import List
import torch
import torch.nn.functional as F

nn = torch.nn


class SupConLoss(nn.Module):
  """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

  def __init__(
      self, temperature=0.07, contrast_mode='all', base_temperature=0.07
  ):
    super(SupConLoss, self).__init__()
    self.temperature = temperature
    self.contrast_mode = contrast_mode
    self.base_temperature = base_temperature

  def forward(self, features, labels=None, weights=None, mask=None):
    device = torch.device('cuda') if features.is_cuda else torch.device('cpu')

    if len(features.shape) < 3:
      raise ValueError(
          '`features` needs to be [bsz, n_views, ...],'
          'at least 3 dimensions are required'
      )
    if len(features.shape) > 3:
      features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
      raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
      mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
      labels = labels.contiguous().view(-1, 1)
      if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
      mask = torch.eq(labels, labels.T).float().to(device)
    else:
      mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if self.contrast_mode == 'one':
      anchor_feature = features[:, 0]
      anchor_count = 1
    elif self.contrast_mode == 'all':
      anchor_feature = contrast_feature
      anchor_count = contrast_count
    else:
      raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), self.temperature
    )
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0,
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
    if weights is not None:
      loss = (
          loss.view(anchor_count, batch_size)
          * weights.view(1, -1)
          / weights.sum()
      )
      loss = loss.sum()
    else:
      loss = loss.view(anchor_count, batch_size).mean()
    return loss


criterion = SupConLoss()


def conv3x3(in_planes, out_planes, stride = 1):
  """Instantiates a 3x3 convolutional layer with no bias.

  Args:
    in_planes: number of input channels
    out_planes: number of output channels
    stride: stride of the convolution

  Returns:
    convolutional layer
  """
  return nn.Conv2d(
      in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
  )


class BasicBlock(nn.Module):
  """The basic block of ResNet."""

  expansion = 1

  def __init__(self, in_planes, planes, stride = 1):
    """Instantiates the basic block of the network.

    Args:
      in_planes: the number of input channels
      planes: the number of channels (to be possibly expanded)
      stride: stride of the convolution

    Returns:
      resnet basic block
    """
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(in_planes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_planes,
              self.expansion * planes,
              kernel_size=1,
              stride=stride,
              bias=False,
          ),
          nn.BatchNorm2d(self.expansion * planes),
      )

  def forward(self, x):
    """Compute a forward pass.

    Args:
      x: input tensor (batch_size, input_size)

    Returns:
      output tensor (10)
    """
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


def off_diagonal(x):
  # return a flattened view of the off-diagonal elements of a square matrix
  n, m = x.shape
  assert n == m
  return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class ResNet(nn.Module):
  """ResNet network architecture. Designed for complex datasets."""

  def __init__(
      self, block, num_blocks, num_classes, nf
  ):
    """Instantiates the layers of the network.

    Args:
      block: the basic ResNet block
      num_blocks: the number of blocks per layer
      num_classes: the number of output classes
      nf: the number of filters
    """
    super(ResNet, self).__init__()
    self.in_planes = nf
    self.block = block
    self.num_classes = num_classes
    self.nf = nf
    self.embdim = self.nf * 8 * block.expansion
    self.conv1 = conv3x3(3, nf * 1)
    self.bn1 = nn.BatchNorm2d(nf * 1)
    self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
    self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    sizes = [512, 64]
    projector_layers = []
    for i in range(len(sizes) - 2):
      projector_layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
      projector_layers.append(nn.BatchNorm1d(sizes[i + 1]))
      projector_layers.append(nn.F.relu(inplace=True))
    projector_layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    self.projector = nn.Sequential(*projector_layers)

    self.contrastive_bn = nn.BatchNorm1d(sizes[-1], affine=False)

    self._features = nn.Sequential(
        self.conv1,
        self.bn1,
        nn.F.relu(),
        self.layer1,
        self.layer2,
        self.layer3,
        self.layer4,
    )
    self.classifier = self.linear

  def _make_layer(
      self, block, planes, num_blocks, stride
  ):
    """Instantiates a ResNet layer.

    Args:
      block: ResNet basic block
      planes: channels across the network
      num_blocks: number of blocks
      stride: stride

    Returns:
      ResNetlayer
    """
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(
      self, x, last = False, freeze = False
  ):
    """Compute a forward pass.

    Args:
      x: input tensor
      last: if True, returns second-last layer embeddings along with model
        outputs
      freeze: if True, gradient computation is kept off

    Returns:
      output tensor
    """
    if freeze:
      with torch.no_grad():
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 64, 32, 32
        out = self.layer2(out)  # 128, 16, 16
        out = self.layer3(out)  # 256, 8, 8
        out = self.layer4(out)  # 512, 4, 4
        out = F.avg_pool2d(out, out.shape[2])  # 512, 1, 1
        emb = out.view(out.size(0), -1)  # 512
    else:
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.layer1(out)  # 64, 32, 32
      out = self.layer2(out)  # 128, 16, 16
      out = self.layer3(out)  # 256, 8, 8
      out = self.layer4(out)  # 512, 4, 4
      out = F.avg_pool2d(out, out.shape[2])  # 512, 1, 1
      emb = out.view(out.size(0), -1)  # 512
    out = self.linear(emb)

    if last:
      return out, emb
    else:
      return out

  def contrastive_forward(self, x1, x2, labels, weights=None):
    x = torch.cat([x1, x2], dim=0)
    bsz = x1.shape[0]
    z = F.normalize(self.linear(self.features(x)), dim=1)
    f1, f2 = torch.split(z, [bsz, bsz], dim=0)
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    loss = criterion(features, labels, weights)
    return loss

  def get_embedding_dim(self):
    return self.embdim

  def features(self, x):
    """Returns the non-activated output of the second-last layer.

    Args:
      x: input tensor

    Returns
      feature embeddings from second-last layer.
    """
    out = self._features(x)
    out = F.avg_pool2d(out, out.shape[2])
    feat = out.view(out.size(0), -1)
    return feat

  def get_params(self):
    """Returns all the parameters concatenated in a single tensor."""
    params = []
    for pp in list(self.parameters()):
      params.append(pp.view(-1))
    return torch.cat(params)

  def set_params(self, new_params):
    """Sets the parameters to a given value."""
    assert new_params.size() == self.get_params().size()
    progress = 0
    for pp in list(self.parameters()):
      cand_params = new_params[
          progress : progress + torch.tensor(pp.size()).prod()
      ].view(pp.size())
      progress += torch.tensor(pp.size()).prod()
      pp.data = cand_params

  def get_grads(self):
    """Returns all the gradients concatenated in a single tensor."""
    grads = []
    for pp in list(self.parameters()):
      grads.append(pp.grad.view(-1))
    return torch.cat(grads)


def resnet18(nclasses, nf = 64):
  """Instantiates a ResNet18 network.

  Args:
    nclasses: number of output classes
    nf: number of filters

  Returns:
    ResNet network
  """
  return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)
