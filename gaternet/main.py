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

"""Loads a GaterNet checkpoint and tests on Cifar-10 test set."""

import argparse
import io
import os
from backbone_resnet import Network as Backbone
from gater_resnet import Gater
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms


def load_from_state(state_dict, model):
  """Loads the state dict of a checkpoint into model."""
  tem_dict = dict()
  for k in state_dict.keys():
    tem_dict[k.replace('module.', '')] = state_dict[k]
  state_dict = tem_dict

  ckpt_key = set(state_dict.keys())
  model_key = set(model.state_dict().keys())
  print('Keys not in current model: {}\n'.format(ckpt_key - model_key))
  print('Keys not in checkpoint: {}\n'.format(model_key - ckpt_key))

  model.load_state_dict(state_dict, strict=True)
  print('Successfully reload from state.')
  return model


def test(backbone, gater, device, test_loader):
  """Tests the model on a test set."""
  backbone.eval()
  gater.eval()
  loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      gate = gater(data)
      output = backbone(data, gate)
      loss += F.cross_entropy(output, target, size_average=False).item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()

  loss /= len(test_loader.dataset)
  acy = 100. * correct / len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
      loss, correct, len(test_loader.dataset), acy))

  return acy


def run(args, device, test_loader):
  """Loads checkpoint into GaterNet and runs test on the test data."""
  with open(args.checkpoint_file, 'rb') as fin:
    inbuffer = io.BytesIO(fin.read())
  state_dict = torch.load(inbuffer, map_location='cpu')
  print('Successfully load checkpoint file.\n')

  backbone = Backbone(depth=args.backbone_depth, num_classes=10)
  print('Loading checkpoint weights into backbone.')
  backbone = load_from_state(state_dict['backbone_state_dict'], backbone)
  backbone = nn.DataParallel(backbone).to(device)
  print('Backbone is ready after loading checkpoint and moving to device:')
  print(backbone)
  n_params_b = sum(
      [param.view(-1).size()[0] for param in backbone.parameters()])
  print('Number of parameters in backbone: {}\n'.format(n_params_b))

  gater = Gater(depth=20,
                bottleneck_size=8,
                gate_size=backbone.module.gate_size)
  print('Loading checkpoint weights into gater.')
  gater = load_from_state(state_dict['gater_state_dict'], gater)
  gater = nn.DataParallel(gater).to(device)
  print('Gater is ready after loading checkpoint and moving to device:')
  print(gater)
  n_params_g = sum(
      [param.view(-1).size()[0] for param in gater.parameters()])
  print('Number of parameters in gater: {}'.format(n_params_g))
  print('Total number of parameters: {}\n'.format(n_params_b + n_params_g))

  print('Running test on test data.')
  test(backbone, gater, device, test_loader)


def parse_flags():
  """Parses input arguments."""
  parser = argparse.ArgumentParser(description='GaterNet')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--backbone-depth', type=int, default=20,
                      help='resnet depth of the backbone subnetwork')
  parser.add_argument('--checkpoint-file', type=str, default=None,
                      help='checkpoint file to run test')
  parser.add_argument('--data-dir', type=str, default=None,
                      help='the directory for storing data')
  args = parser.parse_args()
  return args


def main(args):
  print('Input arguments:\n{}\n'.format(args))

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  print('use_cuda: {}'.format(use_cuda))

  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.backends.cudnn.benchmark = True
  print('device: {}'.format(device))

  if not os.path.isdir(args.data_dir):
    os.mkdir(args.data_dir)

  kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
  normalize_mean = [0.4914, 0.4822, 0.4465]
  normalize_std = [0.2470, 0.2435, 0.2616]
  test_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10(
          args.data_dir,
          train=False,
          download=True,
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(normalize_mean, normalize_std)])
          ),
      batch_size=1000, shuffle=False, drop_last=False, **kwargs)
  print('Successfully get data loader.')

  run(args, device, test_loader)


if __name__ == '__main__':
  main(parse_flags())
