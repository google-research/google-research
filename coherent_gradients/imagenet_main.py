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

# Lint as: python3
"""Imagenet trainer.

This trainer is heavily based on PyTorch Imagenet trainer example that can be
found here:
https://github.com/pytorch/examples/tree/master/imagenet (synced to commit
ee964a2eeb41e1712fe719b83645c79bcbd0ba1a)

There are 3 main differences:
1) It allows you to use RM3 (median of 3) optimizer instead of SGD
2) It allows you to "corrupt" (i.e. generate random labels for) some portion of
the training dataset
3) It produces CSV files containing accuracy and loss stats after every epoch

Example usage (50% of data corrupted, 50000 example in pristine/corrupt
validation sets, no momentum, RM3 optimizer):
  python -m coherent_gradients.imagenet_main /imagenet --corrupt-fraction=0.5
  --output-dir=/tmp/imagenet_training --sample-size=50000 --momentum=0.0
  --optimizer=RM3
"""


import argparse
import csv
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from coherent_gradients.datasets.dataset_with_corruption import VisionDatasetWithCorruption
from coherent_gradients.datasets.dataset_with_indices import VisionDatasetWithIndices
from coherent_gradients.datasets.dataset_with_selection import VisionDatasetWithSelection
from coherent_gradients.optimizers.rm3 import RM3

model_names = ['resnet18', 'vgg13_bn', 'inception_v3']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:12345', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--test-imagenet-dir', default=None, type=str,
                    help='If test dataset should come from '
                         'a different instance of Imagenet dataset '
                         'it should be specified here. '
                         'See README file for more details')
parser.add_argument('--corrupt-fraction', default=0.0, type=float,
                    help='Fraction of training data to corrupt')
parser.add_argument('--sample-size', default=50000, type=int,
                    help='Size of pristine/corrupt validation sets')
parser.add_argument('--output-dir', default='.', type=str,
                    help='Where should we put csv files with stats')
parser.add_argument('--random-augmentations', default=False,
                    action='store_true',
                    help='Random augmentations of training data')
parser.add_argument('--optimizer', default='SGD',
                    help='Optimizer type (either SGD or RM3)')
parser.add_argument('--store-learned', default=False, action='store_true',
                    help='Store list of learned examples after the training')

best_acc1 = 0


def prepare_environment(args):
  """Prepare directories for training."""
  # generate numpy seed for pristine/corrupt
  args.np_seed = args.seed if args.seed is not None else np.random.randint(
      100000)
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  env_file_name = os.path.join(args.output_dir, 'env.txt')

  # Queues for logging training and validation stats from GPUs
  # It maps queue name to a pair of (csv_file_location, queue)
  args.log_queues = {}
  mp_context = mp.get_context('spawn')

  # Log the arguments from main
  with open(env_file_name, 'w+') as env_file:
    env_file.write('Command line arguments: {}\n'.format(args))
    env_file.write('Numpy seed: {}\n'.format(args.np_seed))

  args.log_queues['train'] = mp.SimpleQueue()
  with open(os.path.join(args.output_dir, 'train.csv'), 'w+') as train_file:
    train_file.write('epoch,step,count,loss\n')
  args.log_queues['train'] = (mp_context.SimpleQueue(), train_file.name)

  for queue_name in ('pristine', 'corrupt', 'test', 'train_sample'):
    with open(os.path.join(args.output_dir, queue_name + '.csv'),
              'w+') as samples_file:
      samples_file.write('epoch,count,loss,acc1,acc5\n')
    args.log_queues[queue_name] = (mp_context.SimpleQueue(), samples_file.name)

  for queue_name in ('learned', 'not_learned'):
    with open(
        os.path.join(args.output_dir, queue_name + '_examples.csv'),
        'w+') as learned_file:
      learned_file.write('index\n')
    args.log_queues[queue_name] = (mp_context.SimpleQueue(), learned_file.name)


def main():
  args = parser.parse_args()
  prepare_environment(args)

  if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

  ngpus = torch.cuda.device_count()
  # Use torch.multiprocessing.spawn to launch distributed processes: the
  # main_worker process function
  mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args))
  flush_all_queues(args.log_queues)


def get_model(arch):
  """Default model is Resnet-18, but InceptionV3 and VGG-13 are also available.

  We don't use auxiliary logits in InceptionV3 model for the sake
  of code simplicity. If you want to use auxiliary logits,
  please update the loss function below as well as model calls
  (to handle additional model output) in train() and validate() function.

  Args:
    arch: network architecture

  Returns:
    model
  """
  return models.__dict__[arch](
      aux_logits=False) if arch == 'inception_v3' else models.__dict__[arch]()


def get_lr_scheduler(arch, optimizer, max_epochs):
  if arch == 'resnet18':
    return StepLR(optimizer, step_size=30, gamma=0.1)
  return CosineAnnealingLR(optimizer, max_epochs)


def main_worker(gpu, ngpus, args):
  global best_acc1
  args.gpu = gpu
  print('Use GPU: {} for training'.format(args.gpu))

  dist.init_process_group(
      backend='nccl', init_method=args.dist_url, world_size=ngpus, rank=gpu)
  # create model
  model = get_model(args.arch)
  torch.cuda.set_device(args.gpu)
  model.cuda(args.gpu)
  args.batch_size = int(args.batch_size / ngpus)
  args.workers = int((args.workers + ngpus - 1) / ngpus)
  model = torch.nn.parallel.DistributedDataParallel(
      model, device_ids=[args.gpu])

  # define loss function (criterion) and optimizer
  criterion = nn.CrossEntropyLoss().cuda(args.gpu)

  if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
  elif args.optimizer == 'RM3':
    if args.momentum != 0:
      raise ValueError(
          'Momentum causes instability with RM3, current momentum value: {}'
          .format(args.momentum))
    optimizer = RM3(model.parameters(), args.lr, weight_decay=args.weight_decay)
  else:
    raise ValueError('Unknown optimizer')

  lr_scheduler = get_lr_scheduler(args.arch, optimizer, args.epochs)

  # optionally resume from a checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      if args.gpu is None:
        checkpoint = torch.load(args.resume)
      else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
      # we resume training from epoch specified
      # in command line rather than the one in checkpoint to support
      # adversarial initialization (want correct initial learning rate)
      # args.start_epoch = checkpoint['epoch']
      best_acc1 = checkpoint['best_acc1']
      if args.gpu is not None:
        # best_acc1 may be from a checkpoint from a different GPU
        best_acc1 = best_acc1.to(args.gpu)
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print('=> loaded checkpoint {} (taken at epoch {})'
            'but starting training from epoch {}'
            .format(args.resume, checkpoint['epoch'], args.start_epoch))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  cudnn.benchmark = True

  # Data loading code
  traindir = os.path.join(args.data, 'train')
  testdir = os.path.join(
      args.data, 'val') if args.test_imagenet_dir is None else os.path.join(
          args.test_imagenet_dir, 'val')

  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  input_resize = {
      'resnet18': 256,
      'vgg13_bn': 256,
      'inception_v3': 299
  }[args.arch]
  crop_size = {'resnet18': 224, 'vgg13_bn': 224, 'inception_v3': 299}[args.arch]

  if args.random_augmentations:
    training_transforms = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
  else:
    training_transforms = transforms.Compose([
        transforms.Resize(input_resize),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
  test_transforms = transforms.Compose([
      transforms.Resize(input_resize),
      transforms.CenterCrop(crop_size),
      transforms.ToTensor(),
      normalize,
  ])

  original_train_dataset = datasets.ImageFolder(traindir, training_transforms)
  # Corrupt the train dataset
  rs = np.random.RandomState(seed=args.np_seed)
  n_classes = 1000
  train_dataset = VisionDatasetWithCorruption(original_train_dataset, n_classes,
                                              args.corrupt_fraction, rs)

  pristine_indices = []
  corrupt_indices = []

  perm = rs.permutation(len(train_dataset))

  for i in perm:
    if train_dataset.is_corrupt(i) and len(corrupt_indices) < args.sample_size:
      corrupt_indices.append(i)
    elif not train_dataset.is_corrupt(
        i) and len(pristine_indices) < args.sample_size:
      pristine_indices.append(i)

  mixed_indices = perm[-args.sample_size:] if args.sample_size > 0 else []

  pristine_dataset = VisionDatasetWithSelection(train_dataset, pristine_indices)
  corrupt_dataset = VisionDatasetWithSelection(train_dataset, corrupt_indices)
  train_sample_dataset = VisionDatasetWithSelection(train_dataset,
                                                    mixed_indices)
  test_dataset = datasets.ImageFolder(testdir, test_transforms)

  # Training dataset loading
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True, sampler=train_sampler)

  validation_loaders = []
  for (name, dataset,
       queue_name) in [('Pristine', pristine_dataset, 'pristine'),
                       ('Corrupt', corrupt_dataset, 'corrupt'),
                       ('Train Sample', train_sample_dataset, 'train_sample'),
                       ('Test', test_dataset, 'test')]:
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=sampler)
    validation_loaders.append((name, loader, queue_name))

  if gpu == 0:
    save_checkpoint({
        'epoch': 0,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict()
    }, False, args.output_dir, filename='initial_state.pth.tar')

  for epoch in range(args.start_epoch, args.epochs):
    for (name, loader, queue_name) in validation_loaders:
      acc1 = validate(name, queue_name, epoch, loader, model, criterion, args)
      if name == 'Test':
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if gpu == 0:
          save_checkpoint(
              {
                  'epoch': epoch + 1,
                  'arch': args.arch,
                  'state_dict': model.state_dict(),
                  'best_acc1': best_acc1,
                  'optimizer': optimizer.state_dict(),
              }, is_best, args.output_dir)

    train_sampler.set_epoch(epoch)
    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, args)
    lr_scheduler.step()

  print('Running eval AFTER the last epoch of training')
  for (name, loader, csv_file) in validation_loaders:
    validate(name, csv_file, epoch+1, loader, model, criterion, args)

  if args.store_learned:
    store_learned_examples(train_dataset, model, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
  """Trains the model for one epoch."""
  batch_time = AverageMeter('Time', ':6.3f')
  data_time = AverageMeter('Data', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(train_loader), [batch_time, data_time, losses, top1, top5],
      prefix='Epoch: [{}]'.format(epoch))

  # switch to train mode
  model.train()

  end = time.time()
  for i, (images, target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    if args.gpu is not None:
      images = images.cuda(args.gpu, non_blocking=True)
    target = target.cuda(args.gpu, non_blocking=True)

    # compute output
    output = model(images)
    loss = criterion(output, target)

    # measure accuracy and record loss
    # pylint: disable=unbalanced-tuple-unpacking
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    loss_value = loss.item()
    losses.update(loss_value, images.size(0))
    top1.update(acc1[0], images.size(0))
    top5.update(acc5[0], images.size(0))

    optimizer.zero_grad()
    loss.backward()
    # This is either SGD or RM3 step
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.print_freq == 0:
      progress.display(i)
      args.log_queues['train'][0].put((epoch, i, images.size(0), loss_value))
      if args.gpu == 0:
        write_from_queue(args.log_queues['train'], 10)


def validate(name, queue_name, epoch, test_loader, model, criterion, args):
  """Runs validation on a given dataset."""
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(test_loader), [batch_time, losses, top1, top5],
      prefix='{}: '.format(name))

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (images, target) in enumerate(test_loader):
      if args.gpu is not None:
        images = images.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)
      # compute output
      output = model(images)
      loss = criterion(output, target)

      # measure accuracy and record loss
      # pylint: disable=unbalanced-tuple-unpacking
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % args.print_freq == 0:
        progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
        top1=top1, top5=top5))
    if losses.count > 0:
      args.log_queues[queue_name][0].put(
          (epoch, losses.count, losses.avg, top1.avg.item(), top5.avg.item()))
    if args.gpu == 0:
      write_from_queue(args.log_queues[queue_name], 10)

  return top1.avg


def store_learned_examples(train_dataset, model, args):
  """Stores example that models predicts correctly.

  Go over training dataset one more time and remember which examples
  were predicted correctly ("learned.csv")
  and which were not ("not_learned.csv").

  Args:
    train_dataset: dataset to compute learned examples from
    model: model
    args: command line args for the trainer
  """
  train_dataset_with_indices = VisionDatasetWithIndices(train_dataset)

  sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset_with_indices)
  loader = torch.utils.data.DataLoader(
      train_dataset_with_indices, batch_size=args.batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True, sampler=sampler)

  # switch to evaluate mode
  model.eval()

  print('Finding learned/not learned examples in the training dataset.'
        'This may take a while.')
  with torch.no_grad():
    for _, (images, target, indices) in enumerate(loader):
      if args.gpu is not None:
        images = images.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)
      # compute output
      output = model(images)

      # measure accuracy and record loss
      learned, not_learned = get_learned_examples(output, target, indices)

      for idx in learned:
        args.log_queues['learned'][0].put([idx.item()])

      for idx in not_learned:
        args.log_queues['not_learned'][0].put([idx.item()])

      if args.gpu == 0:
        write_from_queue(args.log_queues['learned'], 256)
        write_from_queue(args.log_queues['not_learned'], 256)
  print('All examples have been classified as either learned or not-learned.')


def flush_all_queues(queues):
  print('Writing all remaining stats to files...')
  for queue_name in queues:
    write_from_queue(queues[queue_name], None)


def write_from_queue(queue_data, limit):
  q, csv_file = queue_data
  with open(csv_file, 'a') as f:
    writer = csv.writer(f)
    while not q.empty() and (limit is None or limit > 0):
      element = q.get()
      if limit is not None:
        limit -= 1
      writer.writerow(element)


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth.tar'):
  file_path = os.path.join(output_dir, filename)
  torch.save(state, file_path)
  if is_best:
    shutil.copyfile(file_path, os.path.join(output_dir, 'model_best.pth.tar'))


class AverageMeter(object):
  """Computes and stores the average and current value."""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
  """Displays training and validation progress."""

  def __init__(self, num_batches, meters, prefix=''):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_learned_examples(output, target, indices):
  predicted = output.argmax(dim=1)
  learned = indices[predicted == target]
  not_learned = indices[predicted != target]
  return learned, not_learned


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k."""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
  main()
