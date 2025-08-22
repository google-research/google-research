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
"""Distill a model."""

import argparse
import logging
import os
import random
import time

from dataset.scannet200_constants import CLASS_LABELS_20
from dataset.scannet200_constants import MATTERPORT_LABELS_21
from dataset.scannet200_constants import NUSCENES_LABELS_16
from dataset.scannet3d import collation_fn_eval_all
from dataset.scannet3d import ScanNet3D
from dataset.scannet3dfeat import collation_fn
from dataset.scannet3dfeat import ScanNet3DFeat
from MinkowskiEngine import SparseTensor
from models.disnet import DisNet as Model
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.backends import cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm
from util import config
from util.util import AverageMeter
from util.util import convert_labels_with_pallete
from util.util import export_pointcloud
from util.util import extract_clip_feature
from util.util import get_new_pallete
from util.util import intersectionanduniongpu
from util.util import poly_learning_rate
from util.util import save_checkpoint

best_iou = 0.0


def worker_init_fn(worker_id):
  random.seed(time.time() + worker_id)


def get_parser():
  """Argument Parser."""
  parser = argparse.ArgumentParser(description='DisNet')
  parser.add_argument(
      '--config',
      type=str,
      default='config/scannet/bpnet_5cm.yaml',
      help='config file')
  parser.add_argument(
      'opts',
      help='see config/scannet/bpnet_5cm.yaml for all options',
      default=None,
      nargs=argparse.REMAINDER)
  args_in = parser.parse_args()
  assert args_in.config is not None
  cfg = config.load_cfg_from_cfg_file(args_in.config)
  if args_in.opts:
    cfg = config.merge_cfg_from_list(cfg, args_in.opts)
  os.makedirs(cfg.save_path, exist_ok=True)
  model_dir = os.path.join(cfg.save_path, 'model')
  result_dir = os.path.join(cfg.save_path, 'result')
  os.makedirs(model_dir, exist_ok=True)
  os.makedirs(result_dir, exist_ok=True)
  os.makedirs(result_dir + '/last', exist_ok=True)
  os.makedirs(result_dir + '/best', exist_ok=True)
  return cfg


def get_logger():
  logger_name = 'main-logger'
  logger_in = logging.getLogger(logger_name)
  logger_in.setLevel(logging.DEBUG)
  handler = logging.StreamHandler()
  fmt = ('[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] '
         '%(message)s')
  handler.setFormatter(logging.Formatter(fmt))
  logger_in.addHandler(handler)
  return logger_in


def main():
  args = get_parser()

  os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.train_gpu)
  cudnn.benchmark = True
  if args.manual_seed is not None:
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)

  # By default we do not use shared memory
  if not hasattr(args, 'use_shm'):
    args.use_shm = False

  print(
      'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s'
      % (torch.__version__, torch.version.cuda, torch.backends.cudnn.version(),
         torch.backends.cudnn.enabled))

  if args.dist_url == 'env://' and args.world_size == -1:
    args.world_size = int(os.environ['WORLD_SIZE'])
  args.distributed = args.world_size > 1 or args.multiprocessing_distributed
  args.ngpus_per_node = len(args.train_gpu)
  if len(args.train_gpu) == 1:
    args.sync_bn = False
    args.distributed = False
    args.multiprocessing_distributed = False
    args.use_apex = False

  main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
  global best_iou
  args = argss

  if args.distributed:
    if args.dist_url == 'env://' and args.rank == -1:
      args.rank = int(os.environ['RANK'])
    if args.multiprocessing_distributed:
      args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank)

  model = get_model(args)
  if args.sync_bn_2d:
    print('using DDP synced BN for 2D')
    model.layer0_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        model.layer0_2d)
    model.layer1_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        model.layer1_2d)
    model.layer2_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        model.layer2_2d)
    model.layer3_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        model.layer3_2d)
    model.layer4_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        model.layer4_2d)
    model.up4_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.up4_2d)
    model.delayer4_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        model.delayer4_2d)
    model.up3_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.up3_2d)
    model.delayer3_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        model.delayer3_2d)
    model.up2_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.up2_2d)
    model.delayer2_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        model.delayer2_2d)
    model.cls_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.cls_2d)

  logger = get_logger()
  writer = SummaryWriter(args.save_path)
  logger.info(args)
  logger.info('=> creating model ...')
  logger.info('Classes: %d', args.classes)

  # ####################### Optimizer ####################### #

  optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
  args.index_split = 0

  if args.distributed:
    torch.cuda.set_device(gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
    args.workers = int(args.workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(), device_ids=[gpu])
  else:
    model = model.cuda()

  criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu)

  if args.resume:
    if os.path.isfile(args.resume):
      logger.info('=> loading checkpoint %s', args.resume)
      checkpoint = torch.load(
          args.resume, map_location=lambda storage, loc: storage.cuda())
      args.start_epoch = checkpoint['epoch']
      model.load_state_dict(checkpoint['state_dict'], strict=True)
      optimizer.load_state_dict(checkpoint['optimizer'])
      best_iou = checkpoint['best_iou']
      logger.info('=> loaded checkpoint %s (epoch %d)', args.resume,
                  checkpoint['epoch'])
    else:
      logger.info('=> no checkpoint found at %s', args.resume)

  # ####################### Data Loader ####################### #
  if args.data_name == 'scannet_3d_feat':
    if not hasattr(args, 'overfit_one_scene'):
      args.overfit_one_scene = False
    if not hasattr(args, 'feat_2d'):
      args.feat_2d = 'lseg'
    if not hasattr(args, 'input_color'):
      args.input_color = True
    train_data = ScanNet3DFeat(
        dataPathPrefix=args.data_root,
        voxelSize=args.voxelSize,
        split='train',
        aug=args.aug,
        memCacheInit=args.use_shm,
        loop=args.loop,
        val_benchmark=args.val_benchmark,
        overfit=args.overfit_one_scene,
        feat_type=args.feat_2d,
        input_color=args.input_color)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collation_fn,
        worker_init_fn=worker_init_fn)
    if args.evaluate:
      val_data = ScanNet3D(
          dataPathPrefix=args.data_root,
          voxelSize=args.voxelSize,
          split='val',
          aug=False,
          memCacheInit=args.use_shm,
          eval_all=True,
          input_color=args.input_color)
      val_sampler = torch.utils.data.distributed.DistributedSampler(
          val_data) if args.distributed else None
      val_loader = torch.utils.data.DataLoader(
          val_data,
          batch_size=args.batch_size_val,
          shuffle=False,
          num_workers=args.workers,
          pin_memory=True,
          drop_last=False,
          collate_fn=collation_fn_eval_all,
          sampler=val_sampler)
  else:
    raise Exception('Dataset not supported yet', args.data_name)

  # ####################### Train ####################### #
  for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
      train_sampler.set_epoch(epoch)
      if args.evaluate:
        val_sampler.set_epoch(epoch)
    if args.data_name == 'scannet_3d_feat':
      loss_train = train_feat(train_loader, model, optimizer, args, epoch,
                              logger, writer)
    else:
      raise NotImplementedError
    epoch_log = epoch + 1
    writer.add_scalar('loss_train', loss_train, epoch_log)

    is_best = False
    if args.evaluate and (epoch_log % args.eval_freq == 0):
      loss_val, miou_val, macc_val, allacc_val = validate(
          val_loader, model, criterion, args, logger)

      writer.add_scalar('loss_val', loss_val, epoch_log)
      writer.add_scalar('mIoU_val', miou_val, epoch_log)
      writer.add_scalar('mAcc_val', macc_val, epoch_log)
      writer.add_scalar('allAcc_val', allacc_val, epoch_log)
      # remember best iou and save checkpoint
      is_best = miou_val > best_iou
      best_iou = max(best_iou, miou_val)

    if epoch_log % args.save_freq == 0:
      save_checkpoint(
          {
              'epoch': epoch_log,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'best_iou': best_iou
          }, is_best, os.path.join(args.save_path, 'model'))
  writer.close()
  logger.info('==>Training done!')
  logger.info('Best Iou: %.3f', best_iou)


def get_model(cfg):
  model = Model(cfg=cfg)
  return model


def train_feat(train_loader, model, optimizer, args, epoch, logger, writer):
  """Train the 3D network."""

  torch.backends.cudnn.enabled = True
  batch_time = AverageMeter()
  data_time = AverageMeter()

  loss_meter = AverageMeter()

  model.train()
  end = time.time()
  max_iter = args.epochs * len(train_loader)

  # obtain the CLIP feature
  if 'scannet_3d' in args.data_root:
    labelset = list(CLASS_LABELS_20)
    labelset[-1] = 'other'
    new_pallete = get_new_pallete()
  elif 'matterport_3d' in args.data_root:
    labelset = list(MATTERPORT_LABELS_21)
    new_pallete = get_new_pallete(colormap='matterport')
  elif 'nuscenes_3d' in args.data_root:
    labelset = list(NUSCENES_LABELS_16)
    new_pallete = get_new_pallete(colormap='nuscenes16')

  if not hasattr(args, 'feat_2d'):
    args.feat_2d = 'lseg'

  if 'lseg' in args.feat_2d:
    text_features, _ = extract_clip_feature(labelset)
  elif 'osegclip' in args.feat_2d and 'scannet_3d' in args.data_root:
    text_features = torch.load(
        'saved_text_embedding/clip_scannet_labels_768.pt').cuda()
  elif 'osegclip' in args.feat_2d and 'matterport_3d' in args.data_root:
    text_features = torch.load(
        'saved_text_embedding/clip_matterport_labels_768.pt').cuda()
  elif 'osegclip' in args.feat_2d and 'nuscenes_3d' in args.data_root:
    text_features = torch.load(
        'saved_text_embedding/clip_nuscenes_labels_768_tmp.pt').cuda()
  else:
    text_features, _ = extract_clip_feature(
        labelset, model_name='ViT-L/14@336px')

  # start the training process
  for i, batch_data in enumerate(train_loader):
    data_time.update(time.time() - end)

    if args.data_name == 'scannet_3d_feat':
      (coords, feat, label_3d, feat_3d, mask) = batch_data
      coords[:, 1:4] += (torch.rand(3) * 100).type_as(coords)
      sinput = SparseTensor(
          feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
      feat_3d, mask = feat_3d.cuda(non_blocking=True), mask.cuda(
          non_blocking=True)

      output_3d = model(sinput)

      output_3d = output_3d[mask]

      if hasattr(args, 'loss_type') and args.loss_type == 'cosine':
        loss = (1 - torch.nn.CosineSimilarity()(output_3d, feat_3d)).mean()
      elif hasattr(args, 'loss_type') and args.loss_type == 'l1':
        loss = torch.nn.L1Loss()(output_3d, feat_3d)
      else:
        raise NotImplementedError

    else:
      raise NotImplementedError
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_meter.update(loss.item(), args.batch_size)
    batch_time.update(time.time() - end)

    # Adjust lr
    current_iter = epoch * len(train_loader) + i + 1
    current_lr = poly_learning_rate(
        args.base_lr, current_iter, max_iter, power=args.power)
    for index in range(0, args.index_split):
      optimizer.param_groups[index]['lr'] = current_lr
    for index in range(args.index_split, len(optimizer.param_groups)):
      optimizer.param_groups[index]['lr'] = current_lr * 10

    # calculate remain time
    remain_iter = max_iter - current_iter
    remain_time = remain_iter * batch_time.avg
    t_m, t_s = divmod(remain_time, 60)
    t_h, t_m = divmod(t_m, 60)
    remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

    if (i + 1) % args.print_freq == 0:
      logger.info('Epoch: [{}/{}][{}/{}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Remain {remain_time} '
                  'Loss {loss_meter.val:.4f} '.format(
                      epoch + 1,
                      args.epochs,
                      i + 1,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      remain_time=remain_time,
                      loss_meter=loss_meter))
    writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
    writer.add_scalar('learning_rate', current_lr, current_iter)

    end = time.time()

  if args.data_name == 'scannet_3d_feat':

    mask_first = (coords[mask][:, 0] == 0)
    output_3d = output_3d[mask_first]
    feat_3d = feat_3d[mask_first]
    logits_pred = output_3d.half() @ text_features.t()
    logits_img = feat_3d.half() @ text_features.t()
    logits_pred = torch.max(logits_pred, 1)[1].cpu().numpy()
    logits_img = torch.max(logits_img, 1)[1].cpu().numpy()
    mask = mask.cpu().numpy()
    logits_gt = label_3d.numpy()[mask][mask_first.cpu().numpy()]
    logits_gt[logits_gt == 255] = args.classes

    pcl = coords[:, 1:].cpu().numpy()

    seg_label_color = convert_labels_with_pallete(
        logits_img, new_pallete, is_3d=True)
    pred_label_color = convert_labels_with_pallete(
        logits_pred, new_pallete, is_3d=True)
    gt_label_color = convert_labels_with_pallete(
        logits_gt, new_pallete, is_3d=True)
    pcl_part = pcl[mask][mask_first.cpu().numpy()]

    export_pointcloud(
        os.path.join(args.save_path, 'result', 'last',
                     '{}_{}.ply'.format(args.feat_2d, epoch)),
        pcl_part,
        colors=seg_label_color)
    export_pointcloud(
        os.path.join(args.save_path, 'result', 'last',
                     'pred_{}.ply'.format(epoch)),
        pcl_part,
        colors=pred_label_color)
    export_pointcloud(
        os.path.join(args.save_path, 'result', 'last',
                     'gt_{}.ply'.format(epoch)),
        pcl_part,
        colors=gt_label_color)
  else:
    raise NotImplementedError

  return loss_meter.avg


def validate(val_loader, model, criterion, args, logger):
  """Evaluation on validation set."""
  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
  torch.backends.cudnn.enabled = False
  loss_meter = AverageMeter()
  intersection_meter = AverageMeter()
  union_meter = AverageMeter()
  target_meter = AverageMeter()

  # obtain the CLIP feature
  if 'scannet_3d' in args.data_root:
    labelset = list(CLASS_LABELS_20)
    labelset[-1] = 'other'
  elif 'matterport_3d' in args.data_root:
    labelset = list(MATTERPORT_LABELS_21)
  elif 'nuscenes_3d' in args.data_root:
    labelset = list(NUSCENES_LABELS_16)
  else:
    labelset = None

  if not hasattr(args, 'feat_2d') or 'lseg' in args.feat_2d:
    text_features, _ = extract_clip_feature(labelset)
  elif 'osegclip' in args.feat_2d and 'scannet_3d' in args.data_root:
    text_features = torch.load(
        'saved_text_embedding/clip_scannet_labels_768.pt').cuda()
  elif 'osegclip' in args.feat_2d and 'matterport_3d' in args.data_root:
    text_features = torch.load(
        'saved_text_embedding/clip_matterport_labels_768.pt').cuda()
  elif 'osegclip' in args.feat_2d and 'nuscenes_3d' in args.data_root:
    text_features = torch.load(
        'saved_text_embedding/clip_nuscenes_labels_768_tmp.pt').cuda()

  model.eval()
  with torch.no_grad():
    for batch_data in tqdm(val_loader):
      (coords, feat, label, inds_reverse) = batch_data
      sinput = SparseTensor(
          feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
      label = label.cuda(non_blocking=True)
      output = model(sinput)
      output = output[inds_reverse, :]
      output = output.half() @ text_features.t()
      loss = criterion(output, label)
      output = torch.max(output, 1)[1]

      intersection, union, target = intersectionanduniongpu(
          output, label.detach(), args.classes, args.ignore_label)
      intersection = intersection.cpu().numpy()
      union = union.cpu().numpy()
      target = target.cpu().numpy()
      intersection_meter.update(intersection)
      union_meter.update(union)
      target_meter.update(target)
      loss_meter.update(loss.item(), args.batch_size)

  iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
  accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
  miou = np.mean(iou_class)
  macc = np.mean(accuracy_class)
  allacc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
  logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
      miou, macc, allacc))
  return loss_meter.avg, miou, macc, allacc


if __name__ == '__main__':
  main()
