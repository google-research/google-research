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
"""Evaluation."""

import argparse
import logging
import os
import random

import cv2
from dataset.scannet200_constants import CLASS_LABELS_20
from dataset.scannet200_constants import MAPPING_NUSCENES_DETAILS
from dataset.scannet200_constants import MATTERPORT_LABELS_21
from dataset.scannet200_constants import MATTERPORT_LABELS_NYU160
from dataset.scannet200_constants import MATTERPORT_LABELS_NYU40
from dataset.scannet200_constants import MATTERPORT_LABELS_NYU80
from dataset.scannet200_constants import NUSCENES_LABELS_16
from dataset.scannet200_constants import NUSCENES_LABELS_DETAILS
from dataset.scannet3dfeat import collation_fn_eval_all
from dataset.scannet3dfeat import ScanNet3DFeat
from metrics import iou
from MinkowskiEngine import SparseTensor
import numpy as np
from tool.train import get_model
import torch
from torch.backends import cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm
from util import config
from util.util import convert_labels_with_pallete
from util.util import export_pointcloud
from util.util import extract_text_feature
from util.util import get_new_pallete
from util.util import visualize_labels

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def worker_init_fn(worker_id):
  random.seed(1463 + worker_id)
  np.random.seed(1463 + worker_id)
  torch.manual_seed(1463 + worker_id)


def get_parser():
  """Argument Parser."""
  parser = argparse.ArgumentParser()
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
  if args_in.opts is not None:
    cfg = config.merge_cfg_from_list(cfg, args_in.opts)
  return cfg


def get_logger():
  logger_name = 'main-logger'
  logger_in = logging.getLogger(logger_name)
  logger_in.setLevel(logging.INFO)
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
  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
  # https://github.com/Microsoft/human-pose-estimation.pytorch/issues/8
  # https://discuss.pytorch.org/t/training-performance-degrades-with-distributeddataparallel/47152/7
  # torch.backends.cudnn.enabled = False

  if args.manual_seed is not None:
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    # cudnn.benchmark = False
    # cudnn.deterministic = True

  print(
      'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s'
      % (torch.__version__, torch.version.cuda, torch.backends.cudnn.version(),
         torch.backends.cudnn.enabled))

  if args.dist_url == 'env://' and args.world_size == -1:
    args.world_size = int(os.environ['WORLD_SIZE'])
  args.distributed = args.world_size > 1 or args.multiprocessing_distributed
  args.ngpus_per_node = len(args.test_gpu)
  if len(args.test_gpu) == 1:
    args.sync_bn = False
    args.distributed = False
    args.multiprocessing_distributed = False
    args.use_apex = False

  if not hasattr(args, 'use_shm'):
    args.use_shm = False

  main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
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
  logger = get_logger()
  logger.info(args)

  if args.distributed:
    torch.cuda.set_device(gpu)
    args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
    args.test_workers = int(args.test_workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(), device_ids=[gpu])
  else:
    model = model.cuda()

  if args.model_path is not None and os.path.isfile(args.model_path):
    logger.info('=> loading checkpoint %s', args.model_path)
    checkpoint = torch.load(
        args.model_path, map_location=lambda storage, loc: storage.cuda())
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    logger.info('=> loaded checkpoint %s (epoch %d)', args.model_path,
                checkpoint['epoch'])
  elif args.pred_type == 'fusion':
    pass
  else:
    raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

  # ####################### Data Loader ####################### #
  if not hasattr(args, 'dataset_name'):
    args.dataset_name = 'scannet'
  if not hasattr(args, 'input_color'):
    args.input_color = True
  if args.data_name == 'scannet_3d_feat' or args.data_name == 'matterport_3d_feat':
    val_data = ScanNet3DFeat(
        dataPathPrefix=args.data_root,
        voxelSize=args.voxelSize,
        split=args.split,
        aug=False,
        memCacheInit=args.use_shm,
        eval_all=True,
        identifier=6797,
        feat_type=args.feat_2d,
        val_benchmark=args.val_benchmark,
        input_color=args.input_color)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collation_fn_eval_all,
        sampler=val_sampler)

  # ####################### Test ####################### #
  dataset_name = args.data_root.split('/')[-1]
  if args.data_name == 'scannet_3d_feat' or args.data_name == 'matterport_3d_feat':
    test_feat_3d(model, val_loader, args, dataset_name)


def test_feat_3d(model,
                 val_data_loader,
                 args,
                 dataset_name='scannet_3d',
                 data_paths=None):
  """Evaluation for our method."""

  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
  torch.backends.cudnn.enabled = False

  if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder, exist_ok=True)

  # obtain the palette
  new_pallete = get_new_pallete(colormap=dataset_name[:-3])

  # obtain the CLIP feature
  if dataset_name == 'scannet_3d':
    labelset = list(CLASS_LABELS_20)
    labelset[-1] = 'other'
  elif dataset_name == 'matterport_3d':
    labelset = list(MATTERPORT_LABELS_21)
    new_pallete = get_new_pallete(colormap='matterport')
  elif 'matterport_nyu40_3d' in dataset_name:
    labelset = list(MATTERPORT_LABELS_NYU40)
    new_pallete = get_new_pallete(colormap='matterport_nyu160')
  elif 'matterport_nyu80_3d' in dataset_name:
    labelset = list(MATTERPORT_LABELS_NYU80)
    new_pallete = get_new_pallete(colormap='matterport_nyu160')
  elif 'matterport_nyu160_3d' in dataset_name:
    labelset = list(MATTERPORT_LABELS_NYU160)
    new_pallete = get_new_pallete(colormap='matterport_nyu160')
  elif 'nuscenes_3d' in dataset_name:
    labelset = list(NUSCENES_LABELS_16)
    new_pallete = get_new_pallete(colormap='nuscenes16')

  mapper = None
  if hasattr(args, 'map_nuscenes_details'):
    labelset = list(NUSCENES_LABELS_DETAILS)
    mapper = torch.tensor(MAPPING_NUSCENES_DETAILS, dtype=int)

  # args.use_img_feat = True
  if hasattr(args, 'use_img_feat') and args.use_img_feat:
    labels = labelset.copy()
    text_features = torch.load(
        'saved_img_embedding/clip_scannet_labels_internet.pt').cuda()
  else:
    text_features, labels = extract_text_feature(labelset, args)
  labels.append('unknown')
  labelset.append('unlabeled')

  mark_no_feature_to_unknown = False
  if hasattr(
      args, 'mark_no_feature_to_unknown'
  ) and args.mark_no_feature_to_unknown and args.pred_type == 'fusion':
    mark_no_feature_to_unknown = True

  with torch.no_grad():
    model.eval()
    store = 0.0
    for rep_i in range(args.test_repeats):
      preds, gts = [], []
      val_data_loader.dataset.offset = rep_i
      pbar = tqdm(total=len(val_data_loader))

      # emsemble!!!
      if rep_i > 0:
        seed = np.random.randint(10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

      if mark_no_feature_to_unknown:
        masks = []

      if args.pred_type == 'ensemble' and hasattr(
          args, 'comp_ensemble_ratio') and args.comp_ensemble_ratio:
        n_feat_2d = 0  # use predictions from 2D features
        n_feat_3d = 0  # use predictions from 3D features

      for i, (coords, feat, label, feat_3d, mask,
              inds_reverse) in enumerate(tqdm(val_data_loader)):
        sinput = SparseTensor(
            feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
        coords = coords[inds_reverse, :]
        pcl = coords[:, 1:].cpu().numpy()

        if hasattr(args, 'pred_type') and args.pred_type == 'distill':
          pred_type = 'distill'
          predictions = model(sinput, None)
          predictions = predictions[inds_reverse, :]
          pred = predictions.half() @ text_features.t()
          logits_pred = torch.max(pred, 1)[1].cpu()
        elif hasattr(args, 'pred_type') and args.pred_type == 'fusion':
          pred_type = 'fusion'
          predictions = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
          pred = predictions.half() @ text_features.t()
          logits_pred = torch.max(pred, 1)[1].detach().cpu()
        elif hasattr(args, 'pred_type') and args.pred_type == 'ensemble':
          pred_type = 'ensemble'
          feat_fuse = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
          # pred_fusion = feat_fuse.half() @ text_features.t()
          pred_fusion = (feat_fuse / (feat_fuse.norm(dim=-1, keepdim=True) +
                                      1e-5)).half() @ text_features.t()

          predictions = model(sinput, None)
          predictions = predictions[inds_reverse, :]
          # pred_distill = predictions.half() @ text_features.t()
          pred_distill = (predictions /
                          (predictions.norm(dim=-1, keepdim=True) +
                           1e-5)).half() @ text_features.t()

          if hasattr(
              args, 'save_ensemble_feature'
          ) and args.save_ensemble_feature:  # save our ensemble features
            out_path = os.path.join(
                '/home/songyou/disk3/matterport_ensemble_feat_nyu{}'.format(
                    args.classes), args.split)
            if not os.path.exists(out_path):
              os.makedirs(out_path)
            feat_ensemble = predictions.clone().half()
            mask_ = pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
            feat_ensemble[mask_] = feat_fuse[mask_]

            name = data_paths[i].split('/')[-1].split('.')[0]
            np.save(
                os.path.join(out_path, '{}.npy'.format(name)),
                feat_ensemble.cpu().numpy())
            continue

          if hasattr(args, 'comp_ensemble_ratio') and args.comp_ensemble_ratio:
            mask_ = pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
            n_feat_2d += mask_.sum()
            n_feat_3d += (~mask_).sum()

          mask_ensem = pred_distill < pred_fusion  # confidence-based ensemble
          pred = pred_distill
          pred[mask_ensem] = pred_fusion[mask_ensem]
          logits_pred = torch.max(pred, 1)[1].detach().cpu()

          # mask_ =  pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
          # logits_distill = torch.max(pred_distill, 1)[1].detach().cpu()
          # logits_pred = logits_distill.clone()
          # logits_pred[mask_] = logits_fusion[mask_]

        elif hasattr(args, 'pred_type') and args.pred_type == 'classifier':
          pred_type = 'classifier'
          feat_3d = feat_3d.cuda(non_blocking=True)
          output_3d = model(feat_3d.float())
          output_3d = output_3d[inds_reverse, :]
          logits_pred = output_3d.detach().max(1)[1].detach().cpu()
        else:
          raise NotImplementedError

        # special case for nuScenes, evaluation points are subset of input
        if 'nuscenes_3d' in dataset_name:
          label_mask = (label != 255)
          label = label[label_mask]
          # pcl = coords[label_mask, 1:].cpu().numpy()
          logits_pred = logits_pred[label_mask]
          pred = pred[label_mask]
          if hasattr(args, 'vis_pred') and args.vis_pred:
            pcl = torch.load(
                val_data_loader.dataset.data_paths[i])[0][label_mask]

        if hasattr(args, 'vis_pred') and args.vis_pred:
          if mapper is not None:
            pred_label_color = convert_labels_with_pallete(
                mapper[logits_pred].numpy(), new_pallete, is_3d=True)
            export_pointcloud(
                os.path.join(args.save_folder, '{}_{}.ply'.format(i,
                                                                  pred_type)),
                pcl,
                colors=pred_label_color)
          else:
            pred_label_color = convert_labels_with_pallete(
                logits_pred.numpy(), new_pallete, is_3d=True)
            export_pointcloud(
                os.path.join(args.save_folder, '{}_{}.ply'.format(i,
                                                                  pred_type)),
                pcl,
                colors=pred_label_color)
            visualize_labels(
                list(np.unique(logits_pred.numpy())),
                labelset,
                new_pallete,
                os.path.join(args.save_folder,
                             '{}_labels_{}.jpg'.format(i, pred_type)),
                ncol=5)

        if hasattr(args, 'vis_gt') and args.vis_gt:
          label[label == 255] = len(labelset) - 1  # for points not evaluating
          gt_label_color = convert_labels_with_pallete(
              label.cpu().numpy(), new_pallete, is_3d=True)
          export_pointcloud(
              os.path.join(args.save_folder, '{}_gt.ply'.format(i)),
              pcl,
              colors=gt_label_color)

          if 'nuscenes_3d' in dataset_name:
            all_digits = np.unique(
                np.concatenate(
                    [np.unique(mapper[logits_pred].numpy()),
                     np.unique(label)]))
            labelset = list(NUSCENES_LABELS_16)
            labelset[4] = 'construct. vehicle'
            labelset[10] = 'road'
            visualize_labels(
                list(all_digits),
                labelset,
                new_pallete,
                os.path.join(args.save_folder, '{}_label.jpg'.format(i)),
                ncol=all_digits.shape[0])

        if mark_no_feature_to_unknown:
          if 'nuscenes_3d' in dataset_name:  # special case
            masks.append(mask[inds_reverse][label_mask])
          else:
            masks.append(mask[inds_reverse])

        if args.test_repeats == 1:
          preds.append(logits_pred)  # save directly the logits
        else:
          preds.append(pred.cpu(
          ))  # only save the dot-product results, for ensemble prediction

        gts.append(label.cpu())
      gt = torch.cat(gts)
      pred = torch.cat(preds)

      if args.pred_type == 'ensemble' and hasattr(
          args, 'comp_ensemble_ratio') and args.comp_ensemble_ratio:
        n_total = gt.shape[0]
        print('Ratio using 2D predictions: ', n_feat_2d / n_total)
        print('Ratio using 3D predictions: ', n_feat_3d / n_total)

      if args.test_repeats > 1:
        pred_logit = pred.float().max(1)[1]
      else:
        pred_logit = pred

      if mapper is not None:
        pred_logit = mapper[pred_logit]

      if mark_no_feature_to_unknown:
        mask = torch.cat(masks)
        pred_logit[~mask] = 256

      stdout = False
      if args.test_repeats == 1:
        stdout = True
        iou.evaluate(
            pred_logit.numpy(), gt.numpy(), dataset=dataset_name, stdout=stdout)
      if rep_i == 0:
        np.save(os.path.join(args.save_folder, 'gt.npy'), gt.numpy())
      if args.test_repeats > 1:
        store = pred + store
        store_logit = store.float().max(1)[1]
        if mapper is not None:
          store_logit = mapper[store_logit]

        if mark_no_feature_to_unknown:
          store_logit[~mask] = 256
        iou.evaluate(
            store_logit.numpy(), gt.numpy(), stdout=True, dataset=dataset_name)
        np.save(os.path.join(args.save_folder, 'pred.npy'), store.cpu().numpy())

      pbar.close()


if __name__ == '__main__':
  main()
