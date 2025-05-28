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

"""Evaluate CaR on segmentation benchmarks."""
# pylint: disable=g-importing-member
import argparse
import numpy as np
import torch
from torch.utils import tensorboard
import torch.utils.data
from torch.utils.data import Subset
import torchvision.transforms as T

# pylint: disable=g-bad-import-order
from modeling.model.car import CaR
from sam.utils import build_sam_config
from utils.utils import Config
from utils.utils import load_yaml
from utils.utils import MetricLogger
from utils.utils import SmoothedValue
from utils.inference_pipeline import inference_car
from utils.merge_mask import merge_masks_simple

# Datasets
# pylint: disable=g-multiple-import
from data.ade import ADE_THING_CLASS, ADE_STUFF_CLASS, ADE_THING_CLASS_ID, ADE_STUFF_CLASS_ID, ADEDataset
from data.ade847 import ADE_847_THING_CLASS_ID, ADE_847_STUFF_CLASS_ID, ADE_847_THING_CLASS, ADE_847_STUFF_CLASS, ADE847Dataset
from data.coco import COCO_OBJECT_CLASSES, COCODataset
from data.context import PASCAL_CONTEXT_STUFF_CLASS_ID, PASCAL_CONTEXT_THING_CLASS_ID, PASCAL_CONTEXT_STUFF_CLASS, PASCAL_CONTEXT_THING_CLASS, CONTEXTDataset
from data.gres import GReferDataset
from data.pascal459 import PASCAL_459_THING_CLASS_ID, PASCAL_459_STUFF_CLASS_ID, PASCAL_459_THING_CLASS, PASCAL_459_STUFF_CLASS, Pascal459Dataset
from data.refcoco import ReferDataset
from data.voc import VOC_CLASSES, VOCDataset


IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512

# set random seed
torch.manual_seed(0)
np.random.seed(0)


def get_dataset(cfg, ds_name, split, transform, data_root=None):
  """Get dataset."""
  data_args = dict(root=data_root) if data_root is not None else {}
  if 'refcoco' in ds_name:
    splitby = cfg.test.splitby if hasattr(cfg.test, 'splitby') else 'unc'
    ds = ReferDataset(
        dataset=ds_name,
        splitBy=splitby,
        split=split,
        image_transforms=transform,
        target_transforms=transform,
        eval_mode=True,
        prompts_augment=cfg.test.prompts_augment,
        **data_args,
    )
  elif ds_name == 'gres':
    ds = GReferDataset(split=split, transform=transform, **data_args)
  elif ds_name == 'voc':
    ds = VOCDataset(
        year='2012',
        split=split,
        transform=transform,
        target_transform=transform,
        **data_args,
    )

  elif ds_name == 'coco':
    ds = COCODataset(transform=transform, **data_args)

  elif ds_name == 'context':
    ds = CONTEXTDataset(
        year='2010', transform=transform, split=split, **data_args
    )
  elif ds_name == 'ade':
    ds = ADEDataset(split=split, transform=transform, **data_args)
  elif ds_name == 'pascal_459':
    ds = Pascal459Dataset(split=split, transform=transform, **data_args)
  elif ds_name == 'ade_847':
    ds = ADE847Dataset(split=split, transform=transform, **data_args)
  else:
    raise ValueError(f'Dataset {ds_name} not implemented')
  return ds


def get_transform():
  transforms = [
      T.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
      T.ToTensor(),
  ]

  return T.Compose(transforms)


def assign_label(
    all_masks,
    scores,
    stuff_masks=None,
    stuff_scores=None,
    id_mapping=None,
    stuff_id_mapping=None,
):
  """Assign labels."""
  label_preds = np.zeros_like(all_masks[0]).astype(np.int32)
  if stuff_masks is not None:
    sorted_idxs = np.argsort(stuff_scores.detach().cpu().numpy())
    stuff_masks = stuff_masks[sorted_idxs]
    stuff_scores = stuff_scores.detach().cpu().numpy()[sorted_idxs]
    for sorted_idx, mask, score in zip(sorted_idxs, stuff_masks, stuff_scores):
      if score > 0:
        # convert mask to boolean
        mask = mask > 0.5
        # assign label
        if stuff_id_mapping is not None:
          label_preds[mask] = stuff_id_mapping[sorted_idx] + 1
        else:
          label_preds[mask] = sorted_idx + 1
  sorted_idxs = np.argsort(scores.detach().cpu().numpy())
  all_masks = all_masks[sorted_idxs]
  scores = scores.detach().cpu().numpy()[sorted_idxs]
  for sorted_idx, mask, score in zip(sorted_idxs, all_masks, scores):
    if score > 0:
      # convert mask to boolean
      mask = mask > 0.5
      # assign label
      if id_mapping is not None:
        label_preds[mask] = id_mapping[sorted_idx] + 1
      else:
        label_preds[mask] = sorted_idx + 1

  return label_preds


def eval_semantic(
    label_space,
    algo,
    cfg,
    model,
    image_path,
    stuff_label_space=None,
    sam_pipeline=None,
):
  """Semantic segmentation evaluation."""

  if label_space is None:
    raise ValueError(
        'label_space must be provided for semantic segmentation evaluation'
    )
  if algo == 'car':
    all_masks, scores = inference_car(
        cfg, model, image_path, label_space, sam_pipeline=sam_pipeline
    )
    if stuff_label_space is not None:
      if cfg.test.ds_name == 'context':
        thing_id_mapping = PASCAL_CONTEXT_THING_CLASS_ID
        stuff_id_mapping = PASCAL_CONTEXT_STUFF_CLASS_ID
      elif cfg.test.ds_name == 'ade':
        thing_id_mapping = ADE_THING_CLASS_ID
        stuff_id_mapping = ADE_STUFF_CLASS_ID
      elif cfg.test.ds_name == 'pascal_459':
        thing_id_mapping = PASCAL_459_THING_CLASS_ID
        stuff_id_mapping = PASCAL_459_STUFF_CLASS_ID
      elif cfg.test.ds_name == 'ade_847':
        thing_id_mapping = ADE_847_THING_CLASS_ID
        stuff_id_mapping = ADE_847_STUFF_CLASS_ID
      else:
        raise ValueError(f'Dataset {cfg.test.ds_name} not supported')

      model.mask_generator.set_bg_cls(label_space)
      model.set_visual_prompt_type(cfg.car.stuff_visual_prompt_type)
      model.set_bg_factor(cfg.car.stuff_bg_factor)
      stuff_masks, stuff_scores = inference_car(
          cfg, model, image_path, stuff_label_space, sam_pipeline=sam_pipeline
      )
      model.mask_generator.set_bg_cls(cfg.car.bg_cls)
      model.set_visual_prompt_type(cfg.car.visual_prompt_type)
      model.set_bg_factor(cfg.car.bg_factor)
      all_masks = all_masks.detach().cpu().numpy()
      stuff_masks = stuff_masks.detach().cpu().numpy()
      label_preds = assign_label(
          all_masks,
          scores,
          stuff_masks=stuff_masks,
          stuff_scores=stuff_scores,
          id_mapping=thing_id_mapping,
          stuff_id_mapping=stuff_id_mapping,
      )
    else:
      all_masks = all_masks.detach().cpu().numpy()
      label_preds = assign_label(all_masks, scores)
    return label_preds.squeeze()
  else:
    raise NotImplementedError(f'algo {algo} not implemented')


def _fast_hist(label_true, label_pred, n_class=21):
  mask = (label_true >= 0) & (label_true < n_class)
  hist = np.bincount(
      n_class * label_true[mask].astype(int) + label_pred[mask],
      minlength=n_class**2,
  ).reshape(n_class, n_class)
  return hist


def semantic_iou(label_trues, label_preds, n_class=21, ignore_background=False):
  """Semantic segmentation IOU."""

  hist = np.zeros((n_class, n_class))
  for lt, lp in zip(label_trues, label_preds):
    hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
  if ignore_background:
    hist = hist[1:, 1:]
  acc = np.diag(hist).sum() / hist.sum()
  acc_cls = np.diag(hist) / hist.sum(axis=1)
  acc_cls = np.nanmean(acc_cls)
  iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
  valid = hist.sum(axis=1) > 0  # added
  if valid.sum() == 0:
    mean_iu = 0
  else:
    mean_iu = np.nanmean(iu[valid])
  freq = hist.sum(axis=1) / hist.sum()
  fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
  if ignore_background:
    cls_iu = dict(zip(range(1, n_class), iu))
  else:
    cls_iu = dict(zip(range(n_class), iu))

  return {
      'Pixel Accuracy': acc,
      'Mean Accuracy': acc_cls,
      'Frequency Weighted IoU': fwavacc,
      'mIoU': mean_iu,
      'Class IoU': cls_iu,
  }


def evaluate(
    data_loader,
    cfg,
    model,
    test_cfg,
    label_space=None,
    stuff_label_space=None,
    sam_pipeline=None,
):
  """Run evaluation."""

  if (
      test_cfg.ds_name
      not in ['voc', 'coco', 'context', 'ade', 'pascal_459', 'ade_847']
      and test_cfg.seg_mode == 'semantic'
  ):
    raise ValueError((
        'Semantic segmentation evaluation is only implemented for voc, '
        'context, coco object, ade, pascal459, ade847 dataset'
    ))

  metric_logger = MetricLogger(delimiter='  ')
  metric_logger.add_meter(
      'mIoU', SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})')
  )
  # evaluation variables
  cum_i, cum_u = 0, 0
  eval_seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]
  seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
  seg_total = 0
  mean_iou = []
  header = 'Test:'

  # all_masks = []
  label_preds, label_gts = [], []
  print(len(data_loader))
  cc = 0
  use_tensorboard = False
  if hasattr(cfg.test, 'use_tensorboard'):
    use_tensorboard = cfg.test.use_tensorboard

  if use_tensorboard:
    writer = tensorboard.SummaryWriter(log_dir=cfg.test.output_path)
  for data in metric_logger.log_every(data_loader, 10, header):
    _, image_paths, target_list, sentences_list = data
    # print(type(target_lis))

    if not isinstance(target_list, list):
      target_list, sentences_list = [target_list], [sentences_list]
    for target, sentences in zip(target_list, sentences_list):
      image_path = image_paths[0]
      # print(image_path)
      if test_cfg.seg_mode == 'refer':
        all_masks, all_scores = inference_car(
            cfg, model, image_path, sentences, sam_pipeline=sam_pipeline
        )
        # final_mask = merge_masks(all_masks, *target.shape[1:])
        final_mask = merge_masks_simple(
            all_masks, *target.shape[1:], scores=all_scores
        )
        intersection, union, cur_iou = compute_iou(final_mask, target)
        # cur_iou = IoU(final_mask, target, 0)
        metric_logger.update(mIoU=cur_iou)
        mean_iou.append(cur_iou)
        if use_tensorboard:
          writer.add_scalar('Mean IoU', cur_iou, cc)
        cum_i += intersection
        cum_u += union
        for n_eval_iou in range(len(eval_seg_iou_list)):
          eval_seg_iou = eval_seg_iou_list[n_eval_iou]
          seg_correct[n_eval_iou] += cur_iou >= eval_seg_iou
        seg_total += 1
      elif test_cfg.seg_mode == 'semantic':
        # torch.cuda.empty_cache()
        label_pred = eval_semantic(
            label_space,
            test_cfg.algo,
            cfg,
            model,
            image_path,
            stuff_label_space,
        )
        label_gt = target.squeeze().cpu().numpy()
        cur_iou = semantic_iou(
            [label_gt],
            [label_pred],
            n_class=cfg.test.n_class,
            ignore_background=cfg.test.ignore_background,
        )['mIoU']
        metric_logger.update(mIoU=cur_iou)
        label_preds.append(label_pred)
        label_gts.append(label_gt)

    cc += 1

  if test_cfg.seg_mode == 'refer':
    mean_iou = np.array(mean_iou)
    m_iou = np.mean(mean_iou)
    if use_tensorboard:
      writer.add_scalar('mIoU', m_iou.item(), len(data_loader))
    print('Final results:')
    print('Mean IoU is %.2f\n' % (m_iou * 100.0))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
      results_str += '    precision@%s = %.2f\n' % (
          str(eval_seg_iou_list[n_eval_iou]),
          seg_correct[n_eval_iou] * 100.0 / seg_total,
      )
    o_iou = cum_i * 100.0 / cum_u
    results_str += '    overall IoU = %.2f\n' % o_iou
    if use_tensorboard:
      writer.add_scalar('oIoU', o_iou, 0)
    print(results_str)
  elif test_cfg.seg_mode == 'semantic':
    iou_score = semantic_iou(
        label_gts,
        label_preds,
        n_class=cfg.test.n_class,
        ignore_background=cfg.test.ignore_background,
    )
    if use_tensorboard:
      writer.add_scalar('mIoU', iou_score['mIoU'].item(), len(data_loader))

    print(iou_score)
  if use_tensorboard:
    writer.close()


def compute_iou(pred_seg, gd_seg):
  """Compute IoU."""
  intersection = torch.sum(torch.logical_and(pred_seg, gd_seg))
  union = torch.sum(torch.logical_or(pred_seg, gd_seg))
  iou = intersection * 1.0 / union
  if union == 0:
    iou = 0
  return intersection, union, iou


def list_of_strings(arg):
  return [a.strip() for a in arg.split(',')]


# pylint: disable=redefined-outer-name
def parse_args():
  """Parse arguments."""
  parser = argparse.ArgumentParser(description='Training')
  parser.add_argument(
      '--cfg-path',
      default='configs/refcoco_test_prompt.yaml',
      help='path to configuration file.',
  )
  parser.add_argument('--index', default=0, type=int, help='split task')
  parser.add_argument('--mask_threshold', default=0.0, type=float)
  parser.add_argument('--confidence_threshold', default=0.0, type=float)
  parser.add_argument('--clipes_threshold', default=0.0, type=float)
  parser.add_argument('--stuff_bg_factor', default=0.0, type=float)
  parser.add_argument('--bg_factor', default=0.0, type=float)
  parser.add_argument('--output_path', default=None, type=str)
  parser.add_argument(
      '--visual_prompt_type', default=None, type=list_of_strings
  )
  parser.add_argument(
      '--stuff_visual_prompt_type', default=None, type=list_of_strings
  )

  args = parser.parse_args()

  return args


def main(args):
  cfg = Config(**load_yaml(args.cfg_path))
  if args.mask_threshold > 0:
    cfg.car.mask_threshold = args.mask_threshold
  if args.confidence_threshold > 0:
    cfg.car.confidence_threshold = args.confidence_threshold
  if args.clipes_threshold > 0:
    cfg.car.clipes_threshold = args.clipes_threshold
  if args.bg_factor > 0:
    cfg.car.bg_factor = args.bg_factor
  if args.stuff_bg_factor > 0:
    cfg.car.stuff_bg_factor = args.stuff_bg_factor
  if args.output_path is not None:
    cfg.test.output_path = args.output_path
  if args.visual_prompt_type is not None:
    cfg.car.visual_prompt_type = args.visual_prompt_type
  if args.stuff_visual_prompt_type is not None:
    cfg.car.stuff_visual_prompt_type = args.stuff_visual_prompt_type

  try:
    data_root = cfg.test.data_root
  except ValueError:
    data_root = None

  dataset_test = get_dataset(
      cfg, cfg.test.ds_name, cfg.test.split, get_transform(), data_root
  )

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  stuff_label_space = None
  if cfg.test.ds_name == 'voc':
    label_space = VOC_CLASSES
  elif cfg.test.ds_name == 'coco':
    label_space = COCO_OBJECT_CLASSES
  elif cfg.test.ds_name == 'context':
    # label_space = PASCAL_CONTEXT_CLASSES
    label_space = PASCAL_CONTEXT_THING_CLASS
    stuff_label_space = PASCAL_CONTEXT_STUFF_CLASS
  elif cfg.test.ds_name == 'ade':
    label_space = ADE_THING_CLASS
    stuff_label_space = ADE_STUFF_CLASS
  elif cfg.test.ds_name == 'pascal_459':
    label_space = PASCAL_459_THING_CLASS
    stuff_label_space = PASCAL_459_STUFF_CLASS
  elif cfg.test.ds_name == 'ade_847':
    label_space = ADE_847_THING_CLASS
    stuff_label_space = ADE_847_STUFF_CLASS
  else:
    label_space = None

  num_chunks, chunk_index = 1, 0
  if hasattr(cfg.test, 'num_chunks'):
    num_chunks = cfg.test.num_chunks
  if hasattr(cfg.test, 'chunk_index'):
    chunk_index = cfg.test.chunk_index
  # Size of each chunk
  chunk_size = len(dataset_test) // num_chunks
  # Choose which chunk to load (0-indexed)
  # Define a subset of the dataset
  subset_indices = range(
      chunk_index * chunk_size, (chunk_index + 1) * chunk_size
  )
  subset_dataset = Subset(dataset_test, indices=subset_indices)

  data_loader_test = torch.utils.data.DataLoader(
      subset_dataset, batch_size=1, shuffle=False, num_workers=1
  )

  car_model = CaR(cfg, device=device, seg_mode=cfg.test.seg_mode)

  car_model = car_model.to(device)

  if not cfg.test.use_pseudo and cfg.test.sam_mask_root is None:
    print('Using sam online')
    # sam_checkpoint, model_type = build_sam_config(cfg)
    build_sam_config(cfg)

  evaluate(
      data_loader_test,
      cfg,
      car_model,
      test_cfg=cfg.test,
      label_space=label_space,
      stuff_label_space=stuff_label_space,
  )


if __name__ == '__main__':
  args = parse_args()
  main(args)
