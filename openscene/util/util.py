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
"""Functions for utilization."""

import os
import shutil

import clip
from dataset.scannet200_constants import MATTERPORT_COLOR_MAP_21
from dataset.scannet200_constants import MATTERPORT_COLOR_MAP_NYU160
from dataset.scannet200_constants import NUSCENES16_COLORMAP
from dataset.scannet200_constants import SCANNET_COLOR_MAP_20
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image
import torch


def save_checkpoint(state, is_best, sav_path, filename='model_last.pth.tar'):
  filename = os.path.join(sav_path, filename)
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, os.path.join(sav_path, 'model_best.pth.tar'))


def extract_clip_feature(labelset, model_name='ViT-B/32'):
  """Extract CLIP text embeddings."""
  # "ViT-L/14@336px" # the big model that OpenSeg uses
  print('Loading CLIP {} model...'.format(model_name))
  clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)
  print('Finish loading')

  if isinstance(labelset, str):
    lines = labelset.split(',')
  elif isinstance(labelset, list):
    lines = labelset
  else:
    raise NotImplementedError

  labels = []
  for line in lines:
    label = line
    labels.append(label)
  text = clip.tokenize(labels)
  text = text.cuda()
  text_features = clip_pretrained.encode_text(text)
  text_features = text_features / text_features.norm(dim=-1, keepdim=True)

  return text_features, labels


def extract_text_feature(labelset, args):
  """Extract text embeddings."""
  # a bit of prompt engineering
  if hasattr(args, 'prompt_eng') and args.prompt_eng:
    print('Use prompt engineering: a XX in a scene')
    labelset = ['a ' + label + ' in a scene' for label in labelset]
    if 'scannet_3d' in args.data_root:
      labelset[-1] = 'other'
    if 'matterport_3d' in args.data_root:
      labelset[-2] = 'other'
  if not hasattr(args, 'feat_2d') or 'lseg' in args.feat_2d:
    text_features, labels = extract_clip_feature(labelset)
  elif 'osegclip' in args.feat_2d:
    text_features, labels = extract_clip_feature(
        labelset, model_name='ViT-L/14@336px')
  else:
    raise NotImplementedError

  return text_features, labels


class AverageMeter(object):
  """Computes and stores the average and current value."""

  def __init__(self):
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


def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
  """Sets the learning rate to the base LR decayed by 10 every step epochs."""
  lr = base_lr * (multiplier**(epoch // step_epoch))
  return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
  """Poly learning rate policy."""
  lr = base_lr * (1 - float(curr_iter) / max_iter)**power
  return lr


def intersectionandunion(output, target, k, ignore_index=255):
  """Calculate intersection and union."""
  assert (output.ndim in [1, 2, 3, 4])
  assert output.shape == target.shape
  output = output.reshape(output.size).copy()
  target = target.reshape(target.size)
  output[np.where(target == ignore_index)[0]] = ignore_index
  intersection = output[np.where(output == target)[0]]
  area_intersection, _ = np.histogram(intersection, bins=np.arange(k + 1))
  area_output, _ = np.histogram(output, bins=np.arange(k + 1))
  area_target, _ = np.histogram(target, bins=np.arange(k + 1))
  area_union = area_output + area_target - area_intersection
  return area_intersection, area_union, area_target


def intersectionanduniongpu(output, target, k, ignore_index=255):
  """Calculate intersection and union on GPU."""
  assert (output.dim() in [1, 2, 3, 4])
  assert output.shape == target.shape
  output = output.view(-1)
  target = target.view(-1)
  output[target == ignore_index] = ignore_index
  intersection = output[output == target]
  area_intersection = torch.histc(
      intersection.float().cpu(), bins=k, min=0, max=k - 1)
  area_output = torch.histc(output.float().cpu(), bins=k, min=0, max=k - 1)
  area_target = torch.histc(target.float().cpu(), bins=k, min=0, max=k - 1)
  area_union = area_output + area_target - area_intersection
  return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def check_mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def check_makedirs(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def export_pointcloud(name, points, colors=None, normals=None):
  """Export pointclouds to ply."""
  if len(points.shape) > 2:
    points = points[0]
    if normals is not None:
      normals = normals[0]
  if isinstance(points, torch.Tensor):
    points = points.detach().cpu().numpy()
    if normals is not None:
      normals = normals.detach().cpu().numpy()
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  if colors is not None:
    pcd.colors = o3d.utility.Vector3dVector(colors)
  if normals is not None:
    pcd.normals = o3d.utility.Vector3dVector(normals)
  o3d.io.write_point_cloud(name, pcd)


def export_mesh(name, v, f, c=None):
  """Export mesh to ply."""
  if len(v.shape) > 2:
    v, f = v[0], f[0]
  if isinstance(v, torch.Tensor):
    v = v.detach().cpu().numpy()
    f = f.detach().cpu().numpy()
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(v)
  mesh.triangles = o3d.utility.Vector3iVector(f)
  if c is not None:
    mesh.vertex_colors = o3d.utility.Vector3dVector(c)
  o3d.io.write_triangle_mesh(name, mesh)


def visualize_labels(u_index,
                     labels,
                     new_pallete,
                     out_name,
                     loc='lower left',
                     ncol=7):
  """Visualize labels colors."""
  patches = []
  for index in u_index:
    label = labels[index]
    cur_color = [
        new_pallete[index * 3] / 255.0, new_pallete[index * 3 + 1] / 255.0,
        new_pallete[index * 3 + 2] / 255.0
    ]
    red_patch = mpatches.Patch(color=cur_color, label=label)
    patches.append(red_patch)
  plt.figure()
  plt.axis('off')
  legend = plt.legend(
      frameon=False,
      handles=patches,
      loc=loc,
      ncol=ncol,
      bbox_to_anchor=(0, -0.3),
      prop={'size': 5},
      handlelength=0.7)
  fig = legend.figure
  fig.canvas.draw()
  bbox = legend.get_window_extent()
  bbox = bbox.from_extents(*(bbox.extents + np.array([-5, -5, 5, 5])))
  bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
  plt.savefig(out_name, bbox_inches=bbox, dpi=300)
  plt.close()


def get_new_pallete(num_cls=21, colormap='scannet'):
  """Obtain the palette for segmentation."""
  if colormap == 'scannet':
    scannet_palette = []
    for _, value in SCANNET_COLOR_MAP_20.items():
      scannet_palette.append(np.array(value))
    pallete = np.concatenate(scannet_palette)
  elif colormap == 'matterport':
    scannet_palette = []
    for _, value in MATTERPORT_COLOR_MAP_21.items():
      scannet_palette.append(np.array(value))
    pallete = np.concatenate(scannet_palette)
  elif colormap == 'matterport_nyu160':
    scannet_palette = []
    for _, value in MATTERPORT_COLOR_MAP_NYU160.items():
      scannet_palette.append(np.array(value))
    pallete = np.concatenate(scannet_palette)
  elif colormap == 'nuscenes16':
    nuscenes16_palette = []
    for _, value in NUSCENES16_COLORMAP.items():
      nuscenes16_palette.append(np.array(value))
    pallete = np.concatenate(nuscenes16_palette)
  else:
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
      lab = j
      pallete[j * 3 + 0] = 0
      pallete[j * 3 + 1] = 0
      pallete[j * 3 + 2] = 0
      i = 0
      while lab > 0:
        pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
        pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
        pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
        i = i + 1
        lab >>= 3
  return pallete


def convert_labels_with_pallete(inputs,
                                new_pallete,
                                is_3d=False,
                                out_label_flag=False,
                                labels=None):
  """Get image color pallete for visualizing masks."""

  if is_3d:
    new_3d = np.zeros((inputs.shape[0], 3))
    u_index = np.unique(inputs)
    for index in u_index:
      if index == 255:
        index_ = 20
      else:
        index_ = index

      new_3d[inputs == index] = np.array([
          new_pallete[index_ * 3] / 255.0, new_pallete[index_ * 3 + 1] / 255.0,
          new_pallete[index_ * 3 + 2] / 255.0
      ])

    if out_label_flag:
      assert labels is not None
      u_index = list(range(len(new_pallete) // 3))  # show all 20 classes
      patches = []
      for index in u_index:
        label = labels[index]
        cur_color = [
            new_pallete[index * 3] / 255.0, new_pallete[index * 3 + 1] / 255.0,
            new_pallete[index * 3 + 2] / 255.0
        ]
        red_patch = mpatches.Patch(color=cur_color, label=label)
        patches.append(red_patch)
    else:
      patches = None
      return new_3d
    return new_3d, patches

  else:
    # put colormap
    out_img = Image.fromarray(inputs.squeeze().astype('uint8'))
    out_img.putpalette(new_pallete)

    if out_label_flag:
      assert labels is not None
      u_index = np.unique(inputs)
      patches = []
      for index in u_index:
        label = labels[index]
        cur_color = [
            new_pallete[index * 3] / 255.0, new_pallete[index * 3 + 1] / 255.0,
            new_pallete[index * 3 + 2] / 255.0
        ]
        red_patch = mpatches.Patch(color=cur_color, label=label)
        patches.append(red_patch)
    else:
      patches = None
    return out_img, patches
