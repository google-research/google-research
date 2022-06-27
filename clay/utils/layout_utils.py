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

"""Utility functions for layout processing."""

import collections
import dataclasses as dc
import random
from typing import Dict, Iterable, List, Tuple, Union

import anytree
import numpy as np
from PIL import Image
import tensorflow as tf

from clay.utils import anytree_utils
from clay.utils import example_utils
from clay.utils import image_utils
from clay.utils.image_utils import BBox
from clay.utils.image_utils import Box


def delete_node(node, set_as_bg = False):
  """Delete node or set its class as background(root cannot be deleted).

  Args:
    node: the node to be deleted.
    set_as_bg: do not delete the node, only set its class as background.
  """
  # Do not delete root node.
  if set_as_bg and node.parent is not None:
    node.set_labels('BACKGROUND', 0)
  else:
    anytree_utils.delete_node(node)


def delete_branch(node, set_as_bg = False):
  """Delete node and all its descendants in tree (root cannot be deleted)."""
  if node.parent is None:
    return
  _delete_descendants(node, set_as_bg)
  delete_node(node, set_as_bg)


def _delete_descendants(node,
                        set_as_bg = False):
  """Delete the descendants of a node in tree."""
  for descendant in node.descendants:
    delete_node(descendant, set_as_bg)


def closest_container_node(root, box):
  """Find tighest containing node."""
  containers = [
      n for n in anytree.PreOrderIter(root) if image_utils.overlap(box, n.bbox)
  ]
  if not containers:
    return None
  sorted_boxes = sorted(
      containers, key=lambda n: image_utils.iou(n.bbox, box), reverse=True)
  return sorted_boxes[0]


def get_nodes_by_class(node,
                       cls):
  """Get nodes by assigned class (label)."""
  if not isinstance(cls, list):
    cls = [cls]
  ns = [n for n in anytree.PreOrderIter(node) if n.bbox.ui_class in cls]
  return ns


def get_nodes_by_android_class(node,
                               cls):
  """Get nodes by Android class name."""
  return [n for n in anytree.PreOrderIter(node) if cls in n.class_name]


def overlaps_with_boxes(box,
                        boxes,
                        threshold = 0.1):
  """Check whether bbox overlaps with bboxes."""
  for other in boxes:
    if image_utils.iou(box, other) >= threshold:
      return True
  return False


def cut_overlapping_box(node,
                        occ,
                        set_as_bg = False):
  """Cut the part of the bounding box in node that overlaps with provided box.

  If the node fully overlaps with the box, we delete the node. If they do not
  overlap at all, we leave the node intact. In the other cases, we remove the
  part of the bounding box that overlaps with box.

  Args:
    node: the node whose box we would like to cut.
    occ: the occluding box.
    set_as_bg: do not delete the node, only set its class as background.
  """
  n_box = node.bbox
  left, right, top, bottom = n_box.left, n_box.right, n_box.top, n_box.bottom
  orig_area = n_box.area
  overlap = image_utils.overlap(n_box, occ)
  if overlap <= 0.:
    return
  if overlap >= 0.9:
    delete_node(node, set_as_bg)
    return

  # Check if there is more than 50% overlap along the height dimension.
  if n_box.bottom - n_box.top > 0 and (min(n_box.bottom, occ.bottom) - max(
      n_box.top, occ.top)) / (n_box.bottom - n_box.top) >= 0.5:
    if n_box.left < occ.left and n_box.right > occ.right:
      delete_node(node, set_as_bg)
      return
    if occ.left <= n_box.left and n_box.left < occ.right < n_box.right:
      left = occ.right
    if occ.right >= n_box.right and n_box.right > occ.left > n_box.left:
      right = occ.left

  # Check if there is more than 50% overlap along the width dimension.
  if n_box.right - n_box.left > 0 and (min(n_box.right, occ.right) - max(
      n_box.left, occ.left)) / (n_box.right - n_box.left) >= 0.5:
    if n_box.top < occ.top and n_box.bottom > occ.bottom:
      delete_node(node, set_as_bg)
      return
    if occ.top <= n_box.top and n_box.top < occ.bottom < n_box.bottom:
      top = occ.bottom
    if occ.bottom >= n_box.bottom and n_box.bottom > occ.top > n_box.top:
      bottom = occ.top
  new_box = dc.replace(n_box, left=left, right=right, top=top, bottom=bottom)

  # Discard if the box shrank too much.
  if new_box.area / orig_area < 0.05:
    delete_node(node, set_as_bg)
    return
  node.bbox = new_box


def shrink_pager_indicators(root):
  """Pager indicators often span the entire screen, shrink them."""
  for pi in get_nodes_by_class(root, 'PAGER_INDICATOR'):
    if pi.bbox.width == 1.0:
      pi.bbox = dc.replace(pi.bbox, left=0.3, right=0.7)


def is_box_empty(box, image, threshold = 2.):
  """Check if box has (almost) uniform color."""
  w, h = image.size
  crop_box = (int(box.left * w), int(box.top * h), int(box.right * w),
              int(box.bottom * h))
  crop = image.crop(crop_box)
  crop_w, crop_h = crop.size
  if crop_w <= 0 or crop_h <= 0:
    return True
  return np.var(crop.convert('L')) < threshold


def remove_blank_boxes(root,
                       image,
                       set_as_bg = False):
  """Delete boxes that have uniform color."""
  for node in list(anytree.PreOrderIter(root)):
    if is_box_empty(node.bbox, image):
      delete_node(node, set_as_bg)


def remove_leaf_boxes(root,
                      cls,
                      set_as_bg = False):
  """Remove all boxes of type cls which do not have any children."""
  for node in get_nodes_by_class(root, cls):
    if not node.children and node.bbox.ui_class != 'KEYBOARD':
      delete_node(node, set_as_bg)


def empty_bboxes(tree,
                 cls,
                 example,
                 set_as_bg = False):
  """Erase content of boxes of class cls."""
  image = example_utils.get_image_pil(example)
  orig_image = image.copy()
  w, h = image.size
  nodes = list(anytree.PreOrderIter(tree))
  for i, node in enumerate(nodes):
    if node.bbox.ui_class != cls:
      continue
    box = node.bbox
    blank_img = Image.new(
        'RGB', (int(w * box.width), int(h * box.height)), color=(255, 255, 255))
    image.paste(blank_img, box=(int(box.left * w), int(box.top * h)))

    # Do not erase occluding elements (paste them back).
    for occ in nodes[i + 1:]:
      box = occ.bbox
      crop_box = (int(box.left * w), int(box.top * h), int(box.right * w),
                  int(box.bottom * h))
      crop = orig_image.crop(crop_box)
      image.paste(crop, box=(int(box.left * w), int(box.top * h)))

    # Finally delete the node of interest.
    delete_node(node, set_as_bg)

  img_bytes = image_utils.img_to_bytes(image)
  example_utils.del_feat(example, 'image/encoded')
  example_utils.add_feat(example, 'image/encoded', img_bytes)
  return image


def pass_check_scale(candidate_box, min_max_area):
  """Check scale of the elements. Returns true if passed check."""
  return candidate_box.area >= min_max_area


def boxes_overlap(box1, box2):
  """Check if boxes overlap, re-using part of iou() in image_utils."""
  inter_top = max(box1.top, box2.top)
  inter_bottom = min(box1.bottom, box2.bottom)
  inter_left = max(box1.left, box2.left)
  inter_right = min(box1.right, box2.right)

  if inter_right <= inter_left or inter_bottom <= inter_top:
    return False  # no intersection
  return True


def bboxes_aligned(bboxes):
  """Return true if bboxes are aligned on the left, right, center, top or bottom."""
  # one bbox is always aligned
  if len(bboxes) < 2:
    return True

  # possible alignment types
  check_attr = ['left', 'x_center', 'top', 'bottom', 'y_center', 'right']

  for attr in check_attr:
    borders = [getattr(bbox, attr, 0) for bbox in bboxes]
    # check if all borders are equal
    if len(set(borders)) < 2:
      return True
  return False


def _generate_random_coords(num_coords = 2):
  return tuple(sorted([random.random() for _ in range(num_coords)]))


def generate_random_boxes(num_boxes, class_to_label,
                          class_name):
  """Generate num_boxes random BBoxRico objects with label as label."""
  for _ in range(num_boxes):
    left, right = _generate_random_coords()
    top, bottom = _generate_random_coords()

    bbox = BBox(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        ui_class=class_name,
        ui_label=class_to_label[class_name])

    yield bbox


def find_str_in_dict(
    term, class_mapping):
  """Finds all key, value in class_mapping s.t key is a substring of term."""
  all_matched_classes = []
  for k, v in class_mapping.items():
    if k in term.lower():
      all_matched_classes.append((v, k))
  return all_matched_classes


def flip_dict(mapping):
  """Function to flip the key/item. Items are lists: mapping is not unique."""
  flipped_dict = collections.defaultdict(list)
  for label, classes_list in mapping.items():
    for raw_class_name in classes_list:
      class_name = raw_class_name.lower()
      flipped_dict[class_name].append(label)
  return flipped_dict
