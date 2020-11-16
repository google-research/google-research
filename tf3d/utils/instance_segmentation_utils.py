# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Useful functions for instance segmentation."""
import sys
import tensorflow as tf


def map_labels_to_0_to_n(labels):
  """Maps the values in labels to numbers between 0 and N.

  Finds the unique values in labels, and maps them to new values.
  The goal is to make sure that the output labels has values starting at 0
  and ending at N. It can be that some pixels have value (-1) which will remain
  as they are.

  Args:
    labels: Input tensor of tf.int32.

  Returns:
    Mapped labels, same size as input labels with values between 0 and N.
  """
  labels_shape = labels.get_shape()
  tf_labels_shape = tf.shape(labels)
  labels = tf.reshape(labels, [-1])
  mask_ignore = tf.cast(tf.less(labels, 0), dtype=tf.int32)
  mask_no_ignore = 1 - mask_ignore
  labels *= mask_no_ignore
  max_label = tf.reduce_max(labels)
  labels += mask_ignore * max_label
  label_map = tf.zeros([max_label + 1], tf.int32)
  unique_labels, _ = tf.unique(labels)
  label_map = tf.reduce_sum(
      tf.one_hot(unique_labels, depth=max_label+1, dtype=tf.float32), axis=0)
  label_map = tf.cast((tf.cumsum(label_map) - 1.0) * label_map, dtype=tf.int32)
  labels = tf.map_fn(lambda l: label_map[l], labels, back_prop=False)
  labels = (labels * mask_no_ignore) - mask_ignore
  labels = tf.reshape(labels, tf_labels_shape)
  labels.set_shape(labels_shape)
  return labels


def randomly_select_one_point_per_segment(labels, include_ignore_label=False):
  """Randomly selects a point for each segment.

  Args:
    labels: A tf.int32 tensor of [N] containing labels for pixels. labels start
            at 0. It can be that some pixels have label (-1) which means this
            function will ignore those pixels.
    include_ignore_label: If this flag is true, the function will not ignore
                          the pixels with ignore label (-1). It increases the
                          labels by 1 and will sample from ignore label too.

  Returns:
    indices: Indices of randomly picked centers. A tensor of type tf.int32.
    labels_one_hot: One hot of the labels. A tensor of type tf.float32.
  """
  if include_ignore_label:
    labels -= tf.reduce_min(labels)
  num_pixels = tf.size(labels)
  num_classes = tf.reduce_max(labels) + 1
  labels_one_hot = tf.one_hot(labels, num_classes, dtype=tf.float32)
  labels_size = tf.reduce_sum(labels_one_hot, 0)
  labels_cumsum = tf.cumsum(labels_one_hot, axis=0)
  labels_cumsum = tf.cast(labels_cumsum * labels_one_hot, dtype=tf.int32)
  rand_inds = tf.random.uniform(
      tf.shape(labels_size), minval=0, maxval=1.0, dtype=tf.float32)
  rand_inds = tf.maximum(
      1, tf.cast(rand_inds * (labels_size + 1.0), dtype=tf.int32))
  rand_inds = tf.minimum(rand_inds, tf.cast(labels_size, dtype=tf.int32))
  rand_inds = tf.expand_dims(rand_inds, 0)
  rand_inds = tf.tile(rand_inds, tf.stack([num_pixels, 1]))
  cond = tf.equal(labels_cumsum, rand_inds)
  indices = tf.where(tf.transpose(cond))
  _, indices = tf.split(value=indices, num_or_size_splits=2, axis=1)
  indices = tf.reshape(indices, [-1])
  return indices, labels_one_hot


def randomly_select_n_points_per_segment(labels, num_points,
                                         include_ignore_label=False):
  """Randomly selects a point for each segment.

  Args:
    labels: A tf.int32 tensor of [N] containing labels for pixels. labels start
            at 0.
    num_points: Number of samples.
    include_ignore_label: If this flag is true, the function will not ignore
                          the pixels with ignore label (-1). It increases the
                          labels by 1 and will sample from ignore label too.

  Returns:
    indices: Indices of randomly picked centers. A tf.int32 tensor of size
             [num_points, num_classes].
  """
  if include_ignore_label:
    labels -= tf.reduce_min(labels)
  i = tf.constant(0, dtype=tf.int32)
  num_classes = tf.reduce_max(labels) + 1
  indices = tf.zeros(tf.stack([num_points, num_classes]), dtype=tf.int32)
  indices_shape = indices.get_shape()

  def body(i, indices):
    indices_local, _ = randomly_select_one_point_per_segment(labels)
    pad0 = tf.stack([i, num_points-i-1])
    pad1 = tf.constant([0, 0], dtype=tf.int32)
    padding = tf.stack([pad0, pad1])
    indices_local = tf.pad(tf.expand_dims(indices_local, 0), padding)
    indices += tf.cast(indices_local, dtype=tf.int32)
    indices.set_shape(indices_shape)
    i += 1
    return (i, indices)

  (i, indices) = tf.while_loop(lambda i, indices: tf.less(i, num_points),
                               body, [i, indices])
  return indices


def inputs_distances_to_centers(inputs, centers):
  """Returns the distances between inputs and centers.

  Args:
    inputs: A tf.float32 tensor of size [N, D] where N is the number of data
      points and D is their dimension.
    centers: A tf.float32 tensor of size [K, D] where K is the number of
      centers.

  Returns:
    A [N, K] tf.float32 tensor of distances.
  """
  num_centers = tf.shape(centers)[0]
  num_inputs = tf.shape(inputs)[0]
  dotprod = tf.matmul(inputs, centers, transpose_b=True)
  inputs_len2 = tf.tile(
      tf.expand_dims(
          tf.square(tf.norm(
              inputs, axis=1)), axis=1), [1, num_centers])
  centers_len2 = tf.tile(
      tf.expand_dims(
          tf.square(tf.norm(
              centers, axis=1)), axis=0), [num_inputs, 1])
  return inputs_len2 + centers_len2 - 2.0 * dotprod


def get_nearest_centers(inputs, centers):
  """Returns the nearest center to each example in input.

  Args:
    inputs: A tf.float32 tensor of size [N, D] where N is the number of data
      points and D is their dimension.
    centers: A tf.float32 tensor of size [K, D] where K is the number of
      centers.

  Returns:
    A [N] tf.int32 tensor of nearest center indices.
  """
  input_distances = inputs_distances_to_centers(inputs, centers)
  return tf.cast(tf.math.argmin(input_distances, axis=1), dtype=tf.int32)


def kmeans_initialize_centers(inputs, num_centers):
  """Randomly initialize centers from inputs.

  Args:
    inputs: A tf.float32 tensor of size [N, D] where N is the number of data
      points and D is their dimension.
    num_centers: Number of centers.

  Returns:
    init_centers: Initialized centers. A tf.float32 tensor of size [C, D].
    indices: Indices of centers in inputs.
  """
  input_num = tf.shape(inputs)[0]
  rand_ind = tf.random.shuffle(tf.range(input_num))[:num_centers]
  init_centers = tf.gather(inputs, rand_ind)
  return init_centers, rand_ind


def kmeans_initialize_centers_plus_plus(inputs, num_centers):
  """Center initialization based on kmeans plus plus algorithm.

  Pick one center at a time. Pick the next center with random probability
  proportional to the distance of the inputs from previously picked points.

  Args:
    inputs: A tf.float32 tensorof size [N, D] where N is the number of data
      points and D is their dimension.
    num_centers: Number of centers.

  Returns:
    init_centers: Initialized centers. A tf.float32 tensor of size [C, D].
    indices: Indices of centers in inputs.
  """
  input_num = tf.shape(inputs)[0]
  input_dims = tf.shape(inputs)[1]
  init_centers = tf.zeros(tf.stack([num_centers, input_dims]), dtype=tf.float32)
  indices = tf.zeros([num_centers], dtype=tf.int32)
  rand_ind_0 = tf.random.uniform(
      [1], minval=0, maxval=input_num, dtype=tf.int32)
  rand_ind_0 = tf.minimum(rand_ind_0, input_num-1)
  init_centers_0 = tf.gather(inputs, rand_ind_0)
  inputs_min_distances = inputs_distances_to_centers(inputs, init_centers_0)
  init_centers += tf.pad(init_centers_0,
                         paddings=tf.stack([tf.stack([0, num_centers-1]),
                                            tf.stack([0, 0])]))
  indices += tf.pad(rand_ind_0, paddings=[[0, num_centers-1]])
  i = tf.constant(1, dtype=tf.int32)

  def body(i, init_centers, indices, inputs_min_distances):
    """while loop body. Pick the next center given previously picked centers.

    Args:
      i: For loop index.
      init_centers: Initial centers that is modified inside the body.
      indices: Indices of picked centers in inputs.
      inputs_min_distances: Minimum distance of the inputs to centers so far.

    Returns:
      i and init_centers.
    """
    best_new_center_ind = tf.squeeze(
        tf.random.categorical(
            tf.math.log(
                tf.transpose(inputs_min_distances) + sys.float_info.min), 1))
    best_new_center_ind = tf.cast(best_new_center_ind, dtype=tf.int32)
    indices += tf.pad(tf.stack([best_new_center_ind]),
                      paddings=[[i, num_centers-i-1]])
    init_centers_i = tf.expand_dims(tf.gather(inputs, best_new_center_ind), 0)
    init_centers += tf.pad(init_centers_i,
                           paddings=tf.stack([tf.stack([i, num_centers-i-1]),
                                              tf.stack([0, 0])]))
    inputs_min_distances = tf.minimum(
        inputs_distances_to_centers(inputs, init_centers_i),
        inputs_min_distances)
    i += 1
    return i, init_centers, indices, inputs_min_distances

  (i, init_centers, indices, inputs_min_distances) = tf.while_loop(
      lambda i, init_centers, indices, inputs_min_distances: i < num_centers,
      body,
      [i, init_centers, indices, inputs_min_distances])
  return init_centers, indices


def get_pairwise_iou_matrix(masks1, masks2):
  """Returns the matrix containing iou score between masks1 and masks2.

  Args:
    masks1: A tf.float32 tensor of size [k1, height, width] where k1 is the
            number of masks1.
    masks2: A tf.float32 tensor of size [k2, height, width] where k2 is the
            number of masks2.

  Returns:
    A tf.float32 tensor of size [k1, k2] containing iou scores.
  """
  masks_shape1 = tf.shape(masks1)
  k1 = masks_shape1[0]
  height = masks_shape1[1]
  width = masks_shape1[2]
  k2 = tf.shape(masks2)[0]
  masks1 = tf.reshape(masks1, tf.stack([k1, height * width]))
  masks2 = tf.reshape(masks2, tf.stack([k2, height * width]))
  intersections = tf.matmul(masks1, masks2, transpose_b=True)
  mask_sizes1 = tf.reduce_sum(masks1, axis=[1])
  mask_sizes1 = tf.tile(tf.expand_dims(mask_sizes1, axis=1), tf.stack([1, k2]))
  mask_sizes2 = tf.reduce_sum(masks2, axis=[1])
  mask_sizes2 = tf.tile(tf.expand_dims(mask_sizes2, axis=0), tf.stack([k1, 1]))
  unions = tf.maximum(mask_sizes1 + mask_sizes2 - intersections, 1.0)
  return intersections / unions


def instance_non_maximum_suppression_1d_scores(masks,
                                               scores,
                                               classes,
                                               min_score_thresh,
                                               min_iou_thresh,
                                               is_class_agnostic=False):
  """Applies non maximum suppression to masks.

  Args:
    masks: A tf.float32 tensor of size [k, height, width] where k is the number
      of masks.
    scores: A tf.float32 tensor of size [k].
    classes: A tf.int32 tensor of size [k].
    min_score_thresh: Minimum score threshold.
    min_iou_thresh: Minimum iou threshold.
    is_class_agnostic: If True, it will suppress two instances from different
      classes that have an iou larger than threshold.

  Returns:
    Masks: A subset of input masks.
    scores: A subset of input scores.
    classes: A subset of input classes.
    b_mask: Boolean mask that is used to remove overlapping masks.
  """
  b_mask = tf.greater_equal(scores, min_score_thresh)
  masks = tf.boolean_mask(masks, b_mask)
  scores = tf.boolean_mask(scores, b_mask)
  classes = tf.boolean_mask(classes, b_mask)
  k = tf.shape(masks)[0]
  scores, indices = tf.nn.top_k(scores, k=k)
  masks = tf.gather(masks, indices)
  classes = tf.gather(classes, indices)
  ious = get_pairwise_iou_matrix(masks, masks)
  ious = tf.greater_equal(ious, min_iou_thresh)
  b_mask = tf.fill(dims=tf.stack([k]), value=tf.constant(True, dtype=tf.bool))
  i = tf.constant(1, tf.int32)
  def body(i, b_mask):
    """While loop body. Computes b_mask containing staying instances in nms.

    Args:
      i: While loop counter.
      b_mask: Mask containing instances in nms.

    Returns:
      Updated i, b_mask.
    """
    b_mask_sub1 = tf.slice(b_mask,
                           begin=tf.stack([0]),
                           size=tf.stack([i]))
    b_mask_sub2 = tf.slice(b_mask,
                           begin=tf.stack([i+1]),
                           size=tf.stack([k-i-1]))
    ious_sub = tf.squeeze(tf.slice(ious,
                                   begin=tf.stack([0, i]),
                                   size=tf.stack([i, 1])),
                          axis=1)
    b_mask_sub1_classes = b_mask_sub1
    if not is_class_agnostic:
      classes_sub = tf.slice(classes,
                             begin=tf.stack([0]),
                             size=tf.stack([i]))
      classes_sub = tf.equal(classes_sub, classes[i])
      b_mask_sub1_classes = tf.logical_and(b_mask_sub1, classes_sub)
    should_add = tf.expand_dims(tf.logical_not(tf.reduce_any(
        tf.logical_and(b_mask_sub1_classes, ious_sub))), axis=0)
    b_mask = tf.concat([b_mask_sub1, should_add, b_mask_sub2], axis=0)
    i += 1
    return i, b_mask
  (i, b_mask) = tf.while_loop(
      lambda i, b_mask: i < k,
      body,
      loop_vars=[i, b_mask],
      shape_invariants=[i.get_shape(), tf.TensorShape([None])],
      parallel_iterations=1)
  new_masks = tf.boolean_mask(masks, b_mask)
  new_scores = tf.boolean_mask(scores, b_mask)
  new_classes = tf.boolean_mask(classes, b_mask)
  return new_masks, new_scores, new_classes, b_mask


def instance_non_maximum_suppression_2d_scores(masks,
                                               scores,
                                               num_classes,
                                               min_score_thresh,
                                               min_iou_thresh,
                                               is_class_agnostic=False):
  """Applies non maximum suppression to masks.

  Args:
    masks: A tf.float32 tensor of size [k, height, width] where k is the number
           of masks.
    scores: A tf.float32 tensor of size [k, num_classes].
    num_classes: Number of classes.
    min_score_thresh: Minimum score threshold.
    min_iou_thresh: Minimum iou threshold.
    is_class_agnostic: If True, it will suppress two instances from different
                       classes that have an iou larger than threshold.

  Returns:
    Masks, scores and classes after non max supperssion.
  """
  if not is_class_agnostic:
    new_masks = []
    new_scores = []
    new_classes = []
    k = tf.shape(scores)[0]
    for i in range(num_classes):
      scores_1d = tf.reshape(
          tf.slice(scores, begin=[0, i], size=tf.stack([k, 1])), [-1])
      classes_1d = tf.fill(dims=tf.stack([k]),
                           value=tf.constant(i, dtype=tf.int32))
      (new_masks_i,
       new_scores_i,
       new_classes_i,
       _) = instance_non_maximum_suppression_1d_scores(
           masks,
           scores_1d,
           classes_1d,
           min_score_thresh,
           min_iou_thresh,
           is_class_agnostic=False)
      new_masks.append(new_masks_i)
      new_scores.append(new_scores_i)
      new_classes.append(new_classes_i)
    new_masks = tf.concat(new_masks, axis=0)
    new_scores = tf.concat(new_scores, axis=0)
    new_classes = tf.concat(new_classes, axis=0)
    return new_masks, new_scores, new_classes
  else:
    scores, classes = tf.nn.top_k(scores, k=1)
    scores = tf.reshape(scores, [-1])
    classes = tf.reshape(classes, [-1])
    (new_masks,
     new_scores,
     new_classes,
     _) = instance_non_maximum_suppression_1d_scores(masks,
                                                     scores,
                                                     classes,
                                                     min_score_thresh,
                                                     min_iou_thresh,
                                                     is_class_agnostic=True)
    return new_masks, new_scores, new_classes


def points_mask_iou(masks1, masks2):
  """Intersection over union between point masks.

  Args:
    masks1: A tensor of size [num_masks1, num_points] with values ranging
      between 0 and 1.
    masks2: A tensor of size [num_masks2, num_points] with values ranging
      between 0 and 1.

  Returns:
    A tensor of size [num_masks1, num_masks2].
  """
  num_masks1 = tf.shape(masks1)[0]
  num_masks2 = tf.shape(masks2)[0]
  masks1 = tf.cast(masks1, dtype=tf.float32)
  masks2 = tf.cast(masks2, dtype=tf.float32)
  intersections = tf.linalg.matmul(masks1, masks2, transpose_b=True)
  mask_sizes1 = tf.math.reduce_sum(masks1, axis=[1])
  mask_sizes1 = tf.tile(
      tf.expand_dims(mask_sizes1, axis=1), tf.stack([1, num_masks2]))
  mask_sizes2 = tf.math.reduce_sum(masks2, axis=[1])
  mask_sizes2 = tf.tile(
      tf.expand_dims(mask_sizes2, axis=0), tf.stack([num_masks1, 1]))
  unions = tf.maximum(mask_sizes1 + mask_sizes2 - intersections, 1.0)
  return intersections / unions


def points_mask_pairwise_iou(masks1, masks2):
  """Intersection over union between pairwise corresponding masks.

  Args:
    masks1: A tensor of size [num_masks, num_points] with values ranging
      between 0 and 1.
    masks2: A tensor of size [num_masks, num_points] with values ranging
      between 0 and 1.

  Returns:
    A tensor of size [num_masks].
  """
  masks1 = tf.cast(masks1, dtype=tf.float32)
  masks2 = tf.cast(masks2, dtype=tf.float32)
  intersections = tf.math.reduce_sum(masks1 * masks2, axis=1)
  mask_sizes1 = tf.math.reduce_sum(masks1, axis=1)
  mask_sizes2 = tf.math.reduce_sum(masks2, axis=1)
  unions = tf.maximum(mask_sizes1 + mask_sizes2 - intersections, 1.0)
  return intersections / unions
