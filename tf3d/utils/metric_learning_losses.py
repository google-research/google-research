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

"""Metric learning losses.

The goal of the metric learning losses is to force the embedding vectors
that are in the same instance to be close to each other and force the embedding
vector of the pixels that belong to different instances be far from each other.

Here are the definition of few terms that are frequently used in this file:
soft_masks: A soft instance segment mask. Soft masks have continuous values
  between 0 and 1, in contrast to hard masks which have binary values.
instance_labels: An instance label, contains pixel values that range from 0 to
  n-1, where n is the number of instances in the image. It is possible to
  convert an instance label to n binary masks.
embedding: A mapping from the points or image pixels to embedding vectors. The
  goal in instance segmentation is to find an embedding space in which points
  or pixels that belong to the same instance correspond to vectors that are very
  close to each other in this space.
"""
import tensorflow as tf
from tf3d.utils import instance_segmentation_utils as isu
from tf3d.utils import metric_learning_utils as eutils


def regularization_loss(embedding, lambda_coef, regularization_type='msq'):
  """Regularization loss.

  It is used for regularization of the embedding vectors. Regularization is the
  process of providing additional information to solve an ill posed problem.
  In this case, we are adding a prior on the length of the embedding vectors to
  be either close to 0 or 1.

  Args:
    embedding: A tf.float32 tensor of size [N, D].
    lambda_coef: Regularization loss coefficient.
    regularization_type: Regularization loss type. Supported values are 'msq'
      and 'unit_length'. 'msq' stands for 'mean square' which penalizes the
      logit vectors if they have a length far from zero. 'unit_length' penalizes
      the logit vectors if they have a length far from one.

  Returns:
    Regularization loss value.

  Raises:
    ValueError: If regularization_type is not known.
  """
  loss = tf.constant(0.0, dtype=tf.float32)
  if regularization_type == 'msq':
    loss = tf.reduce_mean(tf.square(tf.norm(embedding, axis=1))) * 0.5
  elif regularization_type == 'unit_length':
    loss = tf.reduce_mean(tf.abs(tf.square(tf.norm(embedding, axis=1)) - 1.0))
  else:
    raise ValueError('Regularization type is not known')
  return loss * lambda_coef


def npair_loss(embedding,
               target,
               similarity_strategy='dotproduct',
               loss_strategy='softmax'):
  """n-pair loss.

  This loss is based on the following paper:
  Kihyuk Sohn, Improved Deep Metric Learning with Multi-class N-pair Loss
  Objective, NIPS 2016.

  The loss is computed as follows: The dot product between every pair of
  embedding vectors is computed. Given N embedding vectors, this will result
  in a [N, N] matrix. In this matrix, the softmax (or sigmoid) loss for each row
  is computed. The total loss is the average over the losses for all the rows.
  In order to perform the softmax (sigmoid) loss, the one-hot ground-truth
  labels for each row are required. In the row i, the columns that have the same
  target label as this row will be set to 1 and other columns will be set to 0.
  Each row is normalized so the sum of each row is equal to 1.

  Args:
    embedding: A float32 matrix of [N, D] where N is the number of pixels and
               D is the number of embedding dimensions.
    target: A float32 one-hot matrix of [N, K] where K is the number of labels.
    similarity_strategy: Defines the method for computing similarity between
                         embedding vectors. Possible values are 'dotproduct' and
                         'distance'.
    loss_strategy: Defines the type of loss including 'softmax' or 'sigmoid'.

  Returns:
    Total loss value.

  Raises:
    ValueError: If similarity or loss strategy is not known.
  """
  similarity_matrix = eutils.embedding_centers_to_logits(embedding, embedding,
                                                         similarity_strategy)
  target_matrix = tf.matmul(target, target, transpose_b=True)
  if loss_strategy == 'sigmoid':
    if similarity_strategy == 'distance':
      target_matrix *= 0.5
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    losses = loss_fn(y_true=target_matrix, y_pred=similarity_matrix)
    return tf.reduce_mean(losses)
  elif loss_strategy == 'softmax':
    target_matrix_sum = tf.reduce_sum(target_matrix, axis=1)
    target_matrix_sum = tf.expand_dims(target_matrix_sum, axis=1)
    num_pixels = tf.size(target_matrix_sum)
    target_matrix_sum = tf.tile(target_matrix_sum,
                                tf.stack([1, num_pixels]))
    target_matrix /= target_matrix_sum
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    losses = loss_fn(y_true=target_matrix, y_pred=similarity_matrix)
    return tf.reduce_mean(losses)
  else:
    raise ValueError('Unknown strategy')


def weighted_npair_loss(embedding,
                        target,
                        similarity_strategy='dotproduct',
                        loss_strategy='softmax'):
  """Weighted n-pair loss to handle unbalanced number of labels in target.

  This loss is based on the following paper:
  Kihyuk Sohn, Improved Deep Metric Learning with Multi-class N-pair Loss
  Objective, NIPS 2016.

  The loss is computed as follows: The dot product between every pair of
  embedding vectors is computed. Given N embedding vectors, this will result
  in a [N, N] matrix. In this matrix, the softmax (or sigmoid) loss for each row
  is computed. The total loss is the average over the losses for all the rows.
  In order to perform the softmax (sigmoid) loss, the one-hot ground-truth
  labels for each row are required. In the row i, the columns that have the same
  target label as this row will be set to 1 and other columns will be set to 0.
  Each row is normalized so the sum of each row is equal to 1.

  Args:
    embedding: A float32 matrix of [N, D] where N is the number of pixels and
      D is the number of embedding dimensions.
    target: A float32 one-hot matrix of [N, K] where K is the number of labels.
    similarity_strategy: Defines the method for computing similarity between
      embedding vectors. Possible values are 'dotproduct' and 'distance'.
    loss_strategy: Defines the type of loss including 'softmax' or 'sigmoid'.

  Returns:
    Total loss value.

  Raises:
    ValueError: If similarity or loss strategy is not known.
  """
  similarity_matrix = eutils.embedding_centers_to_logits(embedding, embedding,
                                                         similarity_strategy)
  target_matrix = tf.matmul(target, target, transpose_b=True)
  n = tf.shape(target)[0]
  target_sizes = tf.maximum(tf.reduce_sum(target, axis=0), 0.00001)
  target_sizes = tf.tile(tf.expand_dims(target_sizes, axis=0), [n, 1]) * target
  target_weights = 1.0 / tf.reduce_sum(target_sizes, axis=1)
  target_weights_matrix = tf.matmul(tf.expand_dims(target_weights, axis=1),
                                    tf.expand_dims(target_weights, axis=1),
                                    transpose_b=True)
  if loss_strategy == 'sigmoid':
    if similarity_strategy == 'distance':
      target_matrix *= 0.5
    target_weights_matrix_sum = tf.reduce_sum(target_weights_matrix)
    target_weights_matrix *= tf.cast(
        n, dtype=tf.float32) * tf.cast(
            n, dtype=tf.float32)
    target_weights_matrix /= target_weights_matrix_sum
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    losses = loss_fn(
        y_true=tf.reshape(target_matrix, [-1]),
        y_pred=tf.reshape(similarity_matrix, [-1]))
    return tf.reduce_mean(losses * tf.reshape(target_weights_matrix, [-1]))
  elif loss_strategy == 'softmax':
    target_matrix_sum = tf.reduce_sum(target_matrix, axis=1)
    target_matrix_sum = tf.expand_dims(target_matrix_sum, axis=1)
    num_pixels = tf.size(target_matrix_sum)
    target_matrix_sum = tf.tile(target_matrix_sum,
                                tf.stack([1, num_pixels]))
    target_matrix /= target_matrix_sum
    target_weights_sum = tf.reduce_sum(target_weights)
    target_weights *= tf.cast(n, dtype=tf.float32)
    target_weights /= target_weights_sum
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    losses = loss_fn(
        y_true=target_matrix,
        y_pred=similarity_matrix)
    return tf.reduce_mean(losses * target_weights)
  else:
    raise ValueError('Unknown strategy')


def instance_embedding_npair_loss(embedding,
                                  instance_labels,
                                  crop_min_height,
                                  crop_area,
                                  similarity_strategy='dotproduct',
                                  loss_strategy='softmax'):
  """n-pair loss for a cropped box inside the embedding.

  It uses npair_loss (above) to compute the embedding loss given the
  ground-truth instance_labels. instance_labels contains the ground-truth
  labels. The loss is computed as follows: We compute the dot product between
  the embedding vector of each pixel and every other pixel. If we have N pixels,
  this will give us a [N, N] matrix. In this matrix, we compute the
  softmax (or sigmoid) loss for each row, average the losses and return as
  output. In order to perform the softmax (sigmoid) loss, we need the one-hot
  ground-truth labels for each row. In the row i, the pixels that in the
  same instance as the pixel i, will be set to 1, and other pixels will be set
  to 0. Each row is normalized so the sum of each row is equal to 1.

  Args:
    embedding: A tf.float32 tensor of [height, width, embedding_size].
    instance_labels: A tf.int32 tensor of [height, width]. Assumed values in
      target start from 0 and cover 0 to N-1.
    crop_min_height: Minimum height of the crop window.
    crop_area: Area of the crop window.
    similarity_strategy: Defines the method for computing similarity between
      embedding vectors. Possible values are 'dotproduct' and 'distance'.
    loss_strategy: Defines the type of loss including 'softmax' or 'sigmoid'.

  Returns:
    Total loss value.

  Raises:
    ValueError: If loss strategy or similarity strategy are unknown.
  """
  embedding_shape = tf.shape(embedding)
  embedding_height = embedding_shape[0]
  embedding_width = embedding_shape[1]
  embedding_size = embedding_shape[2]
  crop_height = tf.maximum(crop_area // embedding_width, crop_min_height)
  crop_height = tf.maximum(1, tf.minimum(embedding_height - 1, crop_height))
  crop_width = tf.maximum(
      1, tf.minimum(embedding_width - 1, crop_area // crop_height))
  y_start = tf.random.uniform(
      [], minval=0, maxval=tf.maximum(1, embedding_height-crop_height),
      dtype=tf.int32)
  x_start = tf.random.uniform(
      [], minval=0, maxval=tf.maximum(1, embedding_width-crop_width),
      dtype=tf.int32)
  embedding = tf.slice(
      embedding,
      begin=tf.stack([y_start, x_start, 0]),
      size=tf.stack([tf.minimum(crop_height, embedding_height-y_start),
                     tf.minimum(crop_width, embedding_width-x_start),
                     embedding_size]))
  embedding = tf.reshape(embedding, [-1, embedding_size])
  instance_labels = tf.slice(
      instance_labels,
      begin=tf.stack([y_start, x_start]),
      size=tf.stack([tf.minimum(crop_height, embedding_height-y_start),
                     tf.minimum(crop_width, embedding_width-x_start)]))
  instance_labels = tf.reshape(instance_labels, [-1])
  num_instance_labels = tf.reduce_max(instance_labels) + 1
  valid_mask = tf.greater_equal(instance_labels, 0)
  embedding = tf.boolean_mask(embedding, valid_mask)
  instance_labels = tf.boolean_mask(instance_labels, valid_mask)
  unique_labels, _ = tf.unique(instance_labels)
  instance_labels = tf.one_hot(instance_labels,
                               num_instance_labels,
                               dtype=tf.float32)
  instance_labels = tf.transpose(
      tf.gather(tf.transpose(instance_labels), unique_labels))
  return weighted_npair_loss(embedding, instance_labels,
                             similarity_strategy, loss_strategy)


def instance_embedding_npair_random_center_loss(
    embedding,
    instance_labels,
    similarity_strategy='dotproduct',
    loss_strategy='softmax'):
  """Computes n-pair loss by drawing random points from each instance segment.

  It uses npair_loss (above) to compute the embedding loss given the
  ground-truth instance_labels. instance_labels contains the ground-truth
  labels. The loss is computed as follows: We compute the dot product between
  the embedding vector of randomly drawn seed (center) points and every other
  pixel. If we have N pixels, this will give us a [num_inst, N] matrix.
  Since there are num_inst randomly drawn centers (one per instance).
  In this matrix, each column captures the similarity between the pixel
  embedding that it corresponds to with each of the centers. We want the dot
  product be high for the center that belongs to the same instance as the pixel.
  So we can do a softmax loss given this ground-truth for each column and then
  average this over rows.

  Args:
    embedding: A tf.float32 tensor of [height, width, dims].
    instance_labels: A tf.int32 tensor of [height, width]. Assumed values in
      target start from 0 and cover 0 to N-1.
    similarity_strategy: Defines the method for computing similarity between
      embedding vectors. Possible values are 'dotproduct' and 'distance'.
    loss_strategy: Defines the type of loss including 'softmax' or 'sigmoid'.

  Returns:
    Total loss value.

  Raises:
    ValueError: If strategy is not known.
  """
  embedding_shape = tf.shape(embedding)
  embedding_dims = embedding_shape[2]
  embedding = tf.reshape(embedding, tf.stack([-1, embedding_dims]))
  instance_labels = tf.reshape(instance_labels, [-1])
  valid_mask = tf.greater_equal(instance_labels, 0)
  embedding = tf.boolean_mask(embedding, valid_mask)
  instance_labels = tf.boolean_mask(instance_labels, valid_mask)
  (indices,
   instance_label_one_hot) = isu.randomly_select_one_point_per_segment(
       instance_labels)
  centers = tf.gather(embedding, indices)
  embedding_similarity = eutils.embedding_centers_to_logits(centers, embedding,
                                                            similarity_strategy)
  if loss_strategy == 'softmax':
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    losses = loss_fn(y_true=instance_label_one_hot, y_pred=embedding_similarity)
    return tf.reduce_mean(losses)
  elif loss_strategy == 'sigmoid':
    if similarity_strategy == 'distance':
      instance_label_one_hot *= 0.5
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    losses = loss_fn(y_true=instance_label_one_hot, y_pred=embedding_similarity)
    return tf.reduce_mean(losses)
  else:
    raise ValueError('Loss strategy is unknown')


def instance_embedding_npair_random_center_random_sample_loss(
    embedding,
    instance_labels,
    num_samples,
    similarity_strategy='dotproduct',
    loss_strategy='softmax'):
  """Compute n-pair loss by drawing random points and samples from instances.

  This loss is very similar to instance_embedding_npair_random_center_loss,
  with the difference that instead of computing the dot product of every pixel
  with the seeds(centers), we compute the dot product of sampled points with the
  centers. This strategy has the advantage of an even distribution over
  predicted instances in the loss. In particular, large object instances will
  no longer contribute an overwhelming number of the terms in the loss,
  a limitation inherent to `npair_loss`.

  Args:
    embedding: A tf.float32 tensor of [height, width, dims].
    instance_labels: A tf.int32 tensor of [height, width] containing instance
      ids. Assumed values in target start from 0 and cover 0 to N-1.
    num_samples: Number of sampled points from each label.
    similarity_strategy: Defines the method for computing similarity between
      embedding vectors. Possible values are 'dotproduct' and 'distance'.
    loss_strategy: Defines the type of loss including 'softmax' or 'sigmoid'.

  Returns:
    Total loss value.

  Raises:
    ValueError: If strategy is unknown.
  """
  instance_labels = tf.reshape(instance_labels, [-1])
  embedding_shape = tf.shape(embedding)
  embedding_dims = embedding_shape[2]
  embedding = tf.reshape(embedding, tf.stack([-1, embedding_dims]))
  valid_mask = tf.greater_equal(instance_labels, 0)
  embedding = tf.boolean_mask(embedding, valid_mask)
  instance_labels = tf.boolean_mask(instance_labels, valid_mask)
  (center_indices,
   instance_label_one_hot) = isu.randomly_select_one_point_per_segment(
       instance_labels)
  centers = tf.gather(embedding, center_indices)
  sampled_indices = isu.randomly_select_n_points_per_segment(instance_labels,
                                                             num_samples)
  sampled_indices = tf.reshape(sampled_indices, [-1])
  sampled_embedding = tf.gather(embedding, sampled_indices)
  sampled_instance_label_one_hot = tf.gather(instance_label_one_hot,
                                             sampled_indices)
  embedding_similarity = eutils.embedding_centers_to_logits(centers,
                                                            sampled_embedding,
                                                            similarity_strategy)
  if loss_strategy == 'softmax':
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    losses = loss_fn(
        y_true=sampled_instance_label_one_hot, y_pred=embedding_similarity)
    return tf.reduce_mean(losses)
  elif loss_strategy == 'sigmoid':
    if similarity_strategy == 'distance':
      sampled_instance_label_one_hot *= 0.5
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    losses = loss_fn(
        y_true=sampled_instance_label_one_hot, y_pred=embedding_similarity)
    return tf.reduce_mean(losses)
  else:
    raise ValueError('Loss strategy is unknown')


def instance_embedding_npair_random_sample_loss(
    embedding,
    instance_labels,
    num_samples,
    similarity_strategy='dotproduct',
    loss_strategy='softmax'):
  """Compute n-pair loss by drawing random samples from segments.

  Given n samples from each of the k instance, we create a [k x n x n, k]
  matrix of dot products. We compute the softmax loss for the rows of this
  matrix and average.

  Args:
    embedding: A tf.float32 tensor of [height, width, dims].
    instance_labels: A tf.int32 tensor of [height, width] containing
      instance ids. Assumed values in target start from 0 and cover 0 to N-1.
    num_samples: Number of sampled points from each label.
    similarity_strategy: Defines the method for computing similarity between
      embedding vectors. Possible values are 'dotproduct' and 'distance'.
    loss_strategy: Defines the type of loss including 'softmax' or 'sigmoid'.

  Returns:
    Total loss value.

  Raises:
    ValueError: If loss or similarity strategy are unknown.
  """
  embedding_shape = tf.shape(embedding)
  embedding_dims = embedding_shape[2]
  embedding = tf.reshape(embedding, tf.stack([-1, embedding_dims]))
  instance_labels = tf.reshape(instance_labels, [-1])
  valid_mask = tf.greater_equal(instance_labels, 0)
  embedding = tf.boolean_mask(embedding, valid_mask)
  instance_labels = tf.boolean_mask(instance_labels, valid_mask)
  num_instances = tf.reduce_max(instance_labels) + 1
  # num_samples (ns) x num_instances (ni)
  sampled_indices = isu.randomly_select_n_points_per_segment(instance_labels,
                                                             num_samples)
  sampled_indices_ns_ni = tf.reshape(sampled_indices, [-1])
  sampled_indices = tf.transpose(sampled_indices)
  sampled_indices_ni_ns = tf.reshape(sampled_indices, [-1])
  sampled_embedding_ns_ni = tf.gather(embedding, sampled_indices_ns_ni)
  sampled_embedding_ni_ns = tf.gather(embedding, sampled_indices_ni_ns)
  sampled_instance_label_ns_ni = tf.gather(instance_labels,
                                           sampled_indices_ns_ni)
  sampled_instance_label_ns_ni = tf.one_hot(sampled_instance_label_ns_ni,
                                            num_instances)
  sampled_instance_label_ni_ns = tf.gather(instance_labels,
                                           sampled_indices_ni_ns)
  sampled_instance_label_ni_ns = tf.one_hot(sampled_instance_label_ni_ns,
                                            num_instances)
  # [nc x ns, ns x nc]
  target_one_hot = tf.matmul(sampled_instance_label_ni_ns,
                             sampled_instance_label_ns_ni,
                             transpose_b=True)
  embedding_similarity = eutils.embedding_centers_to_logits(
      sampled_embedding_ns_ni, sampled_embedding_ni_ns, similarity_strategy)

  target_one_hot = tf.reshape(target_one_hot, [-1, num_instances])
  embedding_similarity = tf.reshape(embedding_similarity, [-1, num_instances])
  if loss_strategy == 'softmax':
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    losses = loss_fn(
        y_true=target_one_hot, y_pred=embedding_similarity)
    return tf.reduce_mean(losses)
  elif loss_strategy == 'sigmoid':
    if similarity_strategy == 'distance':
      target_one_hot *= 0.5
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    losses = loss_fn(
        y_true=target_one_hot, y_pred=embedding_similarity)
    return tf.reduce_mean(losses)
  else:
    raise ValueError('Loss strategy is unknown')


def instance_embedding_iou_loss(embedding,
                                instance_labels,
                                num_samples,
                                similarity_strategy='dotproduct'):
  """IOU loss on soft masks predicted from embedding.

  Here is how this loss is implemented. First draws one random seed point from
  each instance. Then it computes the similarity between each pixel embedding
  and each of the seed embedding vectors. Ideally, we like pixels that are
  in the same instance with a seed have a large dot product with the embedding
  of that seed and pixels that are in other instances have a very small
  dot product with the embedding of that seed. Assume we call the embedding
  vector of the seed as s and embedding vector of a pixel as e. For each seed,
  we build a soft mask where the mask value at each pixel is
  exp(e.s)/exp(s.s). For each seed mask, we compute its loss as 1.0 - IOU with
  the ground-truth instance that it corresponds to.
  We average these values to compute the final loss.

  This process can happen multiple times. Each time we sample seeds at random,
  compute the loss and at the end average the losses.
  The argument num_samples defines how many times to repeat this process.

  Args:
    embedding: A tf.float32 tensor of [height, width, dims].
    instance_labels: A tf.int32 tensor of [height, width] containing
      instance ids. Assumed values in target start from 0 and cover 0 to N-1.
    num_samples: Number of samples.
    similarity_strategy: Defines the method for computing similarity between
      embedding vectors. Possible values are 'dotproduct' and 'distance'.

  Returns:
    Iou loss of the sigmoid of masks that are grown from random points.
    Scalar tensor of type tf.float32.
  """
  embedding_shape = tf.shape(embedding)
  height = embedding_shape[0]
  width = embedding_shape[1]
  dims = embedding_shape[2]
  embedding = tf.reshape(embedding, tf.stack([-1, dims]))
  instance_labels = tf.reshape(instance_labels, [-1])
  num_inst = tf.reduce_max(instance_labels) + 1
  indices = isu.randomly_select_n_points_per_segment(instance_labels,
                                                     num_samples)
  indices = tf.reshape(indices, [-1])
  init_centers = tf.gather(embedding, indices)
  soft_masks = eutils.embedding_centers_to_soft_masks(
      embedding,
      init_centers,
      similarity_strategy)
  soft_masks = tf.reshape(soft_masks,
                          tf.stack([num_samples * num_inst, height, width]))
  instance_masks = tf.one_hot(instance_labels, num_inst, dtype=tf.float32)
  instance_masks = tf.transpose(instance_masks)
  instance_masks = tf.tile(tf.expand_dims(instance_masks, axis=0),
                           tf.stack([num_samples, 1, 1]))
  instance_masks = tf.reshape(instance_masks,
                              tf.stack([num_samples * num_inst, height, width]))
  # Loss on pixels inside ground-truth segment
  loss_fn = tf.keras.losses.MeanSquaredError()
  losses = loss_fn(
      y_pred=tf.reshape(soft_masks, [-1]),
      y_true=tf.reshape(instance_masks, [-1]))
  loss1 = tf.reduce_mean(losses * tf.reshape(instance_masks, [-1]))
  # Loss on pixels outside ground-truth segment
  loss2 = tf.reduce_mean(losses * (1.0 - tf.reshape(instance_masks, [-1])))
  # Loss on pixels in the difference between ground-truth and predicted segments
  diff_weights = tf.maximum(soft_masks, instance_masks) - tf.minimum(
      soft_masks, instance_masks)
  loss3 = tf.reduce_mean(losses * tf.reshape(diff_weights, [-1]))

  # IOU loss
  loss4 = tf.reduce_mean(
      losses * tf.reshape(tf.maximum(instance_masks, soft_masks), [-1]))

  return (loss1 + loss2 + loss3 + loss4) / 4.0


def get_instance_embedding_loss(embedding,
                                instance_loss_type,
                                instance_labels,
                                crop_area,
                                crop_min_height,
                                num_samples=10,
                                similarity_strategy='dotproduct',
                                loss_strategy='softmax'):
  """Returns the instance embedding loss based on instance_loss_type.

  Args:
    embedding: A tf.float32 tensor of size [height, width, dims] or
               [batch_size, height, width, dims].
    instance_loss_type: A string containing the type of the embedding loss.
    instance_labels: A tf.int32 tensor of size [height, width] or
                     [batch_size, heigh, width] containing instance ids.
                     Assumed values in target start from 0 and cover 0 to N-1.
    crop_area: Area of the crop window. Only used in some cases of embedding
               loss.
    crop_min_height: Minimum height of the crop window. Only used in some cases
                     of embedding loss.
    num_samples: Number of samples. Only used in some cases of embedding loss.
    similarity_strategy: Defines the method for computing similarity between
                         embedding vectors. Possible values are 'dotproduct' and
                         'distance'.
    loss_strategy: Defines the type of loss including 'softmax' or 'sigmoid'.

  Returns:
    Instance embedding loss.

  Raises:
    ValueError: If instance loss type is not known.
  """
  # Handling the case where there is a batch size.
  embedding_shape = embedding.get_shape().as_list()
  if len(embedding_shape) == 4:
    num_batches = embedding_shape[0]
    losses = []
    embedding_list = tf.unstack(embedding)
    instance_label_list = tf.unstack(instance_labels)
    for i in range(num_batches):
      embedding_i = embedding_list[i]
      instance_labels_i = instance_label_list[i]
      loss = get_instance_embedding_loss(embedding_i,
                                         instance_loss_type,
                                         instance_labels_i,
                                         crop_area,
                                         crop_min_height,
                                         num_samples,
                                         similarity_strategy,
                                         loss_strategy)
      losses.append(loss)
    return tf.reduce_mean(tf.stack(losses))
  if instance_loss_type == 'npair':
    return instance_embedding_npair_loss(
        embedding=embedding,
        instance_labels=instance_labels,
        crop_min_height=crop_min_height,
        crop_area=crop_area,
        similarity_strategy=similarity_strategy,
        loss_strategy=loss_strategy)
  elif instance_loss_type == 'npair_r_c':
    return instance_embedding_npair_random_center_loss(
        embedding=embedding,
        instance_labels=instance_labels,
        similarity_strategy=similarity_strategy,
        loss_strategy=loss_strategy)
  elif instance_loss_type == 'npair_r_c_r_s':
    return instance_embedding_npair_random_center_random_sample_loss(
        embedding=embedding,
        instance_labels=instance_labels,
        num_samples=num_samples,
        similarity_strategy=similarity_strategy,
        loss_strategy=loss_strategy)
  elif instance_loss_type == 'npair_r_s':
    return instance_embedding_npair_random_sample_loss(
        embedding=embedding,
        instance_labels=instance_labels,
        num_samples=num_samples,
        similarity_strategy=similarity_strategy,
        loss_strategy=loss_strategy)
  elif instance_loss_type == 'iou':
    return instance_embedding_iou_loss(
        embedding=embedding,
        instance_labels=instance_labels,
        num_samples=num_samples,
        similarity_strategy=similarity_strategy)
  else:
    raise ValueError('Instance loss type is not known')
