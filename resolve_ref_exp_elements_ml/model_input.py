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

"""Input pipeline."""

import collections
import os
from deeplab import feature_extractor
from deeplab import preprocess_utils
import tensorflow as tf

LABEL_ID = "label"
IMAGE_ID = "image"
REF_EXP_ID = "ref_exp"
ELEMENTS_TEXT_ID = "elements_text"
ELEMENTS_NEIGHBORS_ID = "elements_neighbors"
ELEMENTS_REF_MATCH_ID = "elements_ref_match"
ELEMENTS_BOX_ID = "elements_box"
ELEMENTS_EXIST_ID = "elements_exists"
ELEMENTS_MASK_ID = "elements_mask"
IMAGE_PAD_WEIGHTS_ID = "image_pad_weights"
ELEMENTS_TYPE_ID = "elements_type"
SELECTED_CANDIDATE_ID = "selected_candidate"

GROUNDTRUTH_XMIN_ID = "groundTruth/bbox/xmin"
GROUNDTRUTH_XMAX_ID = "groundTruth/bbox/xmax"
GROUNDTRUTH_YMIN_ID = "groundTruth/bbox/ymin"
GROUNDTRUTH_YMAX_ID = "groundTruth/bbox/ymax"

# Information that changes from dataset to dataset.
DatasetDescriptor = collections.namedtuple(
    "DatasetDescriptor",
    [
        "subfolder",
        "num_classes",  # Number of semantic classes.
        "ignore_label",  # Ignore label value.
        "label_id",
        "image_id",
        "elements_box_id",
        "elements_text_id",
        "is_tfrecord",
        "has_candidate",
        "has_elements_boxes",
    ])


def get_resize_dim(width, height, image_size):
  """Calculates the size of each dimension so the aspect ratio is not changed.

  Args:
    width: The original width of the image.
    height: The original height of the image.
    image_size: The desired max image size.

  Returns:
    A tuple of resized dimensions, (resized_width, resized_height).
  """
  max_ = tf.maximum(width, height)
  ratio = tf.to_float(max_) / float(image_size)

  new_width = tf.to_float(width) / ratio
  new_height = tf.to_float(height) / ratio

  return tf.to_int32(new_width), tf.to_int32(new_height)


def resize_im(image, image_size, pad_val, channels, features=None):
  """Decodes and resizes the image.

  Args:
    image: Image to resize.
    image_size: The desired max image size.
    pad_val: The value to pad with.
    channels: The number of channels in the image.
    features: Other features to resize.

  Returns:
    Resized image with possible padded regions,
    and possibly the resized elements boxes.
  """
  [height, width, got_channels] = preprocess_utils.resolve_shape(image, rank=3)

  new_height, new_width = get_resize_dim(height, width, image_size)

  image = tf.reshape(image, [height, width, -1])
  image = tf.cond(
      tf.logical_and(channels == 3, tf.equal(got_channels, 1)),
      true_fn=lambda: tf.image.grayscale_to_rgb(image),
      false_fn=lambda: image,
  )

  image = tf.image.resize_images(image, [new_height, new_width])

  image = preprocess_utils.pad_to_bounding_box(image, 0, 0, image_size,
                                               image_size, pad_val)
  if features is not None:
    width, height = tf.to_float(width), tf.to_float(height)
    max_dim = tf.to_float(tf.maximum(width, height))
    features[ELEMENTS_BOX_ID] = features[ELEMENTS_BOX_ID] / max_dim
    if GROUNDTRUTH_XMIN_ID in features:
      features[GROUNDTRUTH_XMIN_ID] *= width / max_dim
      features[GROUNDTRUTH_XMAX_ID] *= width / max_dim
      features[GROUNDTRUTH_YMIN_ID] *= height / max_dim
      features[GROUNDTRUTH_YMAX_ID] *= height / max_dim
  return image


def assert_or_warn(condition, message, is_assert):
  """Errors or prints a warning when the condition is met."""
  if is_assert:
    return tf.Assert(condition, message)
  else:
    return tf.cond(condition, lambda: condition,
                   lambda: tf.Print(condition, message))


refer_descriptor = DatasetDescriptor(
    subfolder="",
    num_classes=2,
    ignore_label=255,
    label_id="mask",
    image_id=IMAGE_ID,
    elements_text_id=ELEMENTS_TEXT_ID,
    elements_box_id=ELEMENTS_BOX_ID,
    is_tfrecord=True,
    has_candidate=True,
    has_elements_boxes=True,
)

dataset_descriptors = {
    "default":
        DatasetDescriptor(
            subfolder="",
            num_classes=2,
            ignore_label=255,
            label_id="mask",
            image_id=IMAGE_ID,
            elements_text_id=ELEMENTS_TEXT_ID,
            elements_box_id=ELEMENTS_BOX_ID,
            is_tfrecord=True,
            has_candidate=False,
            has_elements_boxes=True,
        )
}


def convert_string_neighbors(string_neighbors):
  split = tf.string_split(string_neighbors, "")
  string_dense = tf.sparse_tensor_to_dense(split, default_value="0")
  num = tf.string_to_number(string_dense, out_type=tf.int32)
  bool_neigh = tf.cast(num, tf.bool)
  return bool_neigh


def input_fn_dataset(dataset, flags):
  """Gets the model input from the given dataset."""
  features = {}
  dataset_descriptor = dataset_descriptors[flags.dataset]

  def process_label(label):
    """Preprocesses the label."""
    label = tf.image.decode_image(label, channels=1)
    ignore_label = 255
    label = tf.cast(label, tf.int32)

    if flags.preprocess_divide_label:
      label /= 255

    label = resize_im(label, flags.image_size, ignore_label, 1)
    label = tf.cast(label, tf.int32)
    return label

  def _parse_function(*args):
    """Parses the tf example."""
    serialized_example = args[-1]

    context_feature_names = {
        dataset_descriptor.image_id: tf.FixedLenFeature([], tf.string),
    }
    sequence_feature_names = {}
    if flags.use_ref_exp:
      context_feature_names[REF_EXP_ID] = tf.FixedLenFeature([], tf.string)

    if flags.use_labels:
      if dataset_descriptor.has_candidate:
        context_feature_names[SELECTED_CANDIDATE_ID] = tf.FixedLenFeature(
            [], tf.int64)
        sequence_feature_names[ELEMENTS_MASK_ID] = tf.FixedLenSequenceFeature(
            [], tf.string)
      else:
        context_feature_names[dataset_descriptor.label_id] = tf.FixedLenFeature(
            [], tf.string)

    if dataset_descriptor.has_elements_boxes:
      sequence_feature_names[
          dataset_descriptor.elements_box_id] = tf.FixedLenSequenceFeature(
              [4], dtype=tf.float32)
    if flags.use_elements_texts:
      sequence_feature_names[
          dataset_descriptor.elements_text_id] = tf.FixedLenSequenceFeature(
              [], dtype=tf.string)
    if flags.use_elements_neighbors:
      sequence_feature_names[
          ELEMENTS_NEIGHBORS_ID] = tf.FixedLenSequenceFeature(
              [], dtype=tf.string)
    if flags.use_elements_ref_match:
      sequence_feature_names[
          ELEMENTS_REF_MATCH_ID] = tf.FixedLenSequenceFeature(
              [], dtype=tf.string)

    if flags.use_groundtruth_box:
      context_feature_names[GROUNDTRUTH_XMIN_ID] = tf.FixedLenFeature(
          [], tf.float32)
      context_feature_names[GROUNDTRUTH_XMAX_ID] = tf.FixedLenFeature(
          [], tf.float32)
      context_feature_names[GROUNDTRUTH_YMIN_ID] = tf.FixedLenFeature(
          [], tf.float32)
      context_feature_names[GROUNDTRUTH_YMAX_ID] = tf.FixedLenFeature(
          [], tf.float32)

    context_features, sequence_features = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_feature_names,
        sequence_features=sequence_feature_names,
    )

    features.update(context_features)
    features.update(sequence_features)

    if flags.use_elements_texts:
      features[ELEMENTS_TEXT_ID] = features.pop(
          dataset_descriptor.elements_text_id)
    if dataset_descriptor.has_elements_boxes:
      features[ELEMENTS_BOX_ID] = features.pop(
          dataset_descriptor.elements_box_id)

    image = features.pop(dataset_descriptor.image_id)
    image = tf.image.decode_image(image, channels=3)

    image = tf.cast(image, tf.float32)
    mean_pixel = tf.reshape(
        feature_extractor.mean_pixel(flags.model_variant), [1, 1, 3])

    features[IMAGE_PAD_WEIGHTS_ID] = tf.ones_like(image[:, :, 0:1])
    features[IMAGE_PAD_WEIGHTS_ID] = resize_im(features[IMAGE_PAD_WEIGHTS_ID],
                                               flags.image_size, 0, 1)
    features[IMAGE_PAD_WEIGHTS_ID] = tf.squeeze(features[IMAGE_PAD_WEIGHTS_ID],
                                                2)

    if dataset_descriptor.has_elements_boxes:
      image = resize_im(image, flags.image_size, mean_pixel, 3, features)
    else:
      image = resize_im(image, flags.image_size, mean_pixel, 3)

    if flags.use_labels:
      if dataset_descriptor.has_candidate:
        features[ELEMENTS_MASK_ID] = tf.map_fn(
            process_label,
            features.pop(ELEMENTS_MASK_ID),
            parallel_iterations=128,
            dtype=tf.int32,
            name="mask_map")
        features[LABEL_ID] = tf.gather_nd(features[ELEMENTS_MASK_ID],
                                          [features[SELECTED_CANDIDATE_ID]])
      else:
        label = features.pop(dataset_descriptor.label_id)
        label = process_label(label)
        features[LABEL_ID] = label

    if flags.use_elements_texts:
      features[ELEMENTS_EXIST_ID] = tf.ones_like(
          features[ELEMENTS_TEXT_ID], dtype=tf.int32)
    elif dataset_descriptor.has_elements_boxes:
      features[ELEMENTS_EXIST_ID] = tf.ones(
          tf.shape(features[ELEMENTS_BOX_ID])[:1], dtype=tf.int32)

    if flags.use_elements_neighbors:
      features[ELEMENTS_NEIGHBORS_ID] = convert_string_neighbors(
          features[ELEMENTS_NEIGHBORS_ID])

    features[IMAGE_ID] = image

    return features

  dataset = dataset.map(
      _parse_function,
      num_parallel_calls=flags.dataset_threads).prefetch(flags.batch_size)

  padded_shapes = {
      IMAGE_ID: [None, None, None],
  }
  if flags.use_labels:
    padded_shapes[LABEL_ID] = [None, None, None]
    if flags.use_groundtruth_box:
      padded_shapes[GROUNDTRUTH_XMIN_ID] = []
      padded_shapes[GROUNDTRUTH_XMAX_ID] = []
      padded_shapes[GROUNDTRUTH_YMIN_ID] = []
      padded_shapes[GROUNDTRUTH_YMAX_ID] = []
  if flags.use_elements_texts:
    padded_shapes[ELEMENTS_TEXT_ID] = [None]
    padded_shapes[ELEMENTS_EXIST_ID] = [None]
  if dataset_descriptor.has_elements_boxes:
    padded_shapes[ELEMENTS_BOX_ID] = [None, None]
    padded_shapes[ELEMENTS_EXIST_ID] = [None]
  if flags.use_elements_neighbors:
    padded_shapes[ELEMENTS_NEIGHBORS_ID] = [None, None]
  if flags.use_elements_ref_match:
    padded_shapes[ELEMENTS_REF_MATCH_ID] = [None]

  padded_shapes[IMAGE_PAD_WEIGHTS_ID] = [None, None]

  if flags.use_ref_exp:
    padded_shapes.update({
        REF_EXP_ID: [],
    })
  if dataset_descriptor.has_candidate:
    padded_shapes.update({
        SELECTED_CANDIDATE_ID: [],
        ELEMENTS_MASK_ID: [None, None, None, None],
    })

  dataset = dataset.padded_batch(flags.batch_size, padded_shapes=padded_shapes)
  dataset = dataset.prefetch(1)

  try:
    iterator = dataset.make_one_shot_iterator()
    feature_map = iterator.get_next()
  except ValueError:
    # This means the input pipeline uses placeholders probably because it's in
    # inference mode.
    feature_map = tf.contrib.data.get_single_element(dataset)

  feature_map[IMAGE_ID] = tf.reshape(
      feature_map[IMAGE_ID], [-1, flags.image_size, flags.image_size, 3])

  assert_ops = []
  if dataset_descriptor.has_elements_boxes:
    assert_ops.append(
        assert_or_warn(
            tf.greater_equal(
                tf.reduce_min(feature_map[ELEMENTS_BOX_ID]), -.001), [
                    "Bounding box is negative",
                    tf.reduce_min(feature_map[ELEMENTS_BOX_ID])
                ], flags.incorrect_boxes_as_errors))

    assert_ops.append(
        assert_or_warn(
            tf.less_equal(
                tf.reduce_max(feature_map[ELEMENTS_BOX_ID][:, :, 0] +
                              feature_map[ELEMENTS_BOX_ID][:, :, 2]), 1.001),
            [
                "Bounding box x dim is too large.",
                tf.reduce_max(feature_map[ELEMENTS_BOX_ID][:, :, 0] +
                              feature_map[ELEMENTS_BOX_ID][:, :, 2])
            ], flags.incorrect_boxes_as_errors))

    assert_ops.append(
        assert_or_warn(
            tf.less_equal(
                tf.reduce_max(feature_map[ELEMENTS_BOX_ID][:, :, 1] +
                              feature_map[ELEMENTS_BOX_ID][:, :, 3]), 1.001),
            [
                "Bounding box y dim is too large.",
                tf.reduce_max(feature_map[ELEMENTS_BOX_ID][:, :, 1] +
                              feature_map[ELEMENTS_BOX_ID][:, :, 3])
            ], flags.incorrect_boxes_as_errors))

  with tf.control_dependencies(assert_ops):
    if dataset_descriptor.has_elements_boxes:
      feature_map[ELEMENTS_BOX_ID].set_shape([None, None, 4])
      feature_map[ELEMENTS_EXIST_ID] = tf.cast(feature_map[ELEMENTS_EXIST_ID],
                                               tf.bool)
    if flags.use_labels:
      if flags.output_mode == "segment" or flags.output_mode == "regression":
        feature_map[LABEL_ID] = tf.reshape(
            feature_map[LABEL_ID], [-1, flags.image_size, flags.image_size, 1])
  return feature_map


def get_input_fn(flags):
  """Returns input_fn."""

  def input_fn():
    """Reads the input features from files."""
    dataset_descriptor = dataset_descriptors[flags.dataset]

    with tf.variable_scope("input"):
      pattern = os.path.join(
          os.path.join(flags.dataset_dir, dataset_descriptor.subfolder),
          flags.split + "*")
      print "Pattern", pattern
      dataset = tf.data.Dataset.list_files(pattern)

      dataset = dataset.shuffle(buffer_size=flags.file_shuffle_buffer_size)
      dataset = dataset.repeat()

      def prefetch_map_fn(filename):
        if dataset_descriptor.is_tfrecord:
          return tf.data.TFRecordDataset(filename).prefetch(flags.batch_size)
        else:
          return tf.data.SSTableDataset(filename).prefetch(flags.batch_size)

      dataset = dataset.interleave(
          prefetch_map_fn, cycle_length=100, block_length=flags.batch_size)

      print "shuffle buffer size", flags.shuffle_buffer_size
      dataset = dataset.shuffle(buffer_size=flags.shuffle_buffer_size)

      return input_fn_dataset(dataset, flags)

  return input_fn


def get_serving_input_receiver_fn(flags):
  """Returns serving_input_receiver_fn."""

  def serving_input_receiver_fn():
    """Used for exporting the model. Expects a serialized tf.Example."""
    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[None], name="input")
    receiver_tensors = serialized_tf_example
    dataset = tf.data.Dataset.from_tensor_slices(serialized_tf_example)

    features = input_fn_dataset(dataset, flags)

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  return serving_input_receiver_fn
