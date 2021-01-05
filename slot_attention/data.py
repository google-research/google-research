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

"""Data utils."""
import collections
import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess_clevr(features, resolution, apply_crop=False,
                     get_properties=True, max_n_objects=10):
  """Preprocess CLEVR."""
  image = tf.cast(features["image"], dtype=tf.float32)
  image = ((image / 255.0) - 0.5) * 2.0  # Rescale to [-1, 1].

  if apply_crop:
    crop = ((29, 221), (64, 256))  # Get center crop.
    image = image[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], :]

  image = tf.image.resize(
      image, resolution, method=tf.image.ResizeMethod.BILINEAR)
  image = tf.clip_by_value(image, -1., 1.)

  if get_properties:
    # One-hot encoding of the discrete features.
    size = tf.one_hot(features["objects"]["size"], 2)
    material = tf.one_hot(features["objects"]["material"], 2)
    shape_obj = tf.one_hot(features["objects"]["shape"], 3)
    color = tf.one_hot(features["objects"]["color"], 8)
    # Originally the x, y, z positions are in [-3, 3].
    # We re-normalize them to [0, 1].
    coords = (features["objects"]["3d_coords"] + 3.) / 6.
    properties_dict = collections.OrderedDict({
        "3d_coords": coords,
        "size": size,
        "material": material,
        "shape": shape_obj,
        "color": color
    })

    properties_tensor = tf.concat(list(properties_dict.values()), axis=1)

    # Add a 1 indicating these are real objects.
    properties_tensor = tf.concat(
        [properties_tensor,
         tf.ones([tf.shape(properties_tensor)[0], 1])], axis=1)

    # Pad the remaining objects.
    properties_pad = tf.pad(
        properties_tensor,
        [[0, max_n_objects - tf.shape(properties_tensor)[0],], [0, 0]],
        "CONSTANT")

    features = {
        "image": image,
        "target": properties_pad
    }

  else:
    features = {"image": image}

  return features


def build_clevr(split, resolution=(128, 128), shuffle=False, max_n_objects=10,
                num_eval_examples=512, get_properties=True, apply_crop=False):
  """Build CLEVR dataset."""
  if split == "train" or split == "train_eval":
    ds = tfds.load("clevr:3.1.0", split="train", shuffle_files=shuffle)
    if split == "train":
      ds = ds.skip(num_eval_examples)
    elif split == "train_eval":
      # Instead of taking the official validation split, we take a smaller split
      # from the training dataset to monitor AP scores during training.
      ds = ds.take(num_eval_examples)
  else:
    ds = tfds.load("clevr:3.1.0", split=split, shuffle_files=shuffle)

  def filter_fn(example, max_n_objects=max_n_objects):
    """Filter examples based on number of objects.

    The dataset only has feature values for visible/instantiated objects. We can
    exploit this fact to count objects.

    Args:
      example: Dictionary of tensors, decoded from tf.Example message.
      max_n_objects: Integer, maximum number of objects (excl. background) for
        filtering the dataset.

    Returns:
      Predicate for filtering.
    """
    return tf.less_equal(tf.shape(example["objects"]["3d_coords"])[0],
                         tf.constant(max_n_objects, dtype=tf.int32))

  ds = ds.filter(filter_fn)

  def _preprocess_fn(x, resolution, max_n_objects=max_n_objects):
    return preprocess_clevr(
        x, resolution, apply_crop=apply_crop, get_properties=get_properties,
        max_n_objects=max_n_objects)
  ds = ds.map(lambda x: _preprocess_fn(x, resolution))
  return ds


def build_clevr_iterator(batch_size, split, **kwargs):
  ds = build_clevr(split=split, **kwargs)
  ds = ds.repeat(-1)
  ds = ds.batch(batch_size, drop_remainder=True)
  return iter(ds)

