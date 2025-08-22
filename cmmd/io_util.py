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

"""IO utilities."""

import os
from cmmd import embedding
import jax
import numpy as np
from PIL import Image
import tensorflow as tf
import tqdm


def _get_image_list(path):
  ext_list = ['png', 'jpg', 'jpeg']
  image_list = []
  for ext in ext_list:
    image_list.extend(tf.io.gfile.glob(os.path.join(path, f'*{ext}')))
    image_list.extend(tf.io.gfile.glob(os.path.join(path, f'*.{ext.upper()}')))
  # Sort the list to ensure a deterministic output.
  image_list.sort()
  return image_list


def _center_crop_and_resize(im, size):
  w, h = im.size
  l = min(w, h)
  top = (h - l) // 2
  left = (w - l) // 2
  box = (left, top, left + l, top + l)
  im = im.crop(box)
  # Note that the following performs anti-aliasing as well.
  return im.resize((size, size), resample=Image.BICUBIC)  # pytype: disable=module-attr


def _read_image(path, reshape_to):
  with tf.io.gfile.GFile(path, 'rb') as f:
    im = Image.open(f)
    im.load()

  if reshape_to > 0:
    im = _center_crop_and_resize(im, reshape_to)

  return np.asarray(im).astype(np.float32)


def _get_img_generator_fn(path, reshape_to, max_count=-1):
  """Returns a generator function that yields one image at a time.

  Args:
    path: Directory to read .jpg and .png imges from.
    reshape_to: If positive, reshape images to a square images of this size.
    max_count: The maximum number of images to read.

  Returns:
    A generation function that yields images.
  """
  img_path_list = _get_image_list(path)
  if max_count > 0:
    img_path_list = img_path_list[:max_count]

  def gen():
    for img_path in img_path_list:
      x = _read_image(img_path, reshape_to)
      if x.ndim == 3:
        yield x
      elif x.ndim == 2:
        # Convert grayscale to RGB by duplicating the channel dimension.
        yield np.tile(x[Ellipsis, np.newaxis], (1, 1, 3))
      else:
        raise ValueError(
            f'Image has {x.ndim} dimensions, which is not supported. Only '
            'images with 1 or 3 color channels are currently supported.'
        )

  return gen, len(img_path_list)


def compute_embeddings_for_dir(
    img_dir,
    embedding_model,
    batch_size,
    max_count = -1,
):
  """Computes embeddings for the images in the given directory.

  This drops the remainder of the images after batching with the provided
  batch_size to enable efficient computation on TPUs. This usually does not
  affect results assuming we have a large number of images in the directory.

  Args:
    img_dir: Directory containing .jpg or .png image files.
    embedding_model: The embedding model to use.
    batch_size: Batch size for the embedding model inference.
    max_count: Max number of images in the directory to use.

  Returns:
    Computed embeddings of shape (num_images, embedding_dim).
  """
  if jax.device_count() != jax.local_device_count():
    raise ValueError('Multi-process environments are not supported yet.')

  if batch_size % jax.device_count():
    raise ValueError(
        f'Batch size ({batch_size}) must be divisible by the '
        f'device count ({jax.device_count()}).'
    )
  generator_fn, count = _get_img_generator_fn(
      img_dir, reshape_to=embedding_model.input_image_size, max_count=max_count
  )
  print(f'Calculating embeddings for {count} images from {img_dir}.')
  dataset = tf.data.Dataset.from_generator(
      generator_fn,
      output_signature=tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
  )
  per_device_batch_size = batch_size // jax.device_count()
  dataset = dataset.batch(per_device_batch_size, drop_remainder=True)
  dataset = dataset.batch(jax.device_count(), drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  all_embs = []
  for batch in tqdm.tqdm(dataset, total=count // batch_size):
    image_batch = jax.tree.map(np.asarray, batch)

    # Normalize to the [0, 1] range.
    image_batch = image_batch / 255.0

    if np.min(image_batch) < 0 or np.max(image_batch) > 1:
      raise ValueError(
          'Image values are expected to be in [0, 1]. Found:'
          f' [{np.min(image_batch)}, {np.max(image_batch)}].'
      )

    # Compute the embeddings using a pmapped function.
    embs = np.asarray(
        embedding_model.parallel_embed(image_batch)
    )  # The output has shape (num_devices, batch_size, embedding_dim).
    embs = embs.reshape((-1,) + embs.shape[2:])
    all_embs.append(embs)

  all_embs = np.concatenate(all_embs, axis=0)

  return all_embs
