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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Applies data augmentation to a dataset useful for SSL.

In this script, we are applying to ImageNet 1k, but you can easily applies to
others.

"""

import pathlib
from dataset import Blur
from dataset import EmojiOverlay
# from dataset import FlipLR
# from dataset import FlipUP
from dataset import Grayscale
from dataset import MemeStyle
from dataset import RandomColorJitter
from dataset import RandomCutOut
from dataset import RandomJPEG
from dataset import RandomResizedCrop
from dataset import RandomRotate
from dataset import TextOverlay

import numpy as np
from PIL import Image
import tensorflow as tf

# Change the next line to the desired output image size dimension.
IMAGE_SIZE = 256
CLASSES = []


def not_flip(image, mask):
  return image, mask


def seq_crop(image, mask):
  """Data Augmented Sequenced to perform a crop.

  Args:
    image:
    mask:

  Returns:

  """
  crop = RandomResizedCrop(scale=(0.08, 0.9), ratio=(3 / 4, 4 / 3))

  # Color Jitter
  color_jitter = RandomColorJitter()

  # FLip
  # flip_up = FlipUP()
  # flip_lr = FlipLR()
  flip = np.random.choice(['flip_up', 'flip_lr', 'not_flip'])

  aug_image, aug_mask = crop(image, mask)
  # aug_image, aug_mask = eval(flip)(aug_image, aug_mask)
  aug_image, aug_mask = color_jitter(aug_image, aug_mask)
  metadata = {'flip': flip, 'blur': 'random_blur', 'crop': 'random_crop'}

  return aug_image, aug_mask, metadata


def seq_sscd(image, mask):
  """Applies the same sequence of data augmentation as.

  Self-Supervised Descriptor for Image Copy Detection (SSCD).

  Args:
    image: Tf.Tensor
    mask: Tf.Tensor
  Returns:
    aug_image, aug_mask, metadata

  """

  # Random FLip
  # flip_up = FlipUP()
  # flip_lr = FlipLR()
  flip = np.random.choice(['flip_up', 'flip_lr', 'not_flip'])
  # aug_image, aug_mask = eval(flip)(image, mask)
  aug_image, aug_mask = not_flip(image, mask)
  metadata = {'flip': flip}

  ## Maybe insert Meme style
  if np.random.choice([0, 1], p=[0.98, 0.02]):
    meme_gen = MemeStyle()
    aug_image, aug_mask = meme_gen(aug_image, aug_mask)
    metadata.update({'meme': 'random_meme'})

  # Maybe insert text
  if np.random.choice([0, 1], p=[0.8, 0.2]):
    text_gen = TextOverlay()
    t_image = tf.identity(aug_image)
    t_mask = aug_mask.copy()
    t_image, t_mask = text_gen(t_image, t_mask, font_size=0.25)
    if len(tf.shape(t_image)) >= 3:
      aug_image = t_image
      aug_mask = t_mask
      metadata.update({'text': 'random_text'})

  # Maybe insert emoji
  if np.random.choice([0, 1], p=[0.8, 0.2]):
    emoji_gen = EmojiOverlay()
    aug_image, aug_mask = emoji_gen(aug_image, aug_mask)
    metadata.update({'emoji': 'random_emoji'})

  # Maybe cutout
  if np.random.choice([0, 1], p=[0.99, 0.01]):
    cut_gen = RandomCutOut()
    aug_image, aug_mask = cut_gen(aug_image, aug_mask)
    metadata.update({'cutout': 'cutout'})

  # Maybe rotate
  if np.random.choice([0, 1], p=[0.9, 0.1]):
    rotate = RandomRotate()
    t_image = tf.identity(aug_image)
    t_mask = aug_mask.copy()
    try:
      t_image, t_mask = rotate(t_image, t_mask)
      metadata.update({'rotate': 'random_rotation'})
      aug_image, aug_mask = t_image, t_mask
    except ValueError:
      pass

  # Crop
  crop = RandomResizedCrop(scale=(0.08, 1), ratio=(3 / 4, 4 / 3))
  aug_image, aug_mask = crop(aug_image, aug_mask)
  metadata.update({'crop': 'random_crop'})

  # Maybe color jitter
  if np.random.choice([0, 1], p=[0.2, 0.8]):
    color_jitter = RandomColorJitter()
    aug_image, aug_mask = color_jitter(aug_image, aug_mask)
    metadata.update({'color_jitter': 'random_color_jitter'})

  # Maybe Gray
  if np.random.choice([0, 1], p=[0.8, 0.2]):
    gray = Grayscale()
    aug_image, aug_mask = gray(aug_image, aug_mask)
    metadata.update({'grayscale': 'grayscale'})

  # Maybe Blur
  if np.random.choice([0, 1], p=[0.5, 0.5]):
    blur = Blur()
    aug_image, aug_mask = blur(aug_image, aug_mask)
    metadata.update({'blur': 'random_blur'})

  # Maybe JPGCompres
  if np.random.choice([0, 1], p=[0.7, 0.3]):
    random_jpg = RandomJPEG()
    aug_image, aug_mask = random_jpg(aug_image, aug_mask)
    metadata.update({'jpeg': 'random_jpeg'})

  return aug_image, aug_mask, metadata


# Convert values to compatible tf.Example types.
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy(
    )  # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_path):
  """Create Tf image example from image path."""

  # Byte images.
  image = Image.open(image_path, mode='rb').convert('RGB')
  image_tensor = tf.keras.utils.img_to_array(image)
  image_tensor = tf.image.resize(image_tensor, (IMAGE_SIZE, IMAGE_SIZE))
  image_tensor = image_tensor / 255.0
  image_tensor = tf.expand_dims(image_tensor, axis=0)

  mask_256 = tf.reshape(tf.range(0, 256**2, dtype=tf.int32), (1, 256, 256))
  mask_64 = tf.reshape(tf.range(0, 64**2, dtype=tf.int32), (1, 64, 64))
  mask_32 = tf.reshape(tf.range(0, 32**2, dtype=tf.int32), (1, 32, 32))
  mask_16 = tf.reshape(tf.range(0, 16**2, dtype=tf.int32), (1, 16, 16))
  mask_8 = tf.reshape(tf.range(0, 8**2, dtype=tf.int32), (1, 8, 8))
  masks = [mask_256, mask_64, mask_32, mask_16, mask_8]

  # Applies data augemtnation to image, resulting on trasnformation 1.
  t1_image, t1_mask, metadata_1 = seq_sscd(image_tensor, masks)

  # Check if t1 was correctly build.
  if len(tf.shape(t1_image)) < 2:
    # Some special combination retrieves a transformed image with shape 0.
    # In this case, we just crop the image.
    image_tensor = tf.keras.utils.img_to_array(image)
    image_tensor = tf.image.resize(image_tensor, (IMAGE_SIZE, IMAGE_SIZE))
    image_tensor = image_tensor / 255.0
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    mask_256 = tf.reshape(tf.range(0, 256**2, dtype=tf.int32), (1, 256, 256))
    mask_64 = tf.reshape(tf.range(0, 64**2, dtype=tf.int32), (1, 64, 64))
    mask_32 = tf.reshape(tf.range(0, 32**2, dtype=tf.int32), (1, 32, 32))
    mask_16 = tf.reshape(tf.range(0, 16**2, dtype=tf.int32), (1, 16, 16))
    mask_8 = tf.reshape(tf.range(0, 8**2, dtype=tf.int32), (1, 8, 8))
    masks = [mask_256, mask_64, mask_32, mask_16, mask_8]

    t1_image, t1_mask, metadata_1 = seq_crop(image_tensor, masks)

  # Applies data augmentation to image, resulting on transformation 2.
  # Reset masks to transformed image 2.
  mask_256 = tf.reshape(tf.range(0, 256**2, dtype=tf.int32), (1, 256, 256))
  mask_64 = tf.reshape(tf.range(0, 64**2, dtype=tf.int32), (1, 64, 64))
  mask_32 = tf.reshape(tf.range(0, 32**2, dtype=tf.int32), (1, 32, 32))
  mask_16 = tf.reshape(tf.range(0, 16**2, dtype=tf.int32), (1, 16, 16))
  mask_8 = tf.reshape(tf.range(0, 8**2, dtype=tf.int32), (1, 8, 8))
  masks = [mask_256, mask_64, mask_32, mask_16, mask_8]

  t2_image, t2_mask, metadata_2 = seq_sscd(image_tensor, masks)

  # Label based on directory structure.
  label = pathlib.Path(image_path).parent.stem
  label = CLASSES.index(label)

  # Dimension.
  image_name = pathlib.Path(image_path).stem
  height, width, bands = np.array(image).shape
  image = tf.keras.utils.array_to_img(image).convert('RGB')
  # Check if t1 was correctly build.
  if len(tf.shape(t2_image)) < 2:
    # Some special combination retrieves a transformed image with shape 0.
    # In this case, we just get a simple crop image.
    image_tensor = tf.keras.utils.img_to_array(image)
    image_tensor = tf.image.resize(image_tensor, (IMAGE_SIZE, IMAGE_SIZE))
    image_tensor = image_tensor / 255.0
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    mask_256 = tf.reshape(tf.range(0, 256**2, dtype=tf.int32), (1, 256, 256))
    mask_64 = tf.reshape(tf.range(0, 64**2, dtype=tf.int32), (1, 64, 64))
    mask_32 = tf.reshape(tf.range(0, 32**2, dtype=tf.int32), (1, 32, 32))
    mask_16 = tf.reshape(tf.range(0, 16**2, dtype=tf.int32), (1, 16, 16))
    mask_8 = tf.reshape(tf.range(0, 8**2, dtype=tf.int32), (1, 8, 8))
    masks = [mask_256, mask_64, mask_32, mask_16, mask_8]

    t2_image, t2_mask, metadata_2 = seq_crop(image_tensor, masks)

  # Encode images to PIL.
  t1_image = tf.keras.utils.array_to_img(tf.squeeze(t1_image)).convert('RGB')
  t2_image = tf.keras.utils.array_to_img(tf.squeeze(t2_image)).convert('RGB')

  # Encode masks.
  t1_mask_256, t1_mask_64, t1_mask_32, t1_mask_16, t1_mask_8 = t1_mask
  t1_mask_256 = tf.squeeze(tf.reshape(t1_mask_256, (1, -1)))
  t1_mask_64 = tf.squeeze(tf.reshape(t1_mask_64, (1, -1)))
  t1_mask_32 = tf.squeeze(tf.reshape(t1_mask_32, (1, -1)))
  t1_mask_16 = tf.squeeze(tf.reshape(t1_mask_16, (1, -1)))
  t1_mask_8 = tf.squeeze(tf.reshape(t1_mask_8, (1, -1)))

  t2_mask_256, t2_mask_64, t2_mask_32, t2_mask_16, t2_mask_8 = t2_mask
  t2_mask_256 = tf.squeeze(tf.reshape(t2_mask_256, (1, -1)))
  t2_mask_64 = tf.squeeze(tf.reshape(t2_mask_64, (1, -1)))
  t2_mask_32 = tf.squeeze(tf.reshape(t2_mask_32, (1, -1)))
  t2_mask_16 = tf.squeeze(tf.reshape(t2_mask_16, (1, -1)))
  t2_mask_8 = tf.squeeze(tf.reshape(t2_mask_8, (1, -1)))

  # Creates features for Tf.records.
  feature = {
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'bands': _int64_feature(bands),
      # Masks 1.
      't1_mask_256': _float_feature(t1_mask_256.numpy().tolist()),
      't1_mask_64': _float_feature(t1_mask_64.numpy().tolist()),
      't1_mask_32': _float_feature(t1_mask_32.numpy().tolist()),
      't1_mask_16': _float_feature(t1_mask_16.numpy().tolist()),
      't1_mask_8': _float_feature(t1_mask_8.numpy().tolist()),
      # Masks 2.
      't2_mask_256': _float_feature(t2_mask_256.numpy().tolist()),
      't2_mask_64': _float_feature(t2_mask_64.numpy().tolist()),
      't2_mask_32': _float_feature(t2_mask_32.numpy().tolist()),
      't2_mask_16': _float_feature(t2_mask_16.numpy().tolist()),
      't2_mask_8': _float_feature(t2_mask_8.numpy().tolist()),
      'image_raw': _bytes_feature(image.tobytes()),
      't1_image': _bytes_feature(t1_image.tobytes()),
      't2_image': _bytes_feature(t2_image.tobytes()),
      'label': _int64_feature(label),
      'image_name': _bytes_feature(image_name.encode()),
      # Metadata w.r.t the augmentation applied
      # assists to track the augmentation on each image.
      'aug1_metadata': _bytes_feature(str(metadata_1).encode()),
      'aug2_metadata': _bytes_feature(str(metadata_2).encode())
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))
