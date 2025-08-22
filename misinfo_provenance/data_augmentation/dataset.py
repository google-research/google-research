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
"""Implement data augmentation functions tailored for self-supervised learning.
"""

import os
import pickle

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tensorflow.compat.v2 as tf
from tensorflow_addons import image as tfa_image
from utils import EMOJI_LIST
from utils import FONT_LIST

tfa_cutout = tfa_image.cutout
tfa_blur = tfa_image.gaussian_filter2d
tfa_rotate = tfa_image.rotate
layers = tf.keras.layers
# Path to font directory.
BASE_FONT_PATH = 'INSERT-PATH-TO-FONTS'

EMOJI_BASE_PATH = 'INSERT-PATH-TO-EMOJIS'

MEME_DEFAULT_FONT = f'{BASE_FONT_PATH}/Raleway-ExtraBold.ttf'
MEME_DEFAULT_COLOR = (0, 0, 0)
WHITE_RGB_COLOR = (255, 255, 255)
RED_RGB_COLOR = (255, 0, 0)
MEME_HEIGHT = 64
IMAGE_SIZE = 256


class RandomResizedCrop(layers.Layer):
  """Perform random resized crop.

  Attributes:
    scale:
    log_ratio:
    seed:
  """

  def __init__(self, scale, ratio, seed=0):
    super().__init__()
    self.scale = scale
    self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
    self.seed = 0

  def apply_transformation_to_mask(self, masks, batch_size, bounding_boxes):

    if len(masks.shape) <= 3:
      # include RGB channel.
      masks = tf.repeat(tf.expand_dims(masks, axis=-1), 3, axis=-1)
    height = tf.shape(masks)[1]
    width = tf.shape(masks)[2]
    masks = tf.image.crop_and_resize(
        masks,
        bounding_boxes,
        tf.range(batch_size),
        (height, width),
        method='nearest',
    )
    # RGB to Single channel.
    masks = masks[:, :, :, 0]

    return masks.numpy()

  def call(self, images, masks=None):
    """Perform Random Resized Crop.

    Args:
      images: tf.Tensor with shame (B,H,W,3).
      masks: list of tf.Tensors, each tf.Tensor has the shape (B,D,D). The same
        operation performed on the images, will be performed in the masks. The
        masks applies nearest interpolation, aiming to get the pixel indeces
        related to the crops.
    Returns:
      images:
    """

    batch_size = tf.shape(images)[0]
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]

    random_scales = tf.random.uniform((batch_size,), self.scale[0],
                                      self.scale[1])
    random_ratios = tf.exp(
        tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1]))

    new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
    new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
    height_offsets = tf.random.uniform((batch_size,),
                                       0,
                                       1 - new_heights,
                                       seed=self.seed)
    width_offsets = tf.random.uniform((batch_size,),
                                      0,
                                      1 - new_widths,
                                      seed=self.seed)

    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    images = tf.image.crop_and_resize(
        images,
        bounding_boxes,
        tf.range(batch_size),
        (height, width),
    )

    if masks is not None:
      masks_256, masks_64, masks_32, masks_16, masks_8 = masks
      masks_256 = self.apply_transformation_to_mask(masks_256, batch_size,
                                                    bounding_boxes)
      masks_64 = self.apply_transformation_to_mask(masks_64, batch_size,
                                                   bounding_boxes)
      masks_32 = self.apply_transformation_to_mask(masks_32, batch_size,
                                                   bounding_boxes)
      masks_16 = self.apply_transformation_to_mask(masks_16, batch_size,
                                                   bounding_boxes)
      masks_8 = self.apply_transformation_to_mask(masks_8, batch_size,
                                                  bounding_boxes)
      masks = [masks_256, masks_64, masks_32, masks_16, masks_8]

      return images, masks

    return images


# Flip
class FlipLR(layers.Layer):
  """Random Flip to Left or Right."""

  def apply_transformation_to_mask(self, masks):
    if len(masks.shape) <= 3:
      # include RGB channel.
      masks = tf.repeat(tf.expand_dims(masks, axis=-1), 3, axis=-1)
    fliped_masks = tf.image.flip_left_right(masks)

    # RGB to Single channel.
    fliped_masks = fliped_masks[:, :, :, 0]
    return fliped_masks.numpy()

  def call(self, images, masks=None):
    fliped_images = tf.image.flip_left_right(images)

    if masks is not None:
      masks_256, masks_64, masks_32, masks_16, masks_8 = masks
      masks_256 = self.apply_transformation_to_mask(masks_256)
      masks_64 = self.apply_transformation_to_mask(masks_64)
      masks_32 = self.apply_transformation_to_mask(masks_32)
      masks_16 = self.apply_transformation_to_mask(masks_16)
      masks_8 = self.apply_transformation_to_mask(masks_8)
      masks = [masks_256, masks_64, masks_32, masks_16, masks_8]

      return fliped_images, masks

    return fliped_images


class FlipUP(layers.Layer):
  """Random Flip Up Down."""

  def __init__(self, seed=0):
    super().__init__()

  def apply_transformation_to_mask(self, masks):
    if len(masks.shape) <= 3:
      # include RGB channel.
      masks = tf.repeat(tf.expand_dims(masks, axis=-1), 3, axis=-1)
    fliped_masks = tf.image.flip_up_down(masks)

    # RGB to Single channel.
    fliped_masks = fliped_masks[:, :, :, 0]
    return fliped_masks.numpy()

  def call(self, images, masks=None):
    fliped_images = tf.image.flip_up_down(images)

    if masks is not None:
      masks_256, masks_64, masks_32, masks_16, masks_8 = masks
      masks_256 = self.apply_transformation_to_mask(masks_256)
      masks_64 = self.apply_transformation_to_mask(masks_64)
      masks_32 = self.apply_transformation_to_mask(masks_32)
      masks_16 = self.apply_transformation_to_mask(masks_16)
      masks_8 = self.apply_transformation_to_mask(masks_8)
      masks = [masks_256, masks_64, masks_32, masks_16, masks_8]
      return fliped_images, masks

    return fliped_images


class RandomColorJitter(layers.Layer):
  """Randon Color Jitter.

  Attributes:
    seed:
  """

  def __init__(self, seed=0):
    super().__init__()
    self.seed = seed

  def call(self, images, masks=None):
    jimages = tf.image.random_hue(images, 0.2)
    jimages = tf.image.random_saturation(jimages, 0.5, 1.5)
    jimages = tf.image.random_brightness(jimages, 0.3)
    jimages = tf.image.random_contrast(jimages, 0.5, 1.5)

    if masks is not None:
      # Masks won't be changed.
      # We're not changing any image block position.
      return jimages, masks
    return jimages


class RandomBrightness(layers.Layer):
  """Applies random brightness.

  Attributes:
    brightness:
  """

  def __init__(self, brightness):
    super().__init__()
    self.brightness = brightness

  def blend(self, images_1, images_2, ratios):
    return tf.clip_by_value(ratios * images_1 + (1.0 - ratios) * images_2, 0, 1)

  def random_brightness(self, images):
    # random interpolation/extrapolation between the image and darkness.
    return self.blend(
        images,
        0,
        tf.random.uniform((tf.shape(images)[0], 1, 1, 1), 1 - self.brightness,
                          1 + self.brightness),
    )

  def call(self, images, masks=None):
    images = self.random_brightness(images)
    if masks is not None:
      return images, masks
    return images


class RandomJPEG(layers.Layer):
  """JPEG Compression."""

  def __init__(self, min_jpeg_quality=50, max_jpeg_quality=99, seed=(0, 1)):

    self.seed = seed
    self.min_jpeg_quality = min_jpeg_quality
    self.max_jpeg_quality = max_jpeg_quality

  def call(self, images, masks):
    jpgs_images = []
    for image in images:
      jpgs_images.append(
          tf.image.stateless_random_jpeg_quality(
              image,
              min_jpeg_quality=self.min_jpeg_quality,
              max_jpeg_quality=self.max_jpeg_quality,
              seed=self.seed))

    jpgs_images = tf.stack(jpgs_images, axis=0)
    if masks is not None:
      return jpgs_images, masks
    return jpgs_images


class Grayscale(layers.Layer):
  """Grayscale transformation."""

  def __init__(self, seed=0):
    super().__init__()

  def call(self, images, masks=None):
    gray_images = tf.repeat(tf.image.rgb_to_grayscale(images), 3, axis=-1)

    if masks is not None:
      return gray_images, masks

    return gray_images


class RandomRotate(layers.Layer):
  """"Random Rotate."""

  def __init__(self, seed=0):
    super().__init__()
    self.seed = seed

  def apply_transformation_to_mask(self, masks, angles):
    if len(masks.shape) <= 3:
      # include RGB channel.
      masks = tf.repeat(tf.expand_dims(masks, axis=-1), 3, axis=-1)

      rotated_masks = tfa_rotate(masks, angles, fill_value=-1)

      # RGB to Single channel.
      rotated_masks = rotated_masks[:, :, :, 0]
      return rotated_masks.numpy()

  def call(self, images, masks=None):
    batch_size = tf.shape(images)[0]
    # Gerate Random Angles.
    angles = tf.random.uniform((batch_size,), 0, 3.1415, seed=self.seed)

    rotated_imgs = tfa_rotate(images, angles)

    if masks is not None:

      masks_256, masks_64, masks_32, masks_16, masks_8 = masks
      masks_256 = self.apply_transformation_to_mask(masks_256, angles)
      masks_64 = self.apply_transformation_to_mask(masks_64, angles)
      masks_32 = self.apply_transformation_to_mask(masks_32, angles)
      masks_16 = self.apply_transformation_to_mask(masks_16, angles)
      masks_8 = self.apply_transformation_to_mask(masks_8, angles)
      masks = [masks_256, masks_64, masks_32, masks_16, masks_8]
      return rotated_imgs, masks

    return rotated_imgs


class Blur(layers.Layer):
  """Perform Global Blur."""

  def __init__(self, seed=0):
    super().__init__()

  def call(self, images, masks=None):
    blur_imgs = tfa_blur(images, filter_shape=[9, 9], sigma=5)

    if masks is not None:
      return blur_imgs, masks

    return blur_imgs


class RandomCutOut(layers.Layer):
  """Generates a cutout transformation.

  Attributes:
    seed:
  """

  def __init__(self, seed=0):
    super().__init__()
    self.seed = seed

  def _random_center(self, mask_dim_length, image_dim_length, batch_size):
    """Return the cetner of the cutout."""

    if mask_dim_length >= image_dim_length:
      return tf.tile([image_dim_length // 2], [batch_size])

    half_mask_dim_length = mask_dim_length // 2
    return tf.random.uniform(
        shape=[batch_size],
        minval=half_mask_dim_length,
        maxval=image_dim_length - half_mask_dim_length,
        dtype=tf.int32,
        seed=self.seed,
    )

  def apply_transformation_to_mask(self, masks, offset, cutout_center_height,
                                   cutout_center_width, image_height,
                                   image_width):

    if len(masks.shape) <= 3:
      # include RGB channel
      masks = tf.repeat(tf.expand_dims(masks, axis=-1), 3, axis=-1)

    mask_height, mask_width = masks.shape[1], masks.shape[2]
    mask_size = tf.shape(masks)[1] // 2
    cutout_center_mask_height = (cutout_center_height /
                                 image_height) * mask_height
    cutout_center_mask_width = (cutout_center_width / image_width) * mask_width

    cutout_center_mask_height = tf.cast(
        cutout_center_mask_height, dtype=tf.int32)
    cutout_center_mask_width = tf.cast(cutout_center_mask_width, dtype=tf.int32)

    offset = tf.transpose([cutout_center_mask_height, cutout_center_mask_width],
                          [1, 0])

    cutout_masks = tfa_cutout(masks, mask_size, offset, constant_values=-1)

    # RGB to Single channel
    cutout_masks = cutout_masks[:, :, :, 0]
    return cutout_masks.numpy()

  def call(self, images, masks=None):
    batch_size = tf.shape(images)[0]
    image_height, image_width = images.shape[1], images.shape[2]
    mask_size = tf.shape(images)[1] // 2

    cutout_center_height = self._random_center(mask_size, image_height,
                                               batch_size)
    cutout_center_width = self._random_center(mask_size, image_width,
                                              batch_size)
    offset = tf.transpose([cutout_center_height, cutout_center_width], [1, 0])
    cutout_images = tfa_cutout(images, mask_size, offset, constant_values=0)

    if masks is not None:
      masks_256, masks_64, masks_32, masks_16, masks_8 = masks
      masks_256 = self.apply_transformation_to_mask(masks_256, offset,
                                                    cutout_center_height,
                                                    cutout_center_width,
                                                    image_height, image_width)
      masks_64 = self.apply_transformation_to_mask(masks_64, offset,
                                                   cutout_center_height,
                                                   cutout_center_width,
                                                   image_height, image_width)
      masks_32 = self.apply_transformation_to_mask(masks_32, offset,
                                                   cutout_center_height,
                                                   cutout_center_width,
                                                   image_height, image_width)
      masks_16 = self.apply_transformation_to_mask(masks_16, offset,
                                                   cutout_center_height,
                                                   cutout_center_width,
                                                   image_height, image_width)
      masks_8 = self.apply_transformation_to_mask(masks_8, offset,
                                                  cutout_center_height,
                                                  cutout_center_width,
                                                  image_height, image_width)
      masks = [masks_256, masks_64, masks_32, masks_16, masks_8]
      return cutout_images, masks

    return cutout_images


class MemeStyle(layers.Layer):
  """Creates an image in a meme style.

  Reference
  https://github.com/facebookresearch/AugLy/blob/
      c7ace6df316e79b44d5caae4e36521ea8e0d19e1/augly/image/functional.py#L874
  """

  alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()[]{}?!'

  def _generate_rand_text(self, number_of_letters=0):
    if number_of_letters == 0:
      number_of_letters = tf.random.uniform((),
                                            minval=4,
                                            maxval=10,
                                            dtype=tf.int32)

    text = []
    for _ in range(number_of_letters):
      rand_var = tf.random.uniform((),
                                   minval=0,
                                   maxval=len(MemeStyle.alphabet),
                                   dtype=tf.int32)
      text.append(MemeStyle.alphabet[rand_var])
    return ''.join(text)

  def _create_meme(self, image, mask=None, text='', opacity=1.0):

    # Generate random text
    if not text:
      text = self._generate_rand_text()

    # Tensor Image -> PIL image
    image = tf.keras.utils.array_to_img(image)

    width, height = image.size
    font_size = MEME_HEIGHT - 10

    while True:
      font = ImageFont.truetype(MEME_DEFAULT_FONT, font_size)
      text_width, text_height = font.getsize_multiline(text)

      if text_width <= (width - 10) and text_height <= (MEME_HEIGHT - 10):
        break

      font_size -= 5

    meme = Image.new('RGB', (width, height), WHITE_RGB_COLOR)
    meme.paste(image, (0, MEME_HEIGHT))

    x_pos = round((width - text_width) / 2)
    y_pos = round((MEME_HEIGHT - text_height) / 2)

    draw = ImageDraw.Draw(meme)
    draw.multiline_text(
        (x_pos, y_pos),
        text,
        font=font,
        fill=(MEME_DEFAULT_COLOR[0], MEME_DEFAULT_COLOR[1],
              MEME_DEFAULT_COLOR[2], round(opacity * 255)),
        align='center',
    )
    image = tf.keras.utils.img_to_array(draw.t_image) / 255.0

    return image

  def apply_transformation_to_mask(self, masks, height, batch_size):
    final_masks = []
    for i in range(batch_size):
      mask = masks[i]
      mask_height, _ = mask.shape
      y_pos = int((MEME_HEIGHT / height) * mask_height)
      if isinstance(mask, np.array):
        mask = mask.numpy()
      mask[:y_pos, :] = -1
      final_masks.append(mask)
    masks = tf.stack(final_masks, axis=0)
    return masks.numpy()

  def call(self, images, masks=None):
    batch_size = tf.shape(images)[0]

    final_images = []
    for i in range(batch_size):
      image = tf.identity(images[i])
      image = self._create_meme(image)

      final_images.append(image)

    images = tf.stack(final_images, axis=0)

    if masks is not None:
      masks_256, masks_64, masks_32, masks_16, masks_8 = masks
      masks_256 = self.apply_transformation_to_mask(masks_256, IMAGE_SIZE,
                                                    batch_size)
      masks_64 = self.apply_transformation_to_mask(masks_64, IMAGE_SIZE,
                                                   batch_size)
      masks_32 = self.apply_transformation_to_mask(masks_32, IMAGE_SIZE,
                                                   batch_size)
      masks_16 = self.apply_transformation_to_mask(masks_16, IMAGE_SIZE,
                                                   batch_size)
      masks_8 = self.apply_transformation_to_mask(masks_8, IMAGE_SIZE,
                                                  batch_size)
      masks = [masks_256, masks_64, masks_32, masks_16, masks_8]
      return images, masks

    return images


class EmojiOverlay(layers.Layer):
  """Insert an emoji into an image.

  Attributes:
    seed:
    emoji:
  """

  def __init__(self, seed=0):
    super().__init__()
    self.seed = seed

  def _load_random_emoji(self):
    emoji = np.random.choice(EMOJI_LIST)
    emoji = Image.open(f'{EMOJI_BASE_PATH}/{emoji}')
    return emoji

  def _overlay_emoji(self, image, mask=None, text='', opacity=0.7):
    # Tensor Image -> PIL image
    image = tf.keras.utils.array_to_img(image)

    # Get from
    x_pos, y_pos = tf.random.uniform((2,), minval=0, maxval=1, seed=self.seed)

    overlay = self.emoji
    overlay_size = 0.3

    im_width, im_height = image.size
    overlay_width, overlay_height = overlay.size
    new_height = max(1, int(im_height * overlay_size))
    new_width = int(overlay_width * new_height / overlay_height)
    overlay = overlay.resize((new_width, new_height))

    try:
      emoji_mask = overlay.convert('RGBA').getchannel('A')
      emoji_mask = Image.fromarray(
          (np.array(emoji_mask) * opacity).astype(np.uint8))
    except ValueError:
      emoji_mask = Image.new(
          mode='L', size=overlay.size, color=int(opacity * 255))

    x = int(im_width * x_pos)
    y = int(im_height * y_pos)

    aug_image = image.convert(mode='RGBA')
    aug_image.paste(im=overlay, box=(x, y), mask=emoji_mask)
    aug_image = aug_image.convert(mode='RGB')

    image = tf.keras.utils.img_to_array(aug_image) / 255.0

    if mask is not None:

      def overlay_masks(mask):
        mask_height, mask_width = mask.shape
        x = int(mask_width * x_pos)
        y = int(mask_height * y_pos)
        new_height = max(1, int(mask_height * overlay_size))
        new_width = int(overlay_width * new_height / overlay_height)

        if isinstance(mask, np.ndarray):
          mask = mask.numpy()
        mask[y:new_height + y, x:new_width + x] = -1
        mask = tf.convert_to_tensor(mask)

        return mask

      masks_256, masks_64, masks_32, masks_16, masks_8 = mask
      masks_256 = overlay_masks(masks_256)
      masks_64 = overlay_masks(masks_64)
      masks_32 = overlay_masks(masks_32)
      masks_16 = overlay_masks(masks_16)
      masks_8 = overlay_masks(masks_8)
      masks = [masks_256, masks_64, masks_32, masks_16, masks_8]

      return image, masks

    return image

  def call(self, images, masks=None):
    batch_size = tf.shape(images)[0]

    final_images = []
    final_mask_256, final_mask_64, final_mask_32, final_mask_16, final_mask_8, = [], [], [], [], []
    self.emoji = self._load_random_emoji()
    for i in range(batch_size):
      if masks is not None:
        masks_256, masks_64, masks_32, masks_16, masks_8 = masks
        masks_i = masks_256[i], masks_64[i], masks_32[i], masks_16[i], masks_8[
            i]
      image = tf.identity(images[i])

      if masks is not None:
        image, masks_i = self._overlay_emoji(image, masks_i)

      else:
        image = self._overlay_emoji(image)

      # Include result image to final list.
      if masks is not None:
        masks_256, masks_64, masks_32, masks_16, masks_8 = masks_i
        final_mask_256.append(masks_256)
        final_mask_64.append(masks_64)
        final_mask_32.append(masks_32)
        final_mask_16.append(masks_16)
        final_mask_8.append(masks_8)

      final_images.append(image)

    images = tf.stack(final_images, axis=0)
    if masks is not None:
      final_mask_256 = tf.stack(final_mask_256, axis=0).numpy()
      final_mask_64 = tf.stack(final_mask_64, axis=0).numpy()
      final_mask_32 = tf.stack(final_mask_32, axis=0).numpy()
      final_mask_16 = tf.stack(final_mask_16, axis=0).numpy()
      final_mask_8 = tf.stack(final_mask_8, axis=0).numpy()
      masks = [
          final_mask_256, final_mask_64, final_mask_32, final_mask_16,
          final_mask_8
      ]
      return images, masks

    return images


class TextOverlay(layers.Layer):
  """Insert a text into an image.

  Attributes:
    seed:
    font_path:
    font:
    chars:
  """

  def __init__(self, seed=0, font_path=BASE_FONT_PATH):
    super().__init__()
    self.seed = seed

  def _generate_text(self, font_size):

    number_of_letters = tf.random.uniform((),
                                          minval=4,
                                          maxval=10,
                                          dtype=tf.int32)
    number_of_words = tf.random.uniform((), minval=1, maxval=7, dtype=tf.int32)

    text_lists = []
    rand_char = self.chars[tf.random.uniform((),
                                             minval=1,
                                             maxval=len(self.chars) - 1,
                                             dtype=tf.int32)]
    for _ in range(number_of_letters):
      for _ in range(number_of_words):
        text_lists.append([rand_char])

    text_strs = [
        # pyre-fixme[16]: Item `int` of `Union[List[int], List[Union[List[int],
        #  int]], int]` has no attribute `__iter__`.
        ''.join([chr(self.chars[c % len(self.chars)])
                 for c in t])
        for t in text_lists
    ]
    return text_strs

  def _overlay_text(self, image, opacity=0.5, font_size=0.1):

    image = tf.keras.utils.array_to_img(image)
    image = image.convert('RGBA')
    color = tf.random.uniform([
        3,
    ],
                              minval=0,
                              maxval=210,
                              dtype=tf.int32,
                              seed=self.seed)
    width, _ = image.size

    text_strs = self._generate_text(font_size)

    draw = ImageDraw.Draw(image)
    for _, text_str in enumerate(text_strs):
      text_width, text_height = self.font.getsize_multiline(text_str)
      x_pos, y_pos = tf.random.uniform((2,),
                                       minval=0,
                                       maxval=1,
                                       dtype=tf.float32).numpy()
      x_pos = round((x_pos * width) - text_width)
      y_pos = round((y_pos * width) - text_height)
      draw.multiline_text(
          (x_pos, y_pos),
          text=text_str,
          # pyre-fixme[6]: Expected `Optional[ImageFont._Font]` for 3rd
          # param but got
          #  `FreeTypeFont`.
          font=self.font,
          fill=(color[0], color[1], color[2], round(opacity * 255)),
      )

    image = image.convert(mode='RGB')

    return tf.keras.utils.img_to_array(image) / 255.0

  def _load_font(self, font_size):
    """Load font."""

    font_path = np.random.choice(FONT_LIST)
    self.font_path = f'{BASE_FONT_PATH}{font_path}'
    self.font = ImageFont.truetype(self.font_path, font_size)
    pkl_file = os.path.splitext(self.font_path)[0] + '.pkl'
    with open(pkl_file, 'rb') as f:
      self.chars = pickle.load(f)

  def call(self, images, masks=None, font_size=0.3):
    batch_size = tf.shape(images)[0]

    # Load Font.
    height, width = tf.shape(images)[1].numpy(), tf.shape(images)[2].numpy()
    font_size = int(min(width, height) * font_size)
    self._load_font(font_size)

    # Generate Random indices.
    final_images = []

    for i in range(batch_size):
      image = tf.identity(images[i])

      try:
        image = self._overlay_text(image)
      except OSError:
        image = tf.identity(images[i])
        continue

      # Include result image to final list.
      final_images.append(image)

    images = tf.stack(final_images, axis=0)
    if masks is not None:
      return images, masks

    return images
