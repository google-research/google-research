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

"""Transforms used in the Augmentation Policies."""

import random
import freq_helpers
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps


IMAGE_SIZE = 32
# What is the dataset mean and std of the images on the training set
MEANS = [0.49139968, 0.48215841, 0.44653091]
STDS = [0.24703223, 0.24348513, 0.26158784]
PARAMETER_MAX = 10  # What is the max 'level' a transform could be predicted


def random_flip(x):
  """Flip the input x horizontally with 50% probability."""
  if np.random.rand(1)[0] > 0.5:
    return np.fliplr(x)
  return x


def mixup_batch(images, labels, alpha=1.0):
  """Apply mixup to a batch.

  Based on
  https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py.

  Args:
    images: Images in this batch. First dimension indexes over the batch.
    labels: One-hot image labels. First dimension indexes over the batch.
    alpha: Controls the amount of mixup. 0 is no mixup, 1 is uniform, inf
      would be exactly halfway between each pair.

  Returns:
    The mixed-up images and labels in the batch.
  """
  if alpha > 0:
    lam = np.random.beta(alpha, alpha)
  else:
    lam = 1
  batch_size = len(labels)
  idx = np.random.permutation(batch_size)
  mixed_images = lam * images + (1 - lam) * images[idx, :]
  mixed_labels = lam * labels + (1 - lam) * labels[idx, :]
  return mixed_images, mixed_labels


def freq_augment(images,
                 labels,
                 amplitude,
                 magnitude=1.0,
                 proportion_f=0.5,
                 probability=0.5):
  """Add image frequency data augmentation, with label smoothing.

  Args:
    images: Images in this batch. First dimension indexes over the batch.
    labels: One-hot image labels. First dimension indexes over the batch.
    amplitude: Amount of label smoothing to apply to augmented images.
    magnitude: Norm of the image perturbation applied in the augmentation.
    proportion_f: Fraction of the image perturbations that have image power
      spectrum f. The remaining image perturbations have image power spectrum
      1/f.
    probability: Fraction of the images that are perturbed.

  Returns:
    The augmented images and labels.
  """
  mask_augment = np.random.rand(len(labels)) < probability
  mask_f = np.random.rand(len(labels)) < proportion_f
  f_img = freq_helpers.get_fourier_composite_image(kind='f')
  f_img = f_img * magnitude / np.linalg.norm(f_img.flatten())
  inv_f_img = freq_helpers.get_fourier_composite_image(kind='1/f')
  inv_f_img = inv_f_img * magnitude / np.linalg.norm(inv_f_img.flatten())
  f_mask = np.logical_and(mask_augment, mask_f)
  inv_f_mask = np.logical_and(mask_augment, np.logical_not(mask_f))
  images[f_mask] += f_img
  images[inv_f_mask] += inv_f_img
  labels[inv_f_mask] = labels[inv_f_mask] * (
      1 - amplitude) + amplitude / labels.shape[1]
  labels[f_mask] = labels[f_mask] * (
      1 - amplitude) + amplitude / labels.shape[1]
  return images, labels


def add_radial_noise(images,
                     labels,
                     frequency,
                     amplitude,
                     noise_class=-1,
                     normalize=False):
  """Add label smoothing where the magnitude of the smoothing is a radial wave.

  Args:
    images: Images in this batch. First dimension indexes over the batch.
    labels: One-hot image labels. First dimension indexes over the batch.
    frequency: Frequency (in image norm space) of the radial wave.
    amplitude: Desired average magnitude of label smoothing over the batch.
      The actual amplitude of the radial wave will be scaled to achieve this
      desired average magnitude.
    noise_class: The class to which label smoothing should be applied. Negative
      numbers denote that label smoothing should be applied to all classes.
    normalize: If True, rescale the actual amplitude over the batch, so that the
      average effective amplitude is as desired.

  Returns:
    The smoothed labels (in one-hot format, but not actually one-hot anymore).
  """
  # Phase is the expected norm of a normalized CIFAR10-sized image
  phase = 55.421115476835411279571604166421314671126762248207653392500044304
  radial_noise = np.sin(
      2 * np.pi * frequency *
      (np.linalg.norm(images.reshape(images.shape[0], -1), axis=1) - phase))
  # Shift radial noise to always be nonnegative
  radial_noise = 1.0 + radial_noise
  if normalize:
    # Rescale amplitude as needed to achieve desired effective amplitude
    # If frequency is sufficiently high then noise_l1 \approx 1
    noise_l1 = np.mean(radial_noise)
    if noise_l1 == 0.0:
      amplitude = 0.0
    else:
      amplitude = amplitude / noise_l1
  radial_noise = amplitude * radial_noise
  radial_noise = radial_noise[:, np.newaxis]
  num_classes = labels.shape[1]
  # Compute a mask for which examples are from noise_class
  mask = np.argmax(labels, axis=1) == noise_class
  if noise_class < 0:
    labels = labels * (1 - radial_noise) + radial_noise / num_classes
  else:
    labels[mask] = labels[mask] * (
        1 - radial_noise[mask]) + radial_noise[mask] / num_classes
  return labels


def add_sinusoidal_noise(images,
                         labels,
                         frequency,
                         amplitude,
                         direction,
                         noise_class=-1,
                         normalize=False):
  """Add label smoothing according to a sine wave along specified direction."""
  # Make sure direction is normalized
  direction = direction / np.linalg.norm(direction.flatten())
  # Compute sinusoid
  proj_dir = np.dot(images.reshape(-1, 32*32*3), direction.flatten())
  sinusoidal_noise = np.sin(2 * np.pi * frequency * proj_dir)
  # Shift sinusoid to be nonnegative
  sinusoidal_noise = 1.0 + sinusoidal_noise
  if normalize:
    # Rescale amplitude as needed to achieve desired effective amplitude
    # If frequency is sufficiently high then noise_l1 \approx 1
    noise_l1 = np.mean(sinusoidal_noise)
    if noise_l1 == 0.0:
      amplitude = 0.0
    else:
      amplitude = amplitude / noise_l1
  sinusoidal_noise = amplitude * sinusoidal_noise
  sinusoidal_noise = sinusoidal_noise[:, np.newaxis]
  num_classes = labels.shape[1]
  # Compute a mask for which examples are from noise_class
  mask = np.argmax(labels, axis=1) == noise_class
  if noise_class < 0:
    labels = labels * (1 - sinusoidal_noise) + sinusoidal_noise / num_classes
  else:
    labels[mask] = labels[mask] * (
        1 - sinusoidal_noise[mask]) + sinusoidal_noise[mask] / num_classes
  return labels


def add_uniform_noise(labels, amplitude, noise_class=-1):
  # Compute a mask for which examples are from noise_class
  mask = np.argmax(labels, axis=1) == noise_class
  if noise_class < 0:
    labels = labels * (1 - amplitude) + amplitude / labels.shape[1]
  else:
    labels[mask] = labels[mask] * (1 - amplitude) + amplitude / labels.shape[1]
  return labels


def zero_pad_and_crop(img, amount=4):
  """Zero pad by `amount` zero pixels on each side then take a random crop.

  Args:
    img: numpy image that will be zero padded and cropped.
    amount: amount of zeros to pad `img` with horizontally and verically.

  Returns:
    The cropped zero padded img. The returned numpy array will be of the same
    shape as `img`.
  """
  padded_img = np.zeros((img.shape[0] + amount * 2, img.shape[1] + amount * 2,
                         img.shape[2]))
  padded_img[amount:img.shape[0] + amount, amount:
             img.shape[1] + amount, :] = img
  top = np.random.randint(low=0, high=2 * amount)
  left = np.random.randint(low=0, high=2 * amount)
  new_img = padded_img[top:top + img.shape[0], left:left + img.shape[1], :]
  return new_img


def create_cutout_mask(img_height, img_width, num_channels, size):
  """Creates a zero mask used for cutout of shape `img_height` x `img_width`.

  Args:
    img_height: Height of image cutout mask will be applied to.
    img_width: Width of image cutout mask will be applied to.
    num_channels: Number of channels in the image.
    size: Size of the zeros mask.

  Returns:
    A mask of shape `img_height` x `img_width` with all ones except for a
    square of zeros of shape `size` x `size`. This mask is meant to be
    elementwise multiplied with the original image. Additionally returns
    the `upper_coord` and `lower_coord` which specify where the cutout mask
    will be applied.
  """
  assert img_height == img_width

  # Sample center where cutout mask will be applied
  height_loc = np.random.randint(low=0, high=img_height)
  width_loc = np.random.randint(low=0, high=img_width)

  # Determine upper right and lower left corners of patch
  upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
  lower_coord = (min(img_height, height_loc + size // 2),
                 min(img_width, width_loc + size // 2))
  mask_height = lower_coord[0] - upper_coord[0]
  mask_width = lower_coord[1] - upper_coord[1]
  assert mask_height > 0
  assert mask_width > 0

  mask = np.ones((img_height, img_width, num_channels))
  zeros = np.zeros((mask_height, mask_width, num_channels))
  mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = (
      zeros)
  return mask, upper_coord, lower_coord


def cutout_numpy(img, size=16):
  """Apply cutout with mask of shape `size` x `size` to `img`.

  The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
  This operation applies a `size`x`size` mask of zeros to a random location
  within `img`.

  Args:
    img: Numpy image that cutout will be applied to.
    size: Height/width of the cutout mask that will be

  Returns:
    A numpy tensor that is the result of applying the cutout mask to `img`.
  """
  img_height, img_width, num_channels = (img.shape[0], img.shape[1],
                                         img.shape[2])
  assert len(img.shape) == 3
  mask, _, _ = create_cutout_mask(img_height, img_width, num_channels, size)
  return img * mask


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / PARAMETER_MAX)


def pil_wrap(img):
  """Convert the `img` numpy tensor to a PIL Image."""
  return Image.fromarray(
      np.uint8((img * STDS + MEANS) * 255.0)).convert('RGBA')


def pil_unwrap(pil_img):
  """Converts the PIL img to a numpy array."""
  pic_array = (np.array(pil_img.getdata()).reshape((32, 32, 4)) / 255.0)
  i1, i2 = np.where(pic_array[:, :, 3] == 0)
  pic_array = (pic_array[:, :, :3] - MEANS) / STDS
  pic_array[i1, i2] = [0, 0, 0]
  return pic_array


def apply_policy(policy, img):
  """Apply the `policy` to the numpy `img`.

  Args:
    policy: A list of tuples with the form (name, probability, level) where
      `name` is the name of the augmentation operation to apply, `probability`
      is the probability of applying the operation and `level` is what strength
      the operation to apply.
    img: Numpy image that will have `policy` applied to it.

  Returns:
    The result of applying `policy` to `img`.
  """
  pil_img = pil_wrap(img)
  for xform in policy:
    assert len(xform) == 3
    name, probability, level = xform
    xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(probability, level)
    pil_img = xform_fn(pil_img)
  return pil_unwrap(pil_img)


class TransformFunction(object):
  """Wraps the Transform function for pretty printing options."""

  def __init__(self, func, name):
    self.f = func
    self.name = name

  def __repr__(self):
    return '<' + self.name + '>'

  def __call__(self, pil_img):
    return self.f(pil_img)


class TransformT(object):
  """Each instance of this class represents a specific transform."""

  def __init__(self, name, xform_fn):
    self.name = name
    self.xform = xform_fn

  def pil_transformer(self, probability, level):

    def return_function(im):
      if random.random() < probability:
        im = self.xform(im, level)
      return im

    name = self.name + '({:.1f},{})'.format(probability, level)
    return TransformFunction(return_function, name)

  def do_transform(self, image, level):
    f = self.pil_transformer(PARAMETER_MAX, level)
    return pil_unwrap(f(pil_wrap(image)))


################## Transform Functions ##################
identity = TransformT('Identity', lambda pil_img, level: pil_img)
flip_lr = TransformT(
    'FlipLR',
    lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
flip_ud = TransformT(
    'FlipUD',
    lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
# pylint:disable=g-long-lambda
auto_contrast = TransformT(
    'AutoContrast',
    lambda pil_img, level: ImageOps.autocontrast(
        pil_img.convert('RGB')).convert('RGBA'))
equalize = TransformT(
    'Equalize',
    lambda pil_img, level: ImageOps.equalize(
        pil_img.convert('RGB')).convert('RGBA'))
invert = TransformT(
    'Invert',
    lambda pil_img, level: ImageOps.invert(
        pil_img.convert('RGB')).convert('RGBA'))
# pylint:enable=g-long-lambda
blur = TransformT(
    'Blur', lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
smooth = TransformT(
    'Smooth',
    lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))


def _rotate_impl(pil_img, level):
  """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
  degrees = int_parameter(level, 30)
  if random.random() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees)


rotate = TransformT('Rotate', _rotate_impl)


def _posterize_impl(pil_img, level):
  """Applies PIL Posterize to `pil_img`."""
  level = int_parameter(level, 4)
  return ImageOps.posterize(pil_img.convert('RGB'), 4 - level).convert('RGBA')


posterize = TransformT('Posterize', _posterize_impl)


def _shear_x_impl(pil_img, level):
  """Applies PIL ShearX to `pil_img`.

  The ShearX operation shears the image along the horizontal axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
  level = float_parameter(level, 0.3)
  if random.random() > 0.5:
    level = -level
  return pil_img.transform((32, 32), Image.AFFINE, (1, level, 0, 0, 1, 0))


shear_x = TransformT('ShearX', _shear_x_impl)


def _shear_y_impl(pil_img, level):
  """Applies PIL ShearY to `pil_img`.

  The ShearY operation shears the image along the vertical axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
  level = float_parameter(level, 0.3)
  if random.random() > 0.5:
    level = -level
  return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, level, 1, 0))


shear_y = TransformT('ShearY', _shear_y_impl)


def _translate_x_impl(pil_img, level):
  """Applies PIL TranslateX to `pil_img`.

  Translate the image in the horizontal direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateX applied to it.
  """
  level = int_parameter(level, 10)
  if random.random() > 0.5:
    level = -level
  return pil_img.transform((32, 32), Image.AFFINE, (1, 0, level, 0, 1, 0))


translate_x = TransformT('TranslateX', _translate_x_impl)


def _translate_y_impl(pil_img, level):
  """Applies PIL TranslateY to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateY applied to it.
  """
  level = int_parameter(level, 10)
  if random.random() > 0.5:
    level = -level
  return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, 0, 1, level))


translate_y = TransformT('TranslateY', _translate_y_impl)


def _crop_impl(pil_img, level, interpolation=Image.BILINEAR):
  """Applies a crop to `pil_img` with the size depending on the `level`."""
  cropped = pil_img.crop((level, level, IMAGE_SIZE - level, IMAGE_SIZE - level))
  resized = cropped.resize((IMAGE_SIZE, IMAGE_SIZE), interpolation)
  return resized


crop_bilinear = TransformT('CropBilinear', _crop_impl)


def _solarize_impl(pil_img, level):
  """Applies PIL Solarize to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had Solarize applied to it.
  """
  level = int_parameter(level, 256)
  return ImageOps.solarize(pil_img.convert('RGB'), 256 - level).convert('RGBA')


solarize = TransformT('Solarize', _solarize_impl)


def _cutout_pil_impl(pil_img, level):
  """Apply cutout to pil_img at the specified level."""
  size = int_parameter(level, 20)
  if size <= 0:
    return pil_img
  img_height, img_width, num_channels = (32, 32, 3)
  _, upper_coord, lower_coord = (
      create_cutout_mask(img_height, img_width, num_channels, size))
  pixels = pil_img.load()  # create the pixel map
  for i in range(upper_coord[0], lower_coord[0]):  # for every col:
    for j in range(upper_coord[1], lower_coord[1]):  # For every row
      pixels[i, j] = (125, 122, 113, 0)  # set the colour accordingly
  return pil_img

cutout = TransformT('Cutout', _cutout_pil_impl)


def _enhancer_impl(enhancer):
  """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""
  def impl(pil_img, level):
    v = float_parameter(level, 1.8) + .1  # going to 0 just destroys it
    return enhancer(pil_img).enhance(v)
  return impl


color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(
    ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))

ALL_TRANSFORMS = [
    identity,
    flip_lr,
    flip_ud,
    auto_contrast,
    equalize,
    invert,
    rotate,
    posterize,
    crop_bilinear,
    solarize,
    color,
    contrast,
    brightness,
    sharpness,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    cutout,
    blur,
    smooth
]

NAME_TO_TRANSFORM = {t.name: t for t in ALL_TRANSFORMS}
TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()
