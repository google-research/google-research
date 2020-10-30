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

# Lint as: python3
"""Augment utility."""

import functools

import deep_representation_one_class.data.augment_ops as aug_ops

registered_ops = {
    'base': aug_ops.base_augment,
    'crop': aug_ops.crop_augment,
    'resize': aug_ops.resize_augment,
    'cnr': aug_ops.crop_and_resize_augment,
    'crop_and_resize': aug_ops.crop_and_resize_augment,
    'shift': aug_ops.shift_augment,
    'hflip': aug_ops.hflip_augment,
    'vflip': aug_ops.vflip_augment,
    'rotate90': aug_ops.rotate90_augment,
    'rotate180': aug_ops.rotate180_augment,
    'rotate270': aug_ops.rotate270_augment,
    'jitter': aug_ops.jitter_augment,
    'blur': aug_ops.blur_augment,
    'gray': aug_ops.gray_augment,
}


def apply_augment(image, ops_list=None):
  """Apply Augmentation Sequence.

  Args:
    image: 3D tensor of (height, width, channel)
    ops_list: list of augmentation operation returned by compose_augment_seq

  Returns:
    Tuple of images
  """
  if ops_list is None:
    return (image,)
  if not isinstance(ops_list, (tuple, list)):
    ops_list = [ops_list]

  def _apply_augment(image, ops):
    for op in ops:
      image = op(image)
    return image

  return tuple([_apply_augment(image, ops) for ops in ops_list])


def compose_augment_seq(aug_list, is_training=False):
  """Compose Augmentation Sequence.

  Args:
    aug_list: List of tuples (aug_type, kwargs)
    is_training: Boolean

  Returns:
    sequence of augmentation ops
  """
  return [
      generate_augment_ops(aug_type, is_training=is_training, **kwargs)
      for aug_type, kwargs in aug_list
  ]


def generate_augment_ops(aug_type, is_training=False, **kwargs):
  """Generate Augmentation Operators.

  Args:
    aug_type: Augmentation type
    is_training: Boolea
    **kwargs: for backward compatibility.

  Returns:
    augmentation ops
  """
  assert aug_type.lower() in registered_ops

  if aug_type.lower() == 'resize':
    size = kwargs['size'] if 'size' in kwargs else 256
    tx_op = aug_ops.Resize(size)

  elif aug_type.lower() == 'crop':
    size = kwargs['size'] if 'size' in kwargs else 224
    tx_op = aug_ops.RandomCrop(size=size)

  elif aug_type.lower() == 'crop_and_resize':
    size = kwargs['size'] if 'size' in kwargs else 224
    min_scale = kwargs['min_scale'] if 'min_scale' in kwargs else 0.5
    tx_op = aug_ops.RandomCropAndResize(size=size, min_scale=min_scale)

  elif aug_type.lower() == 'shift':
    pad = kwargs['pad'] if 'pad' in kwargs else int(0.125 * kwargs['size'])
    tx_op = aug_ops.RandomShift(pad=pad)

  elif aug_type.lower() == 'hflip':
    tx_op = aug_ops.RandomFlipLeftRight()

  elif aug_type.lower() == 'vflip':
    tx_op = aug_ops.RandomFlipUpDown()

  elif aug_type.lower() == 'rotate90':
    tx_op = aug_ops.Rotate90()

  elif aug_type.lower() == 'rotate180':
    tx_op = aug_ops.Rotate180()

  elif aug_type.lower() == 'rotate270':
    tx_op = aug_ops.Rotate270()

  elif aug_type.lower() == 'jitter':
    brightness = kwargs['brightness'] if 'brightness' in kwargs else 0.125
    contrast = kwargs['contrast'] if 'contrast' in kwargs else 0.4
    saturation = kwargs['saturation'] if 'saturation' in kwargs else 0.4
    hue = kwargs['hue'] if 'hue' in kwargs else 0
    tx_op = aug_ops.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue)

  elif aug_type.lower() == 'gray':
    prob = kwargs['prob'] if 'prob' in kwargs else 0.2
    tx_op = aug_ops.RandomGrayScale(prob=prob)

  elif aug_type.lower() == 'blur':
    prob = kwargs['prob'] if 'prob' in kwargs else 0.5
    tx_op = aug_ops.RandomBlur(prob=prob)

  return functools.partial(tx_op, is_training=is_training)


def retrieve_augment(aug_list, **kwargs):
  """Retrieves Augmentation Sequences.

  Args:
    aug_list: list of strings defining augmentations.
    **kwargs: augmentation arguments.

  Returns:
    list of augmentation functions.
  """

  def _get_augment_args(aug_name, **kwargs):
    """Parses aug_name into name of augmentation and arguments.

    Args:
      aug_name: string defining augmentation name and arguments.
      **kwargs: global arguments.

    Returns:
      string of augmentation name and dictionary of arguments.
    """
    aug_args = kwargs
    if aug_name.startswith('jitter'):
      augs = aug_name.replace('jitter', '')
      if augs:
        augs_list = filter(None, augs.split('_'))
        for aug_arg in augs_list:
          if aug_arg[0] == 'b':
            aug_args['brightness'] = float(aug_arg[1:])
          elif aug_arg[0] == 'c':
            aug_args['contrast'] = float(aug_arg[1:])
          elif aug_arg[0] == 's':
            aug_args['saturation'] = float(aug_arg[1:])
          elif aug_arg[0] == 'h':
            aug_args['hue'] = float(aug_arg[1:])
      aug_name = 'jitter'
    elif aug_name.startswith('cnr'):
      min_scale = aug_name.replace('cnr', '')
      if min_scale:
        aug_args['min_scale'] = float(min_scale)
      aug_name = 'cnr'
    elif aug_name.startswith('crop_and_resize'):
      min_scale = aug_name.replace('crop_and_resize', '')
      if min_scale:
        aug_args['min_scale'] = float(min_scale)
      aug_name = 'crop_and_resize'
    elif aug_name.startswith('crop'):
      crop_size = aug_name.replace('crop', '')
      if crop_size:
        aug_args['crop_size'] = int(float(crop_size))
      aug_name = 'crop'
    elif aug_name.startswith('gray'):
      prob = aug_name.replace('gray', '')
      if prob:
        aug_args['prob'] = float(prob)
      aug_name = 'gray'
    elif aug_name.startswith('blur'):
      prob = aug_name.replace('blur', '')
      if prob:
        aug_args['prob'] = float(prob)
      aug_name = 'blur'
    return aug_name, aug_args

  def _retrieve_augment(aug_name, is_training=True):
    if aug_name in registered_ops:
      return functools.partial(
          registered_ops[aug_name], is_training=is_training)
    else:
      raise NotImplementedError

  # Insert augmentation when starting with `+`.
  assert not aug_list[0].startswith(
      '+'), 'Default augmentation cannot start with "+"'
  aug_list = [
      aug_list[0] + aug if aug.startswith('+') else aug for aug in aug_list
  ]
  # Retrieve augmentation ops by chaining ops.
  aug_fn_list = []
  for aug_names in aug_list:
    aug_name_list = filter(None, aug_names.split('+'))
    aug_fn = []
    for aug_name in aug_name_list:
      aug_name, aug_args = _get_augment_args(aug_name, **kwargs)
      if aug_name in ['', 'x']:
        aug_fn = _retrieve_augment('base', is_training=False)(**kwargs)
      else:
        aug_fn += _retrieve_augment(aug_name, is_training=True)(**aug_args)
    aug_fn_list.append(aug_fn)
    if len(aug_fn_list) == 1:
      test_aug_fn = []
      aug_name_list = filter(None, aug_names.split('+'))
      for aug_name in aug_name_list:
        aug_name, aug_args = _get_augment_args(aug_name, **kwargs)
        if aug_name in ['', 'x']:
          test_aug_fn = _retrieve_augment('base', is_training=False)(**kwargs)
        else:
          test_aug_fn += _retrieve_augment(
              aug_name, is_training=False)(**aug_args)
  return aug_fn_list + [test_aug_fn]
