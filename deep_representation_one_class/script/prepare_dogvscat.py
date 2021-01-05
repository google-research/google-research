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

# Lint as: python3
"""Dog vs. cat preparation."""

import os
import cv2
import numpy as np
from tqdm import tqdm

DATA_DIR = os.path.join(os.environ['DATA_DIR'], 'dogs-vs-cats')


def generate_flist(split):
  """Generates file lists."""
  flist = os.listdir(os.path.join(DATA_DIR, split))
  g = open(os.path.join(DATA_DIR, f'{split}.txt'), 'w')
  for f in sorted(flist):
    if f.startswith('dog'):
      label = 1
    elif f.startswith('cat'):
      label = 0
    if f.endswith('.jpg'):
      g.write(f'{os.path.join(split, f)} {label}\n')
  g.close()


def resize_and_crop_image(input_file, output_side_length, greyscale=False):
  """Applies resize and crop of image."""
  img = cv2.imread(input_file)
  img = cv2.cvtColor(img,
                     cv2.COLOR_BGR2RGB if not greyscale else cv2.COLOR_BGR2GRAY)
  height, width = img.shape[:2]
  new_height = output_side_length
  new_width = output_side_length
  if height > width:
    new_height = int(output_side_length * height / width)
  else:
    new_width = int(output_side_length * width / height)
  resized_img = cv2.resize(
      img, (new_width, new_height), interpolation=cv2.INTER_AREA)
  height_offset = (new_height - output_side_length) // 2
  width_offset = (new_width - output_side_length) // 2
  cropped_img = resized_img[height_offset:height_offset + output_side_length,
                            width_offset:width_offset + output_side_length]
  assert cropped_img.shape[:2] == (output_side_length, output_side_length)
  return cropped_img


def generate_numpy_array(split, size):
  """Generates numpy array from image files."""
  txt_name = os.path.join(DATA_DIR, f'{split}.txt')
  if not os.path.exists(txt_name):
    generate_flist(split)
  tgt_name = os.path.join(DATA_DIR, f'{split}_{size}x{size}.npz')
  flist = open(txt_name, 'r').readlines()
  assert len(flist) == 25000, 'Data length is less than 25000.'
  imgs_np = None
  labels_np = None
  for f in tqdm(flist):
    fname, label = f.split()
    label = [int(label)]
    img_np = resize_and_crop_image(os.path.join(DATA_DIR, fname), size)
    img_np = img_np[None, :, :, :]
    label_np = np.array(label)[None, :]
    if imgs_np is None:
      imgs_np = img_np
      labels_np = label_np
    else:
      imgs_np = np.concatenate((imgs_np, img_np), axis=0)
      labels_np = np.concatenate((labels_np, label_np), axis=0)
  np.savez(tgt_name, image=imgs_np, label=labels_np)
  print(f'saved at {tgt_name}')


if __name__ == '__main__':
  generate_numpy_array(split='train', size=64)
