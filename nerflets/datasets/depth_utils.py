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

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import re
import sys


def read_pfm(filename):
  file = open(filename, 'rb')
  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = file.readline().decode('utf-8').rstrip()
  if header == 'PF':
    color = True
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  dim_match = re.fullmatch(r'^(\d+)\s(\d+)\s', file.readline().decode('utf-8'))
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')

  scale = float(file.readline().rstrip())
  if scale < 0:  # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>'  # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)

  data = np.reshape(data, shape)
  data = np.flipud(data)
  file.close()
  return data, scale


def save_pfm(filename, image, scale=1):
  file = open(filename, 'wb')
  color = None

  image = np.flipud(image)

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3:  # color image
    color = True
  elif len(image.shape) == 2 or len(
      image.shape) == 3 and image.shape[2] == 1:  # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
  file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write(('%f\n' % scale).encode('utf-8'))

  image.tofile(file)
  file.close()
