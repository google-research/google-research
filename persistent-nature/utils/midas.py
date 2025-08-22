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

"""utility function to read midas outputs."""
# from midas/utils.py
import re
import numpy as np


def read_pfm(path):
  """Read pfm file.

  Args:
      path (str): path to file

  Returns:
      tuple: (data, scale)

  Raises:
      Exception: for invalid pfm file
  """
  with open(path, "rb") as file:
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == "PF":
      color = True
    elif header.decode("ascii") == "Pf":
      color = False
    else:
      raise Exception("Not a PFM file: " + path)

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
    if dim_match:
      width, height = list(map(int, dim_match.groups()))
    else:
      raise Exception("Malformed PFM header.")

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:
      # little-endian
      endian = "<"
      scale = -scale
    else:
      # big-endian
      endian = ">"

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

    return data, scale
