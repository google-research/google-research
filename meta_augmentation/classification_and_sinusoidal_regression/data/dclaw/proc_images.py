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

"""Processes DClaw images."""
from __future__ import print_function
import glob
import os

from PIL import Image

path_to_images = './../dclaw/'

all_images = glob.glob(path_to_images + '*')
print(all_images)

for subdir, dirs, files in os.walk(path_to_images):
  for file_name in files:
    image_file = os.path.join(subdir, file_name)
    im = Image.open(image_file)
    im = im.resize((84, 84), resample=Image.LANCZOS)
    im.save(image_file)
