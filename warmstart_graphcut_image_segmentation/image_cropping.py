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

"""This file is used to crop the images in the original datasets.

The shape and position of the cropping are already predetermined in the code to
minimize the effort. Nevertheless, one can still change the settings in the main
function to produce different croppings.
"""

import cv2
import numpy as np
import os
import sys
import argparse


def imagegroup(string):
  if string in all_groups or string == 'all':
    return string
  raise argparse.ArgumentTypeError(
      'Currently only support the following image groups:', all_groups)


def crop_all_pics(args):
  folder, group = args.folder, args.group
  x1, x2, y1, y2 = frames[group]
  dir = folder + './' + group
  image_list = os.listdir(dir)
  new_dir = dir + '_cropped'
  if not os.path.exists(new_dir):
    os.makedirs(new_dir)
  for image in image_list:
    img = cv2.imread(dir + './' + image, cv2.IMREAD_GRAYSCALE)
    cropped_image = img[x1:x2, y1:y2]
    cv2.imwrite(new_dir + './' + image.split('.')[0] + '_cropped.jpg',
                cropped_image)
  return


if __name__ == '__main__':
  all_groups = ['head', 'birdhouse', 'shoe', 'dog']
  frames = {
      'head': [50, 650, 200, 800],
      'birdhouse': [50, 650, 400, 1000],
      'shoe': [50, 650, 300, 900],
      'dog': [50, 1050, 400, 1400]
  }
  parser = argparse.ArgumentParser()
  parser.add_argument('--folder', '-f', default='./sequential_datasets')
  parser.add_argument('--group', '-g', default='all', type=imagegroup)
  args = parser.parse_args()
  crop_all_pics(args)
