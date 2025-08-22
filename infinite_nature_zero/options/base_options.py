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

# -*- coding: utf-8 -*-
"""Class definition of base config."""
import argparse
import pickle


class BaseOptions():
  """Base Options."""

  def __init__(self):
    self.initialized = False

  def initialize(self, parser):
    """Initialize base options."""
    parser.add_argument(
        '--name',
        type=str,
        default='infinite-nature-zero',
        help='name of the experiment.')

    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='cuda device rank')

    parser.add_argument(
        '--model', type=str, default='render', help='which model to use')

    parser.add_argument(
        '--compositor',
        type=str,
        default='splat',
        help='point renderer type')

    parser.add_argument(
        '--crop_size',
        type=int,
        default=128,
        help='Crop to the width of crop_size')

    parser.add_argument(
        '--aspect_ratio',
        type=float,
        default=1.,
        help='The ratio width/height.')

    parser.add_argument(
        '--use_dpt', action='store_true', help='read DPT depth, default no')

    parser.add_argument(
        '--ada_sky_shift', action='store_true', help='adaptive sky shift')

    parser.add_argument(
        '--load_from_opt_file',
        action='store_true',
        help='load the options from checkpoints and use that as default')

    self.initialized = True
    return parser

  def gather_options(self):
    """Gather options."""

    # initialize parser with basic options
    if not self.initialized:
      parser = argparse.ArgumentParser(
          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
      parser = self.initialize(parser)

    # get the basic options
    opt, _ = parser.parse_known_args()

    if opt.load_from_opt_file:
      parser = self.update_options_from_file(parser, opt)

    opt = parser.parse_args()
    self.parser = parser
    return opt

  def print_options(self, opt):
    """Print options."""

    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
      comment = ''
      default = self.parser.get_default(k)
      if v != default:
        comment = '\t[default: %s]' % str(default)
      message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

  def update_options_from_file(self, parser, opt):
    new_opt = self.load_options(opt)
    for k, v in sorted(vars(opt).items()):
      if hasattr(new_opt, k) and v != getattr(new_opt, k):
        new_val = getattr(new_opt, k)
        parser.set_defaults(**{k: new_val})
    return parser

  def load_options(self, opt):
    file_name = self.option_file_path(opt, makedir=False)
    new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
    return new_opt

  def parse(self):
    opt = self.gather_options()
    opt.is_train = self.is_train  # train or test

    self.print_options(opt)

    self.opt = opt
    return self.opt
