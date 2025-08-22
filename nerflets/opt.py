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
import configargparse


def get_opts():
  parser = configargparse.ArgumentParser(
      config_file_parser_class=configargparse.YAMLConfigFileParser)
  parser.add_argument(
      '-c', '--config', required=False, is_config_file=True, help='config file')

  parser.add_argument(
      '--root_dir', type=str, required=True, help='root directory of dataset')
  parser.add_argument(
      '--dataset_name',
      type=str,
      default='blender',
      choices=['blender', 'llff', 'toybox5', 'kitti360b', 'scannet'],
      help='which dataset to train/val')
  parser.add_argument(
      '--img_wh',
      nargs='+',
      type=int,
      default=[800, 800],
      help='resolution (img_w, img_h) of the image')
  parser.add_argument(
      '--spheric_poses',
      default=False,
      action='store_true',
      help='whether images are taken in spheric poses (for llff)')

  #### params for nerflets ####
  parser.add_argument(
      '-n', '--N_nerflets', type=int, default=64, help='number of nerflets')
  parser.add_argument(
      '-k',
      '--K_nerflets',
      type=int,
      default=16,
      help='eval only topk of nerflets')
  parser.add_argument(
      '--coverage_log_freq',
      type=int,
      default=10,
      help='log frequency for coverage functions')
  parser.add_argument(
      '--with_semantics',
      action='store_true',
      help='train with semantic supervision')
  parser.add_argument(
      '--N_classes', type=int, default=6, help='number of semantic classes')
  parser.add_argument(
      '--normalize_mlp_inputs',
      action='store_true',
      help='whether or not normalizing the mlp inputs with '
      'scene bounding boxes')

  parser.add_argument(
      '--freeze_nerflets_loc',
      action='store_true',
      help='whether or not freeze the locations of nerflets')
  parser.add_argument(
      '--nerflets_loc_ref_mesh',
      type=str,
      default='',
      help='path to the nerflets reference mesh file')

  parser.add_argument(
      '--coverage_type',
      type=str,
      choices=['rbf', 'avg'],
      default='rbf',
      help='type of coverage functions')

  parser.add_argument(
      '--coverage_type_rbf_softmax_temp', type=float, default=1.0, help='')
  parser.add_argument(
      '--coverage_type_rbf_weight_min', type=float, default=0.0, help='')

  parser.add_argument(
      '--coverage_with_mlp',
      action='store_true',
      help='apply a weight mlp for coverage functions')
  parser.add_argument(
      '--coverage_pen_weight',
      type=float,
      default=1.0,
      help='apply a weight for coverage penalties')

  parser.add_argument(
      '--with_bg_nerf',
      action='store_true',
      help='fit backgrounds with a small bg nerf')
  ###########################

  #### params for networks ####
  parser.add_argument(
      '--N_mlp_depth',
      type=int,
      default=4,
      help='number of layers for sigma and color features')
  parser.add_argument(
      '--N_mlp_width',
      type=int,
      default=32,
      help='number of channels for sigma and color features')
  ###########################

  parser.add_argument(
      '--N_emb_xyz',
      type=int,
      default=10,
      help='number of frequencies in xyz positional encoding')
  parser.add_argument(
      '--N_emb_dir',
      type=int,
      default=4,
      help='number of frequencies in dir positional encoding')
  parser.add_argument(
      '--N_samples', type=int, default=64, help='number of coarse samples')
  parser.add_argument(
      '--N_importance',
      type=int,
      default=128,
      help='number of additional fine samples')
  parser.add_argument(
      '--use_disp',
      default=False,
      action='store_true',
      help='use disparity depth sampling')
  parser.add_argument(
      '--perturb',
      type=float,
      default=1.0,
      help='factor to perturb depth sampling points')
  # TODO: no?
  parser.add_argument(
      '--noise_std',
      type=float,
      default=1.0,
      help='std dev of noise added to regularize sigma')

  parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
  parser.add_argument(
      '--chunk',
      type=int,
      default=32 * 1024,
      help='chunk size to split the input to avoid OOM')
  parser.add_argument(
      '--num_epochs', type=int, default=16, help='number of training epochs')
  parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')

  parser.add_argument(
      '--ckpt_path',
      type=str,
      default=None,
      help='pretrained checkpoint to load (including optimizers, etc)')
  parser.add_argument(
      '--prefixes_to_ignore',
      nargs='+',
      type=str,
      default=['loss'],
      help='the prefixes to ignore in the checkpoint state dict')
  parser.add_argument(
      '--weight_path',
      type=str,
      default=None,
      help='pretrained model weight to load (do not load optimizers, etc)')

  parser.add_argument(
      '--optimizer',
      type=str,
      default='adam',
      help='optimizer type',
      choices=['sgd', 'adam', 'radam', 'ranger'])
  parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
  parser.add_argument(
      '--momentum', type=float, default=0.9, help='learning rate momentum')
  parser.add_argument(
      '--weight_decay', type=float, default=0, help='weight decay')
  parser.add_argument(
      '--lr_scheduler',
      type=str,
      default='steplr',
      help='scheduler type',
      choices=['steplr', 'cosine', 'poly'])
  #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
  parser.add_argument(
      '--warmup_multiplier',
      type=float,
      default=1.0,
      help='lr is multiplied by this factor after --warmup_epochs')
  parser.add_argument(
      '--warmup_epochs',
      type=int,
      default=0,
      help='Gradually warm-up(increasing) learning rate in optimizer')
  ###########################
  #### params for steplr ####
  parser.add_argument(
      '--decay_step',
      nargs='+',
      type=int,
      default=[20],
      help='scheduler decay step')
  parser.add_argument(
      '--decay_gamma',
      type=float,
      default=0.1,
      help='learning rate decay amount')
  ###########################
  #### params for poly ####
  parser.add_argument(
      '--poly_exp',
      type=float,
      default=0.9,
      help='exponent for polynomial learning rate decay')
  ###########################

  parser.add_argument(
      '--exp_name', type=str, default='exp', help='experiment name')

  return parser.parse_args()
