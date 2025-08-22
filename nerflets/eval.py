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
import os
import cv2

from collections import defaultdict
from tqdm import tqdm
import imageio
import configargparse

from models.rendering import render_rays
from models.nerf import *
from models.coverage import *

from utils import load_ckpt
from utils.point_utils import scale_and_shift_points
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

torch.backends.cudnn.benchmark = True


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
      '--scene_name',
      type=str,
      default='test',
      help='scene name, used as output folder name')
  parser.add_argument(
      '--split', type=str, default='test', help='test or test_train')
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
      '--coverage_type_rbf_softmax_temp',
      type=float,
      default=1.0,
      help='type of coverage functions')
  parser.add_argument(
      '--coverage_type_rbf_weight_min', type=float, default=0.0, help='')

  parser.add_argument(
      '--coverage_with_mlp',
      action='store_true',
      help='apply a weight mlp for coverage functions')

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
      '--chunk',
      type=int,
      default=32 * 1024 * 4,
      help='chunk size to split the input to avoid OOM')

  parser.add_argument(
      '--ckpt_path',
      type=str,
      required=True,
      help='pretrained checkpoint path to load')

  parser.add_argument(
      '--save_depth',
      default=False,
      action='store_true',
      help='whether to save depth prediction')
  parser.add_argument(
      '--depth_format',
      type=str,
      default='pfm',
      choices=['pfm', 'bytes'],
      help='which format to save')

  return parser.parse_known_args()[0]


@torch.no_grad()
def batched_inference(models,
                      coverage_models,
                      embeddings,
                      rays,
                      N_samples,
                      N_importance,
                      use_disp,
                      chunk,
                      point_transform_func=None,
                      topk=0):
  """Do batched inference on rays using chunk."""
  B = rays.shape[0]
  results = defaultdict(list)
  for i in range(0, B, chunk):
    rendered_ray_chunks = \
        render_rays(models,
                    coverage_models,
                    embeddings,
                    rays[i:i+chunk],
                    N_samples,
                    use_disp,
                    0,
                    0,
                    N_importance,
                    chunk,
                    dataset.white_back,
                    test_time=True,
                    point_transform_func=point_transform_func,
                    topk=topk)

    for k, v in rendered_ray_chunks.items():
      results[k] += [v.cpu()]

  for k, v in results.items():
    results[k] = torch.cat(v, 0)
  return results


if __name__ == '__main__':
  args = get_opts()
  w, h = args.img_wh

  kwargs = {
      'root_dir': args.root_dir,
      'split': args.split,
      'img_wh': tuple(args.img_wh)
  }
  if args.dataset_name == 'llff':
    kwargs['spheric_poses'] = args.spheric_poses
  dataset = dataset_dict[args.dataset_name](**kwargs)

  if args.normalize_mlp_inputs:
    scene_box = torch.tensor(dataset.scene_box)
    xyz_transformation = lambda x: \
      scale_and_shift_points(scene_box, x)
  else:
    xyz_transformation = None

  coverage_func_class = coverage_funcs[args.coverage_type]

  embedding_xyz = Embedding(args.N_emb_xyz)
  embedding_dir = Embedding(args.N_emb_dir)
  nerf_coarse = Nerflets(
      n=args.N_nerflets,
      D=args.N_mlp_depth,
      W=args.N_mlp_width,
      in_channels_xyz=6 * args.N_emb_xyz + 3,
      in_channels_dir=6 * args.N_emb_dir + 3,
      with_semantics=args.with_semantics,
      n_classes=args.N_classes,
      topk=args.K_nerflets)
  load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
  nerf_coarse.cuda()

  coverage_coarse = coverage_func_class(
      n=args.N_nerflets,
      with_mlp=args.coverage_with_mlp,
      softmax_temp=args.coverage_type_rbf_softmax_temp,
      weight_min=args.coverage_type_rbf_weight_min)
  load_ckpt(coverage_coarse, args.ckpt_path, model_name='coverage_coarse')
  coverage_coarse.cuda()

  models = {'coarse': nerf_coarse}
  coverage_models = {'coarse': coverage_coarse}
  embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

  if args.with_bg_nerf:
    bg_nerf = BgNeRF(
        D=3,
        W=32,
        in_channels_dir=6 * args.N_emb_dir + 3,
        with_semantics=args.with_semantics,
        n_classes=args.N_classes)
    models['bg'] = bg_nerf
    load_ckpt(bg_nerf, args.ckpt_path, model_name='bg_nerf')
    bg_nerf.cuda().eval()

  if args.N_importance > 0:
    nerf_fine = nerf_coarse
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    nerf_fine.cuda()

    coverage_fine = coverage_coarse
    load_ckpt(coverage_fine, args.ckpt_path, model_name='coverage_fine')
    coverage_fine.cuda()

    models['fine'] = nerf_fine
    coverage_models['fine'] = coverage_fine

  imgs, depth_maps, psnrs = [], [], []
  dir_name = f'results/{args.dataset_name}/{args.scene_name}'
  os.makedirs(dir_name, exist_ok=True)

  for i in tqdm(range(len(dataset))):
    sample = dataset[i]
    rays = sample['rays'].cuda()
    results = batched_inference(
        models,
        coverage_models,
        embeddings,
        rays,
        args.N_samples,
        args.N_importance,
        args.use_disp,
        args.chunk,
        xyz_transformation,
        topk=args.K_nerflets)
    typ = 'fine' if 'rgb_fine' in results else 'coarse'

    img_pred = np.clip(results[f'rgb_{typ}'].view(h, w, 3).cpu().numpy(), 0, 1)

    if args.save_depth:
      depth_pred = results[f'depth_{typ}'].view(h, w).cpu().numpy()
      depth_maps += [depth_pred]
      if args.depth_format == 'pfm':
        save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
      else:
        with open(os.path.join(dir_name, f'depth_{i:03d}'), 'wb') as f:
          f.write(depth_pred.tobytes())

    img_pred_ = (img_pred * 255).astype(np.uint8)
    imgs += [img_pred_]
    imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

    if 'rgbs' in sample:
      rgbs = sample['rgbs']
      img_gt = rgbs.view(h, w, 3)
      psnrs += [metrics.psnr(img_gt, img_pred).item()]

  imageio.mimsave(
      os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)

  if args.save_depth:
    min_depth = np.min(depth_maps)
    max_depth = np.max(depth_maps)
    depth_imgs = (depth_maps - np.min(depth_maps)) / (
        max(np.max(depth_maps) - np.min(depth_maps), 1e-8))
    depth_imgs_ = [
        cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET)
        for img in depth_imgs
    ]
    imageio.mimsave(
        os.path.join(dir_name, f'{args.scene_name}_depth.gif'),
        depth_imgs_,
        fps=30)

  if psnrs:
    mean_psnr = np.mean(psnrs)
    print(f'Mean PSNR : {mean_psnr:.2f}')
