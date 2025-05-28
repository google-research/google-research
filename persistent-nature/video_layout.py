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

# pylint: disable=g-multiple-import,invalid-name,redefined-outer-name
"""Figure 8 video for layout model.

Generate a video of camera flying through a sampled synthetic landscape in a
figure 8 pattern, such that the start and end forms a cycle and the video can
loop. This script renders landscapes using the layout representation, which is
slower but achieves better FID compared to the triplane representation.
"""
import argparse
import math
import os

from baukit import renormalize
from models.layout import model_full
import numpy as np
import torch
from tqdm import tqdm
from utils import (video_util, soat_util, sky_util, camera_util,
                   flights, filters, render_settings)

torch.set_grad_enabled(False)
device = 'cuda'

parser = argparse.ArgumentParser(
    description='Generate figure 8 video (layout).'
)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument(
    '--land_ckpt', type=str, default='pretrained/model_terrain.pkl'
)
parser.add_argument(
    '--sky_ckpt', type=str, default='pretrained/model_sky_360.pkl'
)
parser.add_argument('--truncation', type=float, default=0.8)
parser.add_argument('--output_dir', type=str, default='animations/layout')

args = parser.parse_args()
seed = args.seed
output_dir = args.output_dir
os.makedirs(args.output_dir, exist_ok=True)
output_file = os.path.join(args.output_dir, '%04d-' % seed)

print('==================================')
print('RUNNING seed %04d' % seed)

# load models, with added depth-aware smoothing to noise
full_model = (
    model_full.ModelFull(args.land_ckpt, args.sky_ckpt, depth_noise=True)
    .to(device)
    .eval()
)
TRUNCATION = args.truncation

# get nerf rendering parameters
nerf_render_params = render_settings.nerf_render_supersample
nerf_render_fast = render_settings.nerf_render_dense

# use fast rendering for initial camera balancing
full_model.set_nerf_params(**nerf_render_fast)

# initialize SOAT and sky model inputs
G_layout = full_model.terrain_model.layout_model
G_soat = soat_util.init_soat_model(G_layout)
G_sky = full_model.sky_model
grid = sky_util.make_grid(G_sky.G)
input_layer = G_sky.G.synthesis.input

# generate layout features (controlled by seed)
grid_size = 5
layout = soat_util.generate_layout(
    seed,
    grid_h=grid_size,
    grid_w=grid_size,
    device=device,
    truncation_psi=TRUNCATION,
)
z = torch.randn(1, G_layout.layout_generator.z_dim, device=device)
c = None

# sample initial camera
sampled_Rt = G_layout.trajectory_sampler.sample_trajectories(
    G_layout.layout_decoder, layout
)
sampled_camera = camera_util.camera_from_pose(sampled_Rt.squeeze())
print('sampled camera:')
print(sampled_camera)

# noise layout grid
noise_input = torch.randn_like(layout)[
    :, :1
]  # noise input should be a single channel


#### utility functions
def generate_frame(Rt, encode_sky=False, sky_texture=None):
  """render frame from given camera position."""
  camera_params = camera_util.get_full_image_parameters(
      G_layout,
      G_layout.layout_decoder_kwargs.nerf_out_res,
      batch_size=1,
      device=device,
      Rt=Rt,
  )
  # added: noise input to nerf_kwargs
  outputs = full_model(
      z,
      c,
      camera_params,
      truncation=TRUNCATION,
      nerf_kwargs=dict(
          extras=['checkerboard'], cached_layout=layout, noise_input=noise_input
      ),
      sky_texture=sky_texture,
  )
  extras = outputs['extras']
  size = int(math.sqrt(extras['checkerboard'].shape[-1]))
  checker = (
      extras['checkerboard'].view(1, 1, size, size) * 2 - 1
  )  # [-1, 1] normalize
  outputs['checker'] = checker

  if '3dnoise' not in outputs:
    outputs['3dnoise'] = None

  # other composite images
  rgb_overlay_thumb = full_model.composite_sky(
      outputs['rgb_thumb'], outputs['acc_thumb'], outputs['sky_out']
  )
  rgb_overlay_checker = full_model.composite_sky(
      outputs['checker'], outputs['acc_thumb'], outputs['sky_out']
  )
  if outputs['3dnoise'] is not None:
    rgb_overlay_noise = full_model.composite_sky(
        outputs['3dnoise'], outputs['acc_thumb'], outputs['sky_out']
    )

  # move everything to cpu
  outputs_cpu = {
      'rgb_thumb': outputs['rgb_thumb'].cpu(),
      'depth_thumb': outputs['depth_thumb'].cpu(),
      'acc_thumb': outputs['acc_thumb'].cpu(),
      'rgb_up': outputs['rgb_up'].cpu(),
      'depth_up': (
          outputs['depth_up'].cpu() if outputs['depth_up'] is not None else None
      ),
      'acc_up': (
          outputs['acc_up'].cpu() if outputs['acc_up'] is not None else None
      ),
      'checker': checker.cpu(),
      'Rt': Rt.cpu(),
      'sky_mask': outputs['sky_mask'].cpu(),
      'sky_out': outputs['sky_out'].cpu(),
      '3dnoise': (
          outputs['3dnoise'].cpu() if outputs['3dnoise'] is not None else None
      ),
      'rgb_overlay_upsample': outputs['rgb_overlay_upsample'].cpu(),
      'rgb_overlay_thumb': rgb_overlay_thumb.cpu(),
      'rgb_overlay_checker': rgb_overlay_checker.cpu(),
      'rgb_overlay_noise': (
          rgb_overlay_noise.cpu() if outputs['3dnoise'] is not None else None
      ),
  }

  if encode_sky:
    # leave this one on gpu for later
    outputs_cpu['sky_encoder'] = G_sky.encode(outputs['rgb_up'])
  return outputs_cpu


def fly_along_path(cameras, initial_stabilize_frames=20, smooth_end_frames=90):
  """render frames along camera path."""
  frames = []

  # How fast we adjust. Too large and it will overshoot.
  # Too small and it will not react in time to avoid mountains.
  tilt_velocity_scale = (
      0.1  # Keep this small, otherwise you'll get motion sickness.
  )
  offset_velocity_scale = 0.5

  # How far up the image should the horizon be, ideally.
  # Suggested range: 0.5 to 0.7.
  horizon_target = 0.65

  # What proportion of the depth map should be "near" the camera, ideally.
  # The smaller the number, the higher up the camera will fly.
  # Suggested range: 0.05 to 0.2
  near_target = 0.2

  offset = 0
  tilt = 0

  # First get to a stable position for the first camera (not saving frames)
  print('stabilizing camera...')
  for _ in tqdm(range(initial_stabilize_frames)):
    Rt = camera_util.pose_from_camera(
        camera_util.adjust_camera_vertically(cameras[0], offset, tilt)
    ).to(device)[None]
    outputs = generate_frame(Rt, sky_texture=sky_texture)
    tilt, offset = camera_util.update_tilt_and_offset(
        outputs,
        tilt,
        offset,
        horizon_target=horizon_target,
        near_target=near_target,
        tilt_velocity_scale=tilt_velocity_scale,
        offset_velocity_scale=offset_velocity_scale,
    )

  # Remember starting position so we can smoothly interpolate back to it:
  initial_offset = offset
  initial_tilt = tilt

  # update rendering params here
  full_model.set_nerf_params(**nerf_render_params)

  # Now fly along the path
  for i, camera in enumerate(tqdm(cameras)):
    fraction = (i - (len(cameras) - smooth_end_frames)) / smooth_end_frames
    if fraction < 0.0:
      fraction = 0.0
    c = camera_util.adjust_camera_vertically(
        camera,
        camera_util.lerp(offset, initial_offset, fraction),
        camera_util.lerp(tilt, initial_tilt, fraction),
    )
    Rt = camera_util.pose_from_camera(c).to(device)[None]
    outputs = generate_frame(Rt, sky_texture=sky_texture)
    tilt, offset = camera_util.update_tilt_and_offset(
        outputs,
        tilt,
        offset,
        horizon_target=horizon_target,
        near_target=near_target,
        tilt_velocity_scale=tilt_velocity_scale,
        offset_velocity_scale=offset_velocity_scale,
    )
    frames.append(outputs)
  return frames


# get initial camera
initial_camera = sampled_camera
camera_util.INITIAL_CAMERA = sampled_camera

# generate sky texture
print('generating sky...')
Rt = camera_util.pose_from_camera(initial_camera)[None].to(device)
outputs = generate_frame(Rt, encode_sky=True)
sky_encoder_ws = outputs['sky_encoder']
sky_z = z[:, : G_sky.G.z_dim]
start_grid = sky_util.generate_start_grid(seed, input_layer, grid)
sky_pano = sky_util.generate_pano_transform(
    G_sky.G, sky_z, sky_encoder_ws, start_grid
)
sky_texture = sky_pano[None]

# generate flight pattern
cameras = flights.fly_figure8(initial_camera, 30, 901)

# generate frames
frames = fly_along_path(cameras)

# output raw rgb and thumbnail frames
writer = video_util.get_writer(output_file + 'skip00.mp4')
for curr_index in range(len(frames) - 1):
  frame = np.array(
      renormalize.as_image(frames[curr_index]['rgb_overlay_upsample'][0])
  )
  writer.writeFrame(frame)
writer.close()
print('Exported upsample raw')
writer = video_util.get_writer(output_file + 'thumb.mp4')
for curr_index in range(len(frames) - 1):
  frame = np.array(
      renormalize.as_image(frames[curr_index]['rgb_overlay_thumb'][0])
  )
  writer.writeFrame(frame)
writer.close()
print('Exported thumbnail')

# export projected noise if available
if frames[0]['3dnoise'] is not None:
  writer = video_util.get_writer(output_file + '3dnoise.mp4')
  for curr_index in range(len(frames) - 1):
    frame = np.array(renormalize.as_image(frames[curr_index]['3dnoise'][0]))
    writer.writeFrame(frame)
  writer.close()
  print('Exported 3dnoise')

# postprocess: by warping between every 5th frame
skip = 5
writer = video_util.get_writer(output_file + 'skip%02d.mp4' % skip)
for curr_index in tqdm(range(len(frames) - 1)):
  start_index = int(math.floor(curr_index / skip)) * skip
  end_index = start_index + skip
  curr_depth = (
      frames[curr_index]['depth_up']
      if frames[curr_index]['depth_up'] is not None
      else frames[curr_index]['depth_thumb']
  )
  src_warped_start, src_mask_start = filters.align_frames(
      frames[start_index]['rgb_up'],
      frames[start_index]['Rt'],
      frames[curr_index]['Rt'],
      curr_depth,
      G_layout.fov_mean,
      G_layout.layout_decoder_kwargs.far,
  )
  src_warped_end, src_mask_end = filters.align_frames(
      frames[end_index]['rgb_up'],
      frames[end_index]['Rt'],
      frames[curr_index]['Rt'],
      curr_depth,
      G_layout.fov_mean,
      G_layout.layout_decoder_kwargs.far,
  )
  # blend the frames without sky
  weight_start = 1 - (curr_index - start_index) / skip
  weight_end = 1 - (end_index - curr_index) / skip
  warped_blend = weight_start * src_warped_start + weight_end * src_warped_end
  mask_blend = weight_start * src_mask_start + weight_end * src_mask_end

  # get the sky based on current Rt
  Rt = frames[curr_index]['Rt']
  sky_out = sky_util.sample_sky_from_Rt(
      sky_texture, Rt, outputs['rgb_up'].shape[-1], G_layout.fov_mean
  )

  # composite rgb with sky
  warped_composite = full_model.composite_sky(
      warped_blend, frames[curr_index]['sky_mask'], sky_out
  )

  # write the current frame
  frame = np.array(renormalize.as_image(warped_composite[0]))
  writer.writeFrame(frame)
writer.close()
print('Exported upsample processed: skip %d' % skip)

print('Done')
