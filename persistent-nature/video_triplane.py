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
"""Figure 8 video for triplane model.

Generate a video of camera flying through a sampled synthetic landscape in a
figure 8 pattern, such that the start and end forms a cycle and the video can
loop. This script renders landscapes using the triplane representation, which
is faster and more temporally consistent than the layout representation,
although suboptimal to the layout representation in terms of FID.
"""
import argparse
import math
import os

from baukit import renormalize
from models.triplane import model_full
import numpy as np
import torch
from tqdm import tqdm
from utils import (video_util, soat_util_triplane, sky_util,
                   camera_util, flights, filters, noise_util)


torch.set_grad_enabled(False)
device = 'cuda'

parser = argparse.ArgumentParser(
    description='Generate figure 8 video (triplane).'
)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument(
    '--land_ckpt', type=str, default='pretrained/model_triplane.pkl'
)
parser.add_argument(
    '--sky_ckpt', type=str, default='pretrained/model_sky_360.pkl'
)
parser.add_argument('--truncation', type=float, default=1.0)
parser.add_argument('--output_dir', type=str, default='animations/triplane')

args = parser.parse_args()
seed = args.seed
output_dir = args.output_dir
os.makedirs(args.output_dir, exist_ok=True)
output_file = os.path.join(args.output_dir, '%04d-' % seed)

print('==================================')
print('RUNNING seed %04d' % seed)

# load models, with added depth-aware smoothing to noise
full_model = (
    model_full.ModelFull(args.land_ckpt, args.sky_ckpt).to(device).eval()
)

# initialize SOAT and sky model inputs
G = soat_util_triplane.init_soat_model(full_model.ground).eval().cuda()
G_pano = full_model.sky.G
grid = sky_util.make_grid(G_pano)
input_layer = G_pano.synthesis.input

# render settings
fov = 60
box_warp = G.rendering_kwargs['box_warp']
G.rendering_kwargs['ray_end'] *= 2
G.rendering_kwargs['depth_resolution'] *= 2
G.rendering_kwargs['depth_resolution_importance'] *= 2
G.rendering_kwargs['y_clip'] = 8.0
G.rendering_kwargs['decay_start'] = 0.9 * G.rendering_kwargs['ray_end']
G.rendering_kwargs['sample_deterministic'] = True

grid_size = 5
zs, c = soat_util_triplane.prepare_zs(seed, grid_h=grid_size, grid_w=grid_size)
zs = soat_util_triplane.interpolate_zs(zs)

# generate feature planes
xz_soat = soat_util_triplane.generate_xz(zs, c)  # [1, 32, 512, 512]
xy_soat = soat_util_triplane.generate_xy(zs, c)  # 2 x [1, 32, 256, 512]
yz_soat = soat_util_triplane.generate_yz(zs, c)  # 2 x [1, 32, 256, 512]
planes = [xy_soat, xz_soat, yz_soat]

# set up upsampler and sky inputs
z = zs[0, 0]  # extract a z latent for the upsampler
ws = soat_util_triplane.prepare_ws(z, torch.zeros_like(c))
sky_z = z[:, : G_pano.z_dim]

# rendered noise (may not be used depending on noise_mode for upsampler)
noise_gen = noise_util.build_soat_noise(G, grid_size)
noise_input = noise_gen.get_noise(batch_size=1, device=zs.device)

# sample a random camera
sampled_camera, cam2world_matrix, intrinsics = (
    soat_util_triplane.sample_random_camera(fov, box_warp, seed)
)
intrinsics_matrix = intrinsics[None].to(device)
sampled_Rt = camera_util.pose_from_camera(sampled_camera)


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
    adjusted_cam = camera_util.adjust_camera_vertically(
        sampled_camera, offset, tilt
    )
    outputs, horizon, near = soat_util_triplane.generate_frame(
        G, adjusted_cam, planes, ws, intrinsics_matrix, noise_input
    )
    tilt += tilt_velocity_scale * (horizon - horizon_target)
    offset += offset_velocity_scale * (near - near_target)

  # Remember starting position so we can smoothly interpolate back to it:
  initial_offset = offset
  initial_tilt = tilt

  # generate sky frame after balancing camera
  img_w_gray_sky = outputs['image_w_gray_sky']
  sky_encode = full_model.sky.encode(img_w_gray_sky)
  start_grid = sky_util.generate_start_grid(seed, input_layer, grid)
  sky_texture = sky_util.generate_pano_transform(
      G_pano, sky_z, sky_encode, start_grid
  )
  sky_texture = sky_texture.cuda()[None]

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
    outputs, horizon, near = soat_util_triplane.generate_frame(
        G,
        c,
        planes,
        ws,
        intrinsics_matrix,
        noise_input,
        sky_texture=sky_texture,
        to_cpu=True,
    )
    tilt += tilt_velocity_scale * (horizon - horizon_target)
    offset += offset_velocity_scale * (near - near_target)
    frames.append(outputs)
  return frames, sky_texture


# get initial camera
initial_camera = sampled_camera

# generate flight pattern
cameras = flights.fly_figure8(initial_camera, 30, 901)

# generate frames
frames, sky_texture = fly_along_path(cameras)

# output raw rgb and thumbnail frames
writer = video_util.get_writer(output_file + 'skip00.mp4')
for curr_index in range(len(frames) - 1):
  frame = np.array(renormalize.as_image(frames[curr_index]['composite'][0]))
  writer.writeFrame(frame)
writer.close()
print('Exported composite')

writer = video_util.get_writer(output_file + 'thumb.mp4')
for curr_index in range(len(frames) - 1):
  frame = np.array(renormalize.as_image(frames[curr_index]['thumb'][0]))
  writer.writeFrame(frame)
writer.close()
print('Exported thumbnail')

writer = video_util.get_writer(output_file + 'depth.mp4')
for curr_index in range(len(frames) - 1):
  frame = np.array(
      renormalize.as_image(frames[curr_index]['depth'][0], source='pt')
  )
  writer.writeFrame(frame)
writer.close()
print('Exported depth thumb')

writer = video_util.get_writer(output_file + 'acc.mp4')
for curr_index in range(len(frames) - 1):
  frame = np.array(
      renormalize.as_image(frames[curr_index]['mask'][0], source='pt')
  )
  writer.writeFrame(frame)
writer.close()
print('Exported acc thumb')

# export projected noise if available
if G.rendering_kwargs['superresolution_noise_mode'] == '3dnoise':
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
  curr_depth = frames[curr_index]['depth']
  src_warped_start, src_mask_start = filters.align_frames(
      frames[start_index]['image'],
      frames[start_index]['world2cam_matrix'],
      frames[curr_index]['world2cam_matrix'],
      curr_depth,
      fov,
      G.rendering_kwargs['ray_end'],
  )
  src_warped_end, src_mask_end = filters.align_frames(
      frames[end_index]['image'],
      frames[end_index]['world2cam_matrix'],
      frames[curr_index]['world2cam_matrix'],
      curr_depth,
      fov,
      G.rendering_kwargs['ray_end'],
  )
  # blend the frames without sky
  weight_start = 1 - (curr_index - start_index) / skip
  weight_end = 1 - (end_index - curr_index) / skip
  warped_blend = weight_start * src_warped_start + weight_end * src_warped_end
  mask_blend = weight_start * src_mask_start + weight_end * src_mask_end

  # get the sky based on current Rt
  world2cam_matrix = frames[curr_index]['world2cam_matrix']
  cam2world_matrix = world2cam_matrix.inverse().cuda()
  neural_rendering_resolution = warped_blend.shape[-1]
  ray_origins, ray_directions = G.ray_sampler(
      cam2world_matrix, intrinsics_matrix, neural_rendering_resolution
  )
  sky_img = sky_util.sample_sky_from_viewdirs(
      sky_texture, ray_directions, neural_rendering_resolution, fov
  ).cpu()
  warped_composite = sky_util.composite_sky(
      warped_blend, frames[curr_index]['mask'], sky_img
  )

  # write the current frame
  frame = np.array(renormalize.as_image(warped_composite[0]))
  writer.writeFrame(frame)
writer.close()
print('Exported skip %d' % skip)

print('Done')
