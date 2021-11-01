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
"""Baking script for trained Deferred NeRF networks."""
import gc
from os import path

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np
from scipy import ndimage
import tensorflow as tf

from snerg.nerf import datasets
from snerg.nerf import models
from snerg.nerf import utils

from snerg.snerg import baking
from snerg.snerg import culling
from snerg.snerg import eval_and_refine
from snerg.snerg import export
from snerg.snerg import model_utils
from snerg.snerg import params

FLAGS = flags.FLAGS

utils.define_flags()


def main(unused_argv):
  # Hide the GPUs and TPUs from TF so it does not reserve memory on them for
  # dataset loading.
  tf.config.experimental.set_visible_devices([], "GPU")
  tf.config.experimental.set_visible_devices([], "TPU")

  rng = random.PRNGKey(20200823)

  if FLAGS.config is not None:
    utils.update_flags(FLAGS)
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")

  # The viewdir MLP refinement code needs this, as it assumes that both datasets
  # are split into images, rather than a unordered bunch of rays.
  FLAGS.__dict__["batching"] = "single_image"

  train_dataset = datasets.get_dataset("train", FLAGS)
  test_dataset = datasets.get_dataset("test", FLAGS)
  rng, key = random.split(rng)
  model, init_variables = models.get_model(key, test_dataset.peek(), FLAGS)
  optimizer = flax.optim.Adam(FLAGS.lr_init).create(init_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, init_variables

  # Initialize the parameters dictionaries for SNeRG.
  (render_params_init, culling_params_init, atlas_params_init,
   scene_params_init) = params.initialize_params(FLAGS)

  # Also initialize the JAX functions and tensorflow models needed to evaluate
  # image quality.
  quality_evaluator = eval_and_refine.ImageQualityEvaluator()

  last_step = 0
  out_dir = path.join(FLAGS.train_dir, "baked")
  out_render_dir = path.join(out_dir, "test_preds")
  if jax.host_id() == 0:
    utils.makedirs(out_dir)
    utils.makedirs(out_render_dir)

  # Make sure that all JAX hosts have reached this point before proceeding. We
  # need to make sure that out_dir and out_render_dir both exist.
  export.synchronize_jax_hosts()

  if not FLAGS.eval_once:
    summary_writer = tensorboard.SummaryWriter(
        path.join(FLAGS.train_dir, "bake"))

  while True:
    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    step = int(state.optimizer.state.step)
    if step <= last_step:
      continue

    # We interleave explicit calls to garbage collection throughout this loop,
    # with the hope of alleviating out-of-memory errors on systems with limited
    # CPU RAM.
    gc.collect()

    # Extract the MLPs we need for baking a SNeRG.
    (mlp_model, mlp_params, viewdir_mlp_model,
     viewdir_mlp_params) = model_utils.extract_snerg_mlps(
         state.optimizer.target, scene_params_init)

    # Render out the low-res grid used for culling.
    culling_grid_coordinates = baking.build_3d_grid(
        scene_params_init["min_xyz"], culling_params_init["_voxel_size"],
        culling_params_init["_grid_size"],
        scene_params_init["worldspace_T_opengl"],
        np.dtype(scene_params_init["dtype"]))
    _, culling_grid_alpha = baking.render_voxel_block(
        mlp_model, mlp_params, culling_grid_coordinates,
        culling_params_init["_voxel_size"], scene_params_init)

    # Early out in case the culling grid is completely empty.
    if culling_grid_alpha.max() < culling_params_init["alpha_threshold"]:
      if FLAGS.eval_once:
        break
      else:
        continue

    # Using this grid, maximize resolution with a tight crop on the scene.
    (render_params, culling_params, atlas_params,
     scene_params) = culling.crop_alpha_grid(render_params_init,
                                             culling_params_init,
                                             atlas_params_init,
                                             scene_params_init,
                                             culling_grid_alpha)

    # Recompute the low-res grid using the cropped scene bounds.
    culling_grid_coordinates = baking.build_3d_grid(
        scene_params["min_xyz"], culling_params["_voxel_size"],
        culling_params["_grid_size"], scene_params["worldspace_T_opengl"],
        np.dtype(scene_params["dtype"]))
    _, culling_grid_alpha = baking.render_voxel_block(
        mlp_model, mlp_params, culling_grid_coordinates,
        culling_params["_voxel_size"], scene_params)

    # Determine which voxels are visible from the training views.
    num_training_cameras = train_dataset.camtoworlds.shape[0]
    culling_grid_visibility = np.zeros_like(culling_grid_alpha)
    for camera_index in range(0, num_training_cameras,
                              culling_params["visibility_subsample_factor"]):
      culling.integrate_visibility_from_image(
          train_dataset.h * culling_params["visibility_image_factor"],
          train_dataset.w * culling_params["visibility_image_factor"],
          train_dataset.focal * culling_params["visibility_image_factor"],
          train_dataset.camtoworlds[camera_index], culling_grid_alpha,
          culling_grid_visibility, scene_params, culling_params)

    # Finally, using this updated low-res grid, compute the maximum alpha
    # within each macroblock.
    atlas_grid_alpha = culling.max_downsample_grid(culling_params, atlas_params,
                                                   culling_grid_alpha)
    atlas_grid_visibility = culling.max_downsample_grid(
        culling_params, atlas_params, culling_grid_visibility)

    # Make the visibility grid more conservative by dilating it. We need to
    # temporarly cast to float32 here, as ndimage.maximum_filter doesn't work
    # with float16.
    atlas_grid_visibility = ndimage.maximum_filter(
        atlas_grid_visibility.astype(np.float32),
        culling_params["visibility_grid_dilation"]).astype(
            atlas_grid_visibility.dtype)

    # Now we're ready to extract the scene and pack it into a 3D texture atlas.
    atlas, atlas_block_indices = baking.extract_3d_atlas(
        mlp_model, mlp_params, scene_params, render_params, atlas_params,
        culling_params, atlas_grid_alpha, atlas_grid_visibility)

    # Free up CPU memory wherever we can to avoid OOM in the larger scenes.
    del atlas_grid_alpha
    del atlas_grid_visibility
    del culling_grid_alpha
    del culling_grid_visibility
    gc.collect()

    # Convert the atlas to a tensor, so we can use can use tensorflow's massive
    # CPU parallelism for ray marching.
    atlas_block_indices_t = tf.convert_to_tensor(atlas_block_indices)
    del atlas_block_indices
    gc.collect()

    atlas_t_list = []
    for i in range(atlas.shape[2]):
      atlas_t_list.append(tf.convert_to_tensor(atlas[:, :, i, :]))
    del atlas
    gc.collect()

    atlas_t = tf.stack(atlas_t_list, 2)
    del atlas_t_list
    gc.collect()

    # Quantize the atlas to 8-bit precision, as this is the precision will be
    # working with for the exported PNGs.
    uint_multiplier = 2.0**8 - 1.0
    atlas_t *= uint_multiplier
    gc.collect()
    atlas_t = tf.floor(atlas_t)
    gc.collect()
    atlas_t = tf.maximum(0.0, atlas_t)
    gc.collect()
    atlas_t = tf.minimum(uint_multiplier, atlas_t)
    gc.collect()
    atlas_t /= uint_multiplier
    gc.collect()

    # Ray march through the baked SNeRG scene to create training data for the
    # view-depdence MLP.
    (train_rgbs, _, train_directions,
     train_refs) = eval_and_refine.build_sharded_dataset_for_view_dependence(
         train_dataset, atlas_t, atlas_block_indices_t, atlas_params,
         scene_params, render_params)

    # Refine the view-dependence MLP to alleviate the domain gap between a
    # deferred NeRF scene and the baked SNeRG scene.
    refined_viewdir_mlp_params = eval_and_refine.refine_view_dependence_mlp(
        train_rgbs, train_directions, train_refs, viewdir_mlp_model,
        viewdir_mlp_params, scene_params)
    del train_rgbs
    del train_directions
    del train_refs
    gc.collect()

    # Now that we've refined the MLP, create test data with ray marching too.
    (test_rgbs, _, test_directions,
     _) = eval_and_refine.build_sharded_dataset_for_view_dependence(
         test_dataset, atlas_t, atlas_block_indices_t, atlas_params,
         scene_params, render_params)

    # Now run the view-dependence on the ray marched output images to add
    # back view-depdenent effects. Note that we do this both before and after
    # refining the parameters.
    pre_refined_images = eval_and_refine.eval_dataset_and_unshard(
        viewdir_mlp_model, viewdir_mlp_params, test_rgbs, test_directions,
        test_dataset, scene_params)
    post_refined_images = eval_and_refine.eval_dataset_and_unshard(
        viewdir_mlp_model, refined_viewdir_mlp_params, test_rgbs,
        test_directions, test_dataset, scene_params)
    del test_rgbs
    del test_directions
    gc.collect()

    # Evaluate image quality metrics for the baked SNeRG scene, both before and
    # after refining the  view-dependence MLP.
    pre_image_metrics = quality_evaluator.eval_image_list(
        pre_refined_images, test_dataset.images)
    post_image_metrics = quality_evaluator.eval_image_list(
        post_refined_images, test_dataset.images)
    pre_psnr, pre_ssim = pre_image_metrics[0], pre_image_metrics[1]
    post_psnr, post_ssim = post_image_metrics[0], post_image_metrics[1]
    gc.collect()

    # Export the baked scene so we can view it in the web-viewer.
    export.export_snerg_scene(out_dir, atlas_t.numpy(),
                              atlas_block_indices_t.numpy(),
                              refined_viewdir_mlp_params, render_params,
                              atlas_params, scene_params, test_dataset.h,
                              test_dataset.w, test_dataset.focal)
    gc.collect()

    # Compute the size of the exportet SNeRG scene.
    png_size_gb, byte_size_gb, float_size_gb = export.compute_scene_size(
        out_dir, atlas_block_indices_t.numpy(), atlas_params, scene_params)
    gc.collect()

    # Finally, export the rendered test set images and update tensorboard.

    # Parallelize the image export over JAX hosts to speed this up.
    renders_and_paths = []
    paths = []
    for i in range(test_dataset.camtoworlds.shape[0]):
      renders_and_paths.append((post_refined_images[i],
                                path.join(out_render_dir,
                                          "{:03d}.png".format(i))))
    export.parallel_write_images(
        lambda render_and_path: utils.save_img(  # pylint: disable=g-long-lambda
            render_and_path[0],
            render_and_path[1]),
        renders_and_paths)

    if (not FLAGS.eval_once) and (jax.host_id() == 0):
      summary_writer.image("baked_raw_color", pre_refined_images[0], step)
      summary_writer.image("baked_refined_color", post_refined_images[0], step)
      summary_writer.image("baked_target", test_dataset.images[0], step)
      summary_writer.scalar("baked_raw_psnr", pre_psnr, step)
      summary_writer.scalar("baked_raw_ssim", pre_ssim, step)
      summary_writer.scalar("baked_refined_psnr", post_psnr, step)
      summary_writer.scalar("baked_refined_ssim", post_ssim, step)
      summary_writer.scalar("baked_size_png_gb", png_size_gb, step)
      summary_writer.scalar("baked_size_byte_gb", byte_size_gb, step)
      summary_writer.scalar("baked_size_float_gb", float_size_gb, step)

    if FLAGS.save_output and (not FLAGS.render_path) and (jax.host_id() == 0):
      with utils.open_file(path.join(out_dir, "psnr.txt"), "w") as f:
        f.write("{}".format(post_psnr))
      with utils.open_file(path.join(out_dir, "ssim.txt"), "w") as f:
        f.write("{}".format(post_ssim))
      with utils.open_file(path.join(out_dir, "png_gb.txt"), "w") as f:
        f.write("{}".format(png_size_gb))
      with utils.open_file(path.join(out_dir, "byte_gb.txt"), "w") as f:
        f.write("{}".format(byte_size_gb))
      with utils.open_file(path.join(out_dir, "float_gb.txt"), "w") as f:
        f.write("{}".format(float_size_gb))


    if FLAGS.eval_once:
      break

    if int(step) >= FLAGS.max_steps:
      break

    last_step = step


if __name__ == "__main__":
  app.run(main)
