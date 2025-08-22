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

# pylint: skip-file
"""Fast sparse reconstruction library to run COLMAP on a dataset.

This library implements a faster, less stable alternative to
colmap_pipeline.py:reconstruct(). Rather than performing an incremental SfM
reconstruction of the entire scene, adding images one-by-one, this Pipeline
performs an incremental reconstruction for a subset of the whole image,
collection then processes the remaining images in a single batch.
"""
import dataclasses
import os
import re
from typing import List, Optional, Text, Tuple

from absl import logging
import gin
import numpy as np

# Methods for matching images in COLMAP.
EXHAUSTIVE_MATCHER = 'exhaustive_matcher'
MATCHES_IMPORTER = 'matches_importer'
VOCAB_TREE_MATCHER = 'vocab_tree_matcher'


@gin.configurable()
@dataclasses.dataclass
class NerfReconstructOptions(colmap_pipeline.ReconstructOptions):
  """Options for the dataset processor."""

  # If True, use fast_sparse_reconstruct(). If False, use
  # colmap_pipeline.reconstruct().
  use_fast_codepath: bool = True

  # Args passed to the different stages of colmap reconstruction. See also
  # args for other stages defined in colmap_pipeline.py.
  image_registrator_args: Text = ''
  bundle_adjuster_args: Text = ''
  point_triangulator_args: Text = ''
  matches_importer_args: Text = ''

  # Which method to use when matching. One of exhaustive_matcher,
  # vocab_tree_matcher, or matches_importer.
  matcher_method: Text = VOCAB_TREE_MATCHER

  # Knobs for controlling which images are matched when using matcher_method ==
  # 'matches_importer'.
  matches_importer_percent_dense_match: float = 0.1
  matches_importer_window_size: int = 7

  # Path to VocabTree index. Must be compatible with version of COLMAP.
  remote_vocab_tree_path: Text = ''

  # Parameters for choosing initial number of photos to use in incremental SfM.
  # A percentage of the total number of photos is targeted, but if the image
  # collection is too big or too small, explicit lower and upper bounds are
  # used.
  target_percent_images_initial_sfm_pass: float = 0.2
  min_images_initial_sfm_pass: int = 100
  max_images_initial_sfm_pass: int = 500

  # The maximum number of images to use, period. If the number of images in the
  # dataset exceeds this number, the dataset will be culled.
  max_images: Optional[int] = None

  # (Debugging) Flags for skipping parts of the fast codepath. Only use these
  # if you know what you're doing.
  run_feature_extractor: bool = True
  run_matcher: bool = True
  run_mapper: bool = True
  run_first_bundle_adjuster: bool = True
  run_image_registrator: bool = True
  run_point_triangulator: bool = True
  run_second_bundle_adjuster: bool = True

  # If provided, computes the camera name as the "camera_name" named group in
  # the match, and shares the intrinsics for all images of the same camera. For
  # instance, if images are named camera01_image01.jpg, camera02_image02.jpg,
  # etc and we want to share camera parameters for all images of a given camera,
  # the value '(?P<camera_name>[^_]+)_.*' would match in the first group
  # "camera01" or "camera02", and share camera parameters accordingly.
  camera_name_regex_pattern: str = ''


def _compute_camera_id_by_image_name_dict(
    data_dir,
    options,
):
  """Computes the mapping from image name to camera id.

  This can be used to share intrinsics across subsets of images.

  Args:
    data_dir: path to where the images are located, under `images`.
    options: the NerfReconstructOptions object.

  Returns:
    A dict containing the camera id given the image name (basename).
  """
  image_paths = sorted(gfile.Glob(os.path.join(data_dir, 'images', '*')))
  camera_id_by_image_name = {}
  camera_id_by_camera_name = {}
  camera_image_regex = re.compile(options.camera_name_regex_pattern)
  for p in image_paths:
    image_name = os.path.basename(p)
    match = re.fullmatch(camera_image_regex, image_name)
    if not match:
      logging.info(
          'Matching failed for image: %s with pattern: %s',
          image_name,
          options.camera_name_regex_pattern,
      )
      continue
    if 'camera_name' not in match.groupdict():
      logging.info(
          (
              'Match does not contain a "camera_name" group for image: %s with'
              ' pattern: %s, groupdict: %s'
          ),
          image_name,
          options.camera_name_regex_pattern,
          match.groupdict,
      )
      continue
    camera_name = match.group('camera_name')
    if camera_name not in camera_id_by_camera_name:
      # Colmap does not like zero-indexed cameras.
      camera_id_by_camera_name[camera_name] = 1 + len(camera_id_by_camera_name)
    camera_id = camera_id_by_camera_name[camera_name]
    camera_id_by_image_name[image_name] = camera_id
  return camera_id_by_image_name


def reconstruct(
    data_dir,
    options,
    vocab_tree_path,
    required_image_names,
):
  """Runs COLMAP SfM's pipeline.

  Args:
    data_dir: a path to a directory, containing the images to be reconstructed
      in the data_dir/images/ subfolder. The results are written to the same
      path.
    options: fast reconstruction options.
    vocab_tree_path: Path to local file containing VocabTree index.
    required_image_names: Filenames of images that must be part of the COLMAP
      reconstruction.

  Returns:
    List of local files and folders to copy.
  """
  if options.use_fast_codepath:
    return fast_sparse_reconstruct(
        data_dir, options, vocab_tree_path, required_image_names
    )
  else:
    if options.max_images is not None:
      raise ValueError('max_images not supported in default codepath.')
    if options.matcher_method not in [VOCAB_TREE_MATCHER, EXHAUSTIVE_MATCHER]:
      raise ValueError(
          f'Unsupported feature matcher method: {options.matcher_method}'
      )

    extra_args = dict()
    if options.camera_name_regex_pattern:
      extra_args['camera_id_by_image_name'] = (
          _compute_camera_id_by_image_name_dict(data_dir, options)
      )
      logging.info(
          'camera_id_by_image_name computed for pattern `%s`: %s',
          options.camera_name_regex_pattern,
          extra_args['camera_id_by_image_name'],
      )

    colmap_pipeline.reconstruct(
        data_dir, options, vocab_tree_path, **extra_args
    )
    return [
        os.path.join(data_dir, filename)
        for filename in ['database.db', 'sparse']
    ]


def fast_sparse_reconstruct(
    data_dir,
    options,
    vocab_tree_path,
    required_image_names,
):
  """Runs a faster version of COLMAP SfM's pipeline.

  This pipeline performs sparse reconstruction. It runs COLMAP's incremental
  SfM pipeline on a subset of photos, then registers all remaining photos to
  this SfM model, and then performs one final pass of optimization.

  Args:
    data_dir: a path to a directory, containing the images to be reconstructed
      in the data_dir/images/ subfolder. The results are written to the same
      path.
    options: fast reconstruction options.
    vocab_tree_path: Path to local file containing VocabTree index.
    required_image_names: Filenames of images that must be part of the COLMAP
      reconstruction.

  Returns:
    List of local files and folders to copy.
  """
  if options.matcher_method == VOCAB_TREE_MATCHER and not vocab_tree_path:
    raise ValueError('vocab_tree_path is required for fast codepath.')
  image_dir = os.path.join(data_dir, 'images')
  if not gfile.Glob(os.path.join(image_dir, '*')):
    raise ValueError(f'No files found in image path: {image_dir}')
  database_path = os.path.join(data_dir, 'database.db')
  # Local filepaths to copy back
  local_filepaths = [database_path]

  # Write a file containing the filenames of all images to use in this pipeline.
  all_image_filenames = _union(
      _subsample_images(gfile.ListDir(image_dir), options.max_images),
      required_image_names,
  )
  all_image_filenames_filepath = os.path.join(
      data_dir, 'all_image_filenames.txt'
  )
  _write_image_list(all_image_filenames, all_image_filenames_filepath)
  local_filepaths.append(all_image_filenames_filepath)

  # Feature extraction.
  if options.run_feature_extractor:
    logging.info('Calling colmap feature_extractor.')
    colmap_lib.RunFeatureExtractor(
        f'--database_path {database_path} '
        f'--image_path {image_dir} '
        f'--image_list_path {all_image_filenames_filepath} '
        f'{options.feature_extractor_args}'
    )

  # Feature matching.
  if options.run_matcher:
    if options.matcher_method == VOCAB_TREE_MATCHER:
      logging.info('Calling colmap vocab_tree_matcher.')
      colmap_lib.RunVocabTreeMatcher(
          f'--database_path {database_path} '
          f'--VocabTreeMatching.vocab_tree_path {vocab_tree_path} '
          f'{options.vocab_tree_matcher_args}'
      )

    elif options.matcher_method == MATCHES_IMPORTER:
      logging.info('Calling colmap matches_importer.')
      match_pairs = _construct_match_pairs_list(
          all_image_filenames,
          options.matches_importer_window_size,
          options.matches_importer_percent_dense_match,
      )
      logging.info('matching %d pairs of images.', len(match_pairs))
      match_pairs_filepath = os.path.join(data_dir, 'match_list.txt')
      _write_image_pairs_list(match_pairs, match_pairs_filepath)
      local_filepaths.append(match_pairs_filepath)
      colmap_lib.RunMatchesImporter(
          (
              f'--database_path {database_path} '
              f'--match_list_path {match_pairs_filepath} '
              '--match_type pairs '
              f'{options.matches_importer_args}'
          ),
      )

    else:
      raise ValueError(f'Unrecognized matcher method: {options.matcher_method}')

  # Choose images for initial COLMAP run.
  num_initial_images = _num_images_initial_sfm_pass(
      options, len(all_image_filenames)
  )
  initial_image_filenames = _subsample_images(
      all_image_filenames, num_initial_images
  )
  initial_image_filenames_filepath = os.path.join(
      data_dir, 'initial_image_filenames.txt'
  )
  _write_image_list(initial_image_filenames, initial_image_filenames_filepath)
  local_filepaths.append(initial_image_filenames_filepath)

  # Initial SfM reconstruction.
  sparse_dir = os.path.join(data_dir, 'sparse')
  _maybe_make_dir(sparse_dir)
  local_filepaths.append(sparse_dir)

  sparse_0_dir = os.path.join(sparse_dir, '0')
  if not gfile.Exists(sparse_0_dir):
    ############################################################################
    #### START Structure from Motion Pipeline
    ############################################################################
    tmp_sparse_dir = os.path.join(data_dir, 'sparse.tmp')
    _maybe_make_dir(tmp_sparse_dir)
    local_filepaths.append(tmp_sparse_dir)

    # Run default reconstruction pipeline on a subset of images.
    if options.run_mapper:
      logging.info('Calling mapper.')
      colmap_lib.RunMapper(
          f'--database_path {database_path} '
          f'--image_path {image_dir} '
          f'--image_list_path {initial_image_filenames_filepath} '
          f'--output_path {tmp_sparse_dir} '
          f'{options.mapper_args}'
      )

    # Register remaining images.
    mapper_dir = os.path.join(tmp_sparse_dir, '0')
    image_registrator_dir = os.path.join(tmp_sparse_dir, '1_image_registrator')
    _maybe_make_dir(image_registrator_dir)
    if options.run_image_registrator:
      logging.info('Calling image registrator.')
      colmap_lib.RunImageRegistrator(
          f'--database_path {database_path} '
          f'--input_path {mapper_dir} '
          f'--output_path {image_registrator_dir} '
          f'{options.image_registrator_args}'
      )

    # First pass of bundle adjustment with existing point cloud.
    bundle_adjuster_dir = os.path.join(tmp_sparse_dir, '2_bundle_adjuster')
    _maybe_make_dir(bundle_adjuster_dir)
    if options.run_first_bundle_adjuster:
      logging.info('Calling bundle adjuster.')
      colmap_lib.RunBundleAdjuster(
          f'--input_path {image_registrator_dir} '
          f'--output_path {bundle_adjuster_dir} '
          f'{options.bundle_adjuster_args}'
      )

    # Add new points to point cloud.
    point_triangulator_dir = os.path.join(
        tmp_sparse_dir, '3_point_triangulator'
    )
    _maybe_make_dir(point_triangulator_dir)
    if options.run_point_triangulator:
      logging.info('Calling point triangulator.')
      colmap_lib.RunPointTriangulator(
          f'--database_path {database_path} '
          f'--image_path {image_dir} '
          f'--input_path {bundle_adjuster_dir} '
          f'--output_path {point_triangulator_dir} '
          f'{options.point_triangulator_args}'
      )

    # Second pass of bundle adjustment with expanded point cloud.
    final_bundle_adjuster_dir = os.path.join(
        tmp_sparse_dir, '4_bundle_adjuster'
    )
    _maybe_make_dir(final_bundle_adjuster_dir)
    if options.run_second_bundle_adjuster:
      logging.info('Calling bundle adjuster (again).')
      colmap_lib.RunBundleAdjuster(
          f'--input_path {point_triangulator_dir} '
          f'--output_path {final_bundle_adjuster_dir} '
          f'{options.bundle_adjuster_args}'
      )

    # Copy results to final destination.
    gfile.RecursivelyCopyDir(
        final_bundle_adjuster_dir, sparse_0_dir, overwrite=True
    )
    ############################################################################
    #### END Structure from Motion pipeline
    ############################################################################
  else:
    logging.info(
        'Final reconstruction already seems to exist. Skipping SfM pipeline.'
    )
  return local_filepaths


def _subsample_images(
    all_image_filenames, num_images
):
  """Subsample a list of ordered image names.

  Args:
    all_image_filenames: List of image filenames. Should not contain full path.
    num_images: Number of images to subsample.

  Returns:
    List of filenames selected.
  """
  # Select approximately every N-th frame.
  selected_image_names = all_image_filenames
  if num_images is not None:
    selected_image_idxs = np.linspace(
        0,
        len(all_image_filenames),
        num=num_images,
        endpoint=False,
        dtype=np.int32,
    )
    selected_image_idxs = np.unique(selected_image_idxs)
    selected_image_names = [all_image_filenames[i] for i in selected_image_idxs]
  return selected_image_names


def _construct_match_pairs_list(
    all_image_filenames,
    sequential_window_size,
    percent_dense_match,
):
  """Computes unique pairs of images to consider for matching."""
  result = []

  # Match within a sliding window.
  result += _sliding_window_pairs(all_image_filenames, sequential_window_size)

  # Match all pairs in a subset of images.
  target_n = int(len(all_image_filenames) * percent_dense_match)
  subset = _subsample_images(all_image_filenames, target_n)
  result += _all_pairs(subset)

  # Deduplicate pairs.
  result = list(sorted(set(result)))
  return result


def _sliding_window_pairs(
    filenames, window_size
):
  """Computes all unique pairs in a sliding window of a given size."""
  pairs = set()
  window = filenames[:window_size]
  pairs = pairs.union(_all_pairs(window))
  for elem in filenames[window_size:]:
    window = window[1:] + [elem]
    pairs = pairs.union(_all_pairs(window))
  return list(sorted(pairs))


def _all_pairs(filenames):
  """Computes all unique pairs of filenames."""
  pairs = []
  n = len(filenames)
  for i in range(n):
    for j in range(i + 1, n):
      pairs.append((filenames[i], filenames[j]))
  return pairs


def _write_image_list(image_filenames, output_filepath):
  """Write a list of image filenames to a text file."""
  with gfile.Open(output_filepath, 'w') as f:
    f.write('\n'.join(image_filenames))


def _write_image_pairs_list(
    image_pairs, output_filepath
):
  """Write pairs of image filenames to disk."""
  lines = [f'{a} {b}' for a, b in image_pairs]
  _write_image_list(lines, output_filepath)


def _num_images_initial_sfm_pass(
    options, total_num_images
):
  """Calculates number of images to use in initial SfM pass."""
  result = int(
      total_num_images * options.target_percent_images_initial_sfm_pass
  )
  result = np.clip(
      result,
      options.min_images_initial_sfm_pass,
      options.max_images_initial_sfm_pass,
  )
  return result


def _maybe_make_dir(dirname):
  """Creates directory if necessary.

  Args:
    dirname: directory to create.

  Returns:
    True if the director had to be created.
  """
  if gfile.Exists(dirname):
    return False
  gfile.MakeDirs(dirname)
  return True


def _union(*args):
  """Takes the set union of zero or more iterables."""
  result = set()
  for arg in args:
    result = result.union(arg)
  return list(sorted(result))
