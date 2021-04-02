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

"""Video dataset abstraction."""

import logging
import os.path as osp
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from xirl import frame_samplers
from xirl import transforms
from xirl.file_utils import get_subdirs
from xirl.file_utils import load_image
from xirl.tensorizers import ToTensor
from xirl.types import SequenceType

DataArrayPacket = Dict[SequenceType, Union[np.ndarray, str, int]]
DataTensorPacket = Dict[SequenceType, Union[torch.Tensor, str]]


class VideoDataset(Dataset):
  """A dataset for working with videos."""

  def __init__(
      self,
      root_dir,
      frame_sampler,
      augmentor = None,
      max_vids_per_class = -1,
      seed = None,
  ):
    """Constructor.

    Args:
      root_dir: The path to the dataset directory.
      frame_sampler: A sampler specifying the frame sampling strategy.
      augmentor: An instance of transforms.VideoAugmentor. If provided, will
        apply data augmentation to the sampled video data.
      max_vids_per_class: The max number of videos to consider per class. The
        remaining videos are ignored, effectively reducing the total dataset
        size.
      seed: The seed for the rng.

    Raises:
      ValueError: If the root directory is empty.
    """
    super().__init__()

    self._root_dir = root_dir
    self._frame_sampler = frame_sampler
    self._seed = seed
    self._max_vids_per_class = max_vids_per_class
    self._augmentor = augmentor
    self._totensor = ToTensor()

    # Get list of available dirs and ensure that it is not empty.
    self._available_dirs = get_subdirs(
        self._root_dir,
        nonempty=True,
        sort=False,
    )
    if len(self._available_dirs) == 0:  # pylint: disable=g-explicit-length-test
      raise ValueError("{} is an empty directory.".format(root_dir))
    self._allowed_dirs = self._available_dirs

    self.seed_rng()
    self._build_dir_tree()

  def seed_rng(self):
    if self._seed:
      random.seed(self._seed)

  def _build_dir_tree(self):
    """Build a dict of indices for iterating over the dataset."""
    self._dir_tree = {}
    for path in self._allowed_dirs:
      vids = get_subdirs(
          path,
          nonempty=False,
          sort=True,
          sortfunc=lambda x: int(osp.splitext(osp.basename(x))[0]),
      )
      if len(vids) > 0:  # pylint: disable=g-explicit-length-test
        self._dir_tree[path] = vids
    self._restrict_dataset_size()

  def _restrict_dataset_size(self):
    """Restrict the max vid per class or max total vids if specified."""
    if self._max_vids_per_class > 0:
      for vid_class, vid_dirs in self._dir_tree.items():
        self._dir_tree[vid_class] = vid_dirs[:self._max_vids_per_class]

  def restrict_subdirs(self, subdirs):
    """Restrict the set of available subdirectories, i.e.

    classes.

    If using a batch sampler in conjunction with a dataloader, ensure this
    method is called before instantiating the sampler.

    Args:
      subdirs: A list of allowed video classes.

    Raises:
      ValueError: If the restriction leads to an empty directory.
    """
    if not isinstance(subdirs, (tuple, list)):
      subdirs = [subdirs]
    if subdirs:
      len_init = len(self._available_dirs)
      self._allowed_dirs = self._available_dirs
      subdirs = [osp.join(self._root_dir, x) for x in subdirs]
      self._allowed_dirs = list(set(self._allowed_dirs) & set(subdirs))
      if len(self._allowed_dirs) == 0:  # pylint: disable=g-explicit-length-test
        raise ValueError(f"Filtering with {subdirs} returns an empty dataset.")
      len_final = len(self._allowed_dirs)
      logging.debug(  # pylint: disable=logging-format-interpolation
          f"Restricted dataset from {len_init} to {len_final} actions.")
      self._build_dir_tree()
    else:
      logging.debug("Passed in an empty list. No action taken.")

  def _get_video_path(self, class_idx, vid_idx):
    """Return video paths given class and video indices.

    Args:
      class_idx: The index of the action class folder in the dataset directory
        tree.
      vid_idx: The index of the video in the action class folder to retrieve.

    Returns:
      A path to a video to sample in the dataset.
    """
    action_class = list(self._dir_tree)[class_idx]
    return self._dir_tree[action_class][vid_idx]

  def _get_data(self, vid_path):
    """Load video data given a video path.

    Feeds the video path to the frame sampler to retrieve video frames and
    metadata.

    Args:
      vid_path: A path to a video in the dataset.

    Returns:
      A dictionary containing key, value pairs where the key is an enum
      member of `SequenceType` and the value is either an int, a string
      or an ndarray respecting the key type.
    """
    sample = self._frame_sampler.sample(vid_path)

    # Load each frame along with its context frames into an array of shape
    # (S, X, H, W, C), where S is the number of sampled frames and X is the
    # number of context frames.
    frames = np.stack([load_image(f) for f in sample["frames"]])
    frames = np.take(frames, sample["ctx_idxs"], axis=0)

    # Reshape frames into a 4D array of shape (S * X, H, W, C).
    frames = np.reshape(frames, (-1, *frames.shape[2:]))

    frame_idxs = np.asarray(sample["frame_idxs"], dtype=np.int64)

    return {
        SequenceType.FRAMES: frames,
        SequenceType.FRAME_IDXS: frame_idxs,
        SequenceType.VIDEO_NAME: vid_path,
        SequenceType.VIDEO_LEN: sample["vid_len"],
    }

  def __getitem__(self, idxs):
    vid_paths = self._get_video_path(*idxs)
    data_np = self._get_data(vid_paths)
    if self._augmentor:
      data_np = self._augmentor(data_np)
    data_tensor = self._totensor(data_np)
    return data_tensor

  def __len__(self):
    return self.total_vids

  @property
  def num_classes(self):
    """The number of subdirs, i.e. allowed video classes."""
    return len(self._allowed_dirs)

  @property
  def total_vids(self):
    """The total number of videos across all allowed video classes."""
    num_vids = 0
    for vids in self._dir_tree.values():
      num_vids += len(vids)
    return num_vids

  @property
  def dir_tree(self):
    """The directory tree."""
    return self._dir_tree

  def collate_fn(
      self,
      batch,
  ):
    """A custom collate function for video data."""

    def _stack(key):
      return torch.stack([b[key] for b in batch])

    # Convert the keys to their string representation so that a batch can be
    # more easily indexed into without an extra import to SequenceType.
    return {
        str(SequenceType.FRAMES):
            _stack(SequenceType.FRAMES),
        str(SequenceType.FRAME_IDXS):
            _stack(SequenceType.FRAME_IDXS),
        str(SequenceType.VIDEO_LEN):
            _stack(SequenceType.VIDEO_LEN),
        str(SequenceType.VIDEO_NAME): [
            b[SequenceType.VIDEO_NAME] for b in batch
        ],
    }
