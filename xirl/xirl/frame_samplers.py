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

"""Video frame samplers."""

import abc
import random
from absl import logging

import numpy as np

from xirl.file_utils import get_files
# pylint: disable=logging-fstring-interpolation


class FrameSampler(abc.ABC):
  """Video frame sampler base abstraction."""

  def __init__(
      self,
      num_frames,
      num_ctx_frames=1,
      ctx_stride=1,
      pattern="*.png",
      seed=None,
  ):
    """Constructor.

    Args:
      num_frames: How many frames to sample in each video.
      num_ctx_frames: How many context frames to sample for each sampled frame.
        A value of 1 is equivalent to not sampling any context frames.
      ctx_stride: The spacing between sampled context frames.
      pattern: The wildcard pattern for the video frames.
      seed: The seed for the rng.
    """
    assert num_ctx_frames > 0, "num_ctx_frames must be >= 1."
    assert isinstance(num_frames, int), "num_frames must be an int."
    assert isinstance(num_ctx_frames, int), "num_ctx_frames must be an int."
    assert isinstance(ctx_stride, int), "ctx_stride must be an int."

    self._num_frames = num_frames
    self._num_ctx_frames = num_ctx_frames
    self._ctx_stride = ctx_stride
    self._pattern = pattern
    self._seed = seed

    self.seed_rng()

  def seed_rng(self):
    """Reseed the RNG."""
    if self._seed is not None:
      logging.debug("%s seed: %d", self.__class__.__name__, self._seed)
      random.seed(self._seed)

  def _get_context_steps(
      self,
      frame_idxs,
      vid_len,
  ):
    """Generate causal context frame indices for each sampled frame."""
    # Currently, context idxs are sampled up to the current step, i.e. we do
    # not want to encode information from future timesteps.
    ctx_idxs = []
    for idx in frame_idxs:
      idxs = list(
          range(
              idx - (self._num_ctx_frames - 1) * self._ctx_stride,
              idx + self._ctx_stride,
              self._ctx_stride,
          ))
      idxs = np.clip(idxs, a_min=0, a_max=vid_len - 1)
      ctx_idxs.append(idxs)
    return ctx_idxs

  @abc.abstractmethod
  def _sample(self, frames):
    """Subclasses should override this method.

    Args:
      frames: A list where each element if a list of strings containing the
        absolute path to all the frames in a video.

    Returns:
      The indices of the `frames` list to sample.
    """
    pass

  @abc.abstractmethod
  def _load_frames(
      self,
      vid_dirs,
  ):
    """Subclasses should override this method."""
    pass

  def sample(self, vid_dirs):
    """Sample the frames in a video directory.

    Args:
      vid_dirs: A list of video folder paths from which to sample frames.

    Returns:
      A dict containing a list with the sampled frame indices, a list of
      all frame paths in the video directory and a list with indices of
      the context frames for each sampled frame.
    """
    frames = self._load_frames(vid_dirs)
    frame_idxs = self._sample(frames)
    return {
        "frames": frames,
        "frame_idxs": frame_idxs,
        "vid_len": len(frames),
        "ctx_idxs": self._get_context_steps(frame_idxs, len(frames)),
    }

  @property
  def num_frames(self):
    return self._num_frames

  @property
  def num_ctx_frames(self):
    return self._num_ctx_frames


class SingleVideoFrameSampler(FrameSampler):
  """Frame samplers that operate on a single video at a time.

  Subclasses should implemented the `_sample` method.
  """

  def _load_frames(self, vid_dir):
    return get_files(vid_dir, self._pattern, sort_numerical=True)


class StridedSampler(SingleVideoFrameSampler):
  """Sample every n'th frame of a video."""

  def __init__(  # pylint: disable=keyword-arg-before-vararg
      self,
      stride,
      offset=True,
      *args,
      **kwargs,
  ):
    """Constructor.

    Args:
      stride: The spacing between consecutively sampled frames. A stride of 1 is
        equivalent to frame_samplers.AllSampler.
      offset: If set to `True`, a random starting point is chosen along the
        length of the video. Else, the sampling starts at the 0th frame.
      *args: Args.
      **kwargs: Keyword args.
    """
    super().__init__(*args, **kwargs)

    assert stride >= 1, "stride must be >= to 1."
    assert isinstance(stride, int), "stride must be an integer."

    self._offset = offset
    self._stride = stride

  def _sample(self, frames):
    vid_len = len(frames)
    if self._offset:
      # The offset can be set between 0 and the maximum location from
      # which we can get total coverage of the video without having to
      # pad.
      offset = random.randint(0,
                              max(1, vid_len - self._stride * self._num_frames))
    else:
      offset = 0
    cc_idxs = list(
        range(
            offset,
            offset + self._num_frames * self._stride + 1,
            self._stride,
        ))
    cc_idxs = np.clip(cc_idxs, a_min=0, a_max=vid_len - 1)
    return cc_idxs[:self._num_frames]


class AllSampler(StridedSampler):
  """Sample all the frames of a video.

  This should really only be used for evaluation, i.e. when embedding all
  frames of a video, since sampling all frames of a video, especially long
  ones, dramatically increases compute and memory requirements.
  """

  def __init__(self, stride=1, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Constructor.

    Args:
      stride: The spacing between consecutively sampled frames. A stride of 1
        samples all frames in a video sequence. Increase this value for
        high-frame rate videos.
      *args: Args.
      **kwargs: Keyword args.
    """
    kwargs["offset"] = False
    kwargs["num_frames"] = 1
    kwargs["stride"] = stride
    super().__init__(*args, **kwargs)

  def _sample(self, frames):
    self._num_frames = int(np.ceil(len(frames) / self._stride))
    return super()._sample(frames)


class VariableStridedSampler(SingleVideoFrameSampler):
  """Strided sampling based on a video's number of frames."""

  def _sample(self, frames):
    vid_len = len(frames)
    stride = vid_len / self._num_frames
    cc_idxs = np.arange(0.0, vid_len, stride).round().astype(int)
    cc_idxs = np.clip(cc_idxs, a_min=0, a_max=vid_len - 1)
    cc_idxs = cc_idxs[:self._num_frames]
    return cc_idxs


class LastFrameAndRandomFrames(SingleVideoFrameSampler):
  """Sample the last frame and (N-1) random other frames."""

  def _sample(self, frames):
    vid_len = len(frames)
    last_idx = vid_len - 1
    goal_idx = np.random.choice(np.arange(last_idx - 5, last_idx))
    other_idxs = np.random.choice(
        np.arange(0, last_idx - 5), replace=False, size=self._num_frames - 1)
    other_idxs.sort()
    cc_idxs = np.hstack([goal_idx, other_idxs])
    return cc_idxs


class UniformSampler(SingleVideoFrameSampler):
  """Uniformly sample video frames starting from an optional offset."""

  def __init__(self, offset, *args, **kwargs):
    """Constructor.

    Args:
      offset: An offset from which to start the uniform random sampling.
      *args: Args.
      **kwargs: Keyword args.
    """
    super().__init__(*args, **kwargs)

    assert isinstance(offset, int), "`offset` must be an integer."
    self._offset = offset

  def _sample(self, frames):
    vid_len = len(frames)
    cond1 = vid_len >= self._offset
    cond2 = self._num_frames < (vid_len - self._offset)
    if cond1 and cond2:
      cc_idxs = list(range(self._offset, vid_len))
      random.shuffle(cc_idxs)
      cc_idxs = cc_idxs[:self._num_frames]
      return sorted(cc_idxs)
    return list(range(0, self._num_frames))


class WindowSampler(SingleVideoFrameSampler):
  """Samples a contiguous window of frames."""

  def _sample(self, frames):
    vid_len = len(frames)
    if vid_len > self._num_frames:
      range_min = random.randrange(vid_len - self._num_frames)
      range_max = range_min + self._num_frames
      return list(range(range_min, range_max))
    return list(range(0, self._num_frames))


class UniformWithPositivesSampler(SingleVideoFrameSampler):
  """Uniformly sample random frames along with positives within a radius."""

  def __init__(self, pos_window, *args, **kwargs):
    """Constructor.

    Args:
      pos_window: The radius for positive frames.
      *args: Args.
      **kwargs: Keyword args.
    """
    super().__init__(*args, **kwargs)

    assert isinstance(pos_window, int), "`pos_window` must be an integer."
    self._pos_window = pos_window

  def _sample(self, frames):
    vid_len = len(frames)
    cc_idxs = list(range(vid_len))
    random.shuffle(cc_idxs)
    cc_idxs = cc_idxs[:self._num_frames]
    pos_steps = np.asarray([
        np.random.randint(step - self._pos_window, step + 1) for step in cc_idxs
    ])
    return np.concatenate([sorted(pos_steps), sorted(cc_idxs)])
