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

"""API factory."""

import functools
import os.path as osp

import albumentations as alb
import torch
from xirl import evaluators
from xirl import frame_samplers
from xirl import models
from xirl import trainers
from xirl import transforms
from xirl import video_samplers
from xirl.dataset import VideoDataset
from xirl.file_utils import get_subdirs
from xirl.types import SequenceType

# Supported image transformations with default args.
TRANSFORMS = {
    "random_resized_crop":
        functools.partial(
            alb.RandomResizedCrop, scale=(0.8, 1.0), ratio=(0.75, 1.333),
            p=1.0),
    "center_crop":
        functools.partial(alb.CenterCrop, p=1.0),
    "global_resize":
        functools.partial(alb.Resize, p=1.0),
    "grayscale":
        functools.partial(alb.ToGray, p=0.2),
    "vertical_flip":
        functools.partial(alb.VerticalFlip, p=0.5),
    "horizontal_flip":
        functools.partial(alb.HorizontalFlip, p=0.5),
    "gaussian_blur":
        functools.partial(
            alb.GaussianBlur,
            blur_limit=(13, 13),
            sigma_limit=(1.0, 2.0),
            p=0.2,
        ),
    "color_jitter":
        functools.partial(
            alb.ColorJitter,
            brightness=0.4,
            contrast=0.4,
            hue=0.1,
            saturation=0.1,
            p=0.8,
        ),
    "rotate":
        functools.partial(alb.Rotate, limit=(-5, 5), border_mode=0, p=0.5),
    "normalize":
        functools.partial(
            alb.Normalize,
            mean=transforms.PretrainedMeans.IMAGENET,
            std=transforms.PretrainedStds.IMAGENET,
            p=1.0,
        ),
}
FRAME_SAMPLERS = {
    "all": frame_samplers.AllSampler,
    "strided": frame_samplers.StridedSampler,
    "variable_strided": frame_samplers.VariableStridedSampler,
    "uniform": frame_samplers.UniformSampler,
    "uniform_with_positives": frame_samplers.UniformWithPositivesSampler,
    "last_and_randoms": frame_samplers.LastFrameAndRandomFrames,
    "window": frame_samplers.WindowSampler,
}
VIDEO_SAMPLERS = {
    "random": video_samplers.RandomBatchSampler,
    "same_class": video_samplers.SameClassBatchSampler,
    "downstream": video_samplers.SameClassBatchSamplerDownstream,
}
MODELS = {
    "resnet18_linear": models.Resnet18LinearEncoderNet,
    "resnet18_classifier": models.GoalClassifier,
    "resnet18_features": models.Resnet18RawImageNetFeaturesNet,
    "resnet18_linear_ae": models.Resnet18LinearEncoderAutoEncoderNet,
}
TRAINERS = {
    "tcc": trainers.TCCTrainer,
    "lifs": trainers.LIFSTrainer,
    "tcn": trainers.TCNTrainer,
    "goal_classifier": trainers.GoalFrameClassifierTrainer,
}
EVALUATORS = {
    "kendalls_tau": evaluators.KendallsTau,
    "two_way_cycle_consistency": evaluators.TwoWayCycleConsistency,
    "three_way_cycle_consistency": evaluators.ThreeWayCycleConsistency,
    "nn_visualizer": evaluators.NearestNeighbourVisualizer,
    "reward_visualizer": evaluators.RewardVisualizer,
    "embedding_visualizer": evaluators.EmbeddingVisualizer,
    "reconstruction_visualizer": evaluators.ReconstructionVisualizer,
}


def evaluator_from_config(config):
  """Create evaluators from a config."""
  eval_dict = {}
  for eval_name in config.eval.downstream_task_evaluators:
    kwargs = {"distance": config.eval.distance}
    if eval_name == "kendalls_tau":
      kwargs["stride"] = config.eval.kendalls_tau.stride
    elif "cycle_consistency" in eval_name:
      kwargs["stride"] = config.eval.cycle_consistency.stride
    elif eval_name == "nn_visualizer":
      kwargs["num_ctx_frames"] = config.frame_sampler.num_context_frames
      kwargs["num_videos"] = config.eval.nearest_neighbour_visualizer.num_videos
    elif eval_name == "embedding_visualizer":
      kwargs.pop("distance")
      kwargs["num_seqs"] = config.eval.embedding_visualizer.num_seqs
    elif eval_name == "reconstruction_visualizer":
      kwargs.pop("distance")
      kwargs["num_frames"] = config.eval.reconstruction_visualizer.num_frames
      kwargs["num_ctx_frames"] = config.frame_sampler.num_context_frames
    elif eval_name == "reward_visualizer":
      kwargs["num_plots"] = config.eval.reward_visualizer.num_plots
    elif eval_name == "reconstruction_visualizer":
      kwargs.pop("distance")
      kwargs["num_frames"] = config.eval.reconstruction_visualizer.num_frames
    eval_dict[eval_name] = EVALUATORS[eval_name](**kwargs)
  return evaluators.EvalManager(eval_dict)


def trainer_from_config(config, model, optimizer, device):
  return TRAINERS[config.algorithm](model, optimizer, device, config)


def model_from_config(config):
  """Create a model from a config."""
  kwargs = {
      "num_ctx_frames": config.frame_sampler.num_context_frames,
      "normalize_embeddings": config.model.normalize_embeddings,
      "learnable_temp": config.model.learnable_temp,
  }
  if config.model.model_type == "resnet18_linear":
    kwargs["embedding_size"] = config.model.embedding_size
  elif config.model.model_type == "resnet18_linear_ae":
    kwargs["embedding_size"] = config.model.embedding_size
  return MODELS[config.model.model_type](**kwargs)


def optim_from_config(config, model):
  """Create an optimizer from a config."""
  # TODO(kevin): Add SGD and AdamW support.
  return torch.optim.Adam(
      model.parameters(),
      lr=config.optim.lr,
      weight_decay=config.optim.weight_decay,
  )


def create_transform(name, *args, **kwargs):
  """Create an image augmentation from its name and args."""
  # pylint: disable=invalid-name
  if "::" in name:
    # e.g., `rotate::{'limit': (-45, 45)}`
    name, __kwargs = name.split("::")
    _kwargs = eval(__kwargs)  # pylint: disable=eval-used
  else:
    _kwargs = {}
  _kwargs.update(kwargs)
  return TRANSFORMS[name](*args, **_kwargs)


def frame_sampler_from_config(config, downstream):
  """Create a frame sampler from a config."""
  kwargs = {
      "num_frames": config.frame_sampler.num_frames_per_sequence,
      "num_ctx_frames": config.frame_sampler.num_context_frames,
      "ctx_stride": config.frame_sampler.context_stride,
      "pattern": config.frame_sampler.image_ext,
      "seed": config.seed,
  }

  if downstream:
    kwargs.pop("num_frames")
    kwargs["stride"] = config.frame_sampler.all_sampler.stride
    return FRAME_SAMPLERS["all"](**kwargs)

  if config.frame_sampler.strategy == "strided":
    kwargs["stride"] = config.frame_sampler.strided_sampler.stride
    kwargs["offset"] = config.frame_sampler.strided_sampler.offset
  elif config.frame_sampler.strategy == "uniform":
    kwargs["offset"] = config.frame_sampler.uniform_sampler.offset

  return FRAME_SAMPLERS[config.frame_sampler.strategy](**kwargs)


def video_sampler_from_config(config, dir_tree, downstream, sequential):
  """Create a video sampler from a config."""
  kwargs = {
      "dir_tree": dir_tree,
      "batch_size": config.data.batch_size,
      "sequential": sequential,
  }
  if downstream:
    kwargs.pop("batch_size")
    return VIDEO_SAMPLERS["downstream"](**kwargs)
  return VIDEO_SAMPLERS[config.data.pretraining_video_sampler](**kwargs)


def dataset_from_config(config, downstream, split, debug):
  """Create a video dataset from a config."""
  dataset_path = osp.join(config.data.root, split)

  image_size = config.data_augmentation.image_size
  if isinstance(image_size, int):
    image_size = (image_size, image_size)
  image_size = tuple(image_size)

  # Note(kevin): We used to disable data augmentation on all downstream
  # dataloaders. I've decided to keep them for train downstream loaders.
  if debug:
    # The minimum data augmentation we want to keep is resizing when
    # debugging.
    aug_names = ["global_resize"]
  else:
    if split == "train":
      aug_names = config.data_augmentation.train_transforms
    else:
      aug_names = config.data_augmentation.eval_transforms

  # Create a list of data augmentation callables.
  aug_funcs = []
  for name in aug_names:
    if "resize" in name or "crop" in name:
      aug_funcs.append(create_transform(name, *image_size))
    else:
      aug_funcs.append(create_transform(name))

  augmentor = transforms.VideoAugmentor({SequenceType.FRAMES: aug_funcs})

  # Restrict action classes if they have been provided. Else, load all
  # from the data directory.
  c_action_class = (
      config.data.downstream_action_class
      if downstream else config.data.pretrain_action_class)
  if c_action_class:
    action_classes = c_action_class
  else:
    action_classes = get_subdirs(
        dataset_path,
        basename=True,
        nonempty=True,
        sort_lexicographical=True,
    )

  # We need to separate out the dataclasses for each action class when
  # creating downstream datasets.
  if downstream:
    dataset = {}
    for action_class in action_classes:
      frame_sampler = frame_sampler_from_config(config, downstream=True)
      single_class_dataset = VideoDataset(
          dataset_path,
          frame_sampler,
          seed=config.seed,
          augmentor=augmentor,
          max_vids_per_class=config.data.max_vids_per_class,
      )
      single_class_dataset.restrict_subdirs(action_class)
      dataset[action_class] = single_class_dataset
  else:
    frame_sampler = frame_sampler_from_config(config, downstream=False)
    dataset = VideoDataset(
        dataset_path,
        frame_sampler,
        seed=config.seed,
        augmentor=augmentor,
        max_vids_per_class=config.data.max_vids_per_class,
    )
    dataset.restrict_subdirs(action_classes)

  return dataset
