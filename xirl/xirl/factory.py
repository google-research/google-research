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
  for eval_name in config.EVAL.DOWNSTREAM_TASK_EVALUATORS:
    kwargs = {"distance": config.EVAL.DISTANCE}
    if eval_name == "kendalls_tau":
      kwargs["stride"] = config.EVAL.KENDALLS_TAU.STRIDE
    elif "cycle_consistency" in eval_name:
      kwargs["stride"] = config.EVAL.CYCLE_CONSISTENCY.STRIDE
    elif eval_name == "nn_visualizer":
      kwargs["num_ctx_frames"] = config.FRAME_SAMPLER.NUM_CONTEXT_FRAMES
      kwargs["num_videos"] = config.EVAL.NEAREST_NEIGHBOUR_VISUALIZER.NUM_VIDEOS
    elif eval_name == "embedding_visualizer":
      kwargs.pop("distance")
      kwargs["num_seqs"] = config.EVAL.EMBEDDING_VISUALIZER.NUM_SEQS
    elif eval_name == "reconstruction_visualizer":
      kwargs.pop("distance")
      kwargs["num_frames"] = config.EVAL.RECONSTRUCTION_VISUALIZER.NUM_FRAMES
      kwargs["num_ctx_frames"] = config.FRAME_SAMPLER.NUM_CONTEXT_FRAMES
    elif eval_name == "reward_visualizer":
      kwargs["num_plots"] = config.EVAL.REWARD_VISUALIZER.NUM_PLOTS
    elif eval_name == "reconstruction_visualizer":
      kwargs.pop("distance")
      kwargs["num_frames"] = config.EVAL.RECONSTRUCTION_VISUALIZER.NUM_FRAMES
    eval_dict[eval_name] = EVALUATORS[eval_name](**kwargs)
  return evaluators.EvalManager(eval_dict)


def trainer_from_config(config, model, optimizer, device):
  return TRAINERS[config.ALGORITHM](model, optimizer, device, config)


def model_from_config(config):
  """Create a model from a config."""
  kwargs = {
      "num_ctx_frames": config.FRAME_SAMPLER.NUM_CONTEXT_FRAMES,
      "normalize_embeddings": config.MODEL.NORMALIZE_EMBEDDINGS,
      "learnable_temp": config.MODEL.LEARNABLE_TEMP,
  }
  if config.MODEL.MODEL_TYPE == "resnet18_linear":
    kwargs["embedding_size"] = config.MODEL.EMBEDDING_SIZE
  elif config.MODEL.MODEL_TYPE == "resnet18_linear_ae":
    kwargs["embedding_size"] = config.MODEL.EMBEDDING_SIZE
  return MODELS[config.MODEL.MODEL_TYPE](**kwargs)


def optim_from_config(config, model):
  """Create an optimizer from a config."""
  # TODO(kevin): Add SGD and AdamW support.
  return torch.optim.Adam(
      model.parameters(),
      lr=config.OPTIM.LR,
      weight_decay=config.OPTIM.WEIGHT_DECAY,
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
      "num_frames": config.FRAME_SAMPLER.NUM_FRAMES_PER_SEQUENCE,
      "num_ctx_frames": config.FRAME_SAMPLER.NUM_CONTEXT_FRAMES,
      "ctx_stride": config.FRAME_SAMPLER.CONTEXT_STRIDE,
      "pattern": config.FRAME_SAMPLER.IMAGE_EXT,
      "seed": config.SEED,
  }

  if downstream:
    kwargs.pop("num_frames")
    kwargs["stride"] = config.FRAME_SAMPLER.ALL_SAMPLER.STRIDE
    return FRAME_SAMPLERS["all"](**kwargs)

  if config.FRAME_SAMPLER.STRATEGY == "strided":
    kwargs["stride"] = config.FRAME_SAMPLER.STRIDED_SAMPLER.STRIDE
    kwargs["offset"] = config.FRAME_SAMPLER.STRIDED_SAMPLER.OFFSET
  elif config.FRAME_SAMPLER.STRATEGY == "uniform":
    kwargs["offset"] = config.FRAME_SAMPLER.UNIFORM_SAMPLER.OFFSET

  return FRAME_SAMPLERS[config.FRAME_SAMPLER.STRATEGY](**kwargs)


def video_sampler_from_config(config, dir_tree, downstream, sequential):
  """Create a video sampler from a config."""
  kwargs = {
      "dir_tree": dir_tree,
      "batch_size": config.DATA.BATCH_SIZE,
      "sequential": sequential,
  }
  if downstream:
    kwargs.pop("batch_size")
    return VIDEO_SAMPLERS["downstream"](**kwargs)
  return VIDEO_SAMPLERS[config.DATA.PRETRAINING_VIDEO_SAMPLER](**kwargs)


def dataset_from_config(config, downstream, split, debug):
  """Create a video dataset from a config."""
  dataset_path = osp.join(config.DATA.ROOT, split)

  image_size = config.DATA_AUGMENTATION.IMAGE_SIZE
  if isinstance(image_size, int):
    image_size = (image_size, image_size)
  image_size = tuple(image_size)

  if debug or downstream:
    # The minimum data augmentation we want to keep is resizing when
    # debugging.
    aug_names = ["global_resize"]
  else:
    if split == "train":
      aug_names = config.DATA_AUGMENTATION.TRAIN_TRANSFORMS
    else:
      aug_names = config.DATA_AUGMENTATION.EVAL_TRANSFORMS

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
      config.DATA.DOWNSTREAM_ACTION_CLASS
      if downstream else config.DATA.PRETRAIN_ACTION_CLASS)
  if c_action_class:
    action_classes = c_action_class
  else:
    action_classes = get_subdirs(
        dataset_path,
        basename=True,
        nonempty=True,
        sort=False,
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
          seed=config.SEED,
          augmentor=augmentor,
          max_vids_per_class=config.DATA.MAX_VIDS_PER_CLASS,
      )
      single_class_dataset.restrict_subdirs(action_class)
      dataset[action_class] = single_class_dataset
  else:
    frame_sampler = frame_sampler_from_config(config, downstream=False)
    dataset = VideoDataset(
        dataset_path,
        frame_sampler,
        seed=config.SEED,
        augmentor=augmentor,
        max_vids_per_class=config.DATA.MAX_VIDS_PER_CLASS,
    )
    dataset.restrict_subdirs(action_classes)

  return dataset
