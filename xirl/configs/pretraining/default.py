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

"""Default config variables."""

import ml_collections


def get_config():
  """Return a ConfigDict with sensible default values."""

  config = ml_collections.ConfigDict()

  # ============================================== #
  # General experiment params.
  # ============================================== #
  # The root directory where experiments will be saved.
  config.ROOT_DIR = "/tmp/xirl/"
  # RNG seed. Set this to `None` to disable seeding.
  config.SEED = 1
  # CUDNN-related parameters that affect reproducibility.
  config.CUDNN_DETERMINISTIC = False
  config.CUDNN_BENCHMARK = True
  # Pretraining algorithm to use.
  config.ALGORITHM = "tcc"
  # Number of steps between tensorboard logging.
  config.LOGGING_FREQUENCY = 100
  # Number of steps between consecutive checkpoints.
  config.CHECKPOINTING_FREQUENCY = 200

  # ============================================== #
  # Dataset params.
  # ============================================== #
  config.DATA = ml_collections.ConfigDict()

  # Absolute path to the dataset root.
  config.DATA.ROOT = "/home/kevin/datasets/xirl_corl/"  #"/tmp/xirl/datasets/divergent_env/processed/"
  # The mini-batch size. Note this only specifies the number of videos to
  # load frames from in a single batch. The effective batch size is actually
  # larger since we sample multiple frame sequences per video.
  config.DATA.BATCH_SIZE = 4
  # Which action classes to select for creating the pretraining dataset. Leave
  # it empty to load all action classes.
  config.DATA.PRETRAIN_ACTION_CLASS = ()
  # Which action classes to select for creating the dowstream dataset. Leave
  # it empty to load all action classes.
  config.DATA.DOWNSTREAM_ACTION_CLASS = ()
  # Restrict the number of videos per class. This is useful for experiments
  # that test sample complexity based on the number of pretraining
  # demonstrations.
  config.DATA.MAX_VIDS_PER_CLASS = -1
  # This controls how a video batch is created. If set to 'random', videos
  # are sampled randomly from different classes. If set to 'same_class', only
  # videos belonging to the same class folder are sampled within a batch.
  config.DATA.PRETRAINING_VIDEO_SAMPLER = "random"

  # ============================================== #
  # Frame sampling params.
  # ============================================== #
  config.FRAME_SAMPLER = ml_collections.ConfigDict()

  # A wildcard specifying the file extension for images in each video folder.
  # This will usually be either "*.jpg" or "*.png".
  config.FRAME_SAMPLER.IMAGE_EXT = "*.png"
  # This controls the type of sampling we perform on video frames.
  config.FRAME_SAMPLER.STRATEGY = "uniform"
  # The number of frames to sample per video.
  config.FRAME_SAMPLER.NUM_FRAMES_PER_SEQUENCE = 15
  # The number of context frames to sample per frame. This is useful for
  # models that use 3D convolutions.
  config.FRAME_SAMPLER.NUM_CONTEXT_FRAMES = 1
  # The stride between sampled context frames.
  config.FRAME_SAMPLER.CONTEXT_STRIDE = 3

  config.FRAME_SAMPLER.ALL_SAMPLER = ml_collections.ConfigDict()
  config.FRAME_SAMPLER.ALL_SAMPLER.STRIDE = 1

  config.FRAME_SAMPLER.STRIDED_SAMPLER = ml_collections.ConfigDict()
  config.FRAME_SAMPLER.STRIDED_SAMPLER.STRIDE = 3
  config.FRAME_SAMPLER.STRIDED_SAMPLER.OFFSET = True

  config.FRAME_SAMPLER.UNIFORM_SAMPLER = ml_collections.ConfigDict()
  config.FRAME_SAMPLER.UNIFORM_SAMPLER.OFFSET = 0

  # Currently, this frame sampler has no additional kwargs.
  config.FRAME_SAMPLER.WINDOW_SAMPLER = ml_collections.ConfigDict()

  # ============================================== #
  # Data augmentation params.
  # ============================================== #
  config.DATA_AUGMENTATION = ml_collections.ConfigDict()

  # The image resolution to train on.
  config.DATA_AUGMENTATION.IMAGE_SIZE = (112, 112)
  # A list of image augmentations to apply to the training dataset. Note that
  # the order matters, e.g. normalize should be done last if you decide to
  # turn it on.
  config.DATA_AUGMENTATION.TRAIN_TRANSFORMS = [
      "random_resized_crop",
      "color_jitter",
      "grayscale",
      "gaussian_blur",
      # "normalize",
  ]
  # A list of image augmentations to apply to the evaluation dataset.
  config.DATA_AUGMENTATION.EVAL_TRANSFORMS = [
      "global_resize",
      # "normalize",
  ]

  # ============================================== #
  # Evaluator params.
  # ============================================== #
  config.EVAL = ml_collections.ConfigDict()

  # How many iterations of the downstream dataloaders to run. Set to None to
  # evaluate the entire dataloader.
  config.EVAL.VAL_ITERS = 20
  # The number of steps in between every evaluation.
  config.EVAL.EVAL_FREQUENCY = 500
  # A list of downstream task evaluators that will be run sequentially every
  # EVAL_FREQUENCY steps.
  config.EVAL.DOWNSTREAM_TASK_EVALUATORS = [
      "reward_visualizer",
      "kendalls_tau",
  ]
  # What distance metric to use in the embedding space. Should match what was
  # used in the loss computation.
  # Can be one of ['cosine', 'sqeuclidean'].
  config.EVAL.DISTANCE = "sqeuclidean"

  config.EVAL.KENDALLS_TAU = ml_collections.ConfigDict()
  config.EVAL.KENDALLS_TAU.STRIDE = 3

  config.EVAL.REWARD_VISUALIZER = ml_collections.ConfigDict()
  config.EVAL.REWARD_VISUALIZER.NUM_PLOTS = 2

  config.EVAL.CYCLE_CONSISTENCY = ml_collections.ConfigDict()
  config.EVAL.CYCLE_CONSISTENCY.STRIDE = 1

  config.EVAL.NEAREST_NEIGHBOUR_VISUALIZER = ml_collections.ConfigDict()
  config.EVAL.NEAREST_NEIGHBOUR_VISUALIZER.NUM_VIDEOS = 4

  config.EVAL.EMBEDDING_VISUALIZER = ml_collections.ConfigDict()
  config.EVAL.EMBEDDING_VISUALIZER.NUM_SEQS = 2

  config.EVAL.RECONSTRUCTION_VISUALIZER = ml_collections.ConfigDict()
  config.EVAL.RECONSTRUCTION_VISUALIZER.NUM_FRAMES = 2

  # ============================================== #
  # Model params.
  # ============================================== #
  config.MODEL = ml_collections.ConfigDict()

  config.MODEL.MODEL_TYPE = "resnet18_linear"
  config.MODEL.EMBEDDING_SIZE = 32
  config.MODEL.NORMALIZE_EMBEDDINGS = False
  config.MODEL.LEARNABLE_TEMP = False

  # ============================================== #
  # Loss params.
  # ============================================== #
  config.LOSS = ml_collections.ConfigDict()

  ## TCC loss.
  config.LOSS.TCC = ml_collections.ConfigDict()
  config.LOSS.TCC.STOCHASTIC_MATCHING = False
  config.LOSS.TCC.LOSS_TYPE = "regression_mse"
  config.LOSS.TCC.CYCLE_LENGTH = 2
  config.LOSS.TCC.LABEL_SMOOTHING = 0.1
  config.LOSS.TCC.SOFTMAX_TEMPERATURE = 0.1
  config.LOSS.TCC.NORMALIZE_INDICES = True
  config.LOSS.TCC.VARIANCE_LAMBDA = 0.001
  config.LOSS.TCC.HUBER_DELTA = 0.1
  config.LOSS.TCC.SIMILARITY_TYPE = "l2"  # cosine

  ## TCN loss.
  config.LOSS.TCN = ml_collections.ConfigDict()
  config.LOSS.TCN.POS_RADIUS = 1
  config.LOSS.TCN.NEG_RADIUS = 4
  config.LOSS.TCN.NUM_PAIRS = 2
  config.LOSS.TCN.MARGIN = 1.0
  config.LOSS.TCN.TEMPERATURE = 0.1

  ## LIFS loss.
  config.LOSS.LIFS = ml_collections.ConfigDict()
  config.LOSS.LIFS.TEMPERATURE = 1.0

  # ============================================== #
  # Optimizer params
  # ============================================== #
  config.OPTIM = ml_collections.ConfigDict()

  config.OPTIM.TRAIN_MAX_ITERS = 4_000
  # L2 regularization.
  config.OPTIM.WEIGHT_DECAY = 1e-4
  # Learning rate.
  config.OPTIM.LR = 1e-5

  # ============================================== #
  # End of config file
  # ============================================== #

  return config
