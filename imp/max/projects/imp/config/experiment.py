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

"""Configuration of the experiment pipeline."""

import dataclasses
import functools

import fiddle as fdl
from jax import numpy as jnp
import numpy as np
import pyglove as pg

from imp.max.config import registry
from imp.max.config import validators
from imp.max.core import constants
from imp.max.core import utils
from imp.max.data import config as base_data_config
from imp.max.evaluation import config as eval_config
from imp.max.execution import config as exec_config
from imp.max.execution import executors
from imp.max.modeling import config as base_model_config
from imp.max.modeling.garden import config as garden_config
from imp.max.optimization import config as opt_config
from imp.max.projects.imp.config import data as data_config
from imp.max.projects.imp.config import model as model_config

Mode = exec_config.Mode
Task = exec_config.Task
Experiment = exec_config.Experiment
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName
Modality = constants.Modality
ENCODER = DataFeatureRoute.ENCODER
DECODER = DataFeatureRoute.DECODER
TOKEN_RAW = DataFeatureName.TOKEN_RAW
TOKEN_ID = DataFeatureName.TOKEN_ID
LABEL = DataFeatureName.LABEL
LOGITS = DataFeatureName.LOGITS
VISION = Modality.VISION
WAVEFORM = Modality.WAVEFORM
SPECTROGRAM = Modality.SPECTROGRAM
TEXT = Modality.TEXT

register_with_class = registry.Registrar.register_with_class


# -------------------------------
# Global Params
# -------------------------------


# Using bfloat16 in the data pipeline and model results in significantly lower
# memory overhead and slightly faster training times.
DATA_DTYPE = 'bfloat16'
MODEL_DTYPE = jnp.float32
EVAL_MODEL_DTYPE = jnp.bfloat16
OBJECTIVE_DTYPE = jnp.float32
REMAT = 'full'  # Remat significantly reduces memory usage
SCANNED = True  # Scan significantly reduces compilation time on larger models

BASE_TRAIN_BATCH_SIZE = data_config.BASE_TRAIN_BATCH_SIZE
EVAL_BATCH_SIZE = data_config.BASE_EVAL_BATCH_SIZE

BASE_BATCH_SIZE = data_config.BASE_TRAIN_BATCH_SIZE

# Specify the maximum input size for positional encoding.
MAX_VIDEO_INPUT_SIZE = (64, 512, 512, 3)
MAX_WAVEFORM_INPUT_SIZE = data_config.WAVEFORM_PATCH_SIZE * 512
MAX_SPECTROGRAM_INPUT_SIZE = (1024, 128)
MAX_TEXT_INPUT_SIZE = 4096
# Batch size for parameter initialization, large enough for any TPU config
DUMMY_BATCH_SIZE = 1024 * 8

VIDEO_PATCH_SIZE = data_config.VIDEO_PATCH_SIZE
WAVEFORM_PATCH_SIZE = data_config.WAVEFORM_PATCH_SIZE
SPECTROGRAM_PATCH_SIZE = data_config.SPECTROGRAM_PATCH_SIZE

GENERATIVE_VISON_TARGETS = (
    np.prod(VIDEO_PATCH_SIZE) * data_config.BASE_VIDEO_INPUT_SIZE[-1])
GENERATIVE_SPECTROGRAM_TARGETS = np.prod(SPECTROGRAM_PATCH_SIZE)

TEXT_VOCAB_SIZE = constants.TextVocabSize.T5_EN
TOKENIZER = base_data_config.Tokenizer(name=constants.T5_EN)


def perception_model(config,
                     **kwargs):
  """Apply arguments for a perception model."""
  # Use functools to allow overrides of these default args.
  config = functools.partial(
      config,
      input_batch_size=DUMMY_BATCH_SIZE,
      vision_input_size=MAX_VIDEO_INPUT_SIZE,
      vision_patch_size=VIDEO_PATCH_SIZE,
      waveform_input_size=MAX_WAVEFORM_INPUT_SIZE,
      waveform_patch_size=WAVEFORM_PATCH_SIZE,
      spectrogram_input_size=MAX_SPECTROGRAM_INPUT_SIZE,
      spectrogram_patch_size=SPECTROGRAM_PATCH_SIZE,
      text_input_size=MAX_TEXT_INPUT_SIZE,
      vision_classes=data_config.VISION_CLASSES,
      waveform_classes=data_config.AUDIO_CLASSES,
      spectrogram_classes=data_config.AUDIO_CLASSES,
      text_classes=data_config.TEXT_CLASSES,
      scanned_layers=SCANNED)
  return config(**kwargs)


def perception_generation_model(config,
                                **kwargs):
  """Apply arguments for a generation model."""
  # Use functools to allow overrides of these default args.
  config = functools.partial(
      config,
      input_batch_size=DUMMY_BATCH_SIZE,
      vision_input_size=MAX_VIDEO_INPUT_SIZE,
      vision_patch_size=VIDEO_PATCH_SIZE,
      waveform_input_size=MAX_WAVEFORM_INPUT_SIZE,
      waveform_patch_size=WAVEFORM_PATCH_SIZE,
      spectrogram_input_size=MAX_SPECTROGRAM_INPUT_SIZE,
      spectrogram_patch_size=SPECTROGRAM_PATCH_SIZE,
      text_input_size=MAX_TEXT_INPUT_SIZE,
      vision_classes=data_config.VISION_CLASSES,
      waveform_classes=data_config.AUDIO_CLASSES,
      spectrogram_classes=data_config.AUDIO_CLASSES,
      text_classes=data_config.TEXT_CLASSES,
      scanned_layers=SCANNED,
      vision_targets=GENERATIVE_VISON_TARGETS,
      waveform_targets=WAVEFORM_PATCH_SIZE,
      spectrogram_targets=GENERATIVE_SPECTROGRAM_TARGETS,
      text_targets=TEXT_VOCAB_SIZE)
  return config(**kwargs)


# -------------------------------
# Objectives
# -------------------------------


# Note: use the same ordering as the dataloader configs
CrossModalNCEforVisionText = functools.partial(
    opt_config.CrossModalNCE,
    modality_pair_weights=(
        ((Modality.VISION, Modality.TEXT), 1.),
    ),
    hparams_route_key=ENCODER,
    dtype=OBJECTIVE_DTYPE,
)
CrossModalNCEforVideoPerception = functools.partial(
    opt_config.CrossModalNCE,
    modality_pair_weights=(
        ((Modality.VISION, Modality.TEXT), 1.),
        ((Modality.VISION, Modality.WAVEFORM), 1.),
        ((Modality.VISION, Modality.SPECTROGRAM), 1.),
        ((Modality.WAVEFORM, Modality.SPECTROGRAM), 1.),
    ),
    hparams_route_key=ENCODER,
    dtype=OBJECTIVE_DTYPE,
)
SoftmaxCrossEntropyForVisionCls = functools.partial(
    opt_config.SoftmaxCrossEntropy,
    route_key=ENCODER,
    predictions_key=LOGITS,
    targets_key=LABEL,
    modality_token_weights=(
        ((VISION, TOKEN_RAW), 1.),
    ),
    dtype=OBJECTIVE_DTYPE,
)
SoftmaxCrossEntropyForAudioCls = functools.partial(
    opt_config.SoftmaxCrossEntropy,
    route_key=ENCODER,
    predictions_key=LOGITS,
    targets_key=LABEL,
    modality_token_weights=(
        ((SPECTROGRAM, TOKEN_RAW), 1.),
    ),
    dtype=OBJECTIVE_DTYPE,
)

# ----------------------------------------------------------------------
# --------------------- EXAMPLE OBJECTIVE CONFIGS ----------------------
# ----------------------------------------------------------------------

IMAGE_PERCEPTION_PRETRAIN_OBJECTIVES = (
    CrossModalNCEforVisionText(),
    SoftmaxCrossEntropyForVisionCls(predictions_key=f'{LOGITS}_example_3'),
)
VIDEO_PERCEPTION_PRETRAIN_OBJECTIVES = (
    CrossModalNCEforVisionText(),
    SoftmaxCrossEntropyForVisionCls(predictions_key=f'{LOGITS}_example_7'),
)
AUDIO_PERCEPTION_PRETRAIN_OBJECTIVES = (
    opt_config.CrossModalNCE(
        modality_pair_weights=(
            ((Modality.VISION, Modality.TEXT), 1.0),
            ((Modality.VISION, Modality.SPECTROGRAM), 1.0),
            ((Modality.SPECTROGRAM, Modality.TEXT), 1.0),
        ),
        hparams_route_key=ENCODER,
        dtype=OBJECTIVE_DTYPE,
    ),
    SoftmaxCrossEntropyForAudioCls(predictions_key=f'{LOGITS}_example_5'),
)
ALL_PERCEPTION_PRETRAIN_OBJECTIVES = (
    IMAGE_PERCEPTION_PRETRAIN_OBJECTIVES
    + VIDEO_PERCEPTION_PRETRAIN_OBJECTIVES
    + AUDIO_PERCEPTION_PRETRAIN_OBJECTIVES
)
TEXT_UNDERSTANDING_PRETRAIN_OBJECTIVES = (
    opt_config.SoftmaxCrossEntropy(
        modality_token_weights=(((TEXT, TOKEN_ID), 1.0),),
        route_key=DECODER,
        predictions_key=LOGITS,
        targets_key=TOKEN_ID,
        one_hot_targets=False,
        left_shift_targets=False,
        dtype=OBJECTIVE_DTYPE,
    ),
)
IMAGE_GENERATION_PRETRAIN_OBJECTIVES = (
    opt_config.MeanSquaredError(
        modality_token_weights=(((VISION, TOKEN_RAW), 1.0),),
        route_key=DECODER,
        predictions_key=TOKEN_RAW,
        targets_key=TOKEN_RAW,
        z_score_targets=True,
        dtype=OBJECTIVE_DTYPE,
    ),
)
TEXT_GENERATION_PRETRAIN_OBJECTIVES = (
    opt_config.SoftmaxCrossEntropy(
        modality_token_weights=(((TEXT, TOKEN_ID), 1.0),),
        route_key=DECODER,
        predictions_key=LOGITS,
        targets_key=TOKEN_ID,
        one_hot_targets=False,
        left_shift_targets=True,
        dtype=OBJECTIVE_DTYPE,
    ),
)

_PRETRAIN_METRICS_AGGREGATION_NAME_PATTERN = (
    '.*example_1.*/top_1',
    '.*example_2.*/top_1',
    '.*example_3.*/top_1',
    '.*example_4.*/top_1',
    '.*example_5.*/top_1',
    '.*example_6.*/top_1',
    '.*example_7.*/top_1',
)
_EVAL_METRICS_AGGREGATION_NAME_PATTERN = (
    '.*example_5.*/top_1',
    '.*example_5.*/zs_top_1',
    '.*example_4.*/text_to_vision/R1',
    '.*example_4.*/vision_to_text/R1',
)

# ----------------------------------------------------------------------
# -------------------------- END OF EXAMPLES ---------------------------
# ----------------------------------------------------------------------



# -------------------------------
# Dataset configs
# -------------------------------


@validators.validate
@dataclasses.dataclass
class BaseExperimentData(base_data_config.ExperimentData):
  """Base experiment data configuration."""

  vision_spatial_patch_size: tuple[int, int] = VIDEO_PATCH_SIZE[1:]
  vision_temporal_patch_size: int = VIDEO_PATCH_SIZE[0]
  waveform_temporal_patch_size: int = WAVEFORM_PATCH_SIZE
  spectrogram_temporal_patch_size: int = SPECTROGRAM_PATCH_SIZE[0]
  spectrogram_spectoral_patch_size: int = SPECTROGRAM_PATCH_SIZE[1]
  is_training: bool | None = None  # training status is set in loaders
  shuffle: bool | None = None
  tokenizer: base_data_config.Tokenizer = dataclasses.field(
      default_factory=lambda: TOKENIZER
  )
  dtype: str = DATA_DTYPE
  checkpointing: bool = False
  loaders: tuple[base_data_config.Loader, Ellipsis] = ()


@validators.validate
@dataclasses.dataclass
class BasePreTrainExperimentData(BaseExperimentData):
  """Pretrain base experiment data configuration."""

  shuffle: bool = True
  is_training: bool = True
  microbatch_splits: int = 1


@validators.validate
@dataclasses.dataclass
class BaseEvalExperimentData(BaseExperimentData):
  """Eval base experiment data configuration."""

  shuffle: bool = False
  checkpointing: bool = False


# -------------------------------
# Base Optimization configs
# -------------------------------


@validators.validate
@dataclasses.dataclass
class SmallPreTrainOptimization(opt_config.Optimization):
  """Pretrain small optimization configuration."""

  restore_path: str | None = None
  max_checkpoints: int = 20
  total_steps: int = 200_000
  save_checkpoint_freq: int = 5_000
  gradient_clip_norm: float = 1.0
  optimizer: opt_config.Optimizer = dataclasses.field(
      default_factory=lambda: opt_config.AdamWOptimizer(  # pylint: disable=g-long-lambda
          b1=0.9,
          b2=0.999,
          eps=1e-7,
          weight_decay=1e-4,
          learning_rate=opt_config.PreWarmupCosineDecayLearningRate(
              init_value=1e-6,
              peak_value=8e-4,
              warmup_steps=10_000,
              decay_steps=200_000,
              end_value=0.0,
          ),
      )
  )


@validators.validate
@dataclasses.dataclass
class BasePreTrainOptimization(opt_config.Optimization):
  """Pretrain base optimization configuration."""

  restore_path: str | None = None
  max_checkpoints: int = 10
  total_steps: int = 500_000
  save_checkpoint_freq: int = 5_000
  gradient_clip_norm: float = 1.0
  optimizer: opt_config.Optimizer = dataclasses.field(
      default_factory=lambda: opt_config.AdamWOptimizer(  # pylint: disable=g-long-lambda
          b1=0.9,
          b2=0.999,
          eps=1e-7,
          weight_decay=1e-4,
          learning_rate=opt_config.PreWarmupCosineDecayLearningRate(
              init_value=5e-6,
              peak_value=1e-3,
              warmup_steps=25_000,
              decay_steps=500_000,
              end_value=1e-6,
              pre_warmup_steps=2_000,
              pre_warmup_init_value=2e-5,
          ),
      )
  )
  metrics_postprocess_fn: fdl.Partial = dataclasses.field(
      default_factory=lambda: fdl.Partial(  # pylint: disable=g-long-lambda
          utils.aggregate_metrics,
          naming_patterns=_PRETRAIN_METRICS_AGGREGATION_NAME_PATTERN,
          name='aggregated/top_1',
      )
  )


@validators.validate
@dataclasses.dataclass
class LargePreTrainOptimization(BasePreTrainOptimization):
  """Large-scale pretrain base optimization configuration.

  Uses a longer warmup to reduce chances of divergence.
  """

  optimizer: opt_config.Optimizer = dataclasses.field(
      default_factory=lambda: opt_config.AdamWOptimizer(  # pylint: disable=g-long-lambda
          b1=0.9,
          b2=0.999,
          eps=1e-7,
          weight_decay=1e-4,
          learning_rate=opt_config.PreWarmupCosineDecayLearningRate(
              init_value=1e-6,
              peak_value=5e-4,
              warmup_steps=50_000,
              decay_steps=500_000,
              end_value=1e-6,
              pre_warmup_steps=5_000,
              pre_warmup_init_value=1e-5,
          ),
      )
  )


# -------------------------------
# Base Experiment Configs
# -------------------------------


@validators.validate
@dataclasses.dataclass
class BasePreTrainExperiment(exec_config.Experiment):
  """Configuration of a base pretrain experiment."""

  path: str = ''
  mode: Mode = Mode.TRAIN
  task: Task = Task.PRETRAIN
  execution: exec_config.Execution = dataclasses.field(
      default_factory=lambda: exec_config.Execution(  # pylint: disable=g-long-lambda
          partitioning=exec_config.Partitioning(
              num_partitions=1,
              model_parallel_submesh=None,
              params_on_devices=True,
          )
      )
  )


@validators.validate
@dataclasses.dataclass
class BaseEvalExperiment(BasePreTrainExperiment):
  """Configuration of a base evaluation experiment."""

  mode: Mode = Mode.EVAL
  evaluation: eval_config.Evaluation = dataclasses.field(
      default_factory=lambda: eval_config.Evaluation(  # pylint: disable=g-long-lambda
          restore_path=None,
          metric=eval_config.MetricStack(
              classification=(eval_config.Accuracy(),),
              retrieval=(eval_config.RetrievalRecall(),),
          ),
          metrics_postprocess_fn=fdl.Partial(
              utils.aggregate_metrics,
              naming_patterns=_EVAL_METRICS_AGGREGATION_NAME_PATTERN,
          ),
      )
  )
  optimization: opt_config.Optimization = dataclasses.field(
      default_factory=BasePreTrainOptimization
  )


# -------------------------------
# Experiment Configs
# -------------------------------


# -------------------------------
# Perception Experiment configs
# -------------------------------


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class ImpSmallTxtTrainExperiment(BasePreTrainExperiment):
  r"""Small IMP train experiment with text pretraining.

  Follow the instructions in main.py to run an experiment with this config.

  """

  name: str = 'imp_small.txt.train'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SmallIMP,
          text_targets=constants.TextVocabSize.T5_EN,
          remat=REMAT,
          dtype=MODEL_DTYPE,
      )
  )
  data: base_data_config.ExperimentData = dataclasses.field(
      default_factory=lambda: BasePreTrainExperimentData(  # pylint: disable=g-long-lambda
          loaders=data_config.TEXT_UNDERSTANDING_PRETRAIN_LOADERS
      )
  )
  optimization: opt_config.Optimization = dataclasses.field(
      default_factory=lambda: SmallPreTrainOptimization(  # pylint: disable=g-long-lambda
          loss=TEXT_UNDERSTANDING_PRETRAIN_OBJECTIVES
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class ImpSmallTxtEvalExperiment(BaseEvalExperiment):
  r"""Small IMP eval experiment with text pretraining."""

  name: str = 'imp_small.txt.eval'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SmallIMP,
          text_targets=constants.TextVocabSize.T5_EN,
          dtype=EVAL_MODEL_DTYPE,
      )
  )
  data: base_data_config.ExperimentData = dataclasses.field(
      default_factory=lambda: BaseEvalExperimentData(  # pylint: disable=g-long-lambda
          loaders=data_config.BULK_EVAL_LOADERS
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class ImpSmallImgTrainExperiment(BasePreTrainExperiment):
  r"""Small IMP train experiment with image-text pretraining.

  Follow the instructions in main.py to run an experiment with this config.

  """

  name: str = 'imp_small.img.train'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SmallIMP, remat=REMAT, dtype=MODEL_DTYPE
      )
  )
  data: base_data_config.ExperimentData = dataclasses.field(
      default_factory=lambda: BasePreTrainExperimentData(  # pylint: disable=g-long-lambda
          loaders=data_config.IMAGE_PERCEPTION_PRETRAIN_LOADERS
      )
  )
  optimization: opt_config.Optimization = dataclasses.field(
      default_factory=lambda: SmallPreTrainOptimization(  # pylint: disable=g-long-lambda
          loss=IMAGE_PERCEPTION_PRETRAIN_OBJECTIVES
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class ImpSmallImgEvalExperiment(BaseEvalExperiment):
  r"""Small IMP eval experiment with image-text pretraining."""

  name: str = 'imp_small.img.eval'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SmallIMP, dtype=EVAL_MODEL_DTYPE
      )
  )
  data: base_data_config.ExperimentData = dataclasses.field(
      default_factory=lambda: BaseEvalExperimentData(  # pylint: disable=g-long-lambda
          loaders=data_config.BULK_EVAL_LOADERS
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class SparseMoeImpSmallImgTrainExperiment(ImpSmallImgTrainExperiment):
  r"""Sparse MoE Small IMP train experiment with image-text pretraining.

  Follow the instructions in main.py to run an experiment with this config.

  """

  name: str = 'sparse_moe_imp_small.img.train'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SparseMoeSmallIMP,
          num_experts=4,
          num_moe_layers=6,
          # Set a high temperature to allow fast convergence during warmup
          temperature_init=1.0,
          remat=REMAT,
          dtype=MODEL_DTYPE,
      )
  )
  execution: exec_config.Execution = dataclasses.field(
      default_factory=lambda: exec_config.Execution(  # pylint: disable=g-long-lambda
          partitioning=exec_config.Partitioning(
              num_partitions=1,
              num_experts=4,
              model_parallel_submesh=None,
              params_on_devices=True,
          )
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class SparseMoeImpSmallImgEvalExperiment(ImpSmallImgEvalExperiment):
  r"""Sparse MoE Small IMP eval experiment with image-text pretraining."""

  name: str = 'sparse_moe_imp_small.img.eval'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SparseMoeSmallIMP,
          num_experts=4,
          num_moe_layers=6,
          dtype=EVAL_MODEL_DTYPE,
      )
  )
  execution: exec_config.Execution = dataclasses.field(
      default_factory=lambda: exec_config.Execution(  # pylint: disable=g-long-lambda
          partitioning=exec_config.Partitioning(
              num_partitions=1,
              num_experts=4,
              model_parallel_submesh=None,
              params_on_devices=True,
          )
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class ImpBaseImgTrainExperiment(BasePreTrainExperiment):
  r"""Base IMP train experiment with image-text pretraining.

  Follow the instructions in main.py to run an experiment with this config.

  """

  name: str = 'imp_base.img.train'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.BaseIMP, remat=REMAT, dtype=MODEL_DTYPE
      )
  )
  data: base_data_config.ExperimentData = dataclasses.field(
      default_factory=lambda: BasePreTrainExperimentData(  # pylint: disable=g-long-lambda
          loaders=data_config.IMAGE_PERCEPTION_PRETRAIN_LOADERS
      )
  )
  optimization: opt_config.Optimization = dataclasses.field(
      default_factory=lambda: BasePreTrainOptimization(  # pylint: disable=g-long-lambda
          loss=IMAGE_PERCEPTION_PRETRAIN_OBJECTIVES
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class ImpBaseImgEvalExperiment(BaseEvalExperiment):
  r"""Base IMP eval experiment with image-text pretraining.

  Follow the instructions in main.py to run an experiment with this config.

  """

  name: str = 'imp_base.img.eval'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.BaseIMP, dtype=EVAL_MODEL_DTYPE
      )
  )
  data: base_data_config.ExperimentData = dataclasses.field(
      default_factory=lambda: BaseEvalExperimentData(  # pylint: disable=g-long-lambda
          loaders=data_config.BULK_EVAL_LOADERS
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class ImpBaseAllTrainExperiment(BasePreTrainExperiment):
  r"""Base IMP train experiment with video-image-audio-text pretraining.

  Follow the instructions in main.py to run an experiment with this config.

  """

  name: str = 'imp_base.all.train'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.BaseIMP, remat=REMAT, dtype=MODEL_DTYPE
      )
  )
  data: base_data_config.ExperimentData = dataclasses.field(
      default_factory=lambda: BasePreTrainExperimentData(  # pylint: disable=g-long-lambda
          loaders=data_config.ALL_PERCEPTION_PRETRAIN_LOADERS
      )
  )
  optimization: opt_config.Optimization = dataclasses.field(
      default_factory=lambda: BasePreTrainOptimization(  # pylint: disable=g-long-lambda
          loss=ALL_PERCEPTION_PRETRAIN_OBJECTIVES
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class ImpBaseAllEvalExperiment(BaseEvalExperiment):
  r"""Base IMP eval experiment with video-image-audio-text pretraining.

  Follow the instructions in main.py to run an experiment with this config.

  """

  name: str = 'imp_base.all.eval'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.BaseIMP, dtype=EVAL_MODEL_DTYPE
      )
  )
  data: base_data_config.ExperimentData = dataclasses.field(
      default_factory=lambda: BaseEvalExperimentData(  # pylint: disable=g-long-lambda
          loaders=data_config.BULK_EVAL_LOADERS
      )
  )


# TODO(b/233959132): add config verification for MoE experiments
@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class SparseMoeImpBaseImgTrainExperiment(ImpBaseImgTrainExperiment):
  r"""Base Sparse MoE-IMP train experiment.

  Follow the instructions in main.py to run an experiment with this config.

  """

  name: str = 'sparse_moe_imp_base.img.train'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SparseMoeBaseIMP,
          num_experts=16,
          num_moe_layers=6,
          # Set a high temperature to allow fast convergence during warmup
          temperature_init=1.0,
          remat=REMAT,
          dtype=MODEL_DTYPE,
      )
  )
  execution: exec_config.Execution = dataclasses.field(
      default_factory=lambda: exec_config.Execution(  # pylint: disable=g-long-lambda
          partitioning=exec_config.Partitioning(
              num_partitions=1,
              num_experts=16,
              model_parallel_submesh=None,
              params_on_devices=True,
          )
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class SparseMoeImpBaseImgEvalExperiment(ImpBaseImgEvalExperiment):
  """Base Sparse MoE-IMP eval experiment."""

  name: str = 'sparse_moe_imp_base.img.eval'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SparseMoeBaseIMP,
          num_experts=16,
          num_moe_layers=6,
          dtype=EVAL_MODEL_DTYPE,
      )
  )
  execution: exec_config.Execution = dataclasses.field(
      default_factory=lambda: exec_config.Execution(  # pylint: disable=g-long-lambda
          partitioning=exec_config.Partitioning(
              num_partitions=1,
              num_experts=16,
              model_parallel_submesh=None,
              params_on_devices=True,
          )
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class SparseMoeImpBaseAllTrainExperiment(ImpBaseAllTrainExperiment):
  r"""Base Sparse MoE-IMP train experiment.

  Follow the instructions in main.py to run an experiment with this config.

  """

  name: str = 'sparse_moe_imp_base.all.train'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SparseMoeBaseIMP,
          num_experts=16,
          num_moe_layers=6,
          # Set a high temperature to allow fast convergence during warmup
          temperature_init=1.0,
          remat=REMAT,
          dtype=MODEL_DTYPE,
      )
  )
  execution: exec_config.Execution = dataclasses.field(
      default_factory=lambda: exec_config.Execution(  # pylint: disable=g-long-lambda
          partitioning=exec_config.Partitioning(
              num_partitions=1,
              num_experts=16,
              model_parallel_submesh=None,
              params_on_devices=True,
          )
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class SparseMoeImpBaseAllEvalExperiment(ImpBaseAllEvalExperiment):
  """Base Sparse MoE-IMP eval experiment."""

  name: str = 'sparse_moe_imp_base.all.eval'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SparseMoeBaseIMP,
          num_experts=16,
          num_moe_layers=6,
          dtype=EVAL_MODEL_DTYPE,
      )
  )
  execution: exec_config.Execution = dataclasses.field(
      default_factory=lambda: exec_config.Execution(  # pylint: disable=g-long-lambda
          partitioning=exec_config.Partitioning(
              num_partitions=1,
              num_experts=16,
              model_parallel_submesh=None,
              params_on_devices=True,
          )
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class SparseMoeImpLargeAllTrainExperiment(SparseMoeImpBaseAllTrainExperiment):
  r"""Large Sparse MoE-IMP train experiment.

  Follow the instructions in main.py to run an experiment with this config.

  """

  name: str = 'sparse_moe_imp_large.all.train'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SparseMoeLargeIMP,
          num_experts=16,
          num_moe_layers=6,
          # Set a high temperature to allow fast convergence during warmup
          temperature_init=1.0,
          remat=REMAT,
          dtype=MODEL_DTYPE,
      )
  )
  optimization: opt_config.Optimization = dataclasses.field(
      default_factory=lambda: LargePreTrainOptimization(  # pylint: disable=g-long-lambda
          loss=ALL_PERCEPTION_PRETRAIN_OBJECTIVES,
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class SparseMoeImpLargeAllEvalExperiment(SparseMoeImpBaseAllEvalExperiment):
  """Large Sparse MoE-IMP eval experiment."""

  name: str = 'sparse_moe_imp_large.all.eval'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SparseMoeLargeIMP,
          num_experts=16,
          num_moe_layers=6,
          dtype=EVAL_MODEL_DTYPE,
      )
  )
  optimization: opt_config.Optimization = dataclasses.field(
      default_factory=lambda: LargePreTrainOptimization(  # pylint: disable=g-long-lambda
          loss=ALL_PERCEPTION_PRETRAIN_OBJECTIVES,
      )
  )


# ---------------------------------------------------
# Perception & Generation Experiment configs
# ---------------------------------------------------


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class IMPeGeSmallImgTrainExperiment(BasePreTrainExperiment):
  r"""Small IMPeGe train experiment with image-text pretraining.

  Follow the instructions in main.py to run an experiment with this config.

  """

  name: str = 'impege_small.img.train'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_generation_model(  # pylint: disable=g-long-lambda
          model_config.SmallIMPeGe, remat=REMAT, dtype=MODEL_DTYPE
      )
  )
  optimization: opt_config.Optimization = dataclasses.field(
      default_factory=lambda: SmallPreTrainOptimization(  # pylint: disable=g-long-lambda
          loss=(
              IMAGE_GENERATION_PRETRAIN_OBJECTIVES
              + TEXT_GENERATION_PRETRAIN_OBJECTIVES
          ),
      )
  )
  data: base_data_config.ExperimentData = dataclasses.field(
      default_factory=lambda: BasePreTrainExperimentData(  # pylint: disable=g-long-lambda
          loaders=(
              data_config.IMAGE_GENERATION_PRETRAIN_LOADERS
              + data_config.TEXT_GENERATION_PRETRAIN_LOADERS
          ),
      )
  )


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class IMPeGeSmallImgEvalExperiment(BaseEvalExperiment):
  r"""Small IMP eval experiment with image-text pretraining."""

  name: str = 'impege_small.img.eval'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_generation_model(  # pylint: disable=g-long-lambda
          model_config.SmallIMPeGe, dtype=EVAL_MODEL_DTYPE
      )
  )
  data: base_data_config.ExperimentData = dataclasses.field(
      default_factory=lambda: BaseEvalExperimentData(  # pylint: disable=g-long-lambda
          loaders=data_config.BULK_EVAL_LOADERS
      )
  )


# -------------------------------
# Search Experiment configs
# -------------------------------


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class SearchImpBaseImgV1TrainExperiment(ImpBaseImgTrainExperiment):
  r"""Base IMP image search experiment V1 (train).

  Searches over learning rate.

  Follow the instructions in main.py to run an experiment with this config.

  """

  name: str = 'search.imp_base.v1.img.train'
  search_algorithm: str = 'GRID_SEARCH'
  max_num_trials: int = 8

  def with_search_space(self):
    config = self.copy_and_override({})
    config.optimization.optimizer.learning_rate.peak_value = pg.oneof(  # pylint: disable=attribute-error
        (8e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 2e-3), name='learning_rate')
    return config


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class SearchImpBaseImgV1EvalExperiment(ImpBaseImgEvalExperiment):
  r"""Base IMP image search experiment V1 (eval)."""

  name: str = 'search.imp_base.v1.img.eval'
  search_algorithm: str = 'GRID_SEARCH'
  max_num_trials: int = 8

  def with_search_space(self):
    config = self.copy_and_override({})
    config.optimization.optimizer.learning_rate.peak_value = pg.oneof(  # pylint: disable=attribute-error
        (8e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 2e-3), name='learning_rate')
    return config


# -------------------------------
# Export configs
# -------------------------------


@register_with_class(executors.Executor)
@validators.validate
@dataclasses.dataclass
class SparseImpMoeBV1TrainExperiment(SparseMoeImpBaseAllTrainExperiment):
  r"""IMP-MoE-B V1 Config."""

  name: str = 'imp_moe_b_v1'
  model: base_model_config.Model = dataclasses.field(
      default_factory=lambda: perception_model(  # pylint: disable=g-long-lambda
          model_config.SparseMoeBaseIMP,
          num_experts=16,
          num_moe_layers=6,
          dtype=jnp.float32,
          input_batch_size=8,
          vision_input_size=(256, 1024, 1024, 3),
          waveform_input_size=76800,
      )
  )
