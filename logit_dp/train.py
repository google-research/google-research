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

"""Functions used for the train task."""

import functools
import os

from absl import app
from absl import flags
from absl import logging
import jax
import objax

from logit_dp.logit_dp import data_utils
from logit_dp.logit_dp import dp_utils
from logit_dp.logit_dp import encoders
from logit_dp.logit_dp import objax_utils
from logit_dp.logit_dp import trainer_factories
from logit_dp.logit_dp import training_utils


_MODEL_NAME = flags.DEFINE_enum(
    name="model_name",
    default="small_embedding_net",
    enum_values=["small_embedding_net", "ResNet18"],
    help="The name of the model to train.",
    required=False,
)

_OPTIMIZER_NAME = flags.DEFINE_enum(
    name="optimizer_name",
    default="adam",
    enum_values=["adam"],
    help="The name of the optimizer.",
    required=False,
)

_TRAIN_BATCH_SIZE = flags.DEFINE_integer(
    name="train_batch_size",
    default=100,
    help="The training batch size.",
    required=False,
)

_EPOCHS = flags.DEFINE_integer(
    name="epochs",
    default=2,
    help="The epochs.",
    required=False,
)

_STEPS_PER_EVAL = flags.DEFINE_integer(
    name="steps_per_eval",
    default=100,
    help="The number of steps per test metrics evaluation.",
    required=False,
)

_GRADIENT_ACCUMULATION_STEPS = flags.DEFINE_integer(
    name="gradient_accumulation_steps",
    default=None,
    help="The number of gradient accumulation steps for certain trainers.",
    required=False,
)

_SEQUENTIAL_COMPUTATION_STEPS = flags.DEFINE_integer(
    name="sequential_computation_steps",
    default=None,
    help=(
        "The number blocks a batch will be broken into to compute the gradient"
        " sequentially to avoid memory issues."
    ),
    required=False,
)

_LEARNING_RATE = flags.DEFINE_float(
    name="learning_rate",
    default=0.001,
    help="The learning rate.",
    required=False,
)

_WEIGHT_DECAY = flags.DEFINE_float(
    name="weight_decay",
    default=0.0,
    help="The weight decay for adamw optimizer.",
    required=False,
)

_TEMPERATURE = flags.DEFINE_float(
    name="temperature",
    default=0.2,
    help="The loss temperature.",
    required=False,
)

_NUM_EVAL_CONFUSION_MATRIX = flags.DEFINE_integer(
    name="num_eval_confusion_matrix",
    default=500,
    help="The number of samples to use for a confusion matrix class.",
    required=False,
)

_NEAREST_NEIGHBORS_PER_EXAMPLE = flags.DEFINE_integer(
    name="nearest_neighbors_per_example",
    default=3,
    help="The number of neighbors used for label prediction.",
    required=False,
)

_RESULTS_PATH = flags.DEFINE_string(
    name="results_path",
    default=None,
    help="A root folder to save the results.",
    required=True,
)

_EXPERIMENT_OVERRIDE_NAME = flags.DEFINE_string(
    name="experiment_override_name",
    default=None,
    help="A name that replaces the generated experiment name.",
    required=False,
)

_TRAINING_METHOD = flags.DEFINE_enum(
    name="training_method",
    default="non_private",
    enum_values=["non_private", "naive_dp", "logit_dp"],
    help="The type of training to apply.",
    required=False,
)

_DATASET = flags.DEFINE_enum(
    name="dataset",
    default="cifar10",
    enum_values=["cifar10", "cifar100"],
    help="The training dataset.",
    required=False,
)

_NUM_CLASSES = flags.DEFINE_integer(
    name="num_classes",
    default=10,
    help="The number of classes in the dataset.",
    required=False,
)

_NUM_TRAIN_EXAMPLES = flags.DEFINE_integer(
    name="num_train_examples",
    default=50000,
    help="The number of training examples in the training dataset.",
    required=False,
)

_DP_EPSILON = flags.DEFINE_float(
    name="dp_epsilon",
    default=None,
    help="Differential privacy epsilon.",
    required=False,
)

_DP_L2_SENSITIVITY = flags.DEFINE_float(
    name="dp_l2_sensitivity",
    default=None,
    help="Differential privacy L2 sensitivity.",
    required=False,
)

_DP_NOISE_MULTIPLIER = flags.DEFINE_float(
    name="dp_noise_multiplier",
    default=None,
    help="Differential privacy noise multiplier.",
    required=False,
)

_TASK = flags.DEFINE_enum(
    name="task",
    default="pretrain",
    enum_values=["pretrain", "finetune"],
    help="Define if pretrain or finetune on CIFAR100.",
    required=False,
)

_CHECKPOINT_PATH = flags.DEFINE_string(
    name="checkpoint_path",
    default=None,
    help="A path for pretrained model parameters.",
    required=False,
)

_FINAL_CHECKPOINT_FILE_NAME = flags.DEFINE_string(
    name="final_checkpoint_file_name",
    default="pretrain_model_params.npz",
    help="A file name for saving final parameters in `results_path`.",
    required=False,
)

_RESNET_EMBEDDING_DIM = flags.DEFINE_integer(
    name="resnet_embedding_dim",
    default=1000,
    help="The embedding dimension of the ResNet model.",
    required=False,
)

_USE_BATCH_NORM = flags.DEFINE_bool(
    name="use_batch_norm",
    default=False,
    help="Whether to use batch normalization in the ResNet model.",
)


def _get_data():
  """Returns the train and test datasets."""
  is_resnet_model = _MODEL_NAME.value == "ResNet18"
  is_obtax_framework = _TRAINING_METHOD.value == "logit_dp"
  if _DATASET.value == "cifar10":
    train_iterator, test_inputs, test_lookup = (
        data_utils.get_cifar10_data_split(
            train_batch_size=_TRAIN_BATCH_SIZE.value,
            expand_channel_dim=(is_resnet_model and is_obtax_framework),
        )
    )
  elif _DATASET.value == "cifar100":
    train_iterator, test_inputs = data_utils.get_cifar100_data_split(
        _TRAIN_BATCH_SIZE.value
    )
    test_lookup = None
  else:
    raise ValueError(f"Unknown dataset {_DATASET.value}")
  return train_iterator, test_inputs, test_lookup


def _get_trainer(train_iterator, test_inputs, test_lookup):
  """Returns the task trainer."""
  num_classes_finetuning = 0
  if _TASK.value == "finetune":
    # When finetuning must provide a string that does not contain `pretrain`.
    if (
        _FINAL_CHECKPOINT_FILE_NAME.value
        and "pretrain" not in _FINAL_CHECKPOINT_FILE_NAME.value
    ):
      checkpoint_name = _FINAL_CHECKPOINT_FILE_NAME.value
    else:
      if _DATASET.value == "cifar100":
        num_classes_finetuning = 20
      else:
        raise ValueError(
            f"Unknown number of classes for dataset {_DATASET.value}"
        )
      checkpoint_name = "finetune_model_params.npz"
    trainer = trainer_factories.make_objax_finetuning_trainer(
        train_iterator,
        test_inputs,
        checkpoint_name,
        num_classes_finetuning,
    )
    return trainer
  # Set DP params.
  inferred_noise = 0.0
  if _TRAINING_METHOD.value != "non_private":
    # Compute the noise multiplier.
    if _DP_NOISE_MULTIPLIER.value is not None and _DP_EPSILON.value is not None:
      raise ValueError(
          "Only one of DP epsilon or DP noise multiplier can be populated."
      )
    elif _DP_NOISE_MULTIPLIER.value is not None:
      inferred_noise = _DP_NOISE_MULTIPLIER.value
    elif _DP_EPSILON.value is not None:
      grad_acc_steps = _GRADIENT_ACCUMULATION_STEPS.value or 1
      effective_batch_size = grad_acc_steps * _TRAIN_BATCH_SIZE.value
      steps_per_epoch = int(_NUM_TRAIN_EXAMPLES.value / effective_batch_size)
      inferred_noise = dp_utils.compute_noise(
          _DP_EPSILON.value,
          _NUM_TRAIN_EXAMPLES.value,
          _EPOCHS.value,
          steps_per_epoch,
          effective_batch_size,
      )
    else:
      raise ValueError(
          "At least one of DP epsilon or DP noise multiplier must be populated."
      )
  if _TRAINING_METHOD.value == "non_private":
    trainer = trainer_factories.make_non_private_trainer(
        train_iterator,
        test_inputs,
        test_lookup,
        _NUM_CLASSES.value,
        _TEMPERATURE.value,
        _NUM_EVAL_CONFUSION_MATRIX.value,
        _NEAREST_NEIGHBORS_PER_EXAMPLE.value,
        _FINAL_CHECKPOINT_FILE_NAME.value,
    )
  elif _TRAINING_METHOD.value == "naive_dp":
    trainer = trainer_factories.make_naive_dp_trainer(
        train_iterator,
        test_inputs,
        test_lookup,
        _NUM_CLASSES.value,
        _TEMPERATURE.value,
        _DP_L2_SENSITIVITY.value,
        inferred_noise,
        _NUM_EVAL_CONFUSION_MATRIX.value,
        _NEAREST_NEIGHBORS_PER_EXAMPLE.value,
        _FINAL_CHECKPOINT_FILE_NAME.value,
    )
  elif _TRAINING_METHOD.value == "logit_dp":
    trainer = trainer_factories.make_logit_dp_trainer(
        train_iterator,
        test_inputs,
        test_lookup,
        _TEMPERATURE.value,
        _DP_L2_SENSITIVITY.value,
        inferred_noise,
        _GRADIENT_ACCUMULATION_STEPS.value,
        _NUM_EVAL_CONFUSION_MATRIX.value,
        _NEAREST_NEIGHBORS_PER_EXAMPLE.value,
        _SEQUENTIAL_COMPUTATION_STEPS.value,
        _WEIGHT_DECAY.value,
        _FINAL_CHECKPOINT_FILE_NAME.value,
        model_name=_MODEL_NAME.value,
    )
  else:
    raise ValueError(f"Unknown training method: {_TRAINING_METHOD.value}")
  return trainer


def _get_model():
  """Returns the task model."""
  img_in_channels = 3  # RGB images.
  if _TASK.value == "finetune":
    model = objax_utils.ObjaxFinetuningNet()
    training_utils.load_embedding_net_into_finetuning_net(
        _CHECKPOINT_PATH.value, model
    )
  else:
    if _MODEL_NAME.value == "small_embedding_net":
      if _TRAINING_METHOD.value in ["non_private", "naive_dp"]:
        model = objax_utils.ObjaxEmbeddingNet()
      elif _TRAINING_METHOD.value == "logit_dp":
        model = encoders.SmallEmbeddingNet
      else:
        raise ValueError(f"Unknown training method: {_TRAINING_METHOD.value}")
    elif _MODEL_NAME.value == "ResNet18":
      if _TRAINING_METHOD.value in ["non_private", "naive_dp"]:
        model = objax_utils.ObjaxResNet18(
            in_channels=img_in_channels,
            num_classes=_NUM_CLASSES.value,
            use_batch_norm=_USE_BATCH_NORM.value,
        )
      elif _TRAINING_METHOD.value == "logit_dp":
        model = functools.partial(
            encoders.ResNet18, embedding_dim=_RESNET_EMBEDDING_DIM.value
        )
      else:
        raise ValueError(
            f"Unknown training method for ResNet18: {_TRAINING_METHOD.value}"
        )
    else:
      raise ValueError(f"Unknown model name: {_MODEL_NAME.value}")
  return model


def _get_optimizer():
  """Returns the task optimizer."""
  if _OPTIMIZER_NAME.value == "adam":
    if _TASK.value == "finetune" or _TRAINING_METHOD.value in [
        "non_private",
        "naive_dp",
    ]:
      optimizer = objax.optimizer.Adam
    elif _TRAINING_METHOD.value == "logit_dp":
      optimizer = None  # overridden in training
    else:
      raise ValueError(f"Unknown training method: {_TRAINING_METHOD.value}")
  else:
    raise ValueError(f"Unknow optimizer: {_OPTIMIZER_NAME.value}")
  return optimizer


def _get_output_dir():
  """Returns the task output directory."""
  root_dir = _RESULTS_PATH.value
  grad_acc_steps = _GRADIENT_ACCUMULATION_STEPS.value or 1
  seq_comp_steps = _SEQUENTIAL_COMPUTATION_STEPS.value or 1

  effective_batch_size = _TRAIN_BATCH_SIZE.value * grad_acc_steps
  experiment_dir = _EXPERIMENT_OVERRIDE_NAME.value or (
      f"_{_TRAINING_METHOD.value}"
      f"_lr{_LEARNING_RATE.value}"
      f"_l2s{_DP_L2_SENSITIVITY.value}"
      f"_bs{effective_batch_size}"
      f"_ep{_EPOCHS.value}"
      f"_tm{_TEMPERATURE.value}"
      f"_scs{seq_comp_steps}"
      f"_wd{_WEIGHT_DECAY.value}"
  )
  return os.path.join(root_dir, experiment_dir)


def main(_):
  logging.info("JAX Platform: %s", jax.default_backend())
  # Get job data.
  train_iterator, train_inputs, test_lookup = _get_data()
  trainer = _get_trainer(train_iterator, train_inputs, test_lookup)
  model = _get_model()
  optimizer = _get_optimizer()
  output_dir = _get_output_dir()

  _ = trainer(
      model,
      optimizer,
      _LEARNING_RATE.value,
      _EPOCHS.value,
      _STEPS_PER_EVAL.value,
      output_dir,
  )


if __name__ == "__main__":
  app.run(main)
