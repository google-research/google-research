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

"""Config for evolution experiments."""

from typing import List

import ml_collections as mlc


def set_synthesizer_ablation_config(abl_config,
                                    config):
  """Sets synthesizer ablation-related configs."""
  def progressive():
    config.synthesis_name = "progressive"
    config.synthesis = mlc.ConfigDict()
    config.synthesis.mode = "weighted"
    config.synthesis.max_delta = 2
    config.synthesis.min_delta = 2
    config.synthesis.eps = 0.1
    config.synthesis.exp = 2
    config.synthesis.filter_progress = False

  def primer():
    config.synthesis_name = "primer"
    config.synthesis = mlc.ConfigDict()
    config.mutator.p = 1
    config.mutator.max_perc = 1

  def automl_zero():
    config.synthesis_name = "primer"
    config.synthesis = mlc.ConfigDict()
    config.mutator.p = 1
    config.mutator.max_perc = 1
    config.synthesis.max_delta = 2
    config.synthesis.min_delta = 2
    config.synthesis.use_automl_zero = True

  def random_enum():
    config.synthesis_name = "random_enum"
    config.synthesis = mlc.ConfigDict()
    config.synthesis.max_delta = 2
    config.synthesis.min_delta = 2

  key = abl_config[0]
  abl_settings = {
      "progressive": progressive,
      "primer": primer,
      "automl_zero": automl_zero,
      "random_enum": random_enum,
  }
  if key not in abl_settings or len(abl_config) > 1:
    raise ValueError(f"Setting {abl_config} not recognized for synthesizer "
                     "ablations.")
  abl_settings[key]()


def set_property_ablation_config(abl_config, config):
  """Sets property ablation-related configs."""
  if len(abl_config) != 6:
    raise ValueError(f"Setting {abl_config} not recognized for property "
                     "ablations.")

  # The ablation string should be something like:
  # no.shape.mutate.linear.nomutate.depth
  val0, key0, val1, key1, val2, key2 = abl_config

  keys = set([key0, key1, key2])
  vals = [val0, val1, val2]
  if len(keys) != 3:
    raise ValueError(f"Setting {abl_config} not recognized for property "
                     "ablations, contained duplicate keys.")

  key_to_prop = {
      "depth": "depth_property",
      "shape": "shape_property",
      "linop": "linear_property"
  }

  if all([val == "mutate" for val in vals]): return

  for key, val in zip(keys, vals):
    if key not in ["shape", "linop", "depth"]:
      raise ValueError(f"Setting {abl_config} not recognized for property "
                       f"ablations, key {key} not defined.")
    if val not in ["no", "mutate", "nomutate"]:
      raise ValueError(f"Setting {abl_config} not recognized for property "
                       f"ablations, val {val} for key {key} not defined.")

    prop = key_to_prop[key]
    # If val = no, then do not use this property for synthesis.
    # If val = nomutate, then do not mutate this property during synthesis.
    # Any other val is treated as standard behavior (i.e., mutate).
    if val == "no":
      del config.properties[prop]
    elif val == "nomutate":
      config.properties[prop].p = 0.0
    else:
      config.properties[prop].p = .75


def set_ablation_config(ablation, config):
  """Sets ablation-related configs."""
  abl_dict = {
      "synthesizer": set_synthesizer_ablation_config,
      "property": set_property_ablation_config,
  }
  key, abl_config = ablation[0], ablation[1:]
  if key not in abl_dict:
    raise ValueError(f"Ablation {key} not recognized.")
  abl_dict[key](abl_config, config)


def set_dataset_config(dataset, config):
  """Sets dataset-related configs."""
  config.dataset = mlc.ConfigDict()
  if dataset == "imagenet2012":
    config.dataset_name = dataset
    config.dataset.train_split = "train"
    config.dataset.val_split = "validation"
    config.dataset.num_classes = 1000
    config.dataset.input_shape = (224, 224, 3)
  else:
    assert dataset == "cifar10"
    config.dataset_name = dataset
    config.dataset.train_split = "train"
    config.dataset.val_split = "test"
    config.dataset.num_classes = 10
    config.dataset.input_shape = (32, 32, 3)


def set_model_config(model, config):
  """Sets model-related configs."""
  config.model_name = model
  config.model = mlc.ConfigDict()

  # Optimizer section
  config.optim = mlc.ConfigDict()
  if model == "cnn":
    config.optim.optax_name = "trace"  # momentum
    config.optim.optax = mlc.ConfigDict()
    config.optim.optax.decay = 0.9
    config.optim.optax.nesterov = False

    config.optim.wd = 1e-4
    config.optim.wd_mults = [(".*", 1.0)]

    # Learning rate section
    config.optim.lr = 0.01
    config.optim.schedule = mlc.ConfigDict()
    config.optim.schedule.warmup_epochs = 5
    config.optim.schedule.decay_type = "cosine"
    config.optim.scale_with_batchsize = True  # Base batch-size being 256.
  elif "resnet" in model:
    config.optim.optax_name = "trace"  # momentum
    config.optim.optax = mlc.ConfigDict()
    config.optim.optax.decay = 0.9
    config.optim.optax.nesterov = False

    config.optim.wd = 1e-4
    config.optim.wd_mults = [(".*", 1.0)]
    config.optim.grad_clip_norm = 1.0

    # Learning rate section
    config.optim.lr = 0.1
    config.optim.schedule = mlc.ConfigDict()
    config.optim.schedule.warmup_epochs = 5
    config.optim.schedule.decay_type = "cosine"
    # Base batch-size being 256.
    config.optim.schedule.scale_with_batchsize = True
  elif model in [f"b{i}" for i in range(8)]:  # efficientnet
    # Optimizer section
    config.optim.optax_name = "scale_by_rms"  # momentum
    config.optim.optax = mlc.ConfigDict()
    config.optim.optax.decay = 0.9
    config.optim.optax.eps = 1e-3

    config.optim.grad_clip_norm = 1.0
    config.optim.wd = 1e-5
    # config.optim.wd_mults = [(".*", 1.0)]
    config.optim.wd_mults = [(".*", 1.0), (".*/bn.*", 0.0)]  # no decay bn
    config.optim.grad_clip_norm = 1.0

    # Learning rate section
    config.optim.lr = 0.016
    config.optim.schedule = mlc.ConfigDict()
    config.optim.schedule.warmup_epochs = 5
    config.optim.schedule.decay_type = "exponential"
    # At the end of the training, lr should be 1.2% of original value.
    # This mimic the behavior from the efficientnet paper.
    config.optim.schedule.end_lr_ratio = 0.012
    # Base batch-size being 256.
    config.optim.schedule.scale_with_batchsize = True

    config.dataset.preprocess_kwargs = mlc.ConfigDict()
    # config.dataset.preprocess_kwargs.autoaugment = True
    # config.dataset.preprocess_kwargs.label_smoothing = 0.1

    config.ema_decay = 0.9999
    # config.epochs = 350
  else:  # vit
    config.optim.optax_name = "scale_by_adam"
    config.optim.optax = mlc.ConfigDict()
    # config.optim.optax.mu_dtype = "bfloat16"

    config.optim.wd = 0.0001
    config.optim.grad_clip_norm = 1.0

    config.optim.lr = 0.001
    config.optim.schedule = mlc.ConfigDict()
    config.optim.schedule.warmup_steps = 10_000
    config.optim.schedule.decay_type = "cosine"


def get_config(params):
  """Config for training on ImageNet-1k."""
  params = params.split(".")
  assert len(params) >= 2
  if len(params) == 2:
    model, dataset = params
    ablation = None
  else:
    model, dataset, *ablation = params

  config = mlc.ConfigDict()

  config.seed = 0

  ###################
  # Train Config

  config.train = mlc.ConfigDict()
  config.train.seed = 0
  config.train.epochs = 90
  config.train.device_batch_size = 64
  config.train.log_epochs = 1

  ###################
  # Mutator Config
  p = .9
  if dataset == "cifar10":
    p = .75
  config.mutator_name = "random_subgraph"
  config.mutator = mlc.ConfigDict()
  config.mutator.p = p
  config.mutator.max_perc = .5
  if dataset == "cifar10":
    config.mutator.max_perc = .8

  ###################
  # Properties Config
  p = .8
  if dataset == "cifar10":
    p = .5
  config.properties = mlc.ConfigDict()
  config.properties.depth_property = mlc.ConfigDict()
  config.properties.depth_property.p = p
  config.properties.depth_property.delta_max = 2
  config.properties.shape_property = mlc.ConfigDict()
  config.properties.shape_property.p = 0.0
  config.properties.linear_property = mlc.ConfigDict()
  config.properties.linear_property.p = p

  ###################
  # Synthesis Config
  config.synthesis_name = "progressive"
  config.synthesis = mlc.ConfigDict()
  config.synthesis.mode = "weighted"
  config.synthesis.max_delta = 2
  config.synthesis.min_delta = 2
  config.synthesis.eps = 0.1
  config.synthesis.exp = 2
  config.synthesis.filter_progress = False
  config.synthesis.p = .2

  config.synthesis_graph = mlc.ConfigDict()
  config.synthesis_graph.p = .2

  ###################
  # Evolution Config
  config.evolution = mlc.ConfigDict()
  config.evolution.num_seed = 2

  # Mutation section
  config.evolution.mutation = mlc.ConfigDict()
  config.evolution.mutation.mutate_by_block = True
  config.evolution.mutation.synthesis_retries = 1
  config.evolution.mutation.block_delete_prob = 0.2
  config.evolution.mutation.block_add_prob = 0.2
  config.evolution.mutation.block_mutate_prob = 0.2

  # Population section
  config.evolution.population = mlc.ConfigDict()
  config.evolution.population.min_acc = 80 if dataset == "cifar10" else 60
  config.evolution.population.min_to_select = 10
  config.evolution.population.top_perc_to_select = 0.1
  config.evolution.population.top_n_to_select = 0
  config.evolution.population.use_pareto_balanced = True
  config.evolution.population.use_pareto_normalized = True
  config.evolution.population.targeted_evol = False
  if dataset == "imagenet2012":
    if model == "resnet50" or not model.startswith("resnet"):
      config.evolution.population.targeted_evol = True

  config.evolution.population.aging = mlc.ConfigDict()
  config.evolution.population.aging.warm_up_secs = 30 * 60
  config.evolution.population.aging.generations_to_live = 0  # No aging!
  config.evolution.population.aging.cosine = True
  config.evolution.population.aging.cyclic = False
  config.evolution.population.aging.cyclic_impulse = True

  # Dataset section
  set_dataset_config(dataset, config.train)

  # Model section
  set_model_config(model, config.train)

  if ablation:
    set_ablation_config(ablation, config)
  return config
