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

"""Model hyper-parameters and configurations."""


class BaseModelConfig:
  """Base model configurations."""

  def __init__(self,
               final_layer_l2norm=False,
               hidden_dims=None,
               activation_fn="relu",
               softmax_temperature=1.0,
               dro_temperature=0.1,
               streaming_group_loss=False,
               streaming_group_loss_lr=0.01,
               streaming_group_metric_lr=0.01,
               metric_update_freq=1,
               regularized_dro=False,
               kl_regularizer=0.01,
               task_type="erm",
               group_reweight_strategy="loss-dro",
               group=None,
               group_weight_init=None,
               group_loss_init=None,
               group_metric_init=None,
               group_labels=None):
    """Base model config initializations.

    Args:
      final_layer_l2norm: A bool, if `True` will apply l2-normalization on the
        outputs of the embedding tower.
      hidden_dims: A list of integers where the i-th entry represents the number
        of units in the i-th hidden layer.
      activation_fn: A string, activation function, default to use "relu".
      softmax_temperature: A float, temperature of the softmax.
      dro_temperature: A float, temperature of the group re-weighting in DRO. A
        suggested range is between [0.001,0.1] depending on dataset.
      streaming_group_loss: Optional[bool], if `True` will use streaming loss
        estimations.
      streaming_group_loss_lr: Optional[float], between [0.0,1.0], larger value
        will let the estimations of group loss focus more on the current batch.
      streaming_group_metric_lr: Optional[float], between [0.0,1.0], larger
        value will let the estimations of group metric focus more on the current
        batch.
      metric_update_freq: Optional[int], group metric updates every after n
        batch.
      regularized_dro: Optional[bool], if `True` will use KL-regularised version
        of DRO.
      kl_regularizer: Optional[float], between [0.0,1.0], larger value will let
        trade-off between DRO and ERM approach towards to ERM.
      task_type: A string, defines whether the retrieval task is optimized for
        empirical risk or to be distributionally-robust. Shall be one of ["erm"
        or "robust"].
      group_reweight_strategy: Group reweighting strategy. Shall be one of
        ["loss-dro", "metric-dro"]. The vanilla version of "loss-dro" upweights
        by subgroup losses, "metric-dro" uses group metrics to replace group
        loss for reweighting subgroup.
      group: A string, pre-defined subgroup name to apply robust optimization.
      group_weight_init: A list of [num_groups] floats for group weight
        initialization that add up to 1, e.g. [0.3, 0.2, 0.5].
      group_loss_init: A list of [num_groups] floats for group loss
        initialization, e.g. [1.0, 2.0, 3.0].
      group_metric_init: A list of [num_groups] floats for group metric
        initialization, e.g. [0.0, 0.0, 0.0].
      group_labels: A list of integers or strings as group identity labels. Used
        to define subgroups for optimizing robust loss.
    """
    # Model hyperparamters. This part is common for both ERM or DRO.
    self.final_layer_l2norm = final_layer_l2norm
    self.activation_fn = activation_fn
    self.softmax_temperature = softmax_temperature
    if hidden_dims is None:
      self.hidden_dims = [128, 64]

    # Pre-defined subgroups.
    # Here we assume `group` is among the input features.
    self.group = group
    self.group_labels = group_labels

    if task_type not in ["erm", "robust"]:
      raise AssertionError("task_type shall be one of ['erm' or 'robust'].")
    self.task_type = task_type

    # Parameters for robust optimization strategy.
    if task_type == "robust":
      self.dro_temperature = dro_temperature
      if group_reweight_strategy not in ["loss-dro", "metric-dro"]:
        raise AssertionError(
            "group_reweight_strategy shall be one of ['loss-dro', 'metric-dro']."
        )
      self.group_reweight_strategy = group_reweight_strategy
      self.regularized_dro = regularized_dro
      # TODO(xinyang,tyao,jiaxit): define it as coefficient of KL divergence.
      self.kl_regularizer = kl_regularizer
      # KL-regularization will result in a special type of group weight init.
      if regularized_dro:
        self.group_weight_init = group_weight_init**kl_regularizer
      else:
        self.group_weight_init = group_weight_init
      self.group_loss_init = group_loss_init
      self.group_metric_init = group_metric_init

      self.streaming_group_loss = streaming_group_loss
      self.streaming_group_loss_lr = streaming_group_loss_lr
      self.streaming_group_metric_lr = streaming_group_metric_lr
      self.metric_update_freq = metric_update_freq

  def __str__(self):
    return str(self.__dict__)
