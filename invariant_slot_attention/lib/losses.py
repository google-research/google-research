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

"""Loss functions."""

import functools
import inspect
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import ml_collections

_LOSS_FUNCTIONS = {}

Array = Any  # jnp.ndarray somehow doesn't work anymore for pytype.
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ArrayDict = Dict[str, Array]
DictTree = Dict[str, Union[Array, "DictTree"]]  # pytype: disable=not-supported-yet
PRNGKey = Array
LossFn = Callable[[Dict[str, ArrayTree], Dict[str, ArrayTree]],
                  Tuple[Array, ArrayTree]]
ConfigAttr = Any
MetricSpec = Dict[str, str]


def standardize_loss_config(
    loss_config
):
  """Standardize loss configs into a common ConfigDict format.

  Args:
    loss_config: List of strings or ConfigDict specifying loss configuration.
      Valid input formats are: - Option 1 (list of strings), for example,
        `loss_config = ["box", "presence"]` - Option 2 (losses with weights
        only), for example,
            `loss_config = ConfigDict({"box": 5, "presence": 2})` - Option 3
              (losses with weights and other parameters), for example,
            `loss_config = ConfigDict({"box": {"weight": 5, "metric": "l1"},
                                  "presence": {"weight": 2}})`

  Returns:
    Standardized ConfigDict containing the loss configuration.

  Raises:
    ValueError: If loss_config is a list that contains non-string entries.
  """

  if isinstance(loss_config, Sequence):  # Option 1
    if not all(isinstance(loss_type, str) for loss_type in loss_config):
      raise ValueError(f"Loss types all need to be str but got {loss_config}")
    return ml_collections.FrozenConfigDict({k: {} for k in loss_config})

  # Convert all option-2-style weights to option-3-style dictionaries.
  loss_config = {
      k: {
          "weight": v
      } if isinstance(v, (float, int)) else v for k, v in loss_config.items()
  }
  return ml_collections.FrozenConfigDict(loss_config)


def update_loss_aux(loss_aux, update):
  existing_keys = set(update.keys()).intersection(loss_aux.keys())
  if existing_keys:
    raise KeyError(
        f"Can't overwrite existing keys in loss_aux: {existing_keys}")
  loss_aux.update(update)


def compute_full_loss(
    preds, targets,
    loss_config
):
  """Loss function that parses and combines weighted loss terms.

  Args:
    preds: Dictionary of tensors containing model predictions.
    targets: Dictionary of tensors containing prediction targets.
    loss_config: List of strings or ConfigDict specifying loss configuration.
      See @register_loss decorated functions below for valid loss names.
      Valid losses formats are: - Option 1 (list of strings), for example,
        `loss_config = ["box", "presence"]` - Option 2 (losses with weights
        only), for example,
        `loss_config = ConfigDict({"box": 5, "presence": 2})` - Option 3 (losses
          with weights and other parameters), for example,
        `loss_config = ConfigDict({"box": {"weight": 5, "metric": "l1"},
                                   "presence": {"weight": 2}})` - Option 4 (like
                                     3 but decoupling name and loss_type), for
                                     example,
        `loss_config = ConfigDict({"recon_flow": {"loss_type": "recon",
                                                  "key": "flow"},
                                   "recon_video": {"loss_type": "recon",
                                                   "key": "video"}})`

  Returns:
    A 2-tuple of the sum of all individual loss terms and a dictionary of
    auxiliary losses and metrics.
  """

  loss = jnp.zeros([], jnp.float32)
  loss_aux = {}
  loss_config = standardize_loss_config(loss_config)
  for loss_name, cfg in loss_config.items():
    context_kwargs = {"preds": preds, "targets": targets}
    weight, loss_term, loss_aux_update = compute_loss_term(
        loss_name=loss_name, context_kwargs=context_kwargs, config_kwargs=cfg)

    unweighted_loss = jnp.mean(loss_term)
    loss += weight * unweighted_loss
    loss_aux_update[loss_name + "_value"] = unweighted_loss
    loss_aux_update[loss_name + "_weight"] = jnp.ones_like(unweighted_loss)
    update_loss_aux(loss_aux, loss_aux_update)
  return loss, loss_aux


def register_loss(func=None,
                  *,
                  name = None,
                  check_unused_kwargs = True):
  """Decorator for registering a loss function.

  Can be used without arguments:
  ```
  @register_loss
  def my_loss(**_):
    return 0
  ```
  or with keyword arguments:
  ```
  @register_loss(name="my_renamed_loss")
  def my_loss(**_):
    return 0
  ```

  Loss functions may accept
    - context kwargs: `preds` and `targets`
    - config kwargs: any argument specified in the config
    - the special `config_kwargs` parameter that contains the entire loss config
  Loss functions also _need_ to accept a **kwarg argument to support extending
  the interface.
  They should return either:
    - just the computed loss (pre-reduction)
    - or a tuple of the computed loss and a loss_aux_updates dict

  Args:
    func: the decorated function
    name (str): Optional name to be used for this loss in the config. Defaults
      to the name of the function.
    check_unused_kwargs (bool): By default compute_loss_term raises an error if
      there are any unused config kwargs. If this flag is set to False that step
      is skipped. This is useful if the config_kwargs should be passed onward to
      another function.

  Returns:
    The decorated function (or a partial of the decorator)
  """
  # If this decorator has been called with parameters but no function, then we
  # return the decorator again (but with partially filled parameters).
  # This allows using both @register_loss and @register_loss(name="foo")
  if func is None:
    return functools.partial(
        register_loss, name=name, check_unused_kwargs=check_unused_kwargs)

  # No (further) arguments: this is the actual decorator
  # ensure that the loss function includes a **kwargs argument
  loss_name = name if name is not None else func.__name__
  if not any(v.kind == inspect.Parameter.VAR_KEYWORD
             for k, v in inspect.signature(func).parameters.items()):
    raise TypeError(
        f"Loss function '{loss_name}' needs to include a **kwargs argument")
  func.name = loss_name
  func.check_unused_kwargs = check_unused_kwargs
  _LOSS_FUNCTIONS[loss_name] = func
  return func


def compute_loss_term(
    loss_name, context_kwargs,
    config_kwargs):
  """Compute a loss function given its config and context parameters.

  Takes care of:
    - finding the correct loss function based on "loss_type" or name
    - the optional "weight" parameter
    - checking for typos and collisions in config parameters
    - adding the optional loss_aux_updates if omitted by the loss_fn

  Args:
    loss_name: Name of the loss, i.e. its key in the config.losses dict.
    context_kwargs: Dictionary of context variables (`preds` and `targets`)
    config_kwargs: The config dict for this loss.

  Returns:
      1. the loss weight (float)
      2. loss term (Array)
      3. loss aux updates (Dict[str, Array])

  Raises:
    KeyError:
        Unknown loss_type
    KeyError:
        Unused config entries, i.e. not used by the loss function.
        Not raised if using @register_loss(check_unused_kwargs=False)
    KeyError: Config entry with a name that conflicts with a context_kwarg
    ValueError: Non-numerical weight in config_kwargs

  """

  # Make a dict copy of config_kwargs
  kwargs = {k: v for k, v in config_kwargs.items()}

  # Get the loss function
  loss_type = kwargs.pop("loss_type", loss_name)
  if loss_type not in _LOSS_FUNCTIONS:
    raise KeyError(f"Unknown loss_type '{loss_type}'.")
  loss_fn = _LOSS_FUNCTIONS[loss_type]

  # Take care of "weight" term
  weight = kwargs.pop("weight", 1.0)
  if not isinstance(weight, (int, float)):
    raise ValueError(f"Weight for loss {loss_name} should be a number, "
                     f"but was {weight}.")

  # Check for unused config entries (to prevent typos etc.)
  config_keys = set(kwargs)
  if loss_fn.check_unused_kwargs:
    param_names = set(inspect.signature(loss_fn).parameters)
    unused_config_keys = config_keys - param_names
    if unused_config_keys:
      raise KeyError(f"Unrecognized config entries {unused_config_keys} "
                     f"for loss {loss_name}.")

  # Check for key collisions between context and config
  conflicting_config_keys = config_keys.intersection(context_kwargs)
  if conflicting_config_keys:
    raise KeyError(f"The config keys {conflicting_config_keys} conflict "
                   f"with the context parameters ({context_kwargs.keys()}) "
                   f"for loss {loss_name}.")

  # Construct the arguments for the loss function
  kwargs.update(context_kwargs)
  kwargs["config_kwargs"] = config_kwargs

  # Call loss
  results = loss_fn(**kwargs)

  # Add empty loss_aux_updates if necessary
  if isinstance(results, Tuple):
    loss, loss_aux_update = results
  else:
    loss, loss_aux_update = results, {}

  return weight, loss, loss_aux_update


# -------- Loss functions --------
@register_loss
def recon(preds,
          targets,
          key = "video",
          reduction_type = "sum",
          **_):
  """Reconstruction loss (MSE)."""
  squared_l2_norm_fn = jax.vmap(functools.partial(
      squared_l2_norm, reduction_type=reduction_type))
  targets = targets[key]
  loss = squared_l2_norm_fn(preds["outputs"][key], targets)
  if reduction_type == "mean":
    # This rescaling reflects taking the sum over feature axis &
    # mean over space/time axes.
    loss *= targets.shape[-1]  # pytype: disable=attribute-error  # allow-recursive-types
  return jnp.mean(loss)  # pytype: disable=bad-return-type  # jnp-type


def squared_l2_norm(preds, targets,
                    reduction_type = "sum"):
  if reduction_type == "sum":
    return jnp.sum(jnp.square(preds - targets))
  elif reduction_type == "mean":
    return jnp.mean(jnp.square(preds - targets))
  else:
    raise ValueError(f"Unsupported reduction_type: {reduction_type}")
