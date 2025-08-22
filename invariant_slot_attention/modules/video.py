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

"""Video module library."""

import functools
from typing import Any, Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple, Union

from flax import linen as nn
import jax.numpy as jnp
from invariant_slot_attention.lib import utils
from invariant_slot_attention.modules import misc

Shape = Tuple[int]

DType = Any
Array = Any  # jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class CorrectorPredictorTuple(NamedTuple):
  corrected: ProcessorState
  predicted: ProcessorState


class Processor(nn.Module):
  """Recurrent processor module.

  This module is scanned (applied recurrently) over the sequence dimension of
  the input and applies a corrector and a predictor module. The corrector is
  only applied if new inputs (such as a new image/frame) are received and uses
  the new input to correct its internal state.

  The predictor is equivalent to a latent transition model and produces a
  prediction for the state at the next time step, given the current (corrected)
  state.
  """
  corrector: Callable[[ProcessorState, Array], ProcessorState]
  predictor: Callable[[ProcessorState], ProcessorState]

  @functools.partial(
      nn.scan,  # Scan (recurrently apply) over time axis.
      in_axes=(1, 1, nn.broadcast),  # (inputs, padding_mask, train).
      out_axes=1,
      variable_axes={"intermediates": 1},  # Stack intermediates along seq. dim.
      variable_broadcast="params",
      split_rngs={"params": False, "dropout": True})
  @nn.compact
  def __call__(self, state, inputs,
               padding_mask,
               train):

    # Only apply corrector if we receive new inputs.
    if inputs is not None:
      corrected_state = self.corrector(state, inputs, padding_mask, train=train)
    # Otherwise simply use previous state as input for predictor.
    else:
      corrected_state = state

    # Always apply predictor (i.e. transition model).
    predicted_state = self.predictor(corrected_state, train=train)

    # Prepare outputs in a format compatible with nn.scan.
    new_state = predicted_state
    outputs = CorrectorPredictorTuple(
        corrected=corrected_state, predicted=predicted_state)
    return new_state, outputs


class SAVi(nn.Module):
  """Video model consisting of encoder, recurrent processor, and decoder."""

  encoder: Callable[[], nn.Module]
  decoder: Callable[[], nn.Module]
  corrector: Callable[[], nn.Module]
  predictor: Callable[[], nn.Module]
  initializer: Callable[[], nn.Module]
  decode_corrected: bool = True
  decode_predicted: bool = True

  @nn.compact
  def __call__(self, video, conditioning = None,
               continue_from_previous_state = False,
               padding_mask = None,
               train = False):
    """Performs a forward pass on a video.

    Args:
      video: Video of shape `[batch_size, n_frames, height, width, n_channels]`.
      conditioning: Optional jnp.ndarray used for conditioning the initial state
        of the recurrent processor.
      continue_from_previous_state: Boolean, whether to continue from a previous
        state or not. If True, the conditioning variable is used directly as
        initial state.
      padding_mask: Binary mask for padding video inputs (e.g. for videos of
        different sizes/lengths). Zero corresponds to padding.
      train: Indicating whether we're training or evaluating.

    Returns:
      A dictionary of model predictions.
    """
    processor = Processor(
        corrector=self.corrector(), predictor=self.predictor())  # pytype: disable=wrong-arg-types

    if padding_mask is None:
      padding_mask = jnp.ones(video.shape[:-1], jnp.int32)

    # video.shape = (batch_size, n_frames, height, width, n_channels)
    # Vmapped over sequence dim.
    encoded_inputs = self.encoder()(video, padding_mask, train)  # pytype: disable=not-callable
    if continue_from_previous_state:
      assert conditioning is not None, (
          "When continuing from a previous state, the state has to be passed "
          "via the `conditioning` variable, which cannot be `None`.")
      init_state = conditioning[:, -1]  # We currently only use last state.
    else:
      # Same as above but without encoded inputs.
      init_state = self.initializer()(
          conditioning, batch_size=video.shape[0], train=train)  # pytype: disable=not-callable

    # Scan recurrent processor over encoded inputs along sequence dimension.
    _, states = processor(init_state, encoded_inputs, padding_mask, train)
    # type(states) = CorrectorPredictorTuple.
    # states.corrected.shape = (batch_size, n_frames, ..., n_features).
    # states.predicted.shape = (batch_size, n_frames, ..., n_features).

    # Decode latent states.
    decoder = self.decoder()  # Vmapped over sequence dim.
    outputs = decoder(states.corrected,
                      train) if self.decode_corrected else None  # pytype: disable=not-callable
    outputs_pred = decoder(states.predicted,
                           train) if self.decode_predicted else None  # pytype: disable=not-callable

    return {
        "states": states.corrected,
        "states_pred": states.predicted,
        "outputs": outputs,
        "outputs_pred": outputs_pred,
    }


class FrameEncoder(nn.Module):
  """Encoder for single video frame, vmapped over time axis."""

  backbone: Callable[[], nn.Module]
  pos_emb: Callable[[], nn.Module] = misc.Identity
  reduction: Optional[str] = None
  output_transform: Callable[[], nn.Module] = misc.Identity

  # Vmapped application of module, consumes time axis (axis=1).
  @functools.partial(utils.time_distributed, in_axes=(1, 1, None))
  @nn.compact
  def __call__(self, inputs, padding_mask = None,
               train = False):
    del padding_mask  # Unused.

    # inputs.shape = (batch_size, height, width, n_channels)
    x = self.backbone()(inputs, train=train)

    x = self.pos_emb()(x)

    if self.reduction == "spatial_flatten":
      batch_size, height, width, n_features = x.shape
      x = jnp.reshape(x, (batch_size, height * width, n_features))
    elif self.reduction == "spatial_average":
      x = jnp.mean(x, axis=(1, 2))
    elif self.reduction == "all_flatten":
      batch_size, height, width, n_features = x.shape
      x = jnp.reshape(x, (batch_size, height * width * n_features))
    elif self.reduction is not None:
      raise ValueError("Unknown reduction type: {}.".format(self.reduction))

    output_block = self.output_transform()

    if hasattr(output_block, "qkv_size"):
      # Project to qkv_size if used transformer.
      x = nn.relu(nn.Dense(output_block.qkv_size)(x))

    x = output_block(x, train=train)
    return x
