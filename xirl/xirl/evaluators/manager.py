# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""An evaluator manager."""

from absl import logging
from .base import EvaluatorOutput
import torch
# pylint: disable=logging-fstring-interpolation


class EvalManager:
  """Manage a bunch of downstream task evaluators and aggregate their results.

  Specifically, the manager embeds the downstream dataset *once*, and shares
  the embeddings across all evaluators for more efficient evaluation.
  """

  def __init__(self, evaluators):
    """Constructor.

    Args:
      evaluators: A mapping from evaluator name to Evaluator instance.
    """
    self._evaluators = evaluators

  @staticmethod
  @torch.no_grad()
  def embed(
      model,
      downstream_loader,
      device,
      eval_iters,
  ):
    """Run the model on the downstream data and generate embeddings."""
    loader_to_output = {}
    for action_name, valid_loader in downstream_loader.items():
      outs = []
      for batch_idx, batch in enumerate(valid_loader):
        if eval_iters is not None and batch_idx >= eval_iters:
          break
        outs.append(model.infer(batch["frames"].to(device)).numpy())
      loader_to_output[action_name] = outs
    return loader_to_output

  @torch.no_grad()
  def evaluate(
      self,
      model,
      downstream_loader,
      device,
      eval_iters=None,
  ):
    """Evaluate the model on the validation data.

    Args:
      model: The self-supervised model that will embed the frames in the
        downstream loader.
      downstream_loader: A downstream dataloader. Has a batch size of 1 and
        loads all frames of the video.
      device: The compute device.
      eval_iters: The number of time to call `next()` on the downstream
        iterator. Set to None to evaluate on the entire iterator.

    Returns:
      A dict mapping from evaluator name to EvaluatorOutput.
    """
    model.eval()
    logging.debug("Embedding downstream dataset...")
    downstream_outputs = EvalManager.embed(model, downstream_loader, device,
                                           eval_iters)
    eval_to_metric = {}
    for evaluator_name, evaluator in self._evaluators.items():
      logging.debug("\tRunning %s evaluator...", evaluator_name)
      if evaluator.inter_class:
        # Merge all downstream classes into a single list and do one
        # eval computation.
        outs = [
            o for out in downstream_outputs.values() for o in out  # pylint: disable=g-complex-comprehension
        ]
        metric = evaluator.evaluate(outs)
      else:
        # Loop and evaluate over downstream classes separately, then
        # merge into a single EvaluatorOutput whose fields are lists.
        metrics = []
        for outs in downstream_outputs.values():
          metrics.append(evaluator.evaluate(outs))
        metric = EvaluatorOutput.merge(metrics)
      eval_to_metric[evaluator_name] = metric
    return eval_to_metric
