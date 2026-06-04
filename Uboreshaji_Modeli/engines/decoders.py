# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Prediction decoders for composed model engines."""

from collections.abc import Sequence
import re
from typing import Any
import torch

from Uboreshaji_Modeli.engines import base

_BBOX_LOC_PATTERN = re.compile(
    r"""
    <loc(\d{4})>       # y1 coordinate (grid scale 0-1023)
    <loc(\d{4})>       # x1 coordinate (grid scale 0-1023)
    <loc(\d{4})>       # y2 coordinate (grid scale 0-1023)
    <loc(\d{4})>       # x2 coordinate (grid scale 0-1023)
    \s*                # optional separating whitespace
    ([^\s;]+)          # non-whitespace/non-semicolon label
    """,
    re.VERBOSE,
)

_POLY_PATTERN = re.compile(
    r"""
    polygon            # match literal "polygon" keyword
    \s+                # whitespace prefix
    (                  # capture group 1: sequence of point coordinates
      (?:
        <loc\d{4}>     # y coordinate
        <loc\d{4}>     # x coordinate
        \s*            # optional coordinate spacing
      )+
    )
    """,
    re.VERBOSE,
)

_SINGLE_LOC_PATTERN = re.compile(
    r"""
    <loc(\d{4})>       # y coordinate (grid scale 0-1023)
    <loc(\d{4})>       # x coordinate (grid scale 0-1023)
    """,
    re.VERBOSE,
)


class BoundingBoxDecoder(base.PredictionDecoder):
  """A decoder for location tokens to standard normalized float bounding boxes.

  Location tokens are in format: <locYYYY><locXXXX><locYYYY><locXXXX> label.
  Quantized to a 1024-cell grid.
  """

  def decode(
      self, processor, outputs, **kwargs
  ):
    """Decodes location tokens from the model outputs to bounding boxes.

    Args:
      processor: The processor used for the model (e.g., for tokenization).
      outputs: The raw outputs from the model, expected to be a string or a list
        of strings containing location tokens.
      **kwargs: Additional keyword arguments.

    Returns:
      A list of dictionaries, where each dictionary corresponds to an item in
      the outputs and contains:
        - "boxes": A torch.tensor of shape (N, 4) with bounding boxes in
          [x1, y1, x2, y2] format, normalized to [0.0, 1.0].
        - "labels": A list of strings, one for each bounding box.
    """
    del self  # Self is not used in this method.
    results = []

    # If outputs is a raw string, wrap it in a list.
    texts = [outputs] if isinstance(outputs, str) else outputs

    for text in texts:
      boxes = []
      labels = []
      for match in _BBOX_LOC_PATTERN.finditer(text):
        y1, x1, y2, x2, label = match.groups()
        # Dequantize from a 1024-cell grid (0 to 1023).
        box = [
            float(x1) / 1023.0,
            float(y1) / 1023.0,
            float(x2) / 1023.0,
            float(y2) / 1023.0,
        ]
        boxes.append(box)
        labels.append(label)

      results.append({
          "boxes": (
              torch.tensor(boxes, dtype=torch.float32)
              if boxes
              else torch.zeros((0, 4))
          ),
          "labels": labels,
      })
    return results


class PolygonSegmentationDecoder(base.PredictionDecoder):
  """A decoder to map coordinate tokens to instance segmentation polygons.

  Polygons are in format: polygon <locY1><locX1> <locY2><locX2> ...
  """

  def decode(
      self, processor, outputs, **kwargs
  ):
    """Decodes location/pixel coordinate tokens to segmentation polygon points.

    Args:
      processor: The processor used for the model (e.g., tokenizer/processor).
      outputs: The raw outputs from the model, expected to be a string or list
        of strings containing polygon location tokens.
      **kwargs: Additional keyword arguments.

    Returns:
      A list of dictionaries, where each dictionary contains:
        - "polygons": A list of polygons, where each polygon is a list of [x, y]
          coordinate pairs normalized to [0.0, 1.0].
    """
    del self  # Self is not used in this method.
    results = []

    texts = [outputs] if isinstance(outputs, str) else outputs

    for text in texts:
      polygons = []
      for match in _POLY_PATTERN.finditer(text):
        poly_str = match.group(1)
        points = []
        for pt_match in _SINGLE_LOC_PATTERN.finditer(poly_str):
          y, x = pt_match.groups()
          points.append([float(x) / 1023.0, float(y) / 1023.0])
        polygons.append(points)

      results.append({
          "polygons": polygons,
      })
    return results


class TextDecoder(base.PredictionDecoder):
  """A decoder mapping token IDs to plain strings via batch_decode."""

  def decode(self, processor, outputs, **kwargs):
    """Decodes token IDs from model outputs to a list of strings.

    Args:
      processor: The processor (tokenizer) used for decoding.
      outputs: The raw outputs from the model. Expected to contain token IDs.
        This can be a tensor or a list of tensors.
      **kwargs: Additional keyword arguments.

    Returns:
      A list of decoded strings.
    """
    # Outputs can be a list of tensors or a single tensor.
    # `batch_decode` expects a 2D tensor or a list of token sequences.
    # If a single tensor is 1D:
    if hasattr(outputs, "dim") and outputs.dim() == 1:
      outputs = [outputs]
    return processor.batch_decode(outputs, skip_special_tokens=True)


class MmsDecoder(base.PredictionDecoder):
  """A decoder for MMS Wav2Vec2-CTC logits using argmax decoding."""

  def decode(self, processor, outputs, **kwargs):
    import numpy as np  # pylint: disable=g-import-not-at-top
    logits = outputs.logits
    if isinstance(logits, torch.Tensor):
      logits = logits.detach().cpu().numpy()
    predicted_ids = np.argmax(logits, axis=-1)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)


