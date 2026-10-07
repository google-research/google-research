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

"""
This module contains functions to do integer multiplication in a reversible way
for use in deep learning models.
"""

from collections.abc import Sequence
import fractions
import math

from absl import app
from absl import logging
import torch

def ceiling_remainder(a, b):
  """
  Computes the ceil(a/b) * b - a. This is equivalent to (b - (a % b)) % b.

  Args:
    a: The numerator.
    b: The denominator.

  Returns:
    The ceiling remainder of a / b.
  """
  return torch.remainder(b - torch.remainder(a, b), b)

def ceil_div(a, b):
  """
  Computes the ceil(a/b). This is equivalent to (a + b - 1) // b but without
  using floating point numbers.

  Args:
    a: The numerator.
    b: The denominator.

  Returns:
    The ceiling division of a / b.
  """
  return torch.div(a + b - 1, b, rounding_mode="floor")


class GPUReversibleTransform:
  """Exact Reversible Transform executed entirely on GPU with integer limb-packing.

  Supports 10,000+ steps with zero CPU-GPU transfer and zero BigInt overhead.
  """

  def __init__(
      self,
      num_params,
      gamma,
      max_steps = 10000,
      device = "cuda",
      mode = "floor",
  ):
    self.device = torch.device(device)
    self.num_params = num_params
    self.mode = mode
    assert self.mode in ["floor", "ceil"], "Mode must be 'floor' or 'ceil'."

    # Express gamma as exact fraction
    frac = fractions.Fraction(gamma).limit_denominator(1000)
    self.num = frac.numerator
    self.den = frac.denominator

    # Note the assumption that num <= den.
    assert (
        self.num <= self.den
    ), "Numerator must be less than or equal to denominator."

    # Calculate max growth per step and capacity per int64 limb
    # Maximum value active_buffer can hold before (buffer * den + remainder) overflows int64:
    # (buffer * den + den - 1) <= 2^63 - 1
    self.limb_max = (torch.iinfo(torch.int64).max - self.den) // self.den

    # Calculate total limbs required for max_steps
    # At each step, buffer grows as b_{t+1} = (b_t * den + r) // num <= (b_t + 1) * g where g = den / num.
    # Starting from b_0 = 0, after T steps: b_T <= sum_{k=1}^T g^k = g * (g^T - 1) / (g - 1).
    # Setting g * (g^T - 1) / (g - 1) = limb_max gives:
    # T = ln(1 + limb_max * (g - 1) / g) / ln(g)
    growth_per_step = self.den / self.num
    if growth_per_step <= 1.0:
      self.max_limbs = 1
      steps_per_limb = max_steps
    else:
      g = float(growth_per_step)
      t_limb = math.log(1.0 + (float(self.limb_max) * (g - 1.0) / g)) / math.log(g)
      steps_per_limb = max(1, int(t_limb))
      self.max_limbs = max(4, math.ceil(max_steps / steps_per_limb) + 4)
    self.steps_per_limb = steps_per_limb
    self.step_in_limb = 0
    logging.info(
        "max_limbs: %d, steps_per_limb: %d", self.max_limbs, steps_per_limb
    )

    # Pre-allocate limb tape on GPU: shape [max_limbs, num_params]
    self.tape = torch.zeros(
        (self.max_limbs, num_params), dtype=torch.int64, device=self.device
    )
    self.active_buffer = torch.zeros(
        num_params, dtype=torch.int64, device=self.device
    )
    self.limb_idx = 0

  def forward_decay(self, c_vec):
    """Executes forward decay on GPU: c_vec -> c_vec * gamma (exact).This is
    based on the following algorithm:

    Floor:
    buffer <- buffer * den + (c % den)
    c <- (c // den) * num + (buffer % num)
    buffer <- buffer // num

    Ceil:
    r <- ceil_rem(c, den)
    buffer <- buffer * den + r
    c <- (ceil_div(c, den) * num) - (buffer % num)
    buffer <- buffer // num
    """

    # 1. Update buffer with lower remainder
    if self.mode == "ceil":
      rem_den = ceiling_remainder(c_vec, self.den)
    else:
      rem_den = torch.remainder(c_vec, self.den)
    self.active_buffer = (self.active_buffer * self.den) + rem_den
    self.step_in_limb += 1

    # 2. Check if active buffer needs to spill to a new limb on the tape
    # Spill limb based on deterministic step counter (eliminates GPU-CPU sync)
    if self.step_in_limb >= self.steps_per_limb:
      self.step_in_limb = 0
      if self.limb_idx >= self.tape.shape[0]:
        additional_limbs = max(4, self.tape.shape[0] // 2)
        new_tape = torch.zeros(
            (self.tape.shape[0] + additional_limbs, self.num_params),
            dtype=torch.int64,
            device=self.device,
        )
        new_tape[: self.tape.shape[0]] = self.tape
        self.tape = new_tape
        self.max_limbs = self.tape.shape[0]
      self.tape[self.limb_idx] = self.active_buffer
      self.limb_idx += 1
      self.active_buffer = torch.zeros(
          self.num_params, dtype=torch.int64, device=self.device
      )

    # 3. Compute decayed c_vec and pop remainder from buffer
    rem_num = torch.remainder(self.active_buffer, self.num)

    if self.mode == "ceil":
      # This is ceiling division but without using floating point numbers.
      c_vec_decayed = (ceil_div(c_vec, self.den) * self.num) - rem_num
    else:
      c_vec_decayed = (torch.div(c_vec, self.den, rounding_mode="floor") * self.num) + rem_num
    self.active_buffer = torch.div(self.active_buffer, self.num, rounding_mode="floor")

    return c_vec_decayed

  def backward_decay(self, c_vec):
    """Executes exact inverse decay on GPU: c_vec_decayed -> c_vec.
    Floor:
    buffer <- buffer * num + (c % num)
    c <-  (c // num) * den + (buffer % den)
    buffer <- buffer // den
    Ceil:
    r <- ceil_rem(c, num)
    buffer <- buffer * num + r
    c <- (ceil_div(c, num) * den) - (buffer % den)
    buffer <- buffer // den
    """
    # 1. Restore buffer using numerator remainder
    if self.mode == "ceil":
      rem_num = ceiling_remainder(c_vec, self.num)
    else:
      rem_num = torch.remainder(c_vec, self.num)
    self.active_buffer = (self.active_buffer * self.num) + rem_num

    # 2. Check if active buffer underflowed and needs to pop from the tape
    if self.step_in_limb == 0 and self.limb_idx > 0:
      self.limb_idx -= 1
      self.active_buffer = self.tape[self.limb_idx].clone()
      self.tape[self.limb_idx].zero_()
      self.step_in_limb = self.steps_per_limb
    self.step_in_limb -= 1

    # 3. Restore original c_vec and pop denominator remainder from buffer
    rem_den = torch.remainder(self.active_buffer, self.den)
    if self.mode == "ceil":
      c_vec_original = (ceil_div(c_vec, self.num) * self.den) - rem_den
    else:
      c_vec_original = (torch.div(c_vec, self.num, rounding_mode="floor") * self.den) + rem_den
    self.active_buffer = torch.div(self.active_buffer, self.den, rounding_mode="floor")

    return c_vec_original

  def reset(self):
    """Resets the transform state."""
    self.tape.zero_()
    self.active_buffer.zero_()
    self.limb_idx = 0
    self.step_in_limb = 0



def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")


if __name__ == "__main__":
  app.run(main)
