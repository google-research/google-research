# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Tools to generate ASTs by iteratively refining a partial program.

This module defines an interface for weighted AST templates, and functions to
build a random AST by combining those templates until reaching a maximum size.
The set of templates defines a probabilistic grammar, where each nonterminal
includes a small amount of local context (such as variable names that are
currently in scope); we repeatedly choose nonterminals to expand subject to
the constraint that we do not exceed the maximum size.

Note that the interface is not necessarily specific to Python ASTs, but can
generate any sort of grammar-structured object.
"""

import abc
from typing import Any, Callable, Collection, Dict, List, Optional

import dataclasses
import numpy as np

# Allow each task to define its own hole type.
# (pytype has issues with generic types in attributes, so we can't use Python
# generics to be more specific here)
HoleType = Any


@dataclasses.dataclass
class Hole:
  """A hole that needs to be filled by a template.

  Attributes:
    hole_type: The type of value that can fill this hole.
    metadata: Task-specific context (such as variables in scope).
  """
  hole_type: HoleType
  metadata: Any


@dataclasses.dataclass
class ThingWithHoles:
  """Some object that can be constructed by filling each of its holes.

  Attributes:
    cost: A measurement of the size of this object, not counting its holes.
    holes: A list of the holes in this object.
    build: A callable that takes in values for each hole (each as a separate
      argument) and returns the full object.
  """
  cost: int
  holes: List[Hole]
  build: Callable[Ellipsis, Any]


class HoleFillerTemplate(abc.ABC):
  """Abstract base class for a template that fills a hole."""

  @property
  @abc.abstractmethod
  def fills_type(self):
    """The type of hole that this template fills."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def required_cost(self):
    """How much space this template requires.

    For the iterative refinement to work correctly, the required cost should
    be (greater than or) equal to the cost of the partial object generated,
    plus the cost of filling all of its holes with their lowest-cost templates.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def fill(self, hole, rng):
    """Construct a ThingWithHoles that can be substituted into the given hole.

    This method will only be called when hole.hole_type == self.fills_type and
    also self.can_fill(hole) returns True.

    Args:
      hole: The hole to fill.
      rng: Random state to use to fill the hole. Should uniquely determine what
        is chosen.

    Returns:
      An object to substitute for that hole.
    """
    raise NotImplementedError()

  def can_fill(self, hole):
    """Returns True if this template can fill this hole, False otherwise.

    This can be used to filter out templates that only apply in the presence of
    specific hole metadata.

    This method does not need to check that hole.hole_type == self.fills_type;
    that will be checked separately.

    Args:
      hole: Hole to fill.

    Returns:
      Whether this template can fill this hole.
    """
    del hole  # By default we assume the template can fill any hole.
    return True

  def __repr__(self):
    """A default repr for templates."""
    return self.__class__.__qualname__ + "(" + ", ".join(
        f"{k}={repr(v)}" for k, v in self.__dict__.items()) + ")"


@dataclasses.dataclass
class WeightedTemplate:
  """A template with an associated (unnormalized) probability and precedence.

  Attributes:
    template: The template to use to fill this hole.
    weight: The unnormalized probability weight of this template. When sampling
      a template, we normalize the weights across the set of applicable
      templates with the highest precedence.
    precedence: When sampling a template to fill a specific hole, only the set
      of templates with the highest precedence will be considered.
  """
  template: HoleFillerTemplate
  weight: float
  precedence: int = 1


@dataclasses.dataclass
class RefinementDistribution:
  """A distribution over top-down refinement steps.

  First, a hole is selected, where each hole is weighted according to
  `hole_selection_weights`. Then, a template that can fill that hole is selected
  from `weighted_templates`.

  Attributes:
    hole_selection_weights: Weights for each type of hole.
    weighted_templates: Templates to choose from, with weights.
  """
  hole_selection_weights: Dict[HoleType, int]
  weighted_templates: Collection[WeightedTemplate]


def substitute_for_hole(original, hole_index,
                        filler):
  """Substitute `filler` into the `hole_index`th hole of `original`.

  This function builds a new object by inserting the filler object into
  the given hole of the original object. The holes of the new object are then
  the holes of the original object, without the `hole_index` hole, union the
  holes of the filler object.

  Args:
    original: Top-level thing to combine.
    hole_index: Index of one of original's holes to fill.
    filler: Object to substitute for that hole.

  Returns:
    New partial object by substituting the filler into the original.
  """

  def build(*all_hole_values):
    """Builder function for the new thing."""
    # Split the values into those for the filler and those for the original.
    filler_hole_values = all_hole_values[:len(filler.holes)]
    other_hole_values = list(all_hole_values[len(filler.holes):])
    # Build the filler object.
    filler_value = filler.build(*filler_hole_values)
    # Splice it in
    other_hole_values.insert(hole_index, filler_value)
    return original.build(*other_hole_values)

  return ThingWithHoles(
      holes=(filler.holes + original.holes[:hole_index] +
             original.holes[hole_index + 1:]),
      cost=original.cost + filler.cost,
      build=build)


def top_down_construct(root_object,
                       target_cost,
                       refinement_distribution,
                       rng = None):
  """Fills all holes in `root_object`, using no more than `target_cost`.

  Until there are no more holes, repeatedly:
  - chooses a hole at random, weighting each hole proportional to its weight
    in `hole_selection_weights`,
  - finds all templates in `weighted_templates` that can fill this hole,
  - filters down to those with highest precedence,
  - chooses a template to use proportional to their weights, and
  - substitutes that template's partial solution into the current object.

  Args:
    root_object: Object specifying what you want to build.
    target_cost: Upper bound on the cost; generation will try to get as close as
      possible to this without going over.
    refinement_distribution: Distribution that guides example generation.
    rng: Random state to use to generate the example; this determines the
      example that is generated.

  Returns:
    A fully-instantiated version of `root_object` with all holes filled.

  Raises:
    ValueError: If the templates or target cost make it impossible to fill the
      given holes.
  """
  if rng is None:
    rng = np.random.RandomState()

  weighted_templates = refinement_distribution.weighted_templates
  hole_selection_weights = refinement_distribution.hole_selection_weights

  hole_required_costs = {}
  for wt in weighted_templates:
    ft = wt.template.fills_type
    if ft in hole_required_costs:
      hole_required_costs[ft] = min(hole_required_costs[ft],
                                    wt.template.required_cost)
    else:
      hole_required_costs[ft] = wt.template.required_cost

  while root_object.holes:
    # Choose a random hole
    hole_weights = np.array(
        [hole_selection_weights[h.hole_type] for h in root_object.holes])
    hole_idx = rng.choice(
        range(len(root_object.holes)), p=hole_weights / np.sum(hole_weights))
    hole = root_object.holes[hole_idx]

    # Find all templates that can fill this hole
    cost_of_other_holes = sum(hole_required_costs[h.hole_type]
                              for i, h in enumerate(root_object.holes)
                              if i != hole_idx)
    space = target_cost - root_object.cost - cost_of_other_holes
    possible_templates = []
    for wt in weighted_templates:
      if (wt.template.fills_type == hole.hole_type and
          wt.template.required_cost <= space and wt.template.can_fill(hole)):
        possible_templates.append(wt)

    if not possible_templates:
      raise ValueError(
          f"Couldn't find template to fill {hole} using space {space}")

    # Filter to the highest-precedence ones
    max_prec = max(wt.precedence for wt in possible_templates)
    max_prec_templates = [
        wt for wt in possible_templates if wt.precedence == max_prec
    ]

    # Sample a template
    template_weights = np.array([wt.weight for wt in max_prec_templates])
    template_idx = rng.choice(
        range(len(template_weights)),
        p=template_weights / np.sum(template_weights))
    template = max_prec_templates[template_idx].template

    # Fill the hole
    filler = template.fill(hole, rng)
    root_object = substitute_for_hole(root_object, hole_idx, filler)

  return root_object.build()
