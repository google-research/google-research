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

"""Utils for evolution."""

import copy
import dataclasses
import functools
import math
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
import ml_collections as mlc

from abstract_nas.abstract.depth import DepthProperty
from abstract_nas.abstract.fingerprint import fingerprint_graph
from abstract_nas.abstract.linear import LinopProperty
from abstract_nas.abstract.shape import ShapeProperty
from abstract_nas.evolution.mutator.random import RandomMutator
from abstract_nas.evolution.mutator.random_sequential import RandomSequentialMutator
from abstract_nas.model import Model
from abstract_nas.model.block import Block
from abstract_nas.model.concrete import Graph
from abstract_nas.model.subgraph import replace_subgraph
from abstract_nas.model.subgraph import SubgraphSpec
from abstract_nas.synthesis.graph import GraphSynthesizer
from abstract_nas.synthesis.primer_sequential import PrimerSequentialSynthesizer
from abstract_nas.synthesis.prog_sequential import ProgressiveSequentialSynthesizer
from abstract_nas.synthesis.random_enum_sequential import RandomEnumerativeSequentialSynthesizer

Numeric = Union[float, int]
Tensor = Any
EPS = 1e-6


MUTATORS = {
    "random_sequential": RandomSequentialMutator,
    "random_subgraph": RandomMutator,
}


SYNTHESIZERS = {
    "random_enum": RandomEnumerativeSequentialSynthesizer,
    "progressive": ProgressiveSequentialSynthesizer,
    "primer": PrimerSequentialSynthesizer,
}


@dataclasses.dataclass
class Individual:
  """An individual produced during evolution.

  Attributes:
    completion_time_sec: when the individual finished evaluating.
    creation_time_sec: when the individual was created.
    accuracy: accuracy on the test set.
    im_sec_core_train: training latency in images per second per core.
    flops: number of gflops.
    num_params: millions of parameters.
  """
  completion_time_sec: int
  creation_time_sec: int
  accuracy: float
  im_sec_core_train: float
  flops: float
  num_params: float


class PopulationManager:
  """Maintains a population for evolution."""

  def __init__(self, config):
    self.dataset = config.train.dataset_name

    self.min_acc = config.evolution.population.min_acc

    self.min_to_select = config.evolution.population.min_to_select
    self.top_perc_to_select = config.evolution.population.top_perc_to_select
    self.top_n_to_select = config.evolution.population.top_n_to_select

    self.use_pareto_balanced = config.evolution.population.use_pareto_balanced
    self.use_pareto_normalized = config.evolution.population.use_pareto_normalized

    self.targeted_evol = config.evolution.population.targeted_evol

    self.warm_up_secs = config.evolution.population.aging.warm_up_secs
    self.generations_to_live = config.evolution.population.aging.generations_to_live
    self.cosine = config.evolution.population.aging.cosine
    self.cyclic = config.evolution.population.aging.cyclic
    self.cyclic_impulse = config.evolution.population.aging.cyclic_impulse

    if self.cyclic_impulse and self.cyclic:
      raise ValueError("cyclic_impulse and cyclic are mutually exclusive.")

    if not self.cosine and self.cyclic_impulse:
      logging.warn("cyclic_impulse has no affect when not using cosine aging. "
                   "Ignoring the cyclic_impulse option.")

  def age(self, acc, num_generations_alive):
    """Degrades accuracy by the aging strategy."""
    cosine = self.cosine
    cyclic = self.cyclic
    cyclic_impulse = self.cyclic_impulse
    generations_to_live = self.generations_to_live
    if generations_to_live <= 0:
      return acc
    if cosine:
      if cyclic or cyclic_impulse:
        num_generations_alive %= (2 * generations_to_live)
        if num_generations_alive < generations_to_live:
          progress = 0
        else:
          progress_gen = num_generations_alive - generations_to_live
          progress = progress_gen / generations_to_live
          if cyclic:
            progress *= 2
      else:
        progress = num_generations_alive / generations_to_live
        progress -= 1.0
        progress = max(0.0, min(progress, 1.0))
      aged_acc = acc * 0.5 * (1. + math.cos(math.pi * progress))
    else:
      aged_acc = acc
      if cyclic:
        if (num_generations_alive // generations_to_live) % 2:
          aged_acc = 0
      else:
        if num_generations_alive > generations_to_live:
          aged_acc = 0
    acc = .1 * acc + .9 * aged_acc
    return acc

  def filter_fitnesses(
      self,
      base_fitness,
      fitnesses,
      second_high,
      keep_individuals = None,
  ):
    """Filters for fitnesses that are pareto dominant over the base fitness.

    The fitnesses are tuples of (accuracy, secondary objective)

    Args:
      base_fitness: the baseline fitnesses used for filtering.
      fitnesses: the fitnesses to be filtered.
      second_high: whether higher is better for the secondary objective.
      keep_individuals: indices of individuals in idividuals to keep without
        filtering.

    Returns:
      A list of filtered fitnesses, their corresponding indices in the input
        fitnesses, and the anchor used as the filter.
    """
    all_fitnesses = fitnesses
    fitnesses = []
    idxs = []

    if not keep_individuals: keep_individuals = []

    # First, take all the pareto-dominant points.
    for idx, (acc, second) in enumerate(all_fitnesses):
      if idx in keep_individuals:
        fitnesses.append((acc, second))
        idxs.append(idx)
        continue
      if acc + EPS < base_fitness[0]:
        continue
      if second_high and second + EPS < base_fitness[1]:
        continue
      if not second_high and second > base_fitness[1] + EPS:
        continue
      fitnesses.append((acc, second))
      idxs.append(idx)

    if not idxs:
      return [], [], None

    # Set the anchor to a point on the pareto-dominant portion
    # of the curve.
    if self.use_pareto_balanced:
      curve = ParetoCurve(fitnesses, first_high=True,
                          second_high=second_high)
      anchor = curve.sample_point()
    else:
      anchor = None

    # Add other points so that there is at least an equal number of
    # pareto dominant vs. non-dominant points.
    if len(all_fitnesses) > len(idxs):
      select_p = len(idxs) / (
          len(all_fitnesses) - len(idxs))
      for idx, (acc, second) in enumerate(all_fitnesses):
        if idx in idxs: continue
        if random.random() > select_p: continue
        fitnesses.append((acc, second))
        idxs.append(idx)
    return fitnesses, idxs, anchor

  def sample_parent(self, weights,
                    idxs):
    """Samples a parent architecture from a list of indices given weights."""
    if not weights:
      return None

    # Trim the weights.
    weights = list(zip(idxs, weights))
    if self.top_n_to_select:
      top_n = self.top_n_to_select
    else:
      top_n = int(len(weights) * self.top_perc_to_select)
    top_n = min(max(top_n, self.min_to_select), len(weights))
    weights = sorted(weights, key=lambda weight: weight[1])[:top_n]
    idxs, weights = zip(*weights)

    # Weights: normalize to between 0 and 1, lower is better.
    min_weight = min(weights)
    weights = [w - min_weight for w in weights]
    max_weight = max(weights)
    if max_weight:
      weights = [w / max_weight for w in weights]

    # Weights: range from 1 to 1/10, higher is better.
    weights = [10 ** -w for w in weights]
    parent_idx = random.choices(idxs, weights=weights)[0]
    return parent_idx

  def select_individuals(
      self,
      population,
      num_suggestions_hint = 1,
      base_individuals = None):
    """Selects individuals from the population for mutation."""
    if not population: return []

    population_idxs = []
    accuracies = []
    im_sec_core_train = []
    num_params = []
    flops = []
    for idx, individual in enumerate(population):
      if individual.accuracy < self.min_acc: continue

      # Perform the age regularization.
      # Each individual is a new generation. An individual was alive for a given
      # generation if it was completed at least warm_up_secs before the
      # generation was created.
      if individual.completion_time_sec > 0:
        time_completed = individual.completion_time_sec
        num_generations_alive = sum([
            i.creation_time_sec > time_completed + self.warm_up_secs
            for i in population
        ])
        accuracy = self.age(individual.accuracy, num_generations_alive)
      else:
        accuracy = individual.accuracy
      accuracies.append(accuracy)
      im_sec_core_train.append(individual.im_sec_core_train)
      num_params.append(individual.num_params)
      flops.append(individual.flops)
      population_idxs.append(idx)

    targeted_evol = self.targeted_evol
    if random.random() < .5:
      targeted_evol = False
    if targeted_evol and base_individuals:
      base_acc = min([accuracies[bi] for bi in base_individuals])
      base_im_sec_core_train = min(
          [im_sec_core_train[bi] for bi in base_individuals])
      base_num_params = max([num_params[bi] for bi in base_individuals])
      base_flops = max([flops[bi] for bi in base_individuals])
    else:
      targeted_evol = False

    selected = []
    for _ in range(num_suggestions_hint):
      if not population_idxs:
        break

      anchor = None
      current_idxs = list(range(len(population_idxs)))

      # Weights: lower is better.
      if not targeted_evol and random.random() < 0.2:
        top_acc = max(accuracies)
        # Note that this is NOT meant to be a standard normalization, as we
        # later select a trial with probability proportional to the *reciprocal*
        # of the weight.
        weights = [top_acc / a for a in accuracies]
      else:
        base_point = None
        if random.random() < 0.5:
          if self.dataset == "imagenet2012":
            fitnesses = list(zip(accuracies, im_sec_core_train))
            if targeted_evol:
              base_point = (base_acc, base_im_sec_core_train)
            second_high = True
          else:
            fitnesses = list(zip(accuracies, num_params))
            if targeted_evol:
              base_point = (base_acc, base_num_params)
            second_high = False
        else:
          fitnesses = list(zip(accuracies, flops))
          if targeted_evol:
            base_point = (base_acc, base_flops)
          second_high = False

        if base_point:
          fitnesses, parent_idxs, anchor = self.filter_fitnesses(
              base_point, fitnesses, second_high,
              keep_individuals=base_individuals)

          # Since the seed trials which are used to compute the base point are
          # in all_fitnesses, there should always be some parent_idxs.
          assert parent_idxs

        curve = ParetoCurve(fitnesses, first_high=True, second_high=second_high)
        if self.use_pareto_balanced and not anchor:
          anchor = curve.sample_point()
        weights = curve.get_weights(fitnesses,
                                    normalize=self.use_pareto_normalized,
                                    anchor=anchor)

      selected_idx = self.sample_parent(weights, current_idxs)
      selected.append(population_idxs.pop(selected_idx))
      del accuracies[selected_idx]
      del im_sec_core_train[selected_idx]
      del flops[selected_idx]
      del num_params[selected_idx]
    return selected


class ModelMutator:
  """Applies mutations to a parent architecture."""

  def __init__(self, config):
    self.mutate_by_block = config.evolution.mutation.mutate_by_block
    self.block_add_prob = config.evolution.mutation.block_add_prob
    self.block_delete_prob = config.evolution.mutation.block_delete_prob
    self.block_mutate_prob = config.evolution.mutation.block_mutate_prob
    self.synthesis_retries = config.evolution.mutation.synthesis_retries

    self.properties = config.properties

    assert config.mutator_name in MUTATORS
    def mutator_ctr(properties):
      mutator = MUTATORS[config.mutator_name]
      return mutator(properties, **config.mutator)
    self.mutator = mutator_ctr

    assert config.synthesis_name in SYNTHESIZERS
    def synthesizer_ctr(child_id, subg_and_props):
      synthesizer = SYNTHESIZERS[config.synthesis_name]
      if config.mutator_name == "random_subgraph":
        ctr = functools.partial(synthesizer, generation=child_id,
                                **config.synthesis)
        synthesizer = GraphSynthesizer(subg_and_props, sequential_ctr=ctr,
                                       generation=child_id,
                                       **config.synthesis_graph)
      else:
        synthesizer = synthesizer(
            subg_and_props, generation=child_id, **config.synthesis)
      return synthesizer
    self.synthesizer = synthesizer_ctr

    self.inp = {"input": jnp.ones(config.train.dataset.input_shape)}

  def mutate_block(self, block_to_mutate, parent_blocks,
                   subgraph, child_id):
    """Performs mutation by block.

    Args:
      block_to_mutate: The block selected for mutation.
      parent_blocks: A list of all blocks in the parent architecture.
      subgraph: The subgraph to insert into the block.
      child_id: The id of the child.

    Returns:
      A list of blocks with mutations applied.
    """

    new_block_graph = replace_subgraph(block_to_mutate.base_graph, subgraph)
    new_blocks = copy.deepcopy(parent_blocks)
    blocks_to_mutate = [
        block for block in new_blocks
        if block.name == block_to_mutate.name
    ]

    m = re.fullmatch(r"(.+)/trial[0-9]+", block_to_mutate.name)
    if m:
      orig_name = m.group(1)
    else:
      orig_name = block_to_mutate.name
    new_name = f"{orig_name}/trial{child_id}"

    block = random.choice(blocks_to_mutate)
    block.name = new_name
    block.base_graph = new_block_graph

    for block in blocks_to_mutate:
      mutation_p = random.random()
      if mutation_p < self.block_mutate_prob:
        block.name = new_name
        block.base_graph = new_block_graph
    return new_blocks

  def mutate(self, parent_graph, parent_constants,
             parent_blocks,
             model_fn, child_id):
    """Mutates the parent architecture."""

    # Mutate entire graph.
    if not self.mutate_by_block:
      parent_block = Block(f"parent{child_id}", parent_graph,
                           parent_constants)
      parent_blocks = [parent_block]

    # Mutate a block.
    new_graph = None
    new_constants = None
    new_blocks = None
    subgraph_model = None

    if self.mutate_by_block:
      mutation_type_p = random.random()
      logging.info("Mutation type p %.3f", mutation_type_p)
    else:
      # We mutate the entire graph, so no sense in adding / deleting blocks.
      # This just needs to be >= 1.0 to be safe.
      mutation_type_p = 10.

    if mutation_type_p < self.block_add_prob:
      block_id = random.randint(0, len(parent_blocks) - 1)
      logging.info("Duplicate block_id: %d.", block_id)
      new_blocks = copy.deepcopy(parent_blocks)
      new_blocks = (
          new_blocks[:block_id] + [new_blocks[block_id]] +
          new_blocks[block_id:])
      new_graph, new_constants, new_blocks = model_fn(blocks=new_blocks)
    elif (mutation_type_p < (self.block_delete_prob +
                             self.block_add_prob) and
          len(parent_blocks)) > 1:
      block_id = random.randint(0, len(parent_blocks) - 1)
      logging.info("Delete block_id: %d.", block_id)
      new_blocks = copy.deepcopy(parent_blocks)
      new_blocks = new_blocks[:block_id] + new_blocks[block_id + 1:]
      new_graph, new_constants, new_blocks = model_fn(blocks=new_blocks)
    else:
      block_id = random.randint(0, len(parent_blocks) - 1)
      logging.info("Mutate block_id: %d.", block_id)
      parent_fingerprint = fingerprint_graph(parent_graph, parent_constants,
                                             self.inp)

      # Get the block inputs for mutations
      block_input_names = [block.graph.input_names for block in parent_blocks]
      output_names = parent_graph.output_names
      parent_graph.output_names = [
          bi for bis in block_input_names for bi in bis  # pylint: disable=g-complex-comprehension
      ]
      model = Model(parent_graph, parent_constants)
      output, _ = model.init_with_output(jax.random.PRNGKey(0), self.inp)
      block_inputs = [
          {bi: output[bi] for bi in bis} for bis in block_input_names
      ]
      parent_graph.output_names = output_names

      block_inps = []
      for idx, block in enumerate(parent_blocks):
        block_inps_idx = {}
        for cur_key, old_key in zip(block.graph.input_names,
                                    block.base_graph.input_names):
          block_inps_idx[old_key] = block_inputs[idx][cur_key]
        block_inps.append(block_inps_idx)

      block_to_mutate = parent_blocks[block_id]
      blocks = [(block, block_inp)
                for (block, block_inp) in zip(parent_blocks, block_inps)
                if block.name == block_to_mutate.name]

      properties = []
      if "shape_property" in self.properties:
        properties.append(ShapeProperty(**self.properties.shape_property))
      if "depth_property" in self.properties:
        properties.append(DepthProperty(**self.properties.depth_property))
      if "linear_property" in self.properties:
        properties.append(LinopProperty(**self.properties.linear_property))
      mutator = self.mutator(properties)

      for attempt_idx in range(self.synthesis_retries):
        logging.info("Begin mutation attempt %d.", attempt_idx)

        contexts = []
        for block, block_inp in blocks:
          contexts.append((block.base_constants, None, block_inp))

        subg_and_props = mutator.mutate(
            block_to_mutate.base_graph, contexts, abstract=True)
        synthesizer = self.synthesizer(child_id, subg_and_props)

        try:
          subgraph_model = synthesizer.synthesize()[0]
        except (StopIteration, ValueError):
          logging.info("Mutation attempt %d failed: max_len reached.",
                       attempt_idx)
          new_graph = None
          continue

        logging.info("Synthesized:")
        for node in subgraph_model.subgraph:
          logging.info("  %s", node.op.name)
        logging.info("============")

        if self.mutate_by_block:
          new_blocks = self.mutate_block(block_to_mutate, parent_blocks,
                                         subgraph_model.subgraph, child_id)

          new_graph, new_constants, new_blocks = model_fn(blocks=new_blocks)
        else:
          new_graph = subgraph_model.graph
          new_constants = subgraph_model.constants
          new_block = Block(f"model{child_id}", new_graph, new_constants)
          new_blocks = [new_block]

        try:
          child_fingerprint = fingerprint_graph(copy.deepcopy(new_graph),
                                                new_constants,
                                                self.inp)
        except Exception as e:  # pylint: disable=broad-except
          logging.info("Mutation attempt %d failed: model fails to execute with"
                       "error (%s)", attempt_idx, e)
          new_graph = None
          continue

        if child_fingerprint == parent_fingerprint:
          logging.info("Mutation attempt %d failed: child fingerprint "
                       "identical to parent", attempt_idx)
          new_graph = None
          continue
        else:
          logging.info("Mutation attempt %d succeeded.", attempt_idx)
          break

    if not new_graph:
      new_constants = None
      new_blocks = None
      subgraph_model = None
      logging.info("Attempt to create child %d failed.", child_id)

    return new_graph, new_constants, new_blocks, subgraph_model


class ParetoCurve:
  """A convenience class for a two-objective pareto curve and related methods.

  This class (and all methods) are symmetric with respect to the order of the
  objectives.
  """

  def __init__(self, obs, first_high,
               second_high):
    if not obs:
      raise ValueError("Cannot construct a pareto curve without observations.")
    self.first_high = first_high
    self.second_high = second_high
    self.curve = []

    # Sort observations best-to-worse primarily by first objective,
    # secondarily by second objective.
    obs = self.standardize_obs(obs)
    obs.sort(key=lambda ob: (ob[0], ob[1]), reverse=True)

    # Go through and identify all the pareto optimal points.
    last_ob = None
    for ob in obs:
      if not last_ob or ob[1] > last_ob[1]:
        self.curve.append(ob)
        last_ob = ob

  def standardize_ob(
      self, ob
  ):
    """Standardizes obs so that higher is better."""
    first_mul = 1 if self.first_high else -1
    second_mul = 1 if self.second_high else -1
    return (ob[0] * first_mul, ob[1] * second_mul)

  def standardize_obs(
      self,
      obs
  ):
    """Standardizes obs so that higher is better."""
    return [self.standardize_ob(ob) for ob in obs]

  def get_normalization_constants(
      self, anchor = None
  ):
    """Gets normalization according to the pareto curve.

    If the anchor is set, we first identify the points on the pareto curve which
    contain the portion of the curve which is strictly better (worse) than the
    anchor. In other words, we find the two points on the pareto curve such that
      (1) the convex hull created by the two points contains the anchor, and
      (2) the convex hull has the smallest volume.
    We then use this segment for the normalization constants.

    If there is no anchor, we use the entire pareto curve to compute the
    normalization constants.

    The normalization constants are computed from the endpoints of the curve
    such that dividing the points yields a slope of 1. Hence, the normalization
    constants should applied by *dividing* the observations.

    Args:
      anchor: an optional point for local normalization.

    Returns:
      A pair of normalization constants, one for each dimension.

    Raises:
      ValueError: if the anchor point lies outside of the convex hull of the
        pareto curve.
    """
    if len(self.curve) <= 1:
      return 1, 1

    if anchor:
      idx1 = None
      idx2 = None
      for idx, (lo, hi) in enumerate(zip(self.curve, self.curve[1:])):
        if lo[0] >= anchor[0] >= hi[0]:
          if not idx1 or idx2 and idx2 <= idx1:
            idx1 = idx
        if lo[1] <= anchor[1] <= hi[1]:
          if not idx2 or idx1 and idx1 <= idx2:
            idx2 = idx
      if idx1 is None or idx2 is None:
        raise ValueError(f"Anchor point {anchor} lies outside the pareto curve "
                         f"({self.curve[0]}, {self.curve[-1]}) and cannot be "
                         "used for local normalization.")
      lo = self.curve[min(idx1, idx2)]
      hi = self.curve[max(idx1 + 1, idx2 + 1)]
    else:
      lo = self.curve[0]
      hi = self.curve[-1]

    first_range = abs(lo[0] - hi[0])
    second_range = abs(lo[1] - hi[1])
    if first_range == 0 or second_range == 0:
      return 1, 1
    return first_range, second_range

  def get_weights(
      self, obs, normalize = True,
      anchor = None
  ):
    """Gets pareto weights for a list of observations.

    To compute the weights, if there an anchor, the weight of an observtion is
    the l^2 distance to the anchor point. If there is no anchor, we instead
    return the shortest l^2 distance the observation would have to move in order
    to be on the pareto curve.

    Points which are pareto-optimal with respect to the curve are assigned
    negative weight.

    If normalization is true, all distances are first normalized by
    self.get_normalization_constants.

    Args:
      obs: A sequence of tuples, containing the two objectives for which to get
        the pareto weights.
      normalize: whether to normalize the weights.
      anchor: an anchor point for computing local weights.

    Returns:
      A sequence of weights, in the order of obs. Lower is better, and an
        observation which is on the pareto curve has weight 0.
    """
    if not obs: return []

    if normalize:
      norm1, norm2 = self.get_normalization_constants(anchor)
    else:
      norm1, norm2 = 1, 1
    obs = self.standardize_obs(obs)

    def l2(ob1, ob2):
      return math.sqrt(((ob2[0] - ob1[0]) / norm1) ** 2 +
                       ((ob2[1] - ob1[1]) / norm2) ** 2)

    def is_pareto_optimal(ob):
      return not any(
          [po[0] >= ob[0] and po[1] >= ob[1] for po in self.curve])

    # Compute all the weights.
    if anchor:
      weights = []
      for ob in obs:
        weight = l2(ob, anchor)
        if is_pareto_optimal(ob):
          weight = -weight
        weights.append(weight)
      return weights

    best_second = self.curve[-1][1]
    worse_first = self.curve[-1][0]
    best_first = self.curve[0][0]
    worst_second = self.curve[0][1]
    curve = list(zip(self.curve, self.curve[1:]))

    def compute_weight(ob):
      best_so_far = float("inf")
      pareto_optimal = is_pareto_optimal(ob)

      # First, project to the boundary of the pareto curve.
      if pareto_optimal:
        delta1 = max(ob[0] - worse_first, 0)
        delta2 = max(ob[1] - best_second, 0)
        best_so_far = min(best_so_far, l2((delta1, delta2), (0, 0)))
        delta1 = max(ob[0] - best_first, 0)
        delta2 = max(ob[1] - worst_second, 0)
        best_so_far = min(best_so_far, l2((delta1, delta2), (0, 0)))
      else:
        best_so_far = min(best_so_far, abs(ob[0] - best_first))
        best_so_far = min(best_so_far, abs(ob[1] - best_second))

      # Loop over each segment of the pareto curve.
      for lower, upper in curve:

        # First compute side lengths.
        a = l2(lower, upper)
        b = l2(lower, ob)
        c = l2(ob, upper)

        # Compute distance to vertices.
        best_so_far = min(best_so_far, min(b, c))

        # Compute angles, make sure they are acute.
        try:
          theta_c = math.acos((c**2 - a**2 - b**2) / (-2 * a * b))
          if theta_c > math.pi / 2:
            continue
          theta_b = math.acos((b**2 - a**2 - c**2) / (-2 * a * c))
          if theta_b > math.pi / 2:
            continue
        except Exception:  # pylint: disable=broad-except
          logging.error(
              "Error computing pareto weights: %s (lower %s / upper %s)", ob,
              lower, upper)
          continue

        # Compute distance to segment.
        best_so_far = min(best_so_far, b * math.sin(theta_c))
      best_so_far = abs(best_so_far)
      if pareto_optimal: best_so_far = -best_so_far
      return best_so_far

    return [compute_weight(ob) for ob in obs]

  def sample_point(self):
    """Samples a random point on the pareto curve."""
    # By construction, self.curve has at least one point
    if len(self.curve) == 1:
      return self.curve[0]

    segment = random.randrange(len(self.curve) - 1)
    ratio = random.random()

    lo = self.curve[segment]
    hi = self.curve[segment + 1]

    idx1 = lo[0] + (hi[0] - lo[0]) * ratio
    idx2 = lo[1] + (hi[1] - lo[1]) * ratio

    return (idx1, idx2)
