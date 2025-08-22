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

"""Concept extraction utilities."""
import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Tuple, TypeVar, Union

from acme import specs
from acme import types
from acme.wrappers import base
import dm_env
import numpy as np

from concept_marl.utils.concept_types import ConceptType
from concept_marl.utils.concept_types import ObjectType

V = TypeVar('V')


@dataclasses.dataclass
class ConceptHolder:
  base_name: str
  concept_names: List[str]
  concept_ids: List[int]
  concept_shape: Tuple[int, Ellipsis]
  concept_type: ConceptType
  intervene_masks: List[int]
  num_categories: Optional[int] = None


class MAConceptExtractionWrapper(base.EnvironmentWrapper):
  """Wrapper for concept extraction in multiagent environments.

  Default MA-Acme Melting Pot observations are dict-indexed by
  agent and have each concept listed alongside non-concept
  observations in sequence like so:
    observation = {
        0: {
            "POSITION": ...,
            "ORIENTATION": ...,
            "RGB": ...,
            "CONCEPT.1": ...,
            ...
            "CONCEPT.N": ...,
        }
        ...
    }

  This wrapper converts observations of this type into agent-index
  dictionaries with separate entries for raw observations and
  concept observations. Concept observations are further parsed into
  scalar and categorical types. The resulting spec is as follows:
    observation = {
        0: {
            "raw_observations": {
              "POSITION": ...,
              "ORIENTATION": ...,
              "RGB": ...
            }
            "concept_observations": {
                "scalar_concepts": {
                    "concept_names": ...,
                    "concept_values": ...,
                    "concept_types": ...
                },
                "cat_concepts": {
                    "concept_names": ...,
                    "concept_values": ...,
                    "concept_types": ...
                },
            }
        }
        ...
    }

  where concept names, values, and types are stacked into individual
  arrays. The observations are structured this way to more easily use concepts
  in the learner downstream (they must stack nicely to align with the
  vector output of the MLPs used for scalar and categorical predictions,
  respectively).

  This wrapper also assumes that concepts will have a common-prefix in
  the incoming environment's observation spec.
  """

  def __init__(self,
               environment,
               num_agents,
               concept_spec,
               intervene,
               concept_noise = None,
               concepts_to_intervene = None,
               concept_overrides = None,
               mask_agent_self = False):
    """Constructor.

    Args:
      environment: Environment to wrap.
      num_agents: number of agents in environment.
      concept_spec: Dictionary containing concept metadata.
      intervene: whether or not to intervene on concept predictions.
      concept_noise: how much noise (if any) to add during intervention.
      concepts_to_intervene: which concepts (if any) to intervene on.
      concept_overrides: override specific dimensions of concept vector with
        value (agent-indexed).
      mask_agent_self: If false, masks concepts related to other agent. If
        true, masks concepts for self.
    """
    self._environment = environment
    self.num_agents = num_agents
    # Prefix for all concept values.
    # For example, all Melting Pot concepts are prepended with "WORLD.CONCEPT".
    # The prefix helps us parse concepts from other observations in the spec.
    self._concept_prefix = concept_spec['prefix']
    self._concept_spec = concept_spec

    # set up for concept intervention (if any)
    self._intervene = intervene
    self._concepts_to_intervene = concepts_to_intervene
    if self._concepts_to_intervene is None:
      self._concepts_to_intervene = {str(k): [] for k in range(num_agents)}
    self._concept_overrides = concept_overrides
    self._mask_agent_self = mask_agent_self

    # placeholder for all concepts and concept types
    self._concept_map = self._create_concept_map(
        self._environment.observation_spec())

    # process environment specs
    self._observation_spec = self._convert_obs_to_spec(
        self._environment.observation_spec())
    self._reward_spec = self._environment.reward_spec()
    self._action_spec = self._environment.action_spec()

  def _create_concept_map(self,
                          agent_observations
                          ):
    """Returns a map of concepts and concept metadata that are separated from default observation space.

    Note:
      Concept names, types, and shapes are parsed from an agent's observation
      space and stores in a ConceptHolder for later access.
      This only needs to be done once and so is done during initialization.
      This is different from _convert_obs_to_spec, which creates a new
      observation spec that stacks concept information together.

    Args:
      agent_observations: Default Melting Pot observation spec (dict-indexed by
        agent).
    """

    # supports agents with different concept spaces
    parsed_concepts = {k: {} for k in agent_observations.keys()}
    for agent_id, observations in agent_observations.items():
      # pull concepts out from observation dict
      concepts = {
          k: v
          for k, v in observations.items()
          if k.startswith(self._concept_prefix)
      }

      # parse concepts to get name, type, size
      concept_id_idx = 0
      for concept_key, concept in concepts.items():
        # Remove prefix to get base concept name
        # For example:
        #   concept_key = 'WORLD.CONCEPT_AGENT_HAS_DISH'
        #   base_name = 'AGENT_HAS_DISH'
        base_name = '_'.join(concept_key.split('_')[1:])

        # extract concept shape
        concept_shape = concept.shape[1:]
        num_objs = concept_shape[0]
        c_shape = np.prod(concept_shape)

        # parse concept name and assign a concept ID
        # int IDs are used to avoid passing strings through reverb
        c_names = []
        c_ids = []
        num_concept_values = self._concept_spec['concepts'][concept_key][
            'num_values']
        concept_type = self._concept_spec['concepts'][concept_key][
            'concept_type']
        if concept_type == ConceptType.POSITION:
          # position concept
          # make scalar (so learner applies scalar loss to position concepts)
          c_type = ConceptType.SCALAR

          # separate x- and y- coords
          for i in range(num_objs):
            c_names += [base_name + '_X' + str(i), base_name + '_Y' + str(i)]
            c_ids += [concept_id_idx, concept_id_idx + 1]
            concept_id_idx += 2

        else:
          c_type = concept_type
          c_names += [base_name + str(i) for i in range(num_objs)]
          c_ids += [concept_id_idx + i for i in range(num_objs)]
          concept_id_idx += num_objs

          if concept_type == ConceptType.CATEGORICAL:
            # categorical concept (need idxs for each possible category)
            c_type = ConceptType.CATEGORICAL
            c_shape *= num_concept_values
            c_names = list(np.repeat(c_names, num_concept_values))
            c_ids = list(np.repeat(c_ids, num_concept_values))

        # construct concept intervention flags (1 = keep value, 0 = mask value)
        if self._intervene and concept_key in self._concepts_to_intervene[
            agent_id]:
          if self._concept_spec['concepts'][concept_key][
              'object_type'] == ObjectType.AGENT:
            # special handling for agent-related concepts
            # either mask out for agent itself, or agent's teammates
            if self._mask_agent_self:
              # intervene on concepts pertaining to this agent
              intervene_flags = [
                  0 if agent_id in c_name else 1 for c_name in c_names
              ]
            else:
              # intervene on concepts pertaining to other agent
              intervene_flags = [
                  1 if agent_id in c_name else 0 for c_name in c_names
              ]
          else:
            # standard masking for other concepts
            intervene_flags = [0] * len(c_ids)
        else:
          # don't intervene
          intervene_flags = [1] * len(c_ids)

        # store concept information for use in step()
        concept_obj = ConceptHolder(
            base_name=base_name,
            concept_names=c_names,
            concept_ids=c_ids,
            concept_shape=c_shape,
            concept_type=c_type,
            num_categories=num_concept_values,
            intervene_masks=intervene_flags)
        parsed_concepts[agent_id][concept_key] = concept_obj
    return parsed_concepts

  def _one_hot_encode(self, values, num_categories):
    return np.eye(num_categories)[values.astype(np.int64)]

  def _convert_obs_to_spec(self, source):
    """Returns concept observations (converted from dict-indexed Acme observations).

    This observation spec separates raw observations (e.g. RGB, position, etc.)
    from concept observations (e.g. AGENT_HAS_TOMATO), and reshapes the
    resulting arrays following the concept metadata that is already stored in
    self._concept_map. Concept arrays are stacked by type to match how they are
    used downstream (by the agent's learner functions).

    Args:
      source: Dict-indexed Acme observation spec.
    """
    # construct dict that separates raw vs. concept observations
    # and stacks concepts together
    player_observations = {str(i): {} for i in range(self._num_players)}
    for agent_key, agent_observation in source.items():
      # regular observations
      raw_obs = {
          k: v
          for k, v in agent_observation.items()
          if not k.startswith(self._concept_prefix)
      }

      # concept observations
      all_concepts = {
          k: v
          for k, v in agent_observation.items()
          if k.startswith(self._concept_prefix)
      }

      # check that concepts with expected prefix exist in the observation
      if all_concepts:
        first_val = list(all_concepts.values())[0]
        # determine whether observation spec is being initialized (concept
        # value is Array) or used normally during training (concept value
        # is an np array).
        if isinstance(first_val, specs.Array):
          # create scalar concepts spec
          # Overload scalar to mean any non-categorical concept here
          # (e.g. scalar OR binary).
          num_scalar_concepts = np.sum([
              c.concept_shape
              for c in self._concept_map[agent_key].values()
              if c.concept_type != ConceptType.CATEGORICAL
          ])

          scalar_names = specs.Array(
              shape=(num_scalar_concepts,),
              dtype=np.int64,
              name='concept_names')
          scalar_labels = specs.Array(
              shape=(num_scalar_concepts,),
              dtype=np.float64,
              name='concept_values')
          scalar_types = specs.Array(
              shape=(num_scalar_concepts,),
              dtype=np.int64,
              name='concept_types')

          # categorical concepts
          num_cat_concepts = np.sum([
              c.concept_shape
              for c in self._concept_map[agent_key].values()
              if c.concept_type == ConceptType.CATEGORICAL
          ])
          cat_names = specs.Array(
              shape=(num_cat_concepts,), dtype=np.int64, name='concept_names')
          cat_labels = specs.Array(
              shape=(num_cat_concepts,),
              dtype=np.float64,
              name='concept_values')
          cat_types = specs.Array(
              shape=(num_cat_concepts,), dtype=np.int64, name='concept_types')

          # interventions
          intervention_masks = specs.Array(
              shape=(num_scalar_concepts + num_cat_concepts,),
              dtype=np.int64,
              name='intervention_masks')

        else:
          cat_names, cat_labels, cat_types = list(), list(), list()
          scalar_names, scalar_labels, scalar_types = list(), list(), list()
          scalar_intervention_masks, cat_intervention_masks = list(), list()
          for concept_key, concept_value in all_concepts.items():
            concept_obj = self._concept_map[agent_key][concept_key]

            # get concept value for current agent
            concept_value = concept_value[int(agent_key)]

            if concept_obj.concept_type == ConceptType.CATEGORICAL:
              # categorical concepts have shape (num_categories,)
              # make sure names and types have same dimensionality
              cat_names += concept_obj.concept_ids
              cat_types += [concept_obj.concept_type] * len(
                  concept_obj.concept_ids)

              # one hot encode categorical concept labels
              c_label = self._one_hot_encode(
                  concept_value,
                  concept_obj.num_categories).reshape(concept_obj.concept_shape)
              cat_labels.append(c_label)
              cat_intervention_masks += concept_obj.intervene_masks
            else:
              # process scalar concept name and type
              scalar_names += concept_obj.concept_ids
              scalar_types += [concept_obj.concept_type] * len(
                  concept_obj.concept_ids)
              c_label = concept_value.reshape(concept_obj.concept_shape)
              scalar_labels.append(c_label)
              scalar_intervention_masks += concept_obj.intervene_masks

          # convert concepts and intervention masks to np arrays
          scalar_names = np.asarray(scalar_names, dtype=np.int64)
          scalar_labels = np.concatenate(scalar_labels).astype(np.float64)
          scalar_types = np.asarray(scalar_types, dtype=np.int64)
          cat_names = np.asarray(cat_names, dtype=np.int64)
          cat_labels = np.concatenate(cat_labels).astype(np.float64)
          cat_types = np.asarray(cat_types, dtype=np.int64)
          intervention_masks = np.concatenate(
              [scalar_intervention_masks, cat_intervention_masks])
      else:
        # if no concepts, throw exception
        # we throw an exception here to avoid silent failure downstream
        # (e.g. when learner looks for concept values during training)
        raise ValueError(
            'Concept prefix does not match any key in observation spec.')

      # piece together scalar/categorical concepts into one dict
      concept_obs = {
          'scalar_concepts': {
              'concept_names': scalar_names,
              'concept_values': scalar_labels,
              'concept_types': scalar_types
          },
          'cat_concepts': {
              'concept_names': cat_names,
              'concept_values': cat_labels,
              'concept_types': cat_types
          },
      }

      # add interventions if necessary
      if self._intervene:
        interventions = self._intervene_on_concepts(intervention_masks,
                                                    agent_key)

        # add interventions to concept observation dictionary for this agent
        concept_obs['interventions'] = interventions

      # put together observations
      player_observations[agent_key]['raw_observations'] = raw_obs
      player_observations[agent_key]['concept_observations'] = concept_obs

    return player_observations

  def _intervene_on_concepts(self,
                             intervention_masks,
                             agent_id):
    """Create and return intervention masks for this agent."""
    interventions = {}
    if self._concept_overrides:
      # concept overrides replace dimensions of the concept vector
      # with a specific value rather than masking out completely
      intervention_masks = np.ones(intervention_masks.shape[0])
      override_masks = np.zeros(intervention_masks.shape[0])
      for override in self._concept_overrides[agent_id]:
        override_dims = override['dims']
        override_values = override['values']
        # make intervention masks zero at these dims
        intervention_masks[override_dims['min']:override_dims['max']] = 0
        # make override values value from dict at these dims
        override_masks[
            override_dims['min']:override_dims['max']] = override_values
      interventions['override_masks'] = intervention_masks

    interventions['intervention_masks'] = intervention_masks
    return interventions

  def _convert_timestep(self, source):
    """Returns multiplayer timestep from dmlab2d observations."""
    return dm_env.TimeStep(
        step_type=source.step_type,
        reward=source.reward,
        discount=0. if source.discount is None else source.discount,
        observation=self._convert_obs_to_spec(source.observation))

  @property
  def environment(self):
    """Returns the wrapped environment."""
    return self._environment

  def reset(self):
    timestep = self._environment.reset()
    return self._convert_timestep(timestep)

  def step(self, action):
    timestep = self._environment.step(action)
    return self._convert_timestep(timestep)

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

  def reward_spec(self):  # pytype: disable=signature-mismatch
    return self._reward_spec
