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

"""High-level utils for the PathNet library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from learning_parameter_allocation.pathnet import components as pn_components
from learning_parameter_allocation.pathnet import pathnet_lib as pn_lib


def create_wrapped_routed_layer(
    components,
    router,
    router_out,
    combiner,
    in_shape,
    out_shape,
    sparse=True,
    record_summaries_from_components=False):
  """Create a layer of components with a single router at the input.

  This wraps a layer of components by adding appropriately placed identities.
  It allows for a view of the routing proccess where a single input is routed
  through a subset of components, and then aggregated. This is in contrast to
  the default behavior of the pathnet library, where routing happens at the end
  of each component independently.

  Args:
    components: (list) components to route through.
    router: the router used to select the components. It should have
      a `__call__` method with the same arguments as the routers defined
      in pathnet_lib.
    router_out: the router used to route the final output. It should have
      a `__call__` method with the same arguments as the routers defined
      in pathnet_lib.
    combiner: the combiner used to aggregate the outputs.
    in_shape: (sequence of ints) input shape.
    out_shape: (sequence of ints) output shape.
    sparse: (bool) whether to set the `sparse` flag for the components layer.
    record_summaries_from_components: (bool) whether to record summaries
      coming from `components`.

  Returns:
    A list of `ComponentsLayer` with the desired behavior.
  """

  routed_components = []
  for component in components:
    routed_components.append(pn_lib.RoutedComponent(
        component,
        pn_lib.SinglePathRouter(),
        record_summaries=record_summaries_from_components))

  router_layer = pn_lib.ComponentsLayer(
      components=[pn_lib.RoutedComponent(
          pn_components.IdentityComponent(out_shape=in_shape),
          router,
          record_summaries=False)],
      combiner=pn_lib.SumCombiner())

  components_layer = pn_lib.ComponentsLayer(
      components=routed_components,
      combiner=combiner,
      sparse=sparse)

  aggregation_layer = pn_lib.ComponentsLayer(
      components=[pn_lib.RoutedComponent(
          pn_components.IdentityComponent(out_shape=out_shape),
          router_out,
          record_summaries=False)],
      combiner=pn_lib.SumCombiner())

  return [router_layer, components_layer, aggregation_layer]


def create_uniform_layer(
    num_components, component_fn, combiner_fn, router_fn):
  """Creates a layer of components with the same architecture and router.

  Args:
    num_components: (int) number of components.
    component_fn: function that creates a new component.
    combiner_fn: function that creates a combiner used to aggregate the outputs.
    router_fn: function that creates the router used to route into the next
      layer. It should have a `__call__` method with the same arguments as
      the routers defined in pathnet_lib.

  Returns:
    A `ComponentsLayer` with `num_components` components.
  """
  components = [
      pn_lib.RoutedComponent(component_fn(), router_fn())
      for _ in range(num_components)
  ]

  return pn_lib.ComponentsLayer(components, combiner_fn())


def create_identity_input_layer(num_tasks, data_shape, router_out):
  """Creates a layer of identity components used to pass the PathNet input.

  Args:
    num_tasks: (int) number of tasks.
    data_shape: (sequence of ints) input data shape.
    router_out: the router used to route into the next layer. It should have
      a `__call__` method with the same arguments as the routers defined
      in pathnet_lib.

  Returns:
    A `ComponentsLayer` with one `IdentityComponent` per task.
  """
  return create_uniform_layer(
      num_components=num_tasks,
      component_fn=lambda: pn_components.IdentityComponent(data_shape),
      combiner_fn=pn_lib.SelectCombiner,
      router_fn=lambda: router_out)
