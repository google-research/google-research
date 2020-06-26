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

import copy
import logging
import matplotlib.pyplot as plt
import os
from collections import defaultdict, Counter, deque
from strategic_exploration.hrl import abstract_state as AS
from strategic_exploration.hrl import data
from strategic_exploration.hrl.graph import AbstractGraph, DirectedEdge
from strategic_exploration.hrl.graph_update import Traverse
from strategic_exploration.hrl.policy import Policy
from strategic_exploration.hrl.priority import MultiPriorityQueue
from strategic_exploration.hrl.runner import EpisodeRunner
from strategic_exploration.hrl.worker import Worker
from matplotlib.patches import Patch
from scipy.misc import imread
plt.switch_backend("agg")


class Master(Policy):

  @classmethod
  def from_config(cls, config, num_actions, start_state, num_parallel, domain):
    worker = Worker.from_config(config.worker, num_actions,
                                data.room_dir(domain))
    if config.new_reward:
      cls = NewRewardMaster
    return cls(num_actions, worker, config.abstract_graph, start_state,
               config.runner, num_parallel, domain, config.success_weight)

  def __init__(self, num_actions, worker, abstract_graph_config, start_state,
               runner_config, num_parallel, domain, success_weight):
    """Use from_config to construct Master.

        Args:
            num_actions (int): number of possible actions at each state
            worker (Worker): worker to use
            abstract_graph_config (Config): config for AbstractGraph
            start_state (State): state returned by env.reset(). NOTE: assumed
              that there is only a single start state
            runner_config (Config): config for EpisodeRunner
            num_parallel (int): number of workers to run in parallel
            domain (str): environment domain e.g. MontezumaRevengeNoFrameskip-v4
            success_weight (float): how much to weight successes in priority
    """
    super(Master, self).__init__()
    self._num_actions = num_actions
    self._worker = worker
    self._room_dir = data.room_dir(domain)
    self._success_weight = success_weight

    self._path_prioritizer = MultiPriorityQueue(self._priority_fns())
    self._path_prioritizer.add_path([])

    def eval_to_train(edge):
      if not edge.dead:
        worker.mark_failed_evaluation(edge)

    # All paths are added to the queue via new reliable edges or via new
    # edges as an optimization
    def eval_to_reliable(edge):
      worker.mark_reliable(edge)
      feasible_set = self._graph.feasible_set
      for neighbor_edge in edge.end.neighbors:
        if neighbor_edge.end not in feasible_set:
          path = edge.end.path_to_start() + [neighbor_edge]
          self._path_prioritizer.add_path(path)

    def new_edge_callback(new_edge):
      feasible_set = self._graph.feasible_set
      if new_edge.start in feasible_set:
        if new_edge.end not in feasible_set:
          path = new_edge.start.path_to_start() + [new_edge]
          self._path_prioritizer.add_path(path)
      else:
        # Update priority of paths who've gained a new edge
        for parent_edge in new_edge.start.parents:
          if parent_edge.start in feasible_set:
            path = parent_edge.start.path_to_start() + [parent_edge]
            self._path_prioritizer.add_path(path)

      max_edge_degree = abstract_graph_config.max_edge_degree

      # Add higher-distance neighbors
      parent_edges = [
          parent_edge for parent_edge in new_edge.start.parents
          if parent_edge.degree == 1
      ]
      bfs_queue = deque(parent_edges)
      visited = set([parent_edge.start for parent_edge in parent_edges] +
                    [new_edge.end, new_edge.start])
      while len(bfs_queue) > 0:
        edge = bfs_queue.popleft()
        if edge.degree >= max_edge_degree:
          break

        for parent_edge in edge.start.parents:
          parent = parent_edge.start
          # Only traverse degree 1 edges to preserve BFS property
          if parent_edge.degree == 1 and parent not in visited:
            visited.add(parent)
            combined_degree = edge.degree + parent_edge.degree
            combined_reward = edge.reward + parent_edge.reward
            combined_life_lost = edge.life_lost or parent_edge.life_lost
            if not parent.contains_neighbor(edge.end):
              combined_edge = self._graph.get_edge(parent.abstract_state,
                                                   edge.end.abstract_state,
                                                   combined_degree,
                                                   combined_reward,
                                                   combined_life_lost)
              bfs_queue.append(combined_edge)

      # Forwards
      bfs_queue = deque([new_edge])
      visited = set([new_edge.end, new_edge.start])
      while len(bfs_queue) > 0:
        edge = bfs_queue.popleft()
        if edge.degree >= max_edge_degree:
          break

        for neighbor_edge in edge.end.neighbors:
          neighbor = neighbor_edge.end
          # Only traverse degree 1 edges to preserve BFS property
          if neighbor_edge.degree == 1 and neighbor not in visited:
            visited.add(neighbor)
            combined_degree = edge.degree + neighbor_edge.degree
            combined_reward = edge.reward + neighbor_edge.reward
            combined_life_lost = edge.life_lost or neighbor_edge.life_lost
            if not neighbor.contains_parent(edge.start):
              combined_edge = self._graph.get_edge(edge.start.abstract_state,
                                                   neighbor.abstract_state,
                                                   combined_degree,
                                                   combined_reward,
                                                   combined_life_lost)
              bfs_queue.append(combined_edge)

      ##########################
      # END HACK
      ##########################

    edge_callbacks = {
        (DirectedEdge.EVALUATING, DirectedEdge.TRAINING): eval_to_train,
        (DirectedEdge.EVALUATING, DirectedEdge.RELIABLE): eval_to_reliable,
    }
    self._graph = AbstractGraph.from_config(abstract_graph_config,
                                            AS.AbstractState(start_state),
                                            edge_callbacks, new_edge_callback,
                                            domain)

    self._start_node = self._graph.get_node(AS.AbstractState(start_state))

    self._runner_config = runner_config
    # runners[i] is None when previous episode[i] terminated
    self._runners = [None for _ in range(num_parallel)]

  def _start_episode(self, test=False):
    """Returns an EpisodeRunner initialized for the next episode.

        Args:
            test (bool): At evaluation, master chooses highest reward path

        Returns:
            EpisodeRunner
            int: number of steps used up for dead paths
            int: number of episodes used up for dead paths
        """
    # Return highest reward path
    if test or len(self._path_prioritizer) == 0:
      logging.info("Feasible set: {}".format(self._graph.feasible_set))
      # Tie-break by longest path
      highest_reward_node = max((node for node in self._graph.feasible_set),
                                key=lambda x:
                                (x.path_reward(), x.distance_from_start()))
      logging.info("Selected node: {}".format(highest_reward_node))
      logging.info("Path: {}".format(highest_reward_node.path_to_start()))
      return EpisodeRunner.from_config(self._runner_config,
                                       highest_reward_node.path_to_start(),
                                       self._graph, self._worker,
                                       self._num_actions), 0, 0

    wasted_steps = 0
    wasted_episodes = 0
    while True:
      # NOTE: If these are actually multi-processed, the paths should
      # probably NOT overlap, in which case the paths should actually be
      # popped!
      path = self._path_prioritizer.next_path()
      if len(path) > 0 and path[-1].dead:
        wasted_episodes += 1
        teleport_steps = 0
        if path[-1].start.teleport is not None:
          teleport_steps = path[-1].start.teleport.steps
        edge_steps = self._worker.max_steps(path[-1])
        wasted_steps += teleport_steps + edge_steps
        self._update(path, [], [Traverse(path[-1], False)], [])
      elif self._trainable_path(path):
        return EpisodeRunner.from_config(
            self._runner_config, path, self._graph, self._worker,
            self._num_actions), wasted_steps, wasted_episodes
      else:
        # Edge endpoint already reliable
        self._path_prioritizer.remove_path(path)

  def act(self, states, test=False):
    """Returns list of actions for the given states.

        Args: states (list[State])
            test (bool): Only controls the test flag for the actions on index 0

        Returns:
            list[(Action, Justification)]: actions
            int: number of actions used up for dead edges
            int: number of simulated episodes
        """
    tests = [False] * len(states)
    tests[0] = test

    actions = []
    wasted_steps = 0
    wasted_episodes = 0
    for i, (state, test) in enumerate(zip(states, tests)):
      if self._runners[i] is None:
        self._runners[i], steps, episodes = self._start_episode(test)
        wasted_steps += steps
        wasted_episodes += episodes
      actions.append(self._runners[i].act(state, test))
    return actions, wasted_steps, wasted_episodes

  def observe(self, states, actions, rewards, next_states, dones):
    """Updates internal state based on observations from environment.

        Args: states (list[State]) actions (list[Action]) rewards (list[float])
        next_states (list[State]) dones (list[bool])
    """
    experiences = zip(states, actions, rewards, next_states, dones)
    for i, experience in enumerate(experiences):
      state, action, reward, next_state, done = experience
      runner = self._runners[i]
      runner.observe(state, action, reward, next_state, done)
      if done:
        self._update(runner.path, runner.episode, runner.graph_updates,
                     runner.edge_trajectories)
        # Mark None for new EpisodeRunner creation
        self._runners[i] = None

  def stats(self):
    reliable_count = Counter()
    reliable_train_count = Counter()
    training_count = Counter()
    training_train_count = Counter()
    visit_counts = Counter()
    dead_count = Counter()
    dead_train_count = Counter()
    for node in self._graph.nodes:
      room_num = node.abstract_state.room_number
      visit_counts[room_num] += node.visit_count
      for edge in node.neighbors:
        if edge.reliable():
          reliable_count[room_num] += 1
          reliable_train_count[room_num] += edge.train_count
        elif edge.training():
          if edge.dead:
            dead_count[room_num] += 1
            dead_train_count[room_num] += edge.train_count
          else:
            training_count[room_num] += 1
            training_train_count[room_num] += edge.train_count
    total_train_count = reliable_train_count + training_train_count + \
            dead_train_count

    fraction_reliable = {
        k: reliable_train_count[k] /
        (training_train_count[k] + reliable_train_count[k] + 0.001)
        for k in training_train_count
    }
    fraction_reliable_no_blacklist = {
        k: reliable_train_count[k] /
        (training_train_count[k] + reliable_train_count[k] +
         dead_train_count[k] + 0.001) for k in training_train_count
    }

    total_reliable = sum(reliable_train_count.values())
    total_train = sum(training_train_count.values())
    total_blacklisted = sum(dead_train_count.values())
    room_counts = Counter(
        node.abstract_state.room_number for node in self._graph.nodes)
    stats = {
        "NUM_ABSTRACT_STATES/Total":
            len(self._graph.nodes),
        "NUM_EDGES_RELIABLE/Total":
            sum(reliable_count.values()),
        "NUM_EDGES_TRAINING/Total":
            sum(training_count.values()),
        "NUM_EDGES_BLACKLISTED/Total":
            sum(dead_count.values()),
        "TRAINING_COUNTS_RELIABLE/Total":
            total_reliable,
        "TRAINING_COUNTS_TRAINING/Total":
            total_train,
        "TRAINING_COUNTS_BLACKLISTED/Total":
            total_blacklisted,
        "TRAINING_COUNTS_TOTAL/Total":
            total_reliable + total_train + total_blacklisted,
        "FRACTION_RELIABLE_WITH_BLACKLIST/Total":
            float(total_reliable) /
            (total_reliable + total_train + total_blacklisted + 0.001),
        "FRACTION_RELIABLE_WITHOUT_BLACKLIST/Total":
            float(total_reliable) / (total_reliable + total_train + 0.001),
        "VISIT_COUNTS/Total":
            sum(visit_counts.values()),
        "ROOMS/Reliable":
            len(reliable_count.keys()),
        "ROOMS/Visited":
            len(room_counts.keys()),
    }

    def update_stats_with_counter(counter, format_str):
      for k, v in counter.items():
        stats[format_str.format(k)] = v

    update_stats_with_counter(room_counts, "NUM_ABSTRACT_STATES/room_{}_counts")
    update_stats_with_counter(reliable_count,
                              "NUM_EDGES_RELIABLE/room_{}_counts")
    update_stats_with_counter(training_count,
                              "NUM_EDGES_TRAINING/room_{}_counts")
    update_stats_with_counter(dead_count,
                              "NUM_EDGES_BLACKLISTED/room_{}_counts")
    update_stats_with_counter(reliable_train_count,
                              "TRAINING_COUNTS_RELIABLE/room_{}_train_counts")
    update_stats_with_counter(training_train_count,
                              "TRAINING_COUNTS_TRAINING/room_{}_train_counts")
    update_stats_with_counter(
        dead_train_count, "TRAINING_COUNTS_BLACKLISTED/room_{}_train_counts")
    update_stats_with_counter(total_train_count,
                              "TRAINING_COUNTS_TOTAL/room_{}_train_counts")
    update_stats_with_counter(fraction_reliable,
                              "FRACTION_RELIABLE_WITH_BLACKLIST/room_{}")
    update_stats_with_counter(fraction_reliable_no_blacklist,
                              "FRACTION_RELIABLE_WITHOUT_BLACKLIST/room_{}")
    update_stats_with_counter(visit_counts, "VISIT_COUNTS/room_{}")

    stats.update(self._worker.stats())
    return stats

  def visualize(self, save_dir, episode_num):
    save_dir = os.path.join(save_dir, str(episode_num))
    os.mkdir(save_dir)
    self._worker.visualize(save_dir)
    self._visualize_edges(save_dir)

  def state_dict(self):
    """Returns all information necessary to reload this after being

        reconstructed from Config. Reload with load_state_dict.

        Returns:
            dict
        """
    # NOTE: Clobbers default PyTorch state_dict / load_state_dict
    # Doesn't need to save Runners
    # Clobbers default PyTorch saving / loading
    state_dict = {
        "worker": self._worker.partial_state_dict(),
        "graph": self._graph.state_dict(),
        "start_node": self._start_node.abstract_state,
    }
    return state_dict

  def load_state_dict(self, state_dict):
    """Reconstructs Master from state_dict.

        Args:
            state_dict (dict): from state_dict()
    """
    self._graph.load_state_dict(state_dict["graph"])
    self._worker.load_state_dict(state_dict["worker"], self._graph.edges)
    self._start_node = self._graph.get_node(state_dict["start_node"])
    if self._start_node.active():
      self._path_prioritizer.add_path([])

    # Reload PathPrioritizer
    for node in self._graph.feasible_set:
      path = node.path_to_start()
      if node.active():
        self._path_prioritizer.add_path(path)

      for neighbor_edge in node.neighbors:
        if neighbor_edge.end not in self._graph.feasible_set:
          self._path_prioritizer.add_path(path + [neighbor_edge])

  def __str__(self):
    delimiter = "\n{}\n".format("=" * 79)
    s = str(self._worker)
    s += delimiter
    s += str(self._graph)
    s += delimiter
    s += str(self._path_prioritizer)
    return s

  __repr__ = __str__

  def _visualize_edges(self, save_dir):
    # Needs to be sorted in descending order (train_count, color)
    # Edges trained for >= train_count will get colored color
    color_codes = [(800, (0.5, 0, 0.5)), (600, (0, 0, 0.6)),
                   (400, (0.6, 1., 0.2)), (200, (1., 1., 0)), (0, (1, 1, 1))]
    edge_states = [
        DirectedEdge.TRAINING, DirectedEdge.RELIABLE, DirectedEdge.EVALUATING
    ]

    # room_num, match_attrs --> list[DirectedEdge]
    rooms_to_edges = defaultdict(list)
    for node in self._graph.nodes:
      for edge in node.neighbors:
        attrs = [int(edge.start.abstract_state.room_number)]
        attrs.extend(edge.start.abstract_state.match_attributes)
        rooms_to_edges[tuple(attrs)].append(edge)

    fig = plt.figure()
    for edge_state in edge_states:
      for attrs in rooms_to_edges:
        for dead in [True, False]:
          arrow_xs = []
          arrow_ys = []
          arrow_us = []
          arrow_vs = []
          colors = []
          room_num = attrs[0]
          room_path = os.path.join(self._room_dir,
                                   "room-{}.png".format(room_num))
          if not os.path.exists(room_path):
            continue
          plt.imshow(imread(room_path))

          for edge in rooms_to_edges[attrs]:
            if edge.state != edge_state or edge.dead != dead:
              continue
            for (min_train_count, color) in color_codes:
              if edge.train_count >= min_train_count:
                colors.append(color)
                break

            arrow_us.append(edge.end.abstract_state.pixel_x -
                            edge.start.abstract_state.pixel_x)
            arrow_vs.append(edge.end.abstract_state.pixel_y -
                            edge.start.abstract_state.pixel_y)
            arrow_xs.append(edge.start.abstract_state.pixel_x)
            arrow_ys.append(edge.start.abstract_state.pixel_y)

          if len(arrow_xs) > 0:
            legend_elements = [
                Patch(facecolor=color, label=">= {}".format(train_count))
                for (train_count, color) in color_codes
            ]
            plt.legend(handles=legend_elements, fontsize="xx-small")
            plt.quiver(
                arrow_xs,
                arrow_ys,
                arrow_us,
                arrow_vs,
                color=colors,
                scale=1,
                scale_units="xy",
                angles="xy")

            edge_status = "dead" if dead else "fine"
            save_path = os.path.join(
                save_dir, "{}-{}-{}.png".format(edge_state, attrs, edge_status))
            plt.axis("off")
            plt.savefig(save_path, bbox_inches="tight")
          plt.clf()
    plt.close(fig)

  def _update(self, path, episode, graph_updates, edge_trajectories):
    """Updates internal state based on trying to follow path.

        Args:
            path (list[DirectedEdge]): planned path
            episode (list[Experience]): actually executed experiences
            graph_updates (list[GraphUpdate]): updates to make on the graph
            edge_trajectories (dict): map from edges to the experiences for
              those edges, paired with the number of worker steps the worker was
              active at the beginning of the experience (Experience, int)
    """
    # Add all experienced states to the graph
    for experience in episode:
      state = AS.AbstractState(experience.state)
      next_state = AS.AbstractState(experience.next_state)
      if state == next_state:
        # Abstract state must change on positive reward
        # TODO: Fix this: violated by PrivateEye
        #assert experience.reward <= 0, (state, reward, next_state)
        pass

      if not experience.done:
        edge = self._graph.get_edge(state, next_state)
        if edge is not None:
          edge.update_reward(experience.reward, force=self._new_reward)

    # Make graph updates
    for graph_update in graph_updates:
      graph_update.update(self._graph)

    # Add experiences to the worker
    for edge in edge_trajectories:
      # Hindsight-like
      #for edge_to_update in edge.start.neighbors:
      #    if edge_to_update.training() and not edge_to_update.dead:
      trajectory = edge_trajectories[edge]
      for i, (experience, worker_steps, cum_reward) in enumerate(trajectory):
        self._worker.add_experience(edge, experience, worker_steps, cum_reward,
                                    trajectory.success)

    # Add path back to the queue
    self._path_prioritizer.add_path(path)

  def _priority_fns(self):
    """Returns a list of priority functions.

    Each computes the priority of
        a path, where lower priority is better. Each priority function takes as
        input a path (list[DirectedEdge]) and returns a priority (float).
        """

    def neighbor_gain(path):
      priority = 0.
      if len(path) > 0:
        edge = path[-1]
        if edge.reliable():
          priority -= max(edge.end.visit_count - 100, 0)
        else:
          # Prioritize edges
          priority += 2000
          #priority += min(edge.reward, 200)
          priority -= edge.train_count
          priority += edge.total_successes * self._success_weight
          #priority += edge.start.distance_from_start() * 100
          neighbor_gain = any(
              neighbor_edge.degree == 1 and
              neighbor_edge.end not in self._graph.neighbors_of_feasible
              for neighbor_edge in edge.end.neighbors)
          if not neighbor_gain:
            priority -= 5000
          priority -= Traverse.edge_expansion_attempts(edge.degree - 1)
      return -priority

    def path_reward(path):
      priority = -neighbor_gain(path)
      # Use cached reward for speed!
      if len(path) > 0:
        edge = path[-1]
        priority += edge.start.path_reward() + edge.reward
      return -priority

    return [neighbor_gain, path_reward]

  def _trainable_path(self, path):
    if len(path) == 0:
      return self._start_node.active()
    else:
      # Viable paths must either be:
      #   - all reliable ending in an active node OR
      #   - all reliable ending with a not-yet-reliable edge to a
      #     not-yet-feasible node
      last_edge = path[-1]
      if last_edge.end not in self._graph.feasible_set:
        return True
      else:
        return last_edge.reliable() and last_edge.end.active()

  def _new_reward(self):
    return False


class NewRewardMaster(Master):
  """Rewires graph with new reward function and then greedily follows best

    reward path
  """

  def load_state_dict(self, state_dict):
    super(NewRewardMaster, self).load_state_dict(state_dict)
    self._unexplored_graph_paths = []
    self._not_yet_updated_paths = set()

    for node in self._graph.feasible_set:
      is_leaf = not any(edge.reliable() for edge in node.neighbors)
      if is_leaf:
        path = node.path_to_start()
        self._unexplored_graph_paths.append(path)
        self._not_yet_updated_paths.add(path[-1])
    self._cache_cleared = False

  def _start_episode(self, test=False):
    if len(self._unexplored_graph_paths) > 0:
      return EpisodeRunner.from_config(self._runner_config,
                                       self._unexplored_graph_paths.pop(),
                                       self._graph, self._worker,
                                       self._num_actions), 0, 0

    else:
      # Just return highest reward path
      return super(NewRewardMaster, self)._start_episode(True)

  def _update(self, path, episode, graph_updates, edge_trajectories):
    super(NewRewardMaster, self)._update(path, episode, graph_updates,
                                         edge_trajectories)

    # Update reward on deg > 1
    for edge in edge_trajectories:
      edge.update_reward(
          sum(exp.reward for exp, _, _ in edge_trajectories[edge]), force=True)

    # Recompute all path rewards in the graph when done revisiting graph
    if path[-1] in self._not_yet_updated_paths:
      self._not_yet_updated_paths.remove(path[-1])
      print("{} remaining paths".format(len(self._not_yet_updated_paths)))

    if len(self._not_yet_updated_paths) == 0 and not self._cache_cleared:
      self._cache_cleared = True
      for node in self._graph.feasible_set:
        node.clear_path_reward_cache()
        node.path_reward()

  def _new_reward(self):
    return True
