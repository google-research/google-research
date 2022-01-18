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

"""A runner class to run simulations."""

import numpy as np


class Runner:
  """A class running simulations."""

  def __init__(self, env, agent, nsteps=100):
    """Initializes a runner.

    Args:
      env: A EcosystemGymEnv gym environment.
        Initial observation: A dictionary of {
                          `user`: dict(user_id=user_obs),
                          `creator`: dict(creator_id=creator_obs),
                          `doc`: ordered dict(doc_id=document_obs)};
        Step observation: A dictionary of {
                          `user`: dict(user_id=user_obs),
                          `creator`: dict(creator_id=creator_obs),
                          `doc`: ordered dict(doc_id=document_obs),
                          `user_response`: dict(user_id=a list of response_obs)
                          `creator_action`: dict(creator_id=creator_action)`}.
      agent: An agent object to generate recommendations.
      nsteps: Int, maximum steps within one simulation.
    """
    self.env = env
    self.agent = agent
    self.nsteps = nsteps

  def run(self, obs=None):
    """Run simulations with the given initial environment observation.

    Args:
      obs: Initial observation of the environment, either comes from the last
        observation of the last simulation, or None. If None, reset the
        environment and start a new simulation.

    Returns:
      user_dict: {user_obs, user_clicked_docs, user_terminates}:
        user_obs: A dictionary of key=user_id, value=a list of user observations
          at all time steps.
        user_clicked_docs: A dictionary of key=user_id, value=a list of user
          consumed documents (doc, reward, index in the candidate set).
        user_terminates: A dictionary of key=user_id, value=boolean denoting
          whether this user has terminated or not at the end of simulation.
      creator_dict: {creator_obs, creator_recommended_docs,
        creator_clicked_docs, creator_actions, creator_terminates}:
        creator_obs: A dictionary of key=creator_id, value=a list of creator
          observations at all time steps.
        creator_recommended_docs: A dictionary of key=creator_id, value=a list
          of sublists, where each sublist represents the recommended documents
          at current time steps.
        creator_clicked_docs: A dictionary of key=creator_id, value=a list
          of sublists, where each sublist represents the user clicked documents
          (document object, user reward) at current time steps.
        creator_actions: A dictionary of key=creator_id, value=a list of creator
          actions(one of 'create'/'stay'/'leave') at current time step.
        creator_terminates: A dictionary of key=creator_id, value=boolean
          denoting whether this creator has terminated at the end of simulation.
      candidate_set: A list of doc objects in candidate_set at each time step.
      obs: Environment observation after the last action.
      done: Boolean, denotes whether the simulation terminates or not.
    """
    # If initial observation is None, last simulation has terminated, and
    # environment should be reset.
    if obs is None:
      obs = self.env.reset()

    # Initialize return viables.
    user_obs = dict()  # Record user's observation at the current time step.
    user_clicked_docs = dict(
    )  # Record user's click and reward at the current time step.
    user_terminates = dict()  # Record if user leaves.
    for u_id in obs['user']:
      user_obs[u_id] = []
      user_clicked_docs[u_id] = []
      user_terminates[u_id] = False

    creator_obs = dict()
    creator_recommended_docs = dict()
    creator_clicked_docs = dict()
    creator_actions = dict()
    creator_terminates = dict()
    creator_rewards = dict()
    creator_is_saturation = dict()
    for c_id in obs['creator']:
      creator_obs[c_id] = []
      creator_recommended_docs[c_id] = []
      creator_clicked_docs[c_id] = []
      creator_actions[c_id] = []
      creator_terminates[c_id] = False
      creator_rewards[c_id] = []
      creator_is_saturation[c_id] = obs['creator'][c_id][
          'creator_is_saturation']

    # Simulation.
    document_num = []
    creator_num = []
    user_num = []
    selected_probs = []
    policy_probs = []
    user_embedding_states = []
    creator_embedding_states = []
    candidate_documents = []

    # Simulation.
    for t in range(self.nsteps):
      previous_docs = list(obs['doc'].values())
      previous_creators = obs['creator']

      # Record the environment observation at the start of time t.
      for u_id, u_obs in obs['user'].items():
        user_obs[u_id].append(u_obs)
      for c_id, c_obs in obs['creator'].items():
        creator_obs[c_id].append(c_obs)
      document_num.append(self.env.num_documents)
      creator_num.append(self.env.num_creators)
      user_num.append(self.env.num_users)

      # Agent generates recommendations: a dictionary of user_id=slate.
      # Also returns at time t, user embedding states, candidate creator
      # embedding states and candidate creator rnn internal states based on
      # their histories up to time t-1.
      user_dict = dict(
          user_obs=user_obs,
          user_clicked_docs=user_clicked_docs,
          user_terminates=user_terminates)
      creator_dict = dict(
          creator_obs=creator_obs,
          creator_recommended_docs=creator_recommended_docs,
          creator_clicked_docs=creator_clicked_docs,
          creator_actions=creator_actions,
          creator_terminates=creator_terminates,
          creator_is_saturation=creator_is_saturation)
      if self.agent.name == 'EcoAgent':
        preprocessed_candidates = self.agent.preprocess_candidates(
            creator_dict, obs['doc'])
        creator_embedding_states.append(preprocessed_candidates[0])
        creator_rnn_states = preprocessed_candidates[1]
        creator_saturate = preprocessed_candidates[2]
        creator_id = preprocessed_candidates[3]
        candidate_documents.append(preprocessed_candidates[4])

      slates, probs, preprocessed_user = self.agent.step(user_dict, obs['doc'])
      policy_probs.extend(list(probs.values()))
      user_embedding_states.append(preprocessed_user)

      # Record creator current recommendations (recommender feedback).
      ## First initialize to be empty at time t.
      for c_id, c_obs in obs['creator'].items():
        creator_recommended_docs[c_id].append([])
        creator_clicked_docs[c_id].append([])

      # Record recommended docs of creator (recommender feedback).
      for slate in slates.values():
        for idx in slate:
          doc = previous_docs[idx]
          c_id = doc['creator_id']
          creator_recommended_docs[c_id][t].append(doc)

      # Step the environment.
      obs, _, done, _ = self.env.step(slates)

      # Record if user leaves.
      user_terminates = obs['user_terminate']

      # Record click information.
      for u_id, user_responses in obs['user_response'].items():
        for doc_idx, response in zip(slates[u_id], user_responses):
          if response['click']:
            # Record user feedback for creator.
            doc = previous_docs[doc_idx]
            c_id = doc['creator_id']
            creator_clicked_docs[c_id][t].append((doc, response['reward']))
            # Record user clicked doc, user reward, and corresponding clicked
            # creator rnn_states and the satisfaction before this
            # click happens for uplift modeling.
            clicked_creator_previous_satisfaction = previous_creators[c_id][
                'creator_satisfaction']
            if self.agent.name == 'EcoAgent':
              clicked_creator_rnn_state = creator_rnn_states[doc_idx]
              clicked_creator_is_saturation = creator_saturate[doc_idx]
              clicked_creator_id = creator_id[doc_idx]
            else:
              clicked_creator_rnn_state = None
              clicked_creator_is_saturation = None
              clicked_creator_id = None
            user_clicked_docs[u_id].append(
                (doc, response['reward'], doc_idx, clicked_creator_rnn_state,
                 clicked_creator_previous_satisfaction,
                 clicked_creator_is_saturation, clicked_creator_id))
            # Record the probability of selected documents.
            selected_probs.append(probs[u_id][doc_idx])
            break

      # Record creator responses.
      for c_id, (c_action, c_reward) in obs['creator_response'].items():
        creator_actions[c_id].append(c_action)
        if c_action == 'leave':
          creator_terminates[c_id] = True
        creator_rewards[c_id].append(c_reward)

      if done:
        break

    user_dict = dict(
        user_obs=user_obs,
        user_clicked_docs=user_clicked_docs,
        user_terminates=user_terminates)
    creator_dict = dict(
        creator_obs=creator_obs,
        creator_recommended_docs=creator_recommended_docs,
        creator_clicked_docs=creator_clicked_docs,
        creator_actions=creator_actions,
        creator_rewards=creator_rewards,
        creator_terminates=creator_terminates,
        creator_is_saturation=creator_is_saturation)
    env_record = dict(
        document_num=document_num, creator_num=creator_num, user_num=user_num)
    probs = dict(
        selected_probs=np.array(selected_probs),
        policy_probs=np.array(policy_probs))
    preprocessed_user_candidates = [
        user_embedding_states, creator_embedding_states, candidate_documents
    ]
    return (user_dict, creator_dict, preprocessed_user_candidates, env_record,
            probs, obs, done)
