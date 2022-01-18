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

"""User model in the ecosystem.

The user model consists of user state representation, a user sampler, user
state transition model, and user response model.
  - user state representation: includes both observable and unobservable user
      features.
  - user sampler: sample a user.
  - user state transition model: describes the dynamics of user state
      transition after receiving recommendations from the agent.
  - user response model: characterizes how the user responds to the recommended
      slate, e.g document choice and engagement/satisfaction level with it.
"""

from absl import flags
from gym import spaces
import numpy as np
from recsim import choice_model
from recsim import user

from recs_ecosystem_creator_rl.environment import sampling_utils

FLAGS = flags.FLAGS


class UserState(user.AbstractUserState):
  """Class to represent users."""

  def __init__(self, user_id, quality_sensitivity, topic_influence,
               observation_noise_std, viability_threshold, topic_dim,
               topic_preference, initial_satisfaction, satisfaction_decay):
    """Constructor.

    Args:
      user_id: Int representing user id.
      quality_sensitivity: Float representing how sensitive the user is to the
        quality of document when generating rewards.
      topic_influence: Float within [0,1] representing how much topic_preferene
        changes as a response to clicked document.
      observation_noise_std: Float, standard deviation of truncated Guassian
        noise when generating user reward, noise is truncated within [-1, 1].
      viability_threshold: Float, the least satisfaction the user needs to have
        to stay in the platform.
      topic_dim: int representing number of topics,
      topic_preference: Float array of probability representing user's
        preference on topics.
      initial_satisfaction: Float representing user's initial satisfaction with
        the platform.
      satisfaction_decay: Float representing user's satisfaction decay rate.
    """
    self.user_id = user_id

    # Transition hyper-parameters.
    self.quality_sensitivity = quality_sensitivity
    self.topic_influence = topic_influence
    self.observation_noise_std = observation_noise_std
    self.viability_threshold = viability_threshold
    self.satisfaction_decay = satisfaction_decay

    # State variables.
    self.topic_dim = topic_dim
    self.topic_preference = topic_preference
    self.satisfaction = initial_satisfaction

  def create_observation(self):
    """Returns user id since user's state is not observable."""
    # User state (topic_preference) is not observable.
    return int(self.user_id)

  @staticmethod
  def observation_space():
    return spaces.Discrete(np.inf)

  def score_document(self, doc_obs):
    """Returns the user's affinity to the document."""
    # Current document observation is document's topic.
    return np.dot(self.topic_preference, doc_obs['topic'])


class UserSampler(user.AbstractUserSampler):
  """Generates a user with uniformly sampled topic preferences."""

  def __init__(self,
               user_ctor=UserState,
               user_id=0,
               quality_sensitivity=0.3,
               topic_influence=0.2,
               topic_dim=10,
               observation_noise_std=0.1,
               initial_satisfaction=10,
               viability_threshold=0,
               satisfaction_decay=1.0,
               sampling_space='unit ball',
               **kwargs):
    self._state_parameters = {
        'user_id': user_id,
        'quality_sensitivity': quality_sensitivity,
        'topic_dim': topic_dim,
        'topic_influence': topic_influence,
        'observation_noise_std': observation_noise_std,
        'initial_satisfaction': initial_satisfaction,
        'viability_threshold': viability_threshold,
        'satisfaction_decay': satisfaction_decay,
    }
    self.sampling_space = sampling_space
    super(UserSampler, self).__init__(user_ctor, **kwargs)

  def sample_user(self):
    # Uniformly sample initial topic preference from a simplex of dimension
    # `topic_dim`.
    if self.sampling_space == 'unit ball':
      self._state_parameters[
          'topic_preference'] = sampling_utils.sample_from_unit_ball(
              self._rng, self._state_parameters['topic_dim'])
    elif self.sampling_space == 'simplex':
      self._state_parameters[
          'topic_preference'] = sampling_utils.sample_from_simplex(
              self._rng, self._state_parameters['topic_dim'])
    else:
      raise ValueError('Only support sampling from a simplex or a unit ball.')
    return self._user_ctor(**self._state_parameters)


class ResponseModel(user.AbstractResponse):
  """User response class that records user's response to recommended slate."""

  def __init__(self, clicked=False, reward=0.0):
    self.clicked = clicked
    self.reward = reward

  def create_observation(self):
    return {'click': int(self.clicked), 'reward': np.array(self.reward)}

  @staticmethod
  def response_space():
    return spaces.Dict({
        'click':
            spaces.Discrete(2),
        'reward':
            spaces.Box(low=0.0, high=np.inf, dtype=np.float32, shape=tuple())
    })


# TODO(team): Add more details in the class docstring about the User Model.
class UserModel(user.AbstractUserModel):
  """Class that represents an encoding of a user's dynamics including generating responses and state transitioning."""

  def __init__(
      self,
      slate_size,
      user_sampler,
      response_model_ctor,
      choice_model_ctor=lambda: choice_model.MultinomialLogitChoiceModel({})):
    """Initializes a UserModel.

    Args:
      slate_size: Number of items that the agent suggests.
      user_sampler: A UserSampler responsible for providing new users every time
        reset is called.
      response_model_ctor: A response_model class that generates user response
        to recommendations.
      choice_model_ctor: A function that returns a ChoiceModel that will
        determine which doc in the slate the user interacts with.
    """
    super(UserModel, self).__init__(
        slate_size=slate_size,
        user_sampler=user_sampler,
        response_model_ctor=response_model_ctor,
    )
    self.choice_model = choice_model_ctor()

  def simulate_response(self, documents):
    """Simulate user's response to a slate of documents with choice model.

    If the document is not clicked by the user, the default reward for this
    document is -1.
    If the document is clicked by the user, the reward is
      user.quality_sensitivity * document.quality + (1 -
      user.quality_sensitivity) * <user.topic_preference, document.topic> +
      noise.
    The noise is sampled from a truncated Gaussian within range [-1, 1],

    Args:
      documents: A list of Document objects.

    Returns:
      responses: A list of Response objects, one for each document.
    """
    responses = [self._response_model_ctor() for _ in documents]

    # Score each slate item and select one.
    self.choice_model.score_documents(
        self._user_state, [doc.create_observation() for doc in documents])
    selected_index = self.choice_model.choose_item()
    # `choice_model.choose_item()` can return None if the "None of the above"
    # option is given sufficient weight to be chosen, (see e.g.,
    # choice_model.NormalizableChoiceModel.choose_item which always adds an
    # extra item to the slate which signifies choosing nothing from the slate.)
    # If None is returned, no item is clicked.
    if selected_index is not None:
      responses[selected_index].clicked = True
      responses[
          selected_index].reward = self._user_state.quality_sensitivity * documents[
              selected_index].quality + (
                  1 - self._user_state.quality_sensitivity) * np.dot(
                      self._user_state.topic_preference,
                      documents[selected_index].topic
                  ) + sampling_utils.sample_from_truncated_normal(
                      mean=0.0,
                      std=self._user_state.observation_noise_std,
                      clip_a=-1.0,
                      clip_b=1.0) + 2.0  # Shift to positive.

    return responses

  def update_state(self, documents):
    """Update user state and generate user response_observations.

    Use self.simulate_response to generate a list of Response object for each
    documents.

    User's total satisfaction firstly shrinks by rate satisfaction_decay.
    If no document is consumed, user's topic preference remains untouched, and
    the total satisfaction decreases by 1.

    If the user clicks one document, her satisfaction changes by the
    response.reward, and her topic_preference will be:
      1. temporal_topic_preference <- topic_preference + topic_influence *
      response.reward * document.topic.
      2. normalize the temporal_topic_preference to the topic_preference domain
      (unit ball),and set it to be the new user.topic_preference.
    Intuitively, the user topic preference will shift toward the document.topic
    if the response.reward is positive. Otherwise the user will decrease her
    preference on the document's topic.

    Args:
      documents: A list of Document objects in the recommended slate.

    Returns:
      A list of Response observations for the recommended documents.
    """
    responses = self.simulate_response(documents)
    self._user_state.satisfaction *= self._user_state.satisfaction_decay
    click = False
    for doc, response in zip(documents, responses):
      if response.clicked:
        # Update user's satisfaction based on the clicked document.
        self._user_state.satisfaction += response.reward
        # Update user's topic preference based on the clicked document.
        topic_preference = self._user_state.topic_preference + self._user_state.topic_influence * response.reward * doc.topic
        # Normalize the topic_preference to the unit ball.
        topic_preference = topic_preference / np.sqrt(
            np.sum(topic_preference**2))
        self._user_state.topic_preference = topic_preference
        click = True
        break

    # If no click, user satisfaction decreases by 1.
    if not click:
      self._user_state.satisfaction -= 1
    return [response.create_observation() for response in responses]

  def score_document(self, doc_obs):
    return self._user_state.score_document(doc_obs)

  def get_user_id(self):
    return self._user_state.user_id

  def is_terminal(self):
    return self._user_state.satisfaction < self._user_state.viability_threshold
