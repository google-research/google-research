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

"""Creator model in the ecosystem."""

import copy

from absl import flags
from gym import spaces
import numpy as np
from recsim import document

from recs_ecosystem_creator_rl.environment import sampling_utils

FLAGS = flags.FLAGS


class Document(document.AbstractDocument):
  """Class to represent a Document with known topic vector.

  Attributes:
    topic: One-hot array representing the topic of the document.
    quality: Float within [-1, 1] representing document quality.
    creator_id: Int representing the creator from which document originated.
  """

  def __init__(self, doc_id, topic, quality, creator_id):
    super(Document, self).__init__(doc_id)
    # TODO(team): Consider making following fields private.
    self.topic = np.array(topic)
    self.quality = quality
    self.creator_id = creator_id

  def create_observation(self):
    # We can also make quality observable here.
    return {
        'doc_id': int(self._doc_id),
        'topic': self.topic,
        'creator_id': int(self.creator_id),
    }

  def observation_space(self):
    return spaces.Dict({
        'doc_id':
            spaces.Discrete(np.inf),
        'topic':
            spaces.Box(shape=self.topic.shape, dtype=np.float32, low=0, high=1),
        'creator_id':
            spaces.Discrete(np.inf)
    })

  def copy(self, new_doc_id, new_creator_id):
    return Document(new_doc_id, self.topic, self.quality, new_creator_id)

  def equal(self, new_doc):
    return (np.allclose(self.topic, new_doc.topic, 1e-7) and
            np.allclose(self.quality, new_doc.quality))


# TODO(team): Rename class to be more accurate. And provide more details
# into class docstring about state, document generation, state update.
class Creator:
  """Class to represent a content provider.

  Attributes:
    creator_id: Int representing the id of creator.
    topic_preference: Array representing creator's topic preference for next
      created document, log probability for generating topic of next created
      document. Domain: unit ball / simplex.
    satisfaction: Float representing creator's absolute overall satisfaction
      with the platform based on feedback from users and recommender. The
      creator policy is based on saturated satisfaction, see
      self.saturated_satisfaction() for details.
    viability_threshold: Float representing the least saturated satisfaction the
      creator needs to stay in the platform.
    new_document_margin: Positive float, whenever the creator's saturated
      satisfaction increases by this number, the creator will create a new
      document.
    no_recommendation_penalty: Positive float, if there is no recommendation,
      the creator's satisfaction will decrease by this number.
    recommendation_reward: Positive float corresponding to per-recommendation
      increase of creator's satisfaction.
    user_click_reward: Positive float corresponding to per-user-click increase
      of creator's satisfaction.
    satisfaction_decay: Float within [-1,1] representing the satisfaction decay
      rate.
    doc_ctor: A class/constructor for the type of documents that will be created
      by this creator.
    doc_quality_mean: Float representing the mean of the document quality from
      this creator.
    doc_quality_std: Float representing the standard deviation of the document
      quality from this creator. The quality is sampled from a truncated
      Gaussian distribution with mean doc_quality_mean and std doc_quality_std
      in range [-1, 1].
    topic_dim: Int, number of topics among documents.
    topic_influence: Float representing scale of topic_preference change as a
      response to user reward. Default 0, creator's topic sensitivity will not
      change.
    rng: Random seed.
    viable: Boolean representing if this creator is viable, that is, if her
      saturated satisfaction is above the viability_threshold.
    documents: A list of created documents by the creator.
    is_saturation: Boolean representing whether creator's satisfaction gets
      saturated or not.
  """

  def __init__(self,
               creator_id,
               initial_doc_id,
               topic_preference,
               initial_satisfaction,
               viability_threshold,
               new_document_margin,
               no_recommendation_penalty,
               recommendation_reward,
               user_click_reward,
               satisfaction_decay,
               doc_ctor,
               doc_quality_mean,
               doc_quality_std,
               initial_num_docs,
               is_saturation,
               rng,
               topic_influence=0.):

    self.creator_id = creator_id
    self.rng = rng

    # Parameters for document created by this creator.
    self.doc_ctor = doc_ctor
    self.doc_quality_mean = doc_quality_mean
    self.doc_quality_std = doc_quality_std
    self.topic_dim = len(topic_preference)

    # Creator initial state.
    # TODO(team): Check whether deepcopy is needed in other places too.
    self.topic_preference = copy.deepcopy(topic_preference)
    self.satisfaction = initial_satisfaction
    self.viable = True
    self.is_saturation = is_saturation

    # Hyperparameters for creator transition dynamics.
    self.viability_threshold = viability_threshold
    self.topic_influence = topic_influence
    self.new_document_margin = new_document_margin
    self.no_recommendation_penalty = no_recommendation_penalty
    self.recommendation_reward = recommendation_reward
    self.user_click_reward = user_click_reward
    self.satisfaction_decay = satisfaction_decay

    # Initialize creator's documents with one document of doc_id being
    # initial_doc_id.
    self.documents = []
    for i in range(initial_num_docs):
      doc_quality = sampling_utils.sample_from_truncated_normal(
          mean=self.doc_quality_mean,
          std=self.doc_quality_std,
          clip_a=-1,
          clip_b=1)
      # Use softmax probability to sample topic for new document.
      log_prob = self.topic_preference - max(self.topic_preference)
      doc_topic_prob = np.exp(log_prob) / np.sum(np.exp(log_prob))
      doc_topic = np.zeros(self.topic_dim)
      doc_topic[self.rng.choice(len(self.topic_preference),
                                p=doc_topic_prob)] = 1
      self.documents.append(
          self.doc_ctor(
              doc_id=initial_doc_id + i,
              topic=doc_topic,
              quality=doc_quality,
              creator_id=creator_id))

  def update_state(self, doc_count, documents, user_responses):
    """Updates creator state as a response to recommmender's and user's feedback.

    Satisfaction change: Firstly, the old satisfaction will decay by the
    satisfaction_decay rate. Then, if the creator's documents get recommended,
    her satisfaction will increase by #recommendations x recommendation_reward.
    If there is any user that clicks the creator's document, her satisfaction
    will then change by user_click_reward and the user reward.

    Topic_preferece change(optional): Creator's topic preference will change
    based on the user reward and topics of clicked documents. Specifically,
      * temporal_topic_preference <- topic_preference + creator.topic_influence
      * user_reward / self.satisfaction * document.topic
      * creator.topic_preference = normalize(temporal_topic_preference) to a
      unit ball
    Intuitively, if the creator receives positive reward from user, she will
    tend to generate same topic document next time.

    Create-new-document: The creator will create a new document if her
    saturated satisfaction increases by another new_document_margin.

    Viability: The creator will be no longer viable if her saturated
    satisfaction is below the viability_threshold.

    Args:
      doc_count: Int representing number of created documents on the platform,
        used for creating document ID for new document if there is.
      documents: A list of creator's recommended documents at the current time
        step. The document list can contain duplicate documents since one
        document can be recommended to more than one user at a time.
      user_responses: A list of Response observations for the creator's
        recommended documents.

    Returns:
      doc_count: Updated number of existing documents on the platform.
      (action, reward): A tuple of creator response, where action is a string
        describing creator's action, which is one of 'create'/'stay'/'leave';
        reward is creator incremental saturated_satisfaction change.
    """
    # TODO(team): Make this function more modular.
    old_satisfaction = self.satisfaction
    # First decays the satisfaction by satisfaction_decay rate. This allows for
    # capturing the myopic creators, whose actions only depend on current user
    # and recommender feedback.
    self.satisfaction = self.satisfaction * self.satisfaction_decay
    # Default action is to stay.
    action = 'stay'
    if documents:
      # Update creator satisfaction.
      ## Feedback from recommender: increase satisfaction from being recommended
      self.satisfaction += len(documents) * self.recommendation_reward
      ## Feedback from users: modify satisfaction based on user reward
      for response in user_responses:
        if response['click']:
          self.satisfaction += self.user_click_reward
          self.satisfaction += response['reward']

      # Update creator's topic preference based on user reward and the creator's
      # current satisfaction with the platform.
      # Adjust user_reward by old_satisfaction to reflect that popular creators
      # are less affected by one user preference.
      for doc, response in zip(documents, user_responses):
        if response['click']:
          self.topic_preference = self.topic_preference + (
              self.topic_influence * response['reward'] / old_satisfaction *
              doc.topic)
      ## Normalize the creator.topic_preference to the unit ball.
      self.topic_preference = self.topic_preference / np.linalg.norm(
          self.topic_preference)

      # Create new documents.
      # The criterion for creating new documents is motivated by the conjecture
      # that creating a document might be a result of positive feedback from
      # the platform across multiple time steps.
      # Thus we use the accumulated satisfaction here.
      # Every time the creator's satisfaction has increased by another
      # new_document_margin,she will create a new document.
      # For example, consider the new_document_margin=2, if the
      # old_saturated_satisfaction is 1.8 and the new_saturated_satisfaction
      # is 2.1, then the creator will create a new document. On the contrary,
      # if the old_saturated_satisfaction is 1.5 and the
      # new_saturated_satisfaction is 1.9, then the creator will not create
      # a document.
      for _ in range(
          int(
              self.saturated_satisfaction(old_satisfaction) //
              self.new_document_margin),
          int(
              self.saturated_satisfaction(self.satisfaction) //
              self.new_document_margin)):
        # create new content
        doc_quality = sampling_utils.sample_from_truncated_normal(
            mean=self.doc_quality_mean,
            std=self.doc_quality_std,
            clip_a=-1,
            clip_b=1)
        # Use softmax probability to sample topic for new document.
        log_prob = self.topic_preference - max(self.topic_preference)
        doc_topic_prob = np.exp(log_prob) / np.sum(np.exp(log_prob))
        doc_topic = np.zeros(self.topic_dim)
        doc_topic[self.rng.choice(len(self.topic_preference),
                                  p=doc_topic_prob)] = 1
        self.documents.append(
            self.doc_ctor(
                doc_id=doc_count,
                topic=doc_topic,
                quality=doc_quality,
                creator_id=self.creator_id))
        doc_count += 1
        action = 'create'
    else:
      # Decrease the creator satisfaction by no_recommendation_penalty
      self.satisfaction -= self.no_recommendation_penalty

    # Update creator viability.
    self.update_viability()
    if not self.viable:
      action = 'leave'
    creator_reward = self.saturated_satisfaction(
        self.satisfaction) - self.saturated_satisfaction(old_satisfaction)
    return doc_count, (action, creator_reward)

  def update_viability(self):
    # Creator is no longer viable if her saturated satisfaction is below the
    # viability_threshold.
    self.viable = (
        self.saturated_satisfaction(self.satisfaction) >
        self.viability_threshold)

  def saturated_satisfaction(self, satisfaction):
    """Log(1+x) to saturate absolute satisfaction noting diminishing returns.

    The purpose is to distinguish the recommendation effects between popular
    creators versus less popular ones. Intuitively, popular creators are
    influenced less by one recommendation as opposed to less popular ones.

    Args:
      satisfaction: Float representing creator absolute satisfaction with the
        platform based on user and recommender feedback.

    Returns:
      saturated satisfaction by concave function log(1+x).
    """
    # Use a max here to avoid underflow.
    if not self.is_saturation:
      return satisfaction
    return np.log1p(max(satisfaction, np.expm1(self.viability_threshold)))

  def copy(self, new_creator_id, initial_doc_id):
    """Copy current creator to a new creator."""
    new_creator = Creator(
        new_creator_id,
        initial_doc_id,
        topic_preference=self.topic_preference,
        initial_satisfaction=self.satisfaction,
        viability_threshold=self.viability_threshold,
        new_document_margin=self.new_document_margin,
        no_recommendation_penalty=self.no_recommendation_penalty,
        recommendation_reward=self.recommendation_reward,
        user_click_reward=self.user_click_reward,
        satisfaction_decay=self.satisfaction_decay,
        doc_ctor=self.doc_ctor,
        doc_quality_mean=self.doc_quality_mean,
        doc_quality_std=self.doc_quality_std,
        initial_num_docs=0,
        is_saturation=self.is_saturation,
        rng=self.rng,
        topic_influence=self.topic_influence)
    for i, doc in enumerate(self.documents):
      new_creator.documents.append(doc.copy(initial_doc_id + i, new_creator_id))
    return new_creator

  def equal(self, new_creator):
    """Check if new_creator has the same attributes as current creator."""
    if np.linalg.norm(self.topic_preference -
                      new_creator.topic_preference) > 1e-7:
      return False
    if self.satisfaction != new_creator.satisfaction:
      return False
    if self.viability_threshold != new_creator.viability_threshold:
      return False
    if self.new_document_margin != new_creator.new_document_margin:
      return False
    if self.no_recommendation_penalty != new_creator.no_recommendation_penalty:
      return False
    if self.recommendation_reward != new_creator.recommendation_reward:
      return False
    if self.user_click_reward != new_creator.user_click_reward:
      return False
    if self.satisfaction_decay != new_creator.satisfaction_decay:
      return False
    if self.doc_quality_mean != new_creator.doc_quality_mean:
      return False
    if self.doc_quality_std != new_creator.doc_quality_std:
      return False
    if self.topic_influence != new_creator.topic_influence:
      return False
    if len(self.documents) != len(new_creator.documents):
      return False
    if self.is_saturation != new_creator.is_saturation:
      return False
    for doc, new_doc in zip(self.documents, new_creator.documents):
      if not doc.equal(new_doc):
        return False
    return True

  def create_observation(self):
    return {
        'creator_id': int(self.creator_id),
        'creator_satisfaction': self.saturated_satisfaction(self.satisfaction),
        'creator_is_saturation': int(self.is_saturation)
    }

  @staticmethod
  def observation_space():
    return spaces.Dict({
        'creator_id':
            spaces.Discrete(np.inf),
        'creator_satisfaction':
            spaces.Box(shape=(), dtype=np.float32, low=-np.inf, high=np.inf),
        'creator_is_saturation':
            spaces.Discrete(2),
    })


class DocumentSampler(document.AbstractDocumentSampler):
  """Class to sample documents from viable creators."""

  def __init__(
      self,
      doc_ctor=Document,
      creator_ctor=Creator,
      topic_dim=None,
      num_creators=None,
      initial_satisfaction=None,
      viability_threshold=None,
      new_document_margin=None,
      no_recommendation_penalty=None,
      recommendation_reward=None,
      user_click_reward=None,
      satisfaction_decay=None,
      doc_quality_std=None,
      doc_quality_mean_bound=None,
      initial_num_docs=None,
      topic_influence=None,
      is_saturation=None,
      copy_varied_property=None,
      sampling_space='unit ball',
      **kwargs,
  ):
    """Initialize a DocumentSampler.

    Args:
       doc_ctor: A class/constructor for the type of documents that will be
         sampled by this sampler.
       creator_ctor: A class/constructor for the type of creators that will
         generate documents for this sampler.
       topic_dim: int representing number of topics of documents on the
         platform.
       num_creators: Int representing number of creators on the platform.
       initial_satisfaction: A list of float with length num_creators. Each
         entry represents the initial satisfation of the creator.
       viability_threshold: A list of float with length num_creators. Each entry
         represents the least saturated satisfaction the creator needs to have
         to stay in the platform.
       new_document_margin: A list of float with length num_creators. Whenever
         the creator's saturated satisfaction increases by this number, the
         creator will create a new document.
       no_recommendation_penalty: A list of float with length num_creators. If
         there is no recommendation, the creator's satisfaction will decrease by
         this number.
       recommendation_reward: A list of float with length num_creators. Each
         entry represents per-recommendation increase of creator's satisfaction.
       user_click_reward: A list of float with length num_creators. Each entry
         represents the creator's satisfaction increase from per-user-click.
       satisfaction_decay: A list of float with length num_creators. Each entry
         represents the creator's satisfaction decay rate.
       doc_quality_std: A list of float with length num_creators. Each entry
         represents the standard deviation of the document quality created by
         this creator. The quality is sampled from a truncated Gaussian
         distribution with mean doc_quality_mean and std doc_quality_std in
         range [-1, 1].
       doc_quality_mean_bound: A list of float with length num_creators. The
         creator's doc_quality_mean is sampled uniformly from [-val, val].
       initial_num_docs: A list of int representing num of initialized docs for
         each creator.
       topic_influence: A list of float with length num_creators, each entry
         represents how much the creator's topic_preference changes as a
         response to user reward.
       is_saturation: A list of bool with length num_creators, each
         entry represents whether this creator's satisfaction saturated or not.
       copy_varied_property: A string. If none, generate creators based on input
         attribute lists. Otherwise, copy the second half creators from the
         first half creators, but change the attribute `copy_varied_property`
         which now supports `initial_satisfaction` and `recommendation_reward`.
       sampling_space: String describing the domain from which the creator
         topic_preference will be sampled. Valid choices: `unit ball`,`simplex`.
       **kwargs: other arguments used to initialize AbstractDocumentSampler.
    """
    if len(initial_satisfaction) != num_creators:
      raise ValueError(
          'Length of `initial_satisfaction` should be the same as number of creators.'
      )
    if len(viability_threshold) != num_creators:
      raise ValueError(
          'Length of `viability_threshold` should be the same as number of creators.'
      )
    if len(new_document_margin) != num_creators:
      raise ValueError(
          'Length of `new_document_margin` should be the same as number of creators.'
      )
    if len(no_recommendation_penalty) != num_creators:
      raise ValueError(
          'Length of `no_recommendation_penalty` should be the same as number of creators.'
      )
    if len(recommendation_reward) != num_creators:
      raise ValueError(
          'Length of `recommendation_reward` should be the same as number of creators.'
      )
    if len(user_click_reward) != num_creators:
      raise ValueError(
          'Length of `user_click_reward` should be the same as number of creators.'
      )
    if len(satisfaction_decay) != num_creators:
      raise ValueError(
          'Length of `satisfaction_decay` should be the same as number of creators.'
      )
    if len(doc_quality_std) != num_creators:
      raise ValueError(
          'Length of `doc_quality_std` should be the same as number of creators.'
      )
    if len(doc_quality_mean_bound) != num_creators:
      raise ValueError(
          'Length of `doc_quality_mean_bound` should be the same as number of creators.'
      )
    if len(initial_num_docs) != num_creators:
      raise ValueError(
          'Length of `initial_num_docs` should be the same as number of creators.'
      )
    if len(is_saturation) != num_creators:
      raise ValueError(
          'Length of `is_saturation` should be the same as number of creators.')
    if len(topic_influence) != num_creators:
      raise ValueError(
          'Length of `topic_influence` should be the same as number of creators.'
      )

    self.topic_dim = topic_dim
    self.doc_count = 0
    self.sampling_space = sampling_space
    self.doc_quality_mean_bound = doc_quality_mean_bound

    # Creator parameters
    self.num_creators = num_creators
    self.initial_satisfaction = initial_satisfaction
    self.viablibity_threshold = viability_threshold
    self.new_document_margin = new_document_margin
    self.no_recommendation_penalty = no_recommendation_penalty
    self.recommendation_reward = recommendation_reward
    self.user_click_reward = user_click_reward
    self.satisfaction_decay = satisfaction_decay
    self.doc_quality_std = doc_quality_std
    self.initial_num_docs = initial_num_docs
    self.topic_influence = topic_influence
    self.is_saturation = is_saturation
    self.doc_ctor = doc_ctor

    def get_creator(creator_id):
      if self.sampling_space == 'unit ball':
        topic_preference = sampling_utils.sample_from_unit_ball(
            self._rng, self.topic_dim)
      elif self.sampling_space == 'simplex':
        topic_preference = sampling_utils.sample_from_simplex(
            self._rng, self.topic_dim)
      else:
        raise ValueError('Only support sampling from a simplex or a unit ball.')
      # Uniformly sample doc_quality_mean from
      # [-doc_quality_mean_bound, doc_quality_mean_bound].
      doc_quality_mean = self._rng.random_sample(
      ) * 2 * self.doc_quality_mean_bound[
          creator_id] - self.doc_quality_mean_bound[creator_id]
      new_creator = creator_ctor(
          creator_id=creator_id,
          initial_doc_id=self.doc_count,
          topic_preference=topic_preference,
          initial_satisfaction=self.initial_satisfaction[creator_id],
          viability_threshold=self.viablibity_threshold[creator_id],
          new_document_margin=self.new_document_margin[creator_id],
          no_recommendation_penalty=self.no_recommendation_penalty[creator_id],
          recommendation_reward=self.recommendation_reward[creator_id],
          user_click_reward=self.user_click_reward[creator_id],
          satisfaction_decay=self.satisfaction_decay[creator_id],
          doc_ctor=self.doc_ctor,
          doc_quality_mean=doc_quality_mean,
          doc_quality_std=self.doc_quality_std[creator_id],
          initial_num_docs=self.initial_num_docs[creator_id],
          is_saturation=self.is_saturation[creator_id],
          rng=self._rng,
          topic_influence=self.topic_influence[creator_id])
      self.doc_count += self.initial_num_docs[creator_id]
      return new_creator

    def creators_generator():
      # Generates a dictionary of creators.
      # key = creator_id, value = creator object.
      while True:
        # Generator to yield a new set of creators whenever resetting the
        # environment.
        creators = {}
        self.doc_count = 0
        for creator_id in range(num_creators):
          # initialize each creator with one document, creator's preference
          # is uniformly sampled from a unit ball.
          creators[creator_id] = get_creator(creator_id)
        yield creators

    def two_identical_creators_but_generator(varied_property):
      # Generates a dictionary of creators.
      # key = creator_id, value = creator object.
      while True:
        # Generator to yield a new set of creators whenever resetting the
        # environment.
        creators = {}

        # Initialize first half creators.
        self.doc_count = 0
        for creator_id in range(num_creators // 2):
          creators[creator_id] = get_creator(creator_id)

        # Copy the first half creators to the second half.
        for creator_id in range(num_creators // 2, num_creators):
          creators[creator_id] = creators[creator_id - num_creators // 2].copy(
              creator_id, self.doc_count)
          if varied_property == 'satisfaction':
            creators[creator_id].satisfaction = self.initial_satisfaction[
                creator_id]
          elif varied_property == 'recommendation_reward':
            creators[
                creator_id].recommendation_reward = self.recommendation_reward[
                    creator_id]
          else:
            raise NotImplementedError(
                'Only support varying satisfaction and recommendation_reward for now.'
            )
          self.doc_count += len(creators[creator_id].documents)

        yield creators

    if copy_varied_property:
      self.viable_creators_iter = two_identical_creators_but_generator(
          copy_varied_property)
    else:
      self.viable_creators_iter = creators_generator()

    super(DocumentSampler, self).__init__(doc_ctor, **kwargs)
    self.reset_creator()

  def reset_sampler(self):
    self._rng = np.random.RandomState(self._seed)

  def reset_creator(self):
    """Resample all creators and set them as viable."""
    self.viable_creators = next(self.viable_creators_iter)

  def sample_document(self, size=1):
    if self.num_viable_creators:
      documents = []
      for cr in self.viable_creators.values():
        documents.extend(cr.documents)
      # Sampling without replacement.
      return np.random.choice(documents, size=size, replace=False)

  def update_state(self, documents, responses):
    creators_popularity = {
        creator_id: dict(documents=[], user_responses=[])
        for creator_id in self.viable_creators.keys()
    }
    for doc, response in zip(documents, responses):
      creator_id = doc.creator_id
      creators_popularity[creator_id]['documents'].append(doc)
      creators_popularity[creator_id]['user_responses'].append(response)
    creator_response = dict()
    for creator_id, creator_popularity in creators_popularity.items():
      self.doc_count, response = self.viable_creators[creator_id].update_state(
          doc_count=self.doc_count, **creator_popularity)
      creator_response[creator_id] = response
      if not self.viable_creators[creator_id].viable:
        del self.viable_creators[creator_id]

    return creator_response

  @property
  def num_viable_creators(self):
    return len(self.viable_creators)

  @property
  def num_documents(self):
    return sum([len(cr.documents) for cr in self.viable_creators.values()])

  @property
  def topic_documents(self):
    """Show the distribution of documents for topic-diversity metric."""
    topics = []
    for cr in self.viable_creators.values():
      topics.extend([doc.create_observation()['topic'] for doc in cr.documents])
    # topics.shape = (#documents, topic_dim).
    # By taking the mean along dim 0, we can get the document topic
    # distribution.
    return np.mean(topics, axis=0)
