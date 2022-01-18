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

r"""Evaluation of matrix factorization following the protocol of the NCF paper.

Details:
 - Model: Matrix factorization with biases:
     y(u,i) = b + v_{u,1}+v_{i,1}+\sum_{f=2}^d v_{u,f}*v_{i,f}
 - Loss: logistic loss
 - Optimization algorithm: stochastic gradient descent
 - Negatives sampling: Random negatives are added during training
 - Optimization objective (similar to NCF paper)
     argmin_V \sum_{(u,i) \in S} [
          ln(1+exp(-y(u,i)))
        + #neg/|I| * \sum_{j \in I} ln(1+exp(y(u,j)))
        + reg * ||V||_2^2 ]
 - Evaluation follows the protocol from:
   He, X., Liao, L., Zhang, H., Nie, L., Hu, X., and Chua, T.-S.: Neural
   collaborative filtering. WWW 2017
"""

import argparse
# Dataset and evaluation protocols reused from
# https://github.com/hexiangnan/neural_collaborative_filtering
from Dataset import Dataset
from evaluate import evaluate_model
import numpy as np


class MFModel(object):
  """A matrix factorization model trained using SGD and negative sampling."""

  def __init__(self, num_user, num_item, embedding_dim, reg, stddev):
    """Initializes MFModel.

    Args:
      num_user: the total number of users.
      num_item: the total number of items.
      embedding_dim: the embedding dimension.
      reg: the regularization coefficient.
      stddev: embeddings are initialized from a random distribution with this
        standard deviation.
    """
    self.user_embedding = np.random.normal(0, stddev, (num_user, embedding_dim))
    self.item_embedding = np.random.normal(0, stddev, (num_item, embedding_dim))
    self.user_bias = np.zeros([num_user])
    self.item_bias = np.zeros([num_item])
    self.bias = 0.0
    self.reg = reg

  def _predict_one(self, user, item):
    """Predicts the score of a user for an item."""
    return (self.bias + self.user_bias[user] + self.item_bias[item] +
            np.dot(self.user_embedding[user], self.item_embedding[item]))

  def predict(self, pairs, batch_size, verbose):
    """Computes predictions for a given set of user-item pairs.

    Args:
      pairs: A pair of lists (users, items) of the same length.
      batch_size: unused.
      verbose: unused.

    Returns:
      predictions: A list of the same length as users and items, such that
      predictions[i] is the models prediction for (users[i], items[i]).
    """
    del batch_size, verbose
    num_examples = len(pairs[0])
    assert num_examples == len(pairs[1])
    predictions = np.empty(num_examples)
    for i in range(num_examples):
      predictions[i] = self._predict_one(pairs[0][i], pairs[1][i])
    return predictions

  def fit(self, positive_pairs, learning_rate, num_negatives):
    """Trains the model for one epoch.

    Args:
      positive_pairs: an array of shape [n, 2], each row representing a positive
        user-item pair.
      learning_rate: the learning rate to use.
      num_negatives: the number of negative items to sample for each positive.

    Returns:
      The logistic loss averaged across examples.
    """
    # Convert to implicit format and sample negatives.
    user_item_label_matrix = self._convert_ratings_to_implicit_data(
        positive_pairs, num_negatives)
    np.random.shuffle(user_item_label_matrix)

    # Iterate over all examples and perform one SGD step.
    num_examples = user_item_label_matrix.shape[0]
    reg = self.reg
    lr = learning_rate
    sum_of_loss = 0.0
    for i in range(num_examples):
      (user, item, rating) = user_item_label_matrix[i, :]
      user_emb = self.user_embedding[user]
      item_emb = self.item_embedding[item]
      prediction = self._predict_one(user, item)

      if prediction > 0:
        one_plus_exp_minus_pred = 1.0 + np.exp(-prediction)
        sigmoid = 1.0 / one_plus_exp_minus_pred
        this_loss = (np.log(one_plus_exp_minus_pred) +
                     (1.0 - rating) * prediction)
      else:
        exp_pred = np.exp(prediction)
        sigmoid = exp_pred / (1.0 + exp_pred)
        this_loss = -rating * prediction + np.log(1.0 + exp_pred)

      grad = rating - sigmoid

      self.user_embedding[user, :] += lr * (grad * item_emb - reg * user_emb)
      self.item_embedding[item, :] += lr * (grad * user_emb - reg * item_emb)
      self.user_bias[user] += lr * (grad - reg * self.user_bias[user])
      self.item_bias[item] += lr * (grad - reg * self.item_bias[item])
      self.bias += lr * (grad - reg * self.bias)

      sum_of_loss += this_loss

    # Return the mean logistic loss.
    return sum_of_loss / num_examples

  def _convert_ratings_to_implicit_data(self, positive_pairs, num_negatives):
    """Converts a list of positive pairs into a two class dataset.

    Args:
      positive_pairs: an array of shape [n, 2], each row representing a positive
        user-item pair.
      num_negatives: the number of negative items to sample for each positive.
    Returns:
      An array of shape [n*(1 + num_negatives), 3], where each row is a tuple
      (user, item, label). The examples are obtained as follows:
      To each (user, item) pair in positive_pairs correspond:
      * one positive example (user, item, 1)
      * num_negatives negative examples (user, item', 0) where item' is sampled
        uniformly at random.
    """
    num_items = self.item_embedding.shape[0]
    num_pos_examples = positive_pairs.shape[0]
    training_matrix = np.empty([num_pos_examples * (1 + num_negatives), 3],
                               dtype=np.int32)
    index = 0
    for pos_index in range(num_pos_examples):
      u = positive_pairs[pos_index, 0]
      i = positive_pairs[pos_index, 1]

      # Treat the rating as a positive training instance
      training_matrix[index] = [u, i, 1]
      index += 1

      # Add N negatives by sampling random items.
      # This code does not enforce that the sampled negatives are not present in
      # the training data. It is possible that the sampling procedure adds a
      # negative that is already in the set of positives. It is also possible
      # that an item is sampled twice. Both cases should be fine.
      for _ in range(num_negatives):
        j = np.random.randint(num_items)
        training_matrix[index] = [u, j, 0]
        index += 1
    return training_matrix


def evaluate(model, test_ratings, test_negatives, K=10):
  """Helper that calls evaluate from the NCF libraries."""
  (hits, ndcgs) = evaluate_model(model, test_ratings, test_negatives, K=K,
                                 num_thread=1)
  return np.array(hits).mean(), np.array(ndcgs).mean()


def main():
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default='Data/ml-1m',
                      help='Path to the dataset')
  parser.add_argument('--epochs', type=int, default=128,
                      help='Number of training epochs')
  parser.add_argument('--embedding_dim', type=int, default=8,
                      help='Embedding dimensions, the first dimension will be '
                           'used for the bias.')
  parser.add_argument('--regularization', type=float, default=0.0,
                      help='L2 regularization for user and item embeddings.')
  parser.add_argument('--negatives', type=int, default=8,
                      help='Number of random negatives per positive examples.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='SGD step size.')
  parser.add_argument('--stddev', type=float, default=0.1,
                      help='Standard deviation for initialization.')
  args = parser.parse_args()

  # Load the dataset
  dataset = Dataset(args.data)
  train_pos_pairs = np.column_stack(dataset.trainMatrix.nonzero())
  test_ratings, test_negatives = (dataset.testRatings, dataset.testNegatives)
  print('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' % (
      dataset.num_users, dataset.num_items, train_pos_pairs.shape[0],
      len(test_ratings)))

  # Initialize the model
  model = MFModel(dataset.num_users, dataset.num_items,
                  args.embedding_dim-1, args.regularization, args.stddev)

  # Train and evaluate model
  hr, ndcg = evaluate(model, test_ratings, test_negatives, K=10)
  print('Epoch %4d:\t HR=%.4f, NDCG=%.4f\t'
        % (0, hr, ndcg))
  for epoch in range(args.epochs):
    # Training
    _ = model.fit(train_pos_pairs, learning_rate=args.learning_rate,
                  num_negatives=args.negatives)

    # Evaluation
    hr, ndcg = evaluate(model, test_ratings, test_negatives, K=10)
    print('Epoch %4d:\t HR=%.4f, NDCG=%.4f\t'
          % (epoch+1, hr, ndcg))


if __name__ == '__main__':
  main()
