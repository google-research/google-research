# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""How well can a MLP approximate a dot product.
"""

import argparse
import keras
import numpy as np

rmse_best = 0.85
rmse_naive = 1.13


def GenerateData(emb_dim, num_users, num_items, num_train_samples,
                 num_test_samples, num_fresh_samples):
  """Generates a data set where ground truth is a dot product with noise.

  Each generated training case x is a real valued vector of dimension
  2*embedding_dim (this is the concatenation of the two embedding vectors for
  which we want to learn the similarity) and a label y that encodes the
  similarity.

  The data is generated such that a perfect model (the dot product) will have an
  RMSE of rmse_best. The naive model that predicts always 0 will have an RMSE of
  rmse_naive. See the paper for more details.

  Args:
    emb_dim: the embedding dimension.
    num_users: the total number of users.
    num_items: the total number of items.
    num_train_samples: the size of the training set.
    num_test_samples: the size of the first test set.
    num_fresh_samples: the size of the second test set.

  Returns:
  Three datasets are created:
  * train: consists of pairs of user,item embeddings and their label. User and
           item embeddings are drawn from a fixed set of <num_user> and
           <num_items> embeddings
  * test:  same as train but with the constraint that train and test do not
           overlap
  * fresh: same as train but using fresh embeddings, i.e., embeddings are not
           limited to the <num_user> and <num_items> embeddings from train and
           test
  """

  # Calculate standard deviation of embedding distribution and noise
  # distribution such that the data will have the desired RMSE properties.
  sd_noise = rmse_best
  sd_emb = np.sqrt(np.sqrt((np.square(rmse_naive)
                            - np.square(rmse_best))/emb_dim))

  # Generate the embeddings:
  user_embs = np.random.normal(0, sd_emb, size=[num_users, emb_dim])
  item_embs = np.random.normal(0, sd_emb, size=[num_items, emb_dim])

  # Sample n combinations of user x item without replacement
  num_samples = num_train_samples + num_test_samples
  train_x = np.zeros([num_samples*2, 2, emb_dim], dtype=float)
  train_y = np.zeros([num_samples*2], dtype=float)
  sampling_prob = num_samples / (num_users * num_items)
  sampling_prob *= 1.1  # oversample to make sure we have enough samples
  counter = 0
  for u in range(num_users):
    for i in range(num_items):
      if np.random.uniform() < sampling_prob:
        user_emb = user_embs[u]
        item_emb = item_embs[i]
        train_x[counter][0] = user_emb
        train_x[counter][1] = item_emb
        train_y[counter] = (
            np.random.normal(0.0, sd_noise) + np.dot(user_emb, item_emb))
        counter = counter + 1
  counter = np.min([counter, num_samples])

  # discard any additional items
  train_x = train_x[:counter, :, :]
  train_y = train_y[:counter]

  # shuffle
  p = np.random.permutation(train_x.shape[0])
  train_x = train_x[p]
  train_y = train_y[p]

  # Split into 90% training, 10% testing
  train_x, test_x = np.split(train_x,
                             [int((counter * num_train_samples) / num_samples)])
  train_y, test_y = np.split(train_y,
                             [int((counter * num_train_samples) / num_samples)])

  # Second set of holdout interactions, i.e., embeddings are new:
  fresh_x = np.random.normal(0, sd_emb, size=[num_fresh_samples, 2, emb_dim])
  fresh_y = np.zeros([num_fresh_samples], dtype=float)
  for counter in range(num_fresh_samples):
    user_emb = fresh_x[counter][0]
    item_emb = fresh_x[counter][1]
    fresh_y[counter] = (
        np.random.normal(0.0, sd_noise) + np.dot(user_emb, item_emb))

  return train_x, train_y, test_x, test_y, fresh_x, fresh_y


def ComputeRMSE(x, y):
  """Computes the RMSE of a dot product model and a trivial model.

  Args:
    x: the input embeddings, a list of length n, such that x[i] is a pair of
      user and item embeddings.
    y: the labels, a list of length n.

  Returns:
    rmse_trivial: the RMSE of a trivial model that always predicts 0.
    rmse_dot: the RMSE of a dot product model.
  """
  sum_sqr_trivial = 0.0
  sum_sqr_dot = 0.0
  for i in range(x.shape[0]):
    label = y[i]

    prediction = 0
    diff = prediction - label
    sum_sqr_trivial = sum_sqr_trivial + diff * diff

    user_emb = x[i][0]
    item_emb = x[i][1]
    prediction = np.dot(user_emb, item_emb)
    diff = prediction - label
    sum_sqr_dot = sum_sqr_dot + diff * diff

  rmse_trivial = np.sqrt(sum_sqr_trivial / x.shape[0])
  rmse_dot = np.sqrt(sum_sqr_dot / x.shape[0])
  return rmse_trivial, rmse_dot


def TrainMLP(train_x, train_y, test_x, test_y, fresh_x, fresh_y, emb_dim,
             epochs, batch_size, learning_rate, first_layer_mult):
  """Trains a MLP and computes its RMSE on the given datasets."""
  layer_num_hidden = [first_layer_mult * emb_dim*2,
                      first_layer_mult * emb_dim*1,
                      int(first_layer_mult * emb_dim / 2)]

  model = keras.models.Sequential()
  model.add(keras.layers.Flatten(input_shape=(2, emb_dim)))
  for hidden in layer_num_hidden:
    # as suggested in the paper
    model.add(keras.layers.Dense(hidden, activation='relu'))
  model.add(keras.layers.Dense(1, activation='linear'))
  model.summary()
  model.compile(
      loss='mean_squared_error',
      optimizer=keras.optimizers.Adam(lr=learning_rate))  # as suggested

  model.fit(
      train_x, train_y,
      batch_size=batch_size,
      epochs=epochs,
      verbose=2,
      validation_data=(test_x, test_y))

  rmse_train = np.sqrt(model.evaluate(train_x, train_y, verbose=2))
  rmse_test = np.sqrt(model.evaluate(test_x, test_y, verbose=2))
  rmse_fresh = np.sqrt(model.evaluate(fresh_x, fresh_y, verbose=2))

  return rmse_train, rmse_test, rmse_fresh


def main():
  # Command line arguments
  parser = argparse.ArgumentParser()
  # Data related
  parser.add_argument('--embedding_dim', type=int, default=64,
                      help='Embedding dimensions')
  parser.add_argument('--num_users', type=int, default=100000,
                      help='Number of users')
  parser.add_argument('--num_items', type=int, default=100000,
                      help='Number of items')
  # MLP related
  parser.add_argument(
      '--first_layer_mult', type=int, default=1,
      help='The first layer size is 2*emb_dim*first_layer_mult.')
  parser.add_argument('--epochs', type=int, default=32,
                      help='Number of training epochs')
  parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for training')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
  args = parser.parse_args()

  # Generate data and print some statistics
  num_samples = args.num_users * 100      # 100 items per user on average
  train_x, train_y, test_x, test_y, fresh_x, fresh_y = GenerateData(
      emb_dim=args.embedding_dim,
      num_users=args.num_users,
      num_items=args.num_items,
      num_train_samples=int(num_samples*0.9),
      num_test_samples=int(num_samples*0.1), num_fresh_samples=100000)
  print('Num training examples: ', train_x.shape)
  print('Num test examples: ', test_x.shape)
  print('Num fresh examples: ', fresh_x.shape)

  # Evaluate trivial model and dot product model.
  rmse_train_naive, rmse_train_dot = ComputeRMSE(train_x, train_y)
  rmse_test_naive, rmse_test_dot = ComputeRMSE(test_x, test_y)
  rmse_fresh_naive, rmse_fresh_dot = ComputeRMSE(fresh_x, fresh_y)
  print('Trivial model training RMSE: ', rmse_train_naive)
  print('Trivial model test RMSE: ', rmse_test_naive)
  print('Trivial model fresh RMSE: ', rmse_fresh_naive)
  print('Dot training RMSE: ', rmse_train_dot)
  print('Dot test RMSE: ', rmse_test_dot)
  print('Dot fresh RMSE: ', rmse_fresh_dot)

  # Train and evaluate MLP model.
  train_rmse, test_rmse, fresh_rmse = TrainMLP(
      train_x, train_y, test_x, test_y, fresh_x, fresh_y,
      emb_dim=args.embedding_dim, epochs=args.epochs,
      batch_size=args.batch_size, learning_rate=args.learning_rate,
      first_layer_mult=args.first_layer_mult)
  print('MLP training RMSE: ', train_rmse)
  print('MLP test RMSE: ', test_rmse)
  print('MLP fresh RMSE: ', fresh_rmse)

  # Print stats.
  print('\t'.join(map(str, [args.embedding_dim, args.num_users, args.num_items,
                            train_rmse, rmse_train_naive, rmse_train_dot,
                            test_rmse, rmse_test_naive, rmse_test_dot,
                            fresh_rmse, rmse_fresh_naive, rmse_fresh_dot])))

if __name__ == '__main__':
  main()
