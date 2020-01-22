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

"""Helper file to run the discover concept algorithm in the toy dataset."""
# lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from absl import app
import keras.backend as K
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.models import Model

from keras.optimizers import Adam
from keras.optimizers import SGD
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from skimage.segmentation import felzenszwalb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tensorflow.compat.v1 import set_random_seed

seed(0)
set_random_seed(0)
batch_size = 128


def load_xyconcept(n, pretrain):
  """Loads data and create label for toy dataset."""
  concept = np.load('concept_data.npy')
  y = np.zeros((n, 15))
  y[:, 0] = ((1 - concept[:, 0] * concept[:, 2]) + concept[:, 3]) > 0
  y[:, 1] = concept[:, 1] + (concept[:, 2] * concept[:, 3])
  y[:, 2] = (concept[:, 3] * concept[:, 4]) + (concept[:, 1] * concept[:, 2])
  y[:, 3] = np.bitwise_xor(concept[:, 0], concept[:, 1])
  y[:, 4] = concept[:, 1] + concept[:, 4]
  y[:, 5] = (1 - (concept[:, 0] + concept[:, 3] + concept[:, 4])) > 0
  y[:, 6] = np.bitwise_xor(concept[:, 1] * concept[:, 2], concept[:, 4])
  y[:, 7] = concept[:, 0] * concept[:, 4] + concept[:, 1]
  y[:, 8] = concept[:, 2]
  y[:, 9] = np.bitwise_xor(concept[:, 0] + concept[:, 1], concept[:, 3])
  y[:, 10] = (1 - (concept[:, 2] + concept[:, 4])) > 0
  y[:, 11] = concept[:, 0] + concept[:, 3] + concept[:, 4]
  y[:, 12] = np.bitwise_xor(concept[:, 1], concept[:, 2])
  y[:, 13] = (1 - (concept[:, 0] * concept[:, 4] + concept[:, 3])) > 0
  y[:, 14] = np.bitwise_xor(concept[:, 4], concept[:, 3])
  if not pretrain:
    x = np.load('x_data.npy') / 255.0
    return x, y, concept
  return 0, y, concept


def target_category_loss(x, category_index, nb_classes):
  return x * K.one_hot([category_index], nb_classes)


def load_model(x_train, y_train, x_val, y_val, width=216, \
               height=216, channel=3, pretrain=True):
  """Loads pretrain model or train one."""
  input1 = Input(
      shape=(
          width,
          height,
          channel,
      ), name='concat_input')
  conv1 = Conv2D(64, kernel_size=3, activation='relu')
  conv2 = Conv2D(64, kernel_size=3, activation='relu')
  conv3 = Conv2D(32, kernel_size=3, activation='relu')
  conv4 = Conv2D(32, kernel_size=3, activation='relu')
  conv5 = Conv2D(16, kernel_size=3, activation='relu')
  dense1 = Dense(200, activation='relu')
  dense2 = Dense(100, activation='relu')
  predict = Dense(15, activation='sigmoid')
  conv1 = conv1(input1)
  conv2 = conv2(conv1)
  conv3 = conv3(conv2)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = conv4(pool1)
  conv5 = conv5(conv4)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv5)
  pool2f = Flatten()(pool2)
  fc1 = dense1(pool2f)
  fc2 = dense2(fc1)
  softmax1 = predict(fc2)

  mlp = Model(input1, softmax1)
  if pretrain:
    mlp.load_weights('conv_s13.h5')
  mlp.compile(
      loss='binary_crossentropy',
      optimizer=Adam(lr=0.0001),
      metrics=['binary_accuracy'])
  if not pretrain:
    _ = mlp.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=5,
        verbose=1,
        validation_data=(x_val, y_val))
    mlp.save_weights('conv_s13.h5')
  for layer in mlp.layers:
    layer.trainable = False
  feature_dense_model = Model(input1, fc1)
  return dense2, predict, feature_dense_model


def get_ace_concept(concept_arraynew_active, dense2, predict, f_train,
                    n_concept):
  """Calculates ACE/TCAV concepts."""
  concept_input = Input(shape=(200,), name='concept_input')
  fc2_tcav = dense2(concept_input)
  softmax_tcav = predict(fc2_tcav)
  tcav_model = Model(inputs=concept_input, outputs=softmax_tcav)
  tcav_model.layers[-1].activation = None
  tcav_model.layers[-1].trainable = False
  tcav_model.layers[-2].trainable = False
  tcav_model.compile(
      loss='mean_squared_error',
      optimizer=SGD(lr=0.0),
      metrics=['binary_accuracy'])
  tcav_model.summary()

  n_cluster = concept_arraynew_active.shape[0]
  n_percluster = concept_arraynew_active.shape[1]
  print(concept_arraynew_active.shape)
  weight_ace = np.zeros((200, n_cluster))
  tcav_list_rand = np.zeros((15, 200))
  tcav_list_ace = np.zeros((15, n_cluster))
  for i in range(n_cluster):
    y = np.zeros((n_cluster * n_percluster))
    y[i * n_percluster:(i + 1) * n_percluster] = 1
    clf = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        max_iter=10000,
        C=10.0,
        multi_class='ovr').fit(concept_arraynew_active.reshape((-1, 200)), y)
    weight_ace[:, i] = clf.coef_

  weight_rand = np.zeros((200, 200))
  for i in range(200):
    y = np.random.randint(2, size=n_cluster * n_percluster)
    clf = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        max_iter=10000,
        C=10.0,
        multi_class='ovr').fit(concept_arraynew_active.reshape((-1, 200)), y)
    weight_rand[:, i] = clf.coef_

  sig_list = np.zeros(n_cluster)

  for j in range(15):
    grads = (
        K.gradients(target_category_loss(softmax_tcav, j, 15),
                    concept_input)[0])
    gradient_function = K.function([tcav_model.input], [grads])
    grads_val = gradient_function([f_train])[0]
    grad_rand = np.matmul(grads_val, weight_rand)
    grad_ace = np.matmul(grads_val, weight_ace)
    tcav_list_rand[j, :] = np.sum(grad_rand > 0.000, axis=(0))
    tcav_list_ace[j, :] = np.sum(grad_ace > 0.000, axis=(0))
    mean = np.mean(tcav_list_rand[j, :])
    std = np.std(tcav_list_rand[j, :])
    sig_list += (tcav_list_ace[j, :] > mean + std * 1.0).astype(int)
  top_k_index = np.array(sig_list).argsort()[-1 * n_concept:][::-1]
  print(sig_list)
  print(top_k_index)
  return weight_ace[:, top_k_index]


def get_pca_concept(f_train, n_concept):
  pca = PCA()
  pca.fit(f_train)
  weight_pca = np.zeros((200, n_concept))
  for count, pc in enumerate(pca.components_):
    if count >= n_concept:
      break
    weight_pca[:, count] = pc
  return weight_pca


def create_dataset(n_sample=60000):
  """Creates toy dataset and save to disk."""
  concept = np.reshape(np.random.randint(2, size=15 * n_sample),
                       (-1, 15)).astype(np.bool_)
  concept[:15, :15] = np.eye(15)
  fig = Figure(figsize=(3, 3))
  canvas = FigureCanvas(fig)
  axes = fig.gca()
  axes.set_xlim([0, 10])
  axes.set_ylim([0, 10])
  axes.axis('off')
  width, height = fig.get_size_inches() * fig.get_dpi()
  width = int(width)
  height = int(height)
  location = [(1.3, 1.3), (3.3, 1.3), (5.3, 1.3), (7.3, 1.3), (1.3, 3.3),
              (3.3, 3.3), (5.3, 3.3), (7.3, 3.3), (1.3, 5.3), (3.3, 5.3),
              (5.3, 5.3), (7.3, 5.3), (1.3, 7.3), (3.3, 7.3), (5.3, 7.3)]
  location_bool = np.zeros(15)
  x = np.zeros((n_sample, width, height, 3))
  color_array = ['green', 'red', 'blue', 'black', 'orange', 'purple', 'yellow']

  for i in range(n_sample):
    if i % 1000 == 0:
      print('{} images are created'.format(i))
    if concept[i, 5] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'x',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 6] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '3',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 7] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          's',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 8] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'p',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 9] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '_',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 10] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'd',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 11] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'd',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 12] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          11,
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 13] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'o',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 14] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '.',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 0] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '+',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 1] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '1',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 2] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '*',
          color=color_array[np.random.randint(100) % 7],
          markersize=30,
          mew=3,
          ms=5)
    if concept[i, 3] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          '<',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    if concept[i, 4] == 1:
      a = np.random.randint(15)
      while location_bool[a] == 1:
        a = np.random.randint(15)
      location_bool[a] = 1
      axes.plot(
          location[a][0],
          location[a][1],
          'h',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4,
          ms=8)
    canvas.draw()
    image = np.fromstring(
        canvas.tostring_rgb(), dtype='uint8').reshape(width, height, 3)
    x[i, :, :, :] = image
    # imgplot = plt.imshow(image)
    # plt.show()

  # create label by booling functions
  y = np.zeros((n_sample, 15))
  y[:, 0] = ((1 - concept[:, 0] * concept[:, 2]) + concept[:, 3]) > 0
  y[:, 1] = concept[:, 1] + (concept[:, 2] * concept[:, 3])
  y[:, 2] = (concept[:, 3] * concept[:, 4]) + (concept[:, 1] * concept[:, 2])
  y[:, 3] = np.bitwise_xor(concept[:, 0], concept[:, 1])
  y[:, 4] = concept[:, 1] + concept[:, 4]
  y[:, 5] = (1 - (concept[:, 0] + concept[:, 3] + concept[:, 4])) > 0
  y[:, 6] = np.bitwise_xor(concept[:, 1] * concept[:, 2], concept[:, 4])
  y[:, 7] = concept[:, 0] * concept[:, 4] + concept[:, 1]
  y[:, 8] = concept[:, 2]
  y[:, 9] = np.bitwise_xor(concept[:, 0] + concept[:, 1], concept[:, 3])
  y[:, 10] = (1 - (concept[:, 2] + concept[:, 4])) > 0
  y[:, 11] = concept[:, 0] + concept[:, 3] + concept[:, 4]
  y[:, 12] = np.bitwise_xor(concept[:, 1], concept[:, 2])
  y[:, 13] = (1 - (concept[:, 0] * concept[:, 4] + concept[:, 3])) > 0
  y[:, 14] = np.bitwise_xor(concept[:, 4], concept[:, 3])

  np.save('x_data.npy', x)
  np.save('y_data.npy', y)
  np.save('concept_data.npy', concept)

  return width, height


def get_groupacc(finetuned_model_pr, concept_arraynew2, f_train, f_val, concept,
                 n_concept, n_cluster, n0, verbose):
  """Gets the group accuracy for dicovered concepts."""
  print(finetuned_model_pr.summary())
  min_weight = finetuned_model_pr.layers[-5].get_weights()[0]
  sim_array = np.zeros((n_cluster, n_concept))
  for j in range(n_cluster):
    sim_array[j, :] = np.mean(
        np.matmul(concept_arraynew2[j, :100, :], min_weight), axis=0)

  posneg = np.zeros(5)
  sim_array_0mean = sim_array - np.mean(sim_array, axis=0)
  max_cluster = np.argmax(np.abs(sim_array_0mean), axis=0)
  for count in range(5):
    posneg[count] = sim_array_0mean[max_cluster[count], count] > 0
  loss_table = np.zeros((5, 5))
  for count in range(5):
    for count2 in range(5):
      # count2 = max_cluster[count]
      mean0 = np.mean(
          np.matmul(f_train, min_weight[:, count])[concept[:n0,
                                                           count2] == 0]) * 100
      mean1 = np.mean(
          np.matmul(f_train, min_weight[:, count])[concept[:n0,
                                                           count2] == 1]) * 100

      if mean0 < mean1:
        pos = 1
      else:
        pos = -1
      best_err = 1e10
      best_bias = 0
      a = int((mean1 - mean0) / 20)
      if a == 0:
        a = pos
      for bias in range(int(mean0), int(mean1), a):
        if pos == 1:
          if np.sum(
              np.bitwise_xor(
                  concept[:n0, count2],
                  np.matmul(f_train, min_weight[:, count]) >
                  bias / 100.)) < best_err:
            best_err = np.sum(
                np.bitwise_xor(
                    concept[:n0, count2],
                    np.matmul(f_train, min_weight[:, count]) > bias / 100.))
            best_bias = bias
        else:
          if np.sum(
              np.bitwise_xor(
                  concept[:n0, count2],
                  np.matmul(f_train, min_weight[:, count]) <
                  bias / 100.)) < best_err:
            best_err = np.sum(
                np.bitwise_xor(
                    concept[:n0, count2],
                    np.matmul(f_train, min_weight[:, count]) < bias / 100.))
            best_bias = bias
      if pos == 1:
        loss_table[count, count2] = np.sum(
            np.bitwise_xor(
                concept[n0:, count2],
                np.matmul(f_val, min_weight[:, count]) >
                best_bias / 100.)) / 12000
        if verbose:
          print(np.sum(
              np.bitwise_xor(
                  concept[n0:, count2],
                  np.matmul(f_val, min_weight[:, count]) > best_bias / 100.))
                /12000)
      else:
        loss_table[count, count2] = np.sum(
            np.bitwise_xor(
                concept[n0:, count2],
                np.matmul(f_val, min_weight[:, count]) <
                best_bias / 100.)) / 12000
        if verbose:
          print(np.sum(
              np.bitwise_xor(
                  concept[n0:, count2],
                  np.matmul(f_val, min_weight[:, count]) < best_bias / 100.))
                /12000)
  print(np.amin(loss_table, axis=0))
  acc = np.mean(np.amin(loss_table, axis=0))
  print(acc)
  return min_weight, acc


def create_feature(x, width, height, feature_dense_model):
  """Saves embedding to disk to enhance computation."""
  feature_sp = np.zeros((40000 * 10, 200))
  end = 0
  group_array = []
  for i in range(48000):
    if i % 1000 == 0:
      print('{} embeddings are created'.format(i))
    img = x[i, :, :, :]
    _ = plt.imshow(img)
    # plt.show()
    segments_fz = felzenszwalb(img, scale=100, sigma=.2, min_size=50)
    segments = len(np.unique(segments_fz))
    temp_arr = np.ones((segments, width, height, 3))
    for j in range(segments):
      temp_arr[j, segments_fz == j, :] = img[segments_fz == j, :]
      # imgplot = plt.imshow(temp_arr[j,:,:,:])
      # plt.show()
    aa = feature_dense_model.predict(temp_arr)
    if i <= 40000:
      feature_sp[end:end + segments, :] = aa
      end += segments
    group_array.append(aa)
  feature_sp = feature_sp[:end, :]
  all_feature_dense = feature_dense_model.predict(x)
  with open('group_array.pickle', 'wb') as handle:
    pickle.dump(group_array, handle, protocol=pickle.highest_protocol)
  np.save('all_feature_dense.npy', all_feature_dense)
  np.save('feature_sp.npy', feature_sp)


def create_cluster(concept):
  """Creates a self-discovered clustering."""
  with open('group_array.pickle', 'rb') as handle:
    group_array = pickle.load(handle)
  feature_sp = np.load('feature_sp.npy')
  all_feature_dense = np.load('all_feature_dense.npy')
  kmeans = KMeans(n_clusters=20, random_state=0).fit(feature_sp[:100000])
  concept_new = np.zeros((10000, 20))
  for i in range(10000):
    temp_cluster = kmeans.predict(group_array[i])
    concept_new[i, temp_cluster] = 1

  concept_arraynew = np.zeros((20, 300, 200))
  # Returns concepts found in unsupervised way.
  for i in range(20):
    print(i)
    concept_arraynew[i, :] = all_feature_dense[:10000, :][concept_new[:, i] ==
                                                          1, :][:300, :]
  concept_arraynew2 = np.zeros((15, 300, 200))

  # Returns concepts found in supervised way.
  for i in range(15):
    concept_arraynew2[i, :] = all_feature_dense[:60000, :][concept[:, i] ==
                                                           1, :][:300, :]
  np.save('concept_arraynew.npy', concept_arraynew)
  np.save('concept_arraynew2.npy', concept_arraynew2)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
