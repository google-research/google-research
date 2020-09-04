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

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import data
import model
# pylint: skip-file


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--real_path', default='../data/resize128')
parser.add_argument('--fake_path', default='../data/fakedata')
parser.add_argument('--train_label', default='../data/annotations/train_label.txt')
parser.add_argument('--test_label', default='../data/annotations/test_label.txt')
parser.add_argument('--valid_label', default='../data/annotations/val_label.txt')
parser.add_argument('--log_dir', default='test_log', help='Log dir [default: log]')
parser.add_argument('--n_episode', type=int, default=500, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size during training [default: 64]')
parser.add_argument('--n_class', type=int, default=2, help='Number of class [default: 2]')
parser.add_argument('--n_action', type=int, default=2, help='Number of action [default: 2]')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate [default: 0.1]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='momentum', help='adam or momentum [default: momentum]')
FLAGS = parser.parse_args()

#####################  config  #####################

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
tf.set_random_seed(100) # tf.set_random_seed(0)# 0 for 512
REAL_PATH = FLAGS.real_path
FAKE_PATH = FLAGS.fake_path
TRAIN_LABEL = FLAGS.train_label
TEST_LABEL = FLAGS.test_label
VALID_LABEL = FLAGS.valid_label
BATCH_SIZE = FLAGS.batch_size
N_EPISODE = FLAGS.n_episode
N_CLASS = FLAGS.n_class
N_ACTION = FLAGS.n_action
LR = FLAGS.lr
MOMENTUM = FLAGS.momentum
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
LOG_PATH = os.path.join(ROOT_PATH, FLAGS.log_dir)
if not os.path.exists(LOG_PATH): os.mkdir(LOG_PATH)
acc_count = 0
while True:
  if os.path.exists(os.path.join(LOG_PATH, 'log_%02d.txt' % acc_count)): acc_count += 1
  else: break
LOG_FNAME = 'log_%02d.txt' % acc_count
LOG_FOUT = open(os.path.join(LOG_PATH, LOG_FNAME), 'w')

(train_images, train_labels, train_att), train_iters = data.data_train(REAL_PATH, TRAIN_LABEL, BATCH_SIZE)
(fake_images, fake_labels, fake_att), fake_iters = data.data_train(FAKE_PATH, TRAIN_LABEL, BATCH_SIZE)
(valid_images, valid_labels, valid_att), valid_iters = data.data_test(REAL_PATH, VALID_LABEL, BATCH_SIZE)
(test_images, test_labels, test_att), test_iters = data.data_test(REAL_PATH, TEST_LABEL, BATCH_SIZE)

####################################################

def log_string(out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)

def choose_action(prob_actions):
  actions = []
  for i in range(prob_actions.shape[0]):
    action = np.random.choice(range(prob_actions.shape[1]), p=prob_actions[i])
    actions.append(action)
  return np.array(actions)

def vgg_graph(sess, phs):
  VGG = model.VGG()
  Y_score = VGG.build(phs['batch_images'], N_CLASS, phs['is_training_ph'])

  Y_hat = tf.nn.softmax(Y_score)
  Y_pred = tf.argmax(Y_hat, 1)
  Y_label = tf.to_float(tf.one_hot(phs['batch_labels'], N_CLASS))

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Y_score, labels = Y_label)
  loss_op = tf.reduce_mean(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y_label, 1))
  acc_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  update_op = tf.train.MomentumOptimizer(LR, MOMENTUM).minimize(loss_op, var_list=VGG.vars)

  return loss_op, acc_op, cross_entropy, Y_hat, update_op, Y_pred, VGG.vars

def rl_graph(sess, phrl):
  Actor = model.Actor()
  Y_score = Actor.build(phrl['states_rl'], N_ACTION, phrl['is_training_rl'])
  Y_prob =tf.nn.softmax(Y_score)


  neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = Y_score, labels = phrl['actions_rl'])
  loss_op = tf.reduce_mean(neg_log_prob*phrl['values_rl'])

  # update_op = tf.train.MomentumOptimizer(LR, MOMENTUM).minimize(loss_op, var_list=Actor.vars)
  update_op = tf.train.AdamOptimizer(1e-3).minimize(loss_op, var_list=Actor.vars)

  return loss_op, Y_prob, update_op, Actor.vars

def train():
  batch_images = tf.placeholder(tf.float32,[None,128,128,3])
  batch_labels = tf.placeholder(tf.int32,[None,])
  is_training_ph = tf.placeholder(tf.bool)
  lr_ph = tf.placeholder(tf.float32)

  states_rl = tf.placeholder(tf.float32,[None,11])
  actions_rl = tf.placeholder(tf.int32,[None,])
  values_rl = tf.placeholder(tf.float32,[None,])
  is_training_rl = tf.placeholder(tf.bool)
  lr_rl = tf.placeholder(tf.float32)

  phs = {'batch_images': batch_images,
       'batch_labels': batch_labels,
       'is_training_ph': is_training_ph,
       'lr_ph': lr_ph}

  phrl = {'states_rl': states_rl,
       'actions_rl': actions_rl,
       'values_rl': values_rl,
       'is_training_rl': is_training_rl,
       'lr_rl': lr_rl}

  with tf.Session() as sess:
    vgg_loss, vgg_acc, vgg_ce, vgg_prob, vgg_update, vgg_pred, vgg_vars = vgg_graph(sess, phs)
    rl_loss, rl_prob, rl_update, rl_vars = rl_graph(sess, phrl)
    vgg_init = tf.variables_initializer(var_list=vgg_vars)
    saver = tf.train.Saver(vgg_vars)
    all_saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)




    for i in range(N_EPISODE):
      # sess.run(vgg_init)
      all_saver.restore(sess,LOG_PATH+'/all.ckpt')
      saver.restore(sess,LOG_PATH+'/vgg.ckpt')
      # state_list = []
      # action_list = []
      # reward_list = []
      for j in range(train_iters*20):
        tr_images, tr_labels, tr_att = sess.run([train_images,train_labels, train_att])
        fa_images, fa_labels, fa_att = sess.run([fake_images,fake_labels, fake_att])

        train_dict = {phs['batch_images']: tr_images,
                phs['batch_labels']: tr_labels,
                phs['is_training_ph']: False}
        ce, acc, prob, pred = sess.run([vgg_ce, vgg_acc, vgg_prob, vgg_pred], feed_dict=train_dict)
        ce = np.clip(ce, 0, 10)/10.0
        model_stat = list(data.cal_eo(tr_att, tr_labels, pred))
        model_stat.append(np.mean(ce))
        model_stat = np.tile(model_stat,(BATCH_SIZE,1))
        state = np.concatenate((tr_labels[:, np.newaxis], tr_att[:, np.newaxis], prob, ce[:, np.newaxis], model_stat), axis=1)



        rl_dict = {phrl['states_rl']: state,
               phrl['is_training_rl']: False}
        action = choose_action(sess.run(rl_prob, feed_dict=rl_dict))



        bool_train = list(map(bool,action))
        bool_fake = list(map(bool,1-action))
        co_images = np.concatenate((tr_images[bool_train],fa_images[bool_fake]),axis=0)
        co_labels = np.concatenate((tr_labels[bool_train],fa_labels[bool_fake]),axis=0)


        update_dict = {phs['batch_images']: co_images,
                phs['batch_labels']: co_labels,
                phs['is_training_ph']: True}
        _, ce, acc = sess.run([vgg_update, vgg_ce, vgg_acc], feed_dict=update_dict)


        if j % 100 == 0:
          print('====epoch_%d====iter_%d: loss=%.4f, train_acc=%.4f' % (i, j, np.mean(ce), acc))
          print(action, np.sum(action))


      valid_acc = 0.0
      y_pred =[]
      y_label = []
      y_att = []
      for k in range(valid_iters):
        va_images, va_labels, va_att = sess.run([valid_images, valid_labels, valid_att])
        valid_dict = {phs['batch_images']: va_images,
                phs['batch_labels']: va_labels,
                phs['is_training_ph']: False}
        batch_acc, batch_pred = sess.run([vgg_acc,vgg_pred], feed_dict=valid_dict)
        valid_acc += batch_acc
        y_pred += batch_pred.tolist()
        y_label += va_labels.tolist()
        y_att += va_att.tolist()
      valid_acc = valid_acc / float(valid_iters)
      valid_eo = data.cal_eo(y_att, y_label, y_pred)
      log_string('====epoch_%d: valid_acc=%.4f, valid_eo=%.4f' % (i, valid_acc, valid_eo[-1]))
      print('eo: ',valid_eo[0],valid_eo[1])
      print('eo: ',valid_eo[2],valid_eo[3])


if __name__ == "__main__":
  train()
  LOG_FOUT.close()
