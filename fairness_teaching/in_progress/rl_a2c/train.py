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

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import data
import model
# pylint: skip-file

# from data import *
# from architecture import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--real_path', default='/localscratch/wliu328/att/resize128')
# parser.add_argument('--real_path', default='/localscratch/wliu328/att/big/data')
parser.add_argument('--fake_path', default='/localscratch/wliu328/att/output/AttGAN_128/samples_training_2')
parser.add_argument('--train_label', default='/localscratch/wliu328/att/annotations/train_label.txt')
parser.add_argument('--test_label', default='/localscratch/wliu328/att/annotations/test_label.txt')
parser.add_argument('--valid_label', default='/localscratch/wliu328/att/annotations/val_label.txt')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--n_episode', type=int, default=500, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size during training [default: 64]')
parser.add_argument('--n_class', type=int, default=2, help='Number of class [default: 2]')
parser.add_argument('--n_action', type=int, default=2, help='Number of action [default: 2]')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate [default: 0.1]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='momentum', help='adam or momentum [default: momentum]')
parser.add_argument('--random_seed', type=int, default=100,help='random seed')
FLAGS = parser.parse_args()

#####################  config  #####################

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
tf.set_random_seed(FLAGS.random_seed)
np.random.seed(FLAGS.random_seed)
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
(valid_images, valid_labels, valid_att), valid_iters = data.data_test(REAL_PATH, VALID_LABEL, BATCH_SIZE*10)
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
  entropy = tf.reduce_sum(tf.reduce_mean(Y_prob)*tf.math.log(Y_prob))

  neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = Y_score, labels = phrl['actions_rl'])
  loss_op = tf.reduce_mean(neg_log_prob*phrl['values_rl'])
  reg_loss = tf.reduce_sum(Actor.reg_loss)
  loss_op += reg_loss
  loss_op += 1e-3 * entropy
  # update_op = tf.train.MomentumOptimizer(LR, MOMENTUM).minimize(loss_op, var_list=Actor.vars)
  update_op = tf.train.AdamOptimizer(1e-4).minimize(loss_op, var_list=Actor.vars)

  return loss_op, Y_prob, update_op, Actor.vars

def c_graph(sess, phc):
  Critic = model.Critic()

  Y_value = Critic.build(phc['states_c'], phc['is_training_c'])
  loss_op = tf.reduce_mean(tf.square(Y_value-phc['values_c']))
  reg_loss = tf.reduce_sum(Critic.reg_loss)
  loss_op += reg_loss
  # update_op = tf.train.MomentumOptimizer(LR, MOMENTUM).minimize(loss_op, var_list=Critic.vars)
  update_op = tf.train.AdamOptimizer(1e-3).minimize(loss_op, var_list=Critic.vars)

  return loss_op, Y_value, update_op, Critic.vars

def train():
  batch_images = tf.placeholder(tf.float32,[None,128,128,3])
  batch_labels = tf.placeholder(tf.int32,[None,])
  is_training_ph = tf.placeholder(tf.bool)
  lr_ph = tf.placeholder(tf.float32)

  states_rl = tf.placeholder(tf.float32,[None,2])
  actions_rl = tf.placeholder(tf.int32,[None,])
  values_rl = tf.placeholder(tf.float32,[None,])
  is_training_rl = tf.placeholder(tf.bool)
  lr_rl = tf.placeholder(tf.float32)

  states_c = tf.placeholder(tf.float32,[None,7])
  values_c = tf.placeholder(tf.float32,[None,])
  is_training_c = tf.placeholder(tf.bool)
  lr_c= tf.placeholder(tf.float32)

  phs = {'batch_images': batch_images,
       'batch_labels': batch_labels,
       'is_training_ph': is_training_ph,
       'lr_ph': lr_ph}

  phrl = {'states_rl': states_rl,
       'actions_rl': actions_rl,
       'values_rl': values_rl,
       'is_training_rl': is_training_rl,
       'lr_rl': lr_rl}

  phc = {'states_c': states_c,
       'values_c': values_c,
       'is_training_c': is_training_c,
       'lr_c': lr_c}
  with tf.Session() as sess:
    vgg_loss, vgg_acc, vgg_ce, vgg_prob, vgg_update, vgg_pred, vgg_vars = vgg_graph(sess, phs)
    rl_loss, rl_prob, rl_update, rl_vars = rl_graph(sess, phrl)
    c_loss, c_value, c_update, c_vars = c_graph(sess, phc)
    vgg_init = tf.variables_initializer(var_list=vgg_vars)
    saver = tf.train.Saver(vgg_vars)
    all_saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)


    # for epoch in range(4):
    #   for t in range(train_iters):
    #     if t % 50==0: print("pretrain:", t)
    #     tr_images, tr_labels = sess.run([train_images,train_labels])
    #     pre_dict = {phs['batch_images']: tr_images,
    #           phs['batch_labels']: tr_labels,
    #           phs['is_training_ph']: True}
    #     sess.run(vgg_update, feed_dict=pre_dict)
    # saver.save(sess,LOG_PATH+'/vgg.ckpt')
    # valid_acc = 0.0
    # y_pred =[]
    # y_label = []
    # y_att = []
    # for k in range(valid_iters):
    #   va_images, va_labels, va_att = sess.run([valid_images, valid_labels, valid_att])
    #   valid_dict = {phs['batch_images']: va_images,
    #           phs['batch_labels']: va_labels,
    #           phs['is_training_ph']: False}
    #   batch_acc, batch_pred = sess.run([vgg_acc,vgg_pred], feed_dict=valid_dict)
    #   valid_acc += batch_acc
    #   y_pred += batch_pred.tolist()
    #   y_label += va_labels.tolist()
    #   y_att += va_att.tolist()
    # valid_acc = valid_acc / float(valid_iters)
    # valid_eo = data.cal_eo(y_att, y_label, y_pred)
    # log_string('====pretrain: valid_acc=%.4f, valid_eo=%.4f' % (valid_acc, valid_eo[-1]))
    # print(valid_eo)



    va_images, va_labels, va_att = sess.run([valid_images, valid_labels, valid_att])
    for i in range(N_EPISODE):
      sess.run(vgg_init)
      # saver.restore(sess,LOG_PATH+'/vgg.ckpt')
      train_loss = []
      for j in range(train_iters*20):
        tr_images, tr_labels, tr_att = sess.run([train_images,train_labels, train_att])
        fa_images, fa_labels, fa_att = sess.run([fake_images,fake_labels, fake_att])


        train_dict = {phs['batch_images']: tr_images,
                phs['batch_labels']: tr_labels,
                phs['is_training_ph']: False}
        ce, acc, prob, pred = sess.run([vgg_ce, vgg_acc, vgg_prob, vgg_pred], feed_dict=train_dict)
        ce = np.clip(ce, 0, 10)/10.0
        train_loss.append(np.mean(ce))
        model_stat = list(data.cal_eo(tr_att, tr_labels, pred)) #shape [5,]
        model_stat.append(np.mean(ce))
        model_stat.append(j/(train_iters*20))
        # model_stat.append(np.mean(train_loss))
        c_state = np.array(model_stat)[np.newaxis,:]

        # model_stat = np.tile(model_stat,(BATCH_SIZE,1))
        state = np.concatenate((tr_labels[:, np.newaxis], tr_att[:, np.newaxis]), axis=1)



        rl_dict = {phrl['states_rl']: state,
               phrl['is_training_rl']: False}
        action = choose_action(sess.run(rl_prob, feed_dict=rl_dict))

        c_dict = {phc['states_c']: c_state,
              phc['is_training_c']: False}
        base = sess.run(c_value, feed_dict=c_dict)


        bool_train = list(map(bool,action))
        bool_fake = list(map(bool,1-action))
        co_images = np.concatenate((tr_images[bool_train],fa_images[bool_fake]),axis=0)
        co_labels = np.concatenate((tr_labels[bool_train],fa_labels[bool_fake]),axis=0)


        update_dict = {phs['batch_images']: co_images,
                phs['batch_labels']: co_labels,
                phs['is_training_ph']: True}
        _, ce, acc = sess.run([vgg_update, vgg_ce, vgg_acc], feed_dict=update_dict)




        valid_dict = {phs['batch_images']: va_images,
                phs['batch_labels']: va_labels,
                phs['is_training_ph']: False}
        valid_acc, y_pred = sess.run([vgg_acc,vgg_pred], feed_dict=valid_dict)
        valid_eo = data.cal_eo(va_att, va_labels, y_pred)
        if valid_eo[-1]<=0.05:
          value = -2
        else:
          value = -np.log(valid_eo[-1])
        reward = value-base[0]

        c_dict = {phc['states_c']: c_state,
              phc['values_c']: [value],
              phc['is_training_c']: True}
        _, cri_loss = sess.run([c_update, c_loss], feed_dict=c_dict)


        final_reward = np.repeat(reward, BATCH_SIZE)
        learn_dict = {phrl['states_rl']: state,
                phrl['actions_rl']: action,
                phrl['values_rl']: final_reward,
                phrl['is_training_rl']: True}
        sess.run(rl_update, feed_dict=learn_dict)

        if j % 10 == 0:
          log_string('====epoch_%d====iter_%d: student_loss=%.4f, train_acc=%.4f' % (i, j, np.mean(ce), acc))
          log_string('===============: critic_loss=%.4f, reward=%.4f, valid_acc=%.4f, valid_eo=%.4f' % (cri_loss, reward, valid_acc, valid_eo[-1]))
          print('eo: ',valid_eo[0],valid_eo[1])
          print('eo: ',valid_eo[2],valid_eo[3])
          print(action, np.sum(action))


    all_saver.save(sess,LOG_PATH+'/all.ckpt')

      # """
if __name__ == "__main__":
  train()
  LOG_FOUT.close()
