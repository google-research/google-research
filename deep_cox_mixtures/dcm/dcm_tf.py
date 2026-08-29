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

"""Tensorflow 2 implementation of the Deep Cox Mixture model.

This module includes tensorflow implementation of the following models:

1) Cox Mixture
2) Deep Cox Mixture
3) Deep Cox Mixture-VAE

TODO: Add docstrings,
      modularize code for DCM to inherit from CM and,
      DCM-VAE to inherit from DCM.

Not designed to be called directly, would be called when running
dcm.deep_cox_mixture.experiment

"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

import numpy as np

from scipy.interpolate import UnivariateSpline
from sksurv.linear_model.coxph import BreslowEstimator   

from tqdm import tqdm 

import logging

tf.keras.backend.set_floatx('float32')
dtype = tf.float32

FLOAT_64 = True

if FLOAT_64:
  tf.keras.backend.set_floatx('float64')
  dtype = tf.float64
    
lambd = 1

def randargmax(b,**kw):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b==b.max()), **kw)


class DeepCoxMixtureVAE(Model):
  """Tensorflow model definition of the Coupled Deep CPH VAE Survival Model.

  The Coupled Deep CPH VAE model involves learning shared representations for
  the each demographic in the dataset, which are modelled as a latent variable
  using a VAE. The representation then interacts with multiple output heads to
  determine the log-partial hazard for an individual in each group.

  """

  def __init__(self, k, HIDDEN, output_size):
    super(DeepCoxMixtureVAE, self).__init__()
    
    self.k = k
    
    self.hidden = HIDDEN
    self.encoder1 = Dense(HIDDEN, activation='selu', use_bias=False)
    self.encoder2 = Dense(HIDDEN, activation='selu', use_bias=False)
    self.encoder3 = Dense(HIDDEN + HIDDEN, use_bias=False)

    self.decoder1 = Dense(HIDDEN, activation='selu', use_bias=False)
    self.decoder2 = Dense(HIDDEN, activation='selu', use_bias=False)
    self.decoder3 = Dense(output_size, use_bias=False)

    self.gate = Dense(units=k, use_bias=False)
    self.expert = Dense(k, use_bias=False)
    #self.expert = Dense(k, use_bias=False, )
    
  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.hidden), dtype=dtype)
    return self.decode(eps)

  def encode(self, x):
    mean, logvar = tf.split(
        self.encoder3(self.encoder2(self.encoder1(x))), num_or_size_splits=2, axis=1)
    return mean, logvar

  def decode(self, z):
    logits = self.decoder3(self.decoder2(self.decoder1(z)))
    return logits

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape, dtype=dtype)
    return eps * tf.exp(logvar * .5) + mean
    
  def call(self, x):
    #x = self.rep1(x)
    x = tf.nn.selu(self.encode(x)[0])
    return self.gate(x), self.expert(x)

class DeepCoxMixture(Model):
  """Tensorflow model definition of the Deep Cox Mixture Survival Model.

  The Deep Cox Mixture involves learning shared representation for the
  covariates of each individual followed by assuming the survival function
  of the individual to be a mixture of K Cox Models. Conditioned on each
  subgroup Z=k; the PH assumptions are assumed to hold and the baseline
  hazard rates is determined non-parametrically using an spline-interpolated
  Breslow's estimator.

  """

  def __init__(self, k, HIDDEN):
    super(DeepCoxMixture, self).__init__()
    
    self.k = k
    self.rep1 = Dense(units=HIDDEN, use_bias=False, activation='selu')
    self.rep2 = Dense(units=HIDDEN, use_bias=False, activation='selu')
    self.rep3 = Dense(units=HIDDEN, use_bias=False, activation='selu')

    self.gate = Dense(units=k, use_bias=False)
    self.expert = Dense(k, use_bias=False)
    #self.expert = Dense(k, use_bias=False, )
    
  def call(self, x):
    #x = self.rep1(x)
    x = self.rep2(self.rep1(x))
    return self.gate(x), self.expert(x)

class CoxMixture(Model):
  """Tensorflow model definition of the Cox Mixture Survival Model.

  The Cox Mixture involves the assumption that the survival function
  of the individual to be a mixture of K Cox Models. Conditioned on each
  subgroup Z=k; the PH assumptions are assumed to hold and the baseline
  hazard rates is determined non-parametrically using an spline-interpolated
  Breslow's estimator.

  """

  def __init__(self, k):
    super(CoxMixture, self).__init__()
    
    self.k = k
    self.gate = Dense(units=k, use_bias=False)
    self.expert = Dense(k, use_bias=False, kernel_initializer='zeros')
    
  def call(self, x):

    return self.gate(x), self.expert(x)

def partial_ll_loss(lrisks, tb, eb, eps=0.001):

  tb = tb+ eps*np.random.random(len(tb))

  sindex = np.argsort(-tb)

  tb = tb[sindex]
  eb = eb[sindex]
  lrisks = tf.gather(lrisks, sindex)

  lrisksdenom = tf.math.cumulative_logsumexp(lrisks)

  plls = lrisks - lrisksdenom
  pll = plls[eb == 1]
  pll = tf.reduce_sum(pll) 

  return -pll

def fit_spline(t, surv, s=1e-4):
    return UnivariateSpline(t, surv, s=s, ext=3)

def smooth_bl_survival(breslow):
    
    blsurvival = breslow.baseline_survival_
    x, y = blsurvival.x, blsurvival.y 
    return fit_spline(x, y)

def get_probability_(lrisks, ts, spl):
    risks = np.exp(lrisks)
    s0ts = (-risks)*(spl(ts)**(risks-1))
    return s0ts * spl.derivative()(ts)

def get_survival_(lrisks, ts, spl):
    risks = np.exp(lrisks)
    return spl(ts)**risks

def fit_breslow(model, x, t, e, posteriors=None):
    from collections import Counter
    gates, lrisks = model(x)

    if posteriors is None:
        z = np.argmax(gates, axis=1)
        #print (Counter(z))
    else:
        #z = randargmax(posteriors, axis=1)
        z = np.argmax(posteriors, axis=1)
        #print (Counter(z))
        #print (posteriors.shape, z.shape, np.argmax(posteriors, axis=1).shape)
        #print ("*****")

    breslow_splines = {}    
    for i in range(model.k):
        breslowk = BreslowEstimator().fit(lrisks[:, i][z==i], e[z==i], t[z==i])
        breslow_splines[i] = smooth_bl_survival(breslowk)
    return breslow_splines
    
def get_probability(lrisks, breslow_splines, t):
    psurv = []
    for i in range(lrisks.shape[1]):
        p = get_probability_(lrisks[:, i], t, breslow_splines[i])
        psurv.append(p)
    psurv = np.array(psurv).T
    return psurv

def get_survival(lrisks,breslow_splines, t):
    psurv = []
    for i in range(lrisks.shape[1]):
        p = get_survival_(lrisks[:, i], t, breslow_splines[i])
        psurv.append(p)
    psurv = np.array(psurv).T
    return psurv

def get_posteriors_log(probs):
    from scipy.special import logsumexp
    return probs-logsumexp(probs, axis=1).reshape(-1,1)    

def get_posteriors(probs):
    probs_ = probs+1e-10
    return (probs_)/probs_.sum(axis=1).reshape(-1,1)  

def get_hard_z(gates_prob):
    return np.argmax(gates_prob, axis=1)

def sample_hard_z(gates_prob):
    k = gates_prob.shape[1]
    return np.array([int(np.random.choice(k,1,p=p)) for p in gates_prob])

def repair_probs(probs):
    probs[np.isnan(probs)] = 1e-10   
    probs[probs<1e-10] = 1e-10
    return probs

def get_likelihood(model, breslow_splines, xb, tb, eb, log=False):
    
    gates, lrisks = model(xb)
    
    survivals = get_survival(lrisks, breslow_splines, tb)
    probability = get_probability(lrisks, breslow_splines, tb)

    event_probs = np.array([survivals,probability])
    event_probs = event_probs[eb.astype('int'), range(len(eb)), :]

    if log:
        gates_prob = tf.nn.log_softmax(gates)
        probs = gates_prob+event_probs        
        probs = probs.numpy()
        
    else:
        gates_prob = tf.keras.layers.Softmax()(gates)
        probs = gates_prob*event_probs
        probs = probs.numpy()
        
    return probs


def log_normal_pdf(sample, mean, logvar, raxis=1):

  log2pi = tf.cast(tf.math.log(2. * np.pi), dtype=dtype)
  return tf.reduce_mean(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def vae_loss(model, x):
  """The Loss from the VAE component of the Coupled Deep CPH VAE.

  Args:
    model:
      instance of CoupledDeepCPHVAE class.
    x:
      a numpy array of input features (Training Data).

  Returns:
    a differentiable tensorflow variable with the mean loss over the batch.

  """

  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  mse = tf.keras.losses.MSE(x, x_logit)

  logpx_z = -mse
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)

  return -tf.reduce_sum(logpx_z + logpz - logqz_x)

def q_function(model, xb, tb, eb, posteriors, typ='soft'):
    
    if typ == 'hard':
        zb = get_hard_z(posteriors)
    else:
        zb = sample_hard_z(posteriors)
    
    gates, lrisks = model(xb)

    k = model.k
    
    loss = 0
    for i in range(k):
        lrisks_ = tf.boolean_mask(lrisks[:, i], zb == i)            
        loss += partial_ll_loss(lrisks_, tb[zb == i], eb[zb == i])
    
    log_smax_loss = -tf.nn.log_softmax(gates)
                
    gate_loss = 0
    for i in range(k):
        gate_loss += posteriors[:,i]*log_smax_loss[:,i]
    
    gate_loss = tf.reduce_sum(gate_loss)     
    loss+=gate_loss
    
    return loss

def e_step(model, breslow_splines, xb, tb, eb):
    
    if breslow_splines is None:
        # If Breslow splines are not available, like in the first
        # iteration of learning, we only use the gate output.
        gates, lrisks = model(xb)
        gates_prob = tf.keras.layers.Softmax()(gates)
        probs = gates_prob.numpy()
    else:
        probs = get_likelihood(model, breslow_splines, xb, tb, eb)

    probs = repair_probs(probs)
    posteriors = get_posteriors(probs)

    return posteriors
    
def m_step(model, optimizer, xb, tb, eb, posteriors, typ='soft'):
            
    with tf.GradientTape() as tape:
        loss = q_function(model, xb, tb, eb, posteriors, typ)
        
        if type(model).__name__  == 'DeepCoxMixtureVAE':
            vaeloss = vae_loss(model, xb)
            loss += lambd*vaeloss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return float(loss)
    
def get_survival_risk(model, x, breslow_splines, t):
    
    gates, lrisks = model(x) 
    
    psurv = []
    for i in range(risks.shape[1]):
        p = get_survival_(risks[:, i], t, breslow_splines[i])
        psurv.append(p)
        
    psurv = np.array(psurv).T
    return psurv

def train_step(model, x, t, e, a, breslow_splines, optimizer,
               bs=256, seed=100, typ='soft', use_posteriors=False):
    
  from sklearn.utils import shuffle
  
  x, t, e, a = shuffle(x, t, e, a, random_state=seed)
  n = x.shape[0]

  batches = (n // bs) + 1
  
  epoch_loss = 0
  for i in range(batches):

    xb = x[i*bs:(i+1)*bs]
    tb = t[i*bs:(i+1)*bs]
    eb = e[i*bs:(i+1)*bs]
    ab = a[i*bs:(i+1)*bs]
    # E-Step !!!
    posteriors = e_step(model, breslow_splines, xb, tb, eb)
    # M-Step !!!
    loss = m_step(model, optimizer, xb, tb, eb, posteriors, typ=typ)      
    #print (loss)
    # Fit Breslow on entire Data !!
    try:
      if i%1 == 0:
        posteriors = e_step(model, breslow_splines, x, t, e)
        if use_posteriors:
          breslow_splines = fit_breslow(model, x, t, e, posteriors=posteriors)
        else:
          breslow_splines = fit_breslow(model, x, t, e, posteriors=None)
    except Exception as exce:
      logging.warning("Couldn't fit splines, reusing from previous epoch")
    epoch_loss+=loss
  #print (epoch_loss/n)
  return breslow_splines

def test_step(model, x, t, e, a, breslow_splines, loss='q',typ='soft'):    
  
  if loss == 'q':
    posteriors = e_step(model, breslow_splines, x, t, e)
    loss = q_function(model, x, t, e, posteriors, typ=typ)

  return float(loss/x.shape[0])

def train(model, xt, tt, et, at, xv, tv, ev, av, epochs=50,
          patience=2, vloss='q', bs=256, typ='soft', lr=1e-3,
          use_posteriors=False, debug=False):
    
  logging.info("Running Monte-Carlo EM for: " + str(epochs) +
               " epochs; with a batch size of: "+ str(bs))

  optimizer = tf.keras.optimizers.Adam(lr=lr)

  valc = np.inf
  patience_ = 0
    
  breslow_splines = None

  for epoch in tqdm(range(epochs)):

    breslow_splines = train_step(model, xt, tt, et, at, breslow_splines, 
                                 optimizer, bs=bs, seed=epoch, typ=typ,
                                 use_posteriors=use_posteriors)
    valcn = test_step(model, xv, tv, ev, av, breslow_splines, loss=vloss,typ=typ)
        
    if epoch % 1 == 0:
      if debug:
        print(patience_, epoch, valcn)

    if valcn > valc:
      patience_ += 1
    else:
      patience_ = 0

    if patience_ == patience:
      return (model, breslow_splines)

    valc = valcn

  return (model, breslow_splines)

def predict_scores(model, x, t ):
    
  model, breslow_splines = model
  gates, lrisks = model(x)
    
  t = np.array([t for i in range(len(x))])
    
  expert_output = get_survival(lrisks, breslow_splines, t)
  gate_probs = tf.keras.layers.Softmax()(gates).numpy()
    
  return (gate_probs*expert_output).sum(axis=1)
  
    