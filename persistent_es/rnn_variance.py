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

"""This script measures the empirical variance of PES using a tiny toy LSTM on a subset of
the Penn TreeBank dataset.

Examples:
---------
python rnn_variance.py --scenario=real
python rnn_variance.py --scenario=random
python rnn_variance.py --scenario=repeat
"""
import os
import sys
import csv
import ipdb
import copy
import argparse
import pickle as pkl
from functools import partial
from collections import Counter

import numpy as onp

import jax
print(jax.devices())
import jax.numpy as jnp
from jax import flatten_util
from jax.tree_util import tree_flatten, tree_unflatten

import haiku as hk

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('bright')


parser = argparse.ArgumentParser(description='Empirical variance measurement for a small LSTM')
parser.add_argument('--scenario', type=str, default='real', choices=['real', 'random', 'repeat'],
                    help='Scenario for the data distribution')
args = parser.parse_args()


def count_params(params):
    value_flat, value_tree = tree_flatten(params)
    return sum([v.size for v in value_flat])

def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.shape[0] // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0:nbatch*batch_size]
    # Evenly divide the data across the batch_size batches.
    data = data.reshape(batch_size, -1).T
    return data

def get_batch(source, i, seq_len=40, flatten_targets=True):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    if flatten_targets:
        target = source[i+1:i+1+seq_len].reshape(-1)
    else:
        target = source[i+1:i+1+seq_len]
    return data, target

def flatten(parameters):
    leaves, treedef = tree_flatten(parameters)
    concat_flat_params = jnp.concatenate([value.reshape(-1) for value in leaves])
    return concat_flat_params

def flat_norm(parameters):
    concat_flat_params = flatten(parameters)
    total_norm = jnp.linalg.norm(concat_flat_params, ord=2)
    return total_norm

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = onp.zeros(tokens, dtype=onp.int32)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

batch_size = 1
print('Producing dataset...')
corpus = Corpus(os.path.join('data/pennchar'))
ntokens = len(corpus.dictionary)
train_data = batchify(corpus.train, batch_size)
print('Total trainining sequence length: {}'.format(train_data.shape))

class LSTMCell(hk.Module):
    def __init__(self, nhid, name=None):
        super().__init__(name=name)
        self.nhid = nhid

        self.i2h = hk.Linear(4*nhid)
        self.h2h = hk.Linear(4*nhid)

    def __call__(self, input, hidden):
        def update(state, x):
            nhid = self.nhid

            h, cell = state[0], state[1]
            h = h.squeeze()
            cell = cell.squeeze()

            x_components = self.i2h(x)
            h_components = self.h2h(h)

            preactivations = x_components + h_components

            gates_together = jax.nn.sigmoid(preactivations[:, 0:3*nhid])
            forget_gate = gates_together[:, 0:nhid]
            input_gate = gates_together[:, nhid:2*nhid]
            output_gate = gates_together[:, 2*nhid:3*nhid]
            new_cell = jnp.tanh(preactivations[:, 3*nhid:4*nhid])

            cell = forget_gate * cell + input_gate * new_cell
            h = output_gate * jnp.tanh(cell)

            new_state = jnp.stack([h, cell])
            return new_state, h

        #   hidden.shape == (2, 32, 100)
        #   input.shape == (1, 32, 100)
        new_state, hidden_list = jax.lax.scan(update, hidden, input)  # new_state.shape == (2, 32, 100), hidden_list.shape == (1, 32, 100)
        hidden_stacked = jnp.stack(hidden_list)  # hidden_stacked.shape == (70, 40, 650) OR (1, 32, 100)
        return hidden_stacked, new_state

class RNNModel(hk.Module):
    def __init__(self, model='lstm', ntoken=10000, nhid=650, nlayers=1, dropoute=0.0,
                 dropouti=0.0, dropouth=0.0, dropouto=0.0, tie_weights=False, use_embeddings=True, with_bias=True):
        super().__init__()
        self.nhid = nhid
        self.ntoken = ntoken
        self.nlayers = nlayers
        self.dropoute = dropoute
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropouto = dropouto
        self.tie_weights = tie_weights
        self.use_embeddings = use_embeddings

        if model == 'lstm':
            self.layers = [LSTMCell(nhid) for _ in range(nlayers)]

        initrange = 0.1
        if use_embeddings:
            self.embedding = hk.Embed(ntoken, nhid, w_init=hk.initializers.RandomUniform(-initrange, initrange))

        if self.tie_weights:
            self.decoder_bias = hk.Bias(b_init=hk.initializers.Constant(0.0))
        else:
            self.decoder = hk.Linear(ntoken,
                                     with_bias=with_bias,
                                     # w_init=hk.initializers.RandomUniform(-initrange, initrange),
                                     # w_init=hk.initializers.RandomNormal(0.01),
                                     b_init=hk.initializers.Constant(0.0),
                                    )

    def __call__(self, input, hidden, key, training=True, return_h=False):
        if training:
            key1, key2, key3, key4 = jax.random.split(key, 4)
        else:
            key1, key2, key3, key4 = None, None, None, None

        if self.use_embeddings:
            emb = embedded_dropout(key1, self.embedding, input, self.ntoken, dropout=self.dropoute if training else 0)
            emb = locked_dropout(emb, self.dropouti, key2, training=training)
        else:
            emb = jax.nn.one_hot(input, self.ntoken)

        raw_output = emb
        new_hidden = []  # A list because it contains hiddens for diff. layers
        raw_outputs = []
        outputs = []
        for l, cell in enumerate(self.layers):
            current_input = raw_output
            raw_output, new_h = cell(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = locked_dropout(raw_output, self.dropouth, key3, training=training)
                outputs.append(raw_output)

        hidden = jnp.stack(new_hidden)  # For PES with vmap

        output = locked_dropout(raw_output, self.dropouto, key4, training=training)
        outputs.append(output)

        if self.tie_weights:
            decoded = jnp.matmul(output.reshape(output.shape[0]*output.shape[1], output.shape[2]), self.embedding.embeddings.T)
            decoded = self.decoder_bias(decoded)
        else:
            decoded = self.decoder(output.reshape(output.shape[0]*output.shape[1], output.shape[2]))

        result = decoded.reshape(output.shape[0], output.shape[1], decoded.shape[1])
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden, outputs

def locked_dropout(x, rate, key=None, training=True):
    """The difference between this and regular dropout is that here
       we use the same dropout mask for every step of the sequence.

       This is based on the Haiku dropout implementation at:
       https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/basic.py#L278
    """
    if not training:
        return x

    keep_rate = 1 - rate
    m = jax.random.bernoulli(key, keep_rate, shape=(1, x.shape[1], x.shape[2]))
    mask = m / keep_rate  # mask.shape == (1, 40, 650)
    return mask * x  # mask is (1, 40, 650) and x is (70, 40, 650), AND BROADCASTING WORKS, so we don't need expand_as for the mask

def embedded_dropout(key, embed, words, ntoken, dropout=0.1):
    if dropout:
        keep_rate = 1 - dropout
        mask = jax.random.bernoulli(key, keep_rate, (embed.embeddings.shape[0], 1)) / keep_rate
        masked_embed_weight = mask * embed.embeddings
    else:
        masked_embed_weight = embed.embeddings

    X = masked_embed_weight[words.reshape(-1)].reshape(*words.shape, -1)
    return X
# ===============================================================================================

nlayers = 1
emsize = 5
nhid = 5
sigma = 1e-3
T = 10000

def init_hidden(batch_size, nhid, nlayers):
    return jnp.stack([jnp.zeros((2, batch_size, nhid)) for l in range(nlayers)])  # The 2 is for the hidden state and cell of the LSTM

def forward_fn(input, hidden, randomness=None, training=False):
    model = RNNModel(ntoken=ntokens, nhid=nhid, nlayers=nlayers)
    return model(input, hidden, randomness, training)

@jax.jit
def cross_entropy(logits, targets):
    return jnp.sum(-jnp.sum(jax.nn.log_softmax(logits) * targets, axis=1))

def loss_fn(params, data, targets, hidden):
    # result, hidden, outputs = apply_jit(params, data, hidden)
    result, hidden, outputs = apply_jit(params, data, hidden, None, False)
    loss = cross_entropy(result.reshape(-1, ntokens), hk.one_hot(targets, ntokens))
    return loss, hidden

loss_jit = jax.jit(loss_fn)
loss_grad = jax.jit(jax.grad(loss_fn, has_aux=True))
loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

# -------------------------------------------------------------------------
# Initialize the model parameters
# -------------------------------------------------------------------------
forward = hk.without_apply_rng(hk.transform(forward_fn))
apply_jit = jax.jit(forward.apply, static_argnums=(4,))

inputs, targets = get_batch(train_data, 0, seq_len=10)
hidden = init_hidden(batch_size, nhid, nlayers)

key = jax.random.PRNGKey(42)
params = forward.init(key, inputs, hidden, randomness=key, training=False)
flat_params, params_unravel_pytree = flatten_util.ravel_pytree(params)
num_params = len(flat_params)

print('Num parameters: {}'.format(count_params(params)))
# -------------------------------------------------------------------------

@jax.jit
def loss_fn_flat(param_vector, hidden, data, targets):
    params = params_unravel_pytree(param_vector)
    result, hidden, outputs = apply_jit(params, data, hidden, None, False)
    loss = cross_entropy(result.reshape(-1, ntokens), hk.one_hot(targets, ntokens))
    return loss, hidden

@partial(jax.jit, static_argnums=(5,6))
def es_grad_estimate(key, params, hidden, data, targets, sigma, N):
    params_vector, _ = flatten_util.ravel_pytree(params)
    pos_pert = jax.random.normal(key, (N//2, len(params_vector))) * sigma
    neg_pert = -pos_pert
    perts = jnp.concatenate([pos_pert, neg_pert])
    losses, hiddens = jax.vmap(loss_fn_flat, in_axes=(0,None,None,None))(params_vector + perts, hidden, data, targets.reshape(-1))
    gradient_estimate = jnp.sum(losses.reshape(-1, 1) * perts, axis=0) / (N * sigma**2)
    gradient_estimate = params_unravel_pytree(gradient_estimate)
    return gradient_estimate

@partial(jax.jit, static_argnums=(6,7))
def pes_grad_estimate(key, params, all_hidden_states, all_perturbation_accums, data, targets, sigma, N):
    params_vector, _ = flatten_util.ravel_pytree(params)
    pos_pert = jax.random.normal(key, (N//2, len(params_vector))) * sigma
    neg_pert = -pos_pert
    perts = jnp.concatenate([pos_pert, neg_pert])
    losses, all_hidden_states = jax.vmap(loss_fn_flat, in_axes=(0,0,None,None))(params_vector + perts, all_hidden_states, data, targets.reshape(-1))
    all_perturbation_accums += perts
    gradient_estimate = jnp.sum(losses.reshape(-1, 1) * all_perturbation_accums, axis=0) / (N * sigma**2)
    gradient_estimate = params_unravel_pytree(gradient_estimate)
    return gradient_estimate, all_hidden_states, all_perturbation_accums

def full_pes_grad(key, params, data, targets, K, sigma, N):
    hidden = init_hidden(batch_size, nhid, nlayers)
    all_hidden_states_list = []
    for _ in range(N):
        all_hidden_states_list.append(copy.deepcopy(hidden))
    all_hidden_states = jnp.stack(all_hidden_states_list)
    all_perturbation_accums = jnp.zeros((N, num_params))

    T = len(data)
    t = 0
    gradient_estimate = jax.tree_map(lambda x: jnp.zeros(x.shape), params)
    while t < T:
        key, skey = jax.random.split(key)
        current_data = data[t:t+K]
        current_targets = targets[t:t+K]
        grad_pes_term, all_hidden_states, all_perturbation_accums = pes_grad_estimate(skey, params, all_hidden_states, all_perturbation_accums, current_data, current_targets, sigma, N)
        gradient_estimate = jax.tree_multimap(lambda x, y: x + y, gradient_estimate, grad_pes_term)
        t += K
    return key, gradient_estimate, all_perturbation_accums


if args.scenario == 'random':  # For the case where we want the data to be randomly generated
    key_for_randint = jax.random.PRNGKey(5)
    train_data = jax.random.randint(key_for_randint, (20000,1), 0, ntokens)
elif args.scenario == 'repeat':
    # This is for the case where we want to have all gradients equal to each other ---> just repeat the same character for the whole sequence
    train_data = jnp.array([train_data[0]] * 20000)

data, targets = get_batch(train_data, 0, T)  # Get a single fixed sequence for which to compute the gradient
hidden = init_hidden(batch_size, nhid, nlayers)

# ---------------------------------------------------------------------------
base_grad = es_grad_estimate(key, params, hidden, data, targets, sigma, 1000)
flat_base_grad, _ = flatten_util.ravel_pytree(base_grad)
print('Finished computing base grad')
# ---------------------------------------------------------------------------

# es_variance_dict = {}
# for N in [2, 10, 30, 100, 1000, 10000]:
#     grad_diff_list = []
#     for i in range(1000):
#         key, skey = jax.random.split(key)
#         es_grad = es_grad_estimate(key, params, hidden, data, targets, sigma, N)
#         grad_diff = tree_l2_dist(es_grad_base, es_grad)**2
#         grad_diff_list.append(grad_diff)
#     es_variance_dict[N] = onp.mean(grad_diff_list)
#     print('ES | N: {:5d} | variance: {:6.4e}'.format(N, es_variance_dict[N]))

# with open('es_variance.pkl', 'wb') as file:
#     pkl.dump(es_variance_dict, file)

pes_variance_dict = {}
pes_sum_of_pert_dict = {}
for K in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    for N in [1000, 100, 30, 10]:
        print('K {} | N {}'.format(K, N))
        grad_diff_list = []
        sum_pert_list = []
        for i in range(50):
            key, skey = jax.random.split(key)
            key, pes_grad, total_perturbation_accum = full_pes_grad(skey, params, data, targets, K, sigma, N)
            flat_pes_grad, _ = flatten_util.ravel_pytree(pes_grad)
            grad_diff = jnp.linalg.norm(flat_base_grad - flat_pes_grad)**2
            grad_diff_list.append(grad_diff)
            print('K: {} | N: {} | i: {}'.format(K, N, i))
            sys.stdout.flush()
            sum_pert = onp.mean([flat_norm(pert)**2 for pert in total_perturbation_accum])
            sum_pert_list.append(sum_pert)

        pes_sum_of_pert_dict[(K,N)] = onp.mean(sum_pert_list)
        pes_variance_dict[(K,N)] = onp.mean(grad_diff_list)
        print('PES | K: {:5d} | N: {} | variance: {:6.4e} | pert: {:6.4e}'.format(
               K, N, pes_variance_dict[(K,N)], pes_sum_of_pert_dict[(K,N)]))
        sys.stdout.flush()

with open('pes_lstm_variance_dict_{}.pkl'.format(args.scenario), 'wb') as file:
    pkl.dump(pes_variance_dict, file)

# ---------------------------------
# Plot variance
# ---------------------------------
with open('pes_lstm_variance_dict_{}.pkl'.format(args.scenario), 'rb') as f:
    pes_variance_dict = pkl.load(f)

plt.figure(figsize=(7,5))
for N_pert in [10, 30, 100, 1000]:
    Ks = []
    num_unrolls = []
    variances = []
    for (K, N) in pes_variance_dict:
        if N == N_pert:
            var = pes_variance_dict[(K, N)] / (jnp.linalg.norm(flat_base_grad)**2)
            Ks.append(K)
            num_unrolls.append(T / K)
            variances.append(var)

    plt.plot(num_unrolls, variances, linewidth=2, marker='o', label='P={}'.format(N_pert))

plt.xlabel('# Unrolls', fontsize=20)
plt.ylabel('Variance', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xscale('log')
plt.yscale('log')
if args.scenario == 'real':
    plt.ylim(5e-1, 1e3)
elif args.scenario == 'repeat':
    plt.ylim(1e-1, 1e3)
elif args.scenario == 'random':
    plt.ylim(1, 1e6)
plt.legend(fontsize=18, fancybox=True, framealpha=0.3, loc='upper left', ncol=2)
sns.despine()
plt.savefig('pes_lstm_variance_{}.pdf'.format(args.scenario), bbox_inches='tight', pad_inches=0)
plt.savefig('pes_lstm_variance_{}.png'.format(args.scenario), bbox_inches='tight', pad_inches=0)
