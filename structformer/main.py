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

# Lint as: python3
"""Masked language model training script for StructFormer."""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from structformer import data_penn
from structformer import data_ptb
from structformer import structformer
from structformer import test_phrase_grammar
from structformer.utils import batchify

parser = argparse.ArgumentParser(
    description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument(
    '--data',
    type=str,
    default='data/penn/',
    help='location of the data corpus')
parser.add_argument('--dict_thd', type=int, default=1, help='upper epoch limit')
parser.add_argument(
    '--model',
    type=str,
    default='structformer',
    help='type of neural net (structformer, transformer)')
parser.add_argument(
    '--nhid', type=int, default=512, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=8, help='number of layers')
parser.add_argument(
    '--n_parser_layers', type=int, default=3, help='number of layers')
parser.add_argument('--nheads', type=int, default=8, help='number of layers')
parser.add_argument(
    '--conv_size',
    type=int,
    default=9,
    help='number of convolution window size')
parser.add_argument(
    '--lr', type=float, default=0.0003, help='initial learning rate')
parser.add_argument(
    '--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument(
    '--batch_size', type=int, default=4096, metavar='N', help='batch size')
parser.add_argument(
    '--dropout',
    type=float,
    default=0.1,
    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument(
    '--dropatt',
    type=float,
    default=0,
    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument(
    '--mask_rate',
    type=float,
    default=0.3,
    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--relative_bias', action='store_true', help='use CUDA')
parser.add_argument('--pos_emb', action='store_true', help='use CUDA')
parser.add_argument(
    '--weight_act', type=str, default='softmax', help='use CUDA')
parser.add_argument(
    '--relations', type=str, default='head,child', help='relation list')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--nonmono', type=int, default=5, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument(
    '--log-interval',
    type=int,
    default=100,
    metavar='N',
    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--test_grammar', action='store_true', help='test grammar')
args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
  if not args.cuda:
    print('WARNING: You have a CUDA device, '
          'so you should probably run with --cuda')
  else:
    torch.cuda.manual_seed(args.seed)

if args.cuda:
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

mask_bernoulli = torch.distributions.Bernoulli(args.mask_rate)


###############################################################################
# Load data
###############################################################################


def model_save(fn):
  with open(fn, 'wb') as f:
    torch.save([model, criterion, optimizer, scheduler], f)


def model_load(fn):
  global model, criterion, optimizer, scheduler
  with open(fn, 'rb') as f:
    model, criterion, optimizer, scheduler = torch.load(f)

print('Loading dataset...')
corpus = data_penn.Corpus(args.data, thd=args.dict_thd)
ptb_corpus = data_ptb.Corpus(args.data)

pad_token = corpus.dictionary.word2idx['<pad>']
mask_token = corpus.dictionary.word2idx['<mask>']
unk_token = corpus.dictionary.word2idx['<unk>']

val_data = batchify(corpus.valid, args.batch_size, device, pad=pad_token)
test_data = batchify(corpus.test, args.batch_size, device, pad=pad_token)

###############################################################################
# Build the model
###############################################################################

criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

ntokens = len(corpus.dictionary)
print('Number of tokens: ', ntokens)
if args.model == 'structformer':
  model = structformer.StructFormer(
      args.nhid,
      args.nlayers,
      ntokens,
      args.nheads,
      args.dropout,
      args.dropatt,
      args.relative_bias,
      args.pos_emb,
      pad=pad_token,
      n_parser_layers=args.n_parser_layers,
      conv_size=args.conv_size,
      relations=args.relations.split(','),
      weight_act=args.weight_act)
elif args.model == 'transformer':
  model = structformer.Transformer(
      args.nhid,
      args.nlayers,
      ntokens,
      args.nheads,
      args.dropout,
      args.dropatt,
      args.relative_bias,
      args.pos_emb,
      pad=pad_token)
else:
  raise Exception
###
if args.resume:
  print('Resuming model ...')
  model_load(args.resume)
###
if args.cuda:
  model = model.cuda()
###
params = list(model.parameters())
total_params = sum(np.prod(x.size()) for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


###############################################################################
# Training code
###############################################################################


def mask_data(data):
  """randomly mask input sequence."""
  mask = mask_bernoulli.sample(data.shape).to(device).bool()
  mask = mask * (data != pad_token) * (data != unk_token)
  targets = data.masked_fill(~mask, pad_token)
  data = data.masked_fill(mask, mask_token)
  return data, targets


def evaluate(data_source):
  """Evaluate the model on given dataset."""
  model.eval()
  total_loss = 0
  total_count = 0
  for data in data_source:
    data, targets = mask_data(data)
    pos = torch.arange(data.size(1), device=device)[None, :]

    output, _ = model(data, pos)

    loss = criterion(output, targets.reshape(-1))
    count = (targets != pad_token).float().sum().data
    total_loss += loss.data * count
    total_count += count

  return total_loss / total_count


def train():
  """One epoch of training."""
  model.train()
  # Turn on training mode which enables dropout.
  if args.model == 'QRNN': model.reset()
  total_loss = 0
  start_time = time.time()
  batch = 0
  train_data = batchify(
      corpus.train, args.batch_size, device, pad=pad_token, shuffle=True)
  while batch < len(train_data):
    data = train_data[batch]
    data, targets = mask_data(data)
    pos = torch.arange(data.size(1), device=device)[None, :]

    optimizer.zero_grad()

    output, _ = model(data, pos)
    loss = criterion(output, targets.reshape(-1))
    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem.
    if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
    optimizer.step()

    total_loss += loss.data

    if batch % args.log_interval == 0 and batch > 0:
      cur_loss = total_loss / args.log_interval
      elapsed = time.time() - start_time
      print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | '
            'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} '.format(
                epoch, batch, len(train_data), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss,
                math.exp(cur_loss)))
      total_loss = 0
      start_time = time.time()
    ###
    batch += 1


# Loop over epochs.
lr = args.lr
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
  if not args.resume:
    optimizer = torch.optim.Adam(
        params, lr=args.lr, eps=1e-9, weight_decay=args.wdecay)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 0.5, patience=2, threshold=0)

  model_save(args.save)

  for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()

    train()

    val_loss = evaluate(val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss,
              math.exp(val_loss), val_loss / math.log(2)))
    print('-' * 89)
    # test_phrase_grammar.test(model, ptb_corpus, device)
    # print('-' * 89)

    if val_loss < stored_loss:
      model_save(args.save)
      print('Saving model (new best validation)')
      stored_loss = val_loss
    scheduler.step(val_loss)

    print('PROGRESS: {}%'.format((epoch / args.epochs) * 100))

except KeyboardInterrupt:
  print('-' * 89)
  print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | '
      'test bpc {:8.3f}'.format(test_loss, math.exp(test_loss),
                                test_loss / math.log(2)))
print('=' * 89)
if args.test_grammar:
  test_phrase_grammar.test(model, ptb_corpus, device)
  print('=' * 89)
