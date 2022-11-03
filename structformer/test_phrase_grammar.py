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

"""Test grammar induction performance of StructFormer."""

import argparse
import collections

import matplotlib.pyplot as plt
from nltk.parse import DependencyGraph
import numpy
import torch

from structformer import data_ptb
from structformer import tree_utils
from structformer.hinton import plot


def mean(x):
  return sum(x) / len(x)


@torch.no_grad()
def test(parser, corpus, device, prt=False, gap=0):
  """Compute UF1 and UAS scores.

  Args:
    parser: pretrained model
    corpus: labeled corpus
    device: cpu or gpu
    prt: bool, whether print examples
    gap: distance gap for building non-binary tree
  Returns:
    UF1: unlabeled F1 score for constituency parsing
  """
  parser.eval()

  prec_list = []
  reca_list = []
  f1_list = []
  dtree_list = []
  corpus_sys = {}
  corpus_ref = {}
  nsens = 0

  word2idx = corpus.dictionary.word2idx
  dataset = zip(corpus.test_sens, corpus.test_trees, corpus.test_nltktrees)

  for sen, sen_tree, sen_nltktree in dataset:
    x = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in sen]
    data = torch.LongTensor([x]).to(device)
    pos = torch.LongTensor([list(range(len(sen)))]).to(device)

    _, p_dict = parser(data, pos)
    block = p_dict['block']
    cibling = p_dict['cibling']
    head = p_dict['head']
    distance = p_dict['distance']
    height = p_dict['height']

    distance = distance.clone().squeeze(0).cpu().numpy().tolist()
    height = height.clone().squeeze(0).cpu().numpy().tolist()
    head = head.clone().squeeze(0).cpu().numpy()
    max_height = numpy.max(height)

    parse_tree = tree_utils.build_tree(distance, sen, gap=gap)

    model_out, _ = tree_utils.get_brackets(parse_tree)
    std_out, _ = tree_utils.get_brackets(sen_tree)
    overlap = model_out.intersection(std_out)

    corpus_sys[nsens] = tree_utils.mrg(parse_tree)
    corpus_ref[nsens] = tree_utils.mrg_labeled(sen_nltktree)

    prec = float(len(overlap)) / (len(model_out) + 1e-8)
    reca = float(len(overlap)) / (len(std_out) + 1e-8)
    if not std_out:
      reca = 1.
      if not model_out:
        prec = 1.
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    prec_list.append(prec)
    reca_list.append(reca)
    f1_list.append(f1)

    new_words = []
    true_words = sen_nltktree.pos()
    for d, c, w, ph in zip(distance, height, sen, head):
      next_word = true_words.pop(0)
      while next_word[1] not in data_ptb.WORD_TAGS:
        next_word = true_words.pop(0)
      new_words.append({
          'address': len(new_words) + 1,
          'word': next_word[0],
          'lemma': None,
          'ctag': None,
          'tag': next_word[1],
          'feats': None,
          'head': numpy.argmax(ph) + 1 if c < max_height else 0,
          'deps': collections.defaultdict(list),
          'rel': None,
          'distance': d,
          'height': c
      })
    while true_words:
      next_word = true_words.pop(0)
      assert next_word[1] not in data_ptb.WORD_TAGS

    dtree = DependencyGraph()
    for w in new_words:
      dtree.add_node(w)

    dtree_list.append(dtree)

    if prt and len(dtree_list) % 100 == 0:
      cibling = cibling.clone().squeeze(0).cpu().numpy()
      block = block.clone().squeeze(0).cpu().numpy()
      for word_i, d_i, imp_i, block_i, cibling_i, head_i in zip(
          sen, distance, height, block, cibling, head):
        print('%20s\t%10.2f\t%5.2f\t%s\t%s\t%s' %
              (word_i, d_i, imp_i, plot(block_i, max_val=1.),
               plot(head_i, max_val=1), plot(cibling_i, max_val=1.)))
      print('Standard output:', sen_tree)
      print('Model output:', parse_tree)
      print(dtree.to_conll(10))
      print()

      fig_i, ax_i = plt.subplots()
      ax_i.set_xticks(numpy.arange(len(sen)))
      ax_i.set_yticks(numpy.arange(len(sen)))
      ax_i.set_xticklabels(sen)
      ax_i.set_yticklabels(sen)

      plt.setp(
          ax_i.get_xticklabels(),
          rotation=45,
          ha='right',
          rotation_mode='anchor')

      for row in range(len(sen)):
        for col in range(len(sen)):
          _ = ax_i.text(
              col,
              row,
              '%.2f' % (head[row, col]),
              ha='center',
              va='center',
              color='w')
      fig_i.tight_layout()
      plt.savefig(
          './figures/sentence-%d.png' % (len(dtree_list)),
          dpi=300,
          format='png')

    nsens += 1

  print('Constituency parsing performance:')
  print('Mean Prec: %.4f, Mean Reca: %.4f, Mean F1: %.4f' %
        (mean(prec_list), mean(reca_list), mean(f1_list)))
  correct, total = tree_utils.corpus_stats_labeled(corpus_sys, corpus_ref)
  print(correct)
  print(total)
  print('SBAR: %.4f' % (correct['SBAR'] / total['SBAR']))
  print('NP: %.4f' % (correct['NP'] / total['NP']))
  print('VP: %.4f' % (correct['VP'] / total['VP']))
  print('PP: %.4f' % (correct['PP'] / total['PP']))
  print('ADJP: %.4f' % (correct['ADJP'] / total['ADJP']))
  print('ADVP: %.4f' % (correct['ADVP'] / total['ADVP']))
  print(tree_utils.corpus_average_depth(corpus_sys))

  print('-' * 89)

  print('Dependency parsing performance:')
  print('Stanford Style:')
  tree_utils.evald(dtree_list, '../data/ptb/test.stanford', directed=True)
  tree_utils.evald(dtree_list, '../data/ptb/test.stanford', directed=False)
  print('Conll Style:')
  tree_utils.evald(dtree_list, '../data/ptb/test.conll', directed=True)
  tree_utils.evald(dtree_list, '../data/ptb/test.conll', directed=False)

  return mean(f1_list)


if __name__ == '__main__':
  marks = [' ', '-', '=']

  numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)

  argpr = argparse.ArgumentParser(description='PyTorch PTB Language Model')

  # Model parameters.
  argpr.add_argument(
      '--data',
      type=str,
      default='data/penn/',
      help='location of the data corpus')
  argpr.add_argument(
      '--checkpoint',
      type=str,
      default='PTB.pt',
      help='model checkpoint to use')
  argpr.add_argument('--seed', type=int, default=1111, help='random seed')
  argpr.add_argument('--gap', type=float, default=0, help='random seed')
  argpr.add_argument('--print', action='store_true', help='use CUDA')
  argpr.add_argument('--cuda', action='store_true', help='use CUDA')
  argpr.add_argument('--wsj10', action='store_true', help='use WSJ10')
  args = argpr.parse_args()

  # Set the random seed manually for reproducibility.
  torch.manual_seed(args.seed)

  # Load model
  print('Loading model...')
  with open(args.checkpoint, 'rb') as f:
    model, _, _, _ = torch.load(f)
    torch.cuda.manual_seed(args.seed)
    model.cpu()
    if args.cuda:
      model.cuda()

  # Load data
  print('Loading PTB dataset...')
  ptb_corpus = data_ptb.Corpus(args.data)

  print('Evaluating...')
  if args.cuda:
    eval_device = torch.device('cuda:0')
  else:
    eval_device = torch.device('cpu')

  print('=' * 89)
  test(model, ptb_corpus, eval_device, prt=args.print, gap=args.gap)
  print('=' * 89)

  rel_weight = model.rel_weight.detach().cpu().numpy()
  fig, axs = plt.subplots(8, 8, sharex=True, sharey=True)

  names = ['p', 'd']

  for i in range(rel_weight.shape[0]):
    for j in range(rel_weight.shape[1]):
      print(plot(rel_weight[i, j], max_val=1.), end=' ')
      values = rel_weight[i, j]
      if i == 0:
        axs[i, j].set_title('%d' % (j,))
      if j == 0:
        axs[i, j].set_ylabel('%d' % (i,))
      axs[i, j].bar(names, values)
    print()

  plt.savefig('./figures/mask_weights.png', dpi=300, format='png')
