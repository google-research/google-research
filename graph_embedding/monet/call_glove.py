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

"""Interface to TensorFlow GloVe trainer.
"""

from glove_model_tf import GloVeModelTf


def GloVe(sentences, session, vector_size, covariate_size=0, iters=5,
          random_seed=12345, metadata=None, window_size=5, batch_size=128,
          alpha=0.75, xmax=10, eta=0.05, use_monet=False, print_every=-1,
          super_verbose=False, weight_verbose=False, checkpoint_every=-1,
          checkpoint_dir=None, db_level=1.0,
          kill_after=-1,
          init_weight_dir=None,
          adv_dim=None, adv_lam=0.2, adv_labels=None, adv_lr=0.05,
          use_w2v=False, neg_samples=5, w2v_logit_max=10.0,
          w2v_neg_sample_mean=False):
  """Runs the GloVe model and returns or saves weight matrices.

  Args:
    sentences: a list of token lists
    session: a tensorflow session
    vector_size: desired size of the topology dimension
    covariate_size: desired size of the metadata dimension in the model
    iters: how many iterations through the co-occurrences to do
    random_seed: your favorite integer
    metadata: a keyed list of float lists, where each key identifies
          a token in the corpus, and each float list is a row of covariate data
    window_size: how far to look around each center word for cooccurrences
    batch_size: number of co-occurrences to process per training step
    alpha: parameter in exponent of GloVe loss weight function
    xmax: cutoff in GloVe loss weight function
    eta: learning rate
    use_monet: whether to do an orthogonalization op on topology dimensions
    print_every: how often to print status to console (counted in batches)
    super_verbose: prints a lot of extra console output you probably don't want
    weight_verbose: whether to print weight/update values (probably don't want)
    checkpoint_every: how often to save the model (counted in batches)
    checkpoint_dir: where to save checkpoints (must be a local dir, no cns)
    db_level: ("debias level") - a double between 0.0 and 1.0 inclusive giving
        the strength of the debiasing. 0.0 is no debiasing, 1.0 is full.
    kill_after: if >= 0, will stop the training after this many batches
    init_weight_dir: where to get initial weights from (don't use unless you
      know what you're doing)
    adv_dim: dimension of hidden layer of MLP adversary
    adv_lam: tuning parameter for adversarial loss. For now, this assumes you
      have input metadata and that the metadata is binary and one-dimensional.
      The adversary uses a length(adv_dim)-layer MLP with leaky ReLU
      activations. The loss is softmax cross-entropy.
    adv_labels: a {token, v} dict where v is a 2-length list of one-hot floats.
    adv_lr: learning rate for all adversarial train ops
    use_w2v: uses word2vec-like loss, adapted for co-occurrence counts
      *NOTE*: the w2v loss is experimental and does not carry any guarantees of
          good performance. It was not benchmarked in the associated
          publication, nor was it otherwise tested in any rigorous way.
    neg_samples: number of negative samples for word2vec
    w2v_logit_max: logits in w2v model are capped (in absolute value) at this
    w2v_neg_sample_mean: negative sample loss is averaged, if True
  Returns:
    out: a dict with the following items (they are None for certain models):
        {'topo_input': input node representations
         'topo_outpt': output node representations
         'meta_input': input metadata representations
         'meta_outpt': output metadata representations
         'meta_trans_input': the input metadata transformation weights
         'meta_trans_outpt': the outpt metadata transformation weights
         'topo_input_raw': if using monet, the un-orthogonalized node embeddings
         'topo_outpt_raw': same as above but for output}
  """
  trainer = GloVeModelTf(vector_size, covariate_size=covariate_size,
                         random_seed=random_seed, random_walks=sentences,
                         covariate_data=metadata, window_size=window_size)
  out = trainer.train_model(session, iters, alpha, xmax, eta, use_monet,
                            print_every=print_every,
                            print_extra_diagnostics=super_verbose,
                            print_weight_diagnostics=weight_verbose,
                            checkpoint_every=checkpoint_every,
                            checkpoint_dir=checkpoint_dir, db_level=db_level,
                            batch_size=batch_size, kill_after=kill_after,
                            init_weight_dir=init_weight_dir,
                            neg_samples=neg_samples,
                            adv_dim=adv_dim, adv_lam=adv_lam,
                            adv_labels=adv_labels, adv_lr=adv_lr,
                            use_w2v=use_w2v, w2v_logit_max=w2v_logit_max,
                            w2v_neg_sample_mean=w2v_neg_sample_mean)
  return out
