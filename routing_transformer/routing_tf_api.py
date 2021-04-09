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

import os
import pdb

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.get_logger().setLevel('ERROR')
from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
from tensor2tensor.data_generators import text_encoder

from tensor2tensor.data_generators import problem

from routing_transformer.problems import pg19
from tqdm import tqdm

from routing_transformer.sparse_transformer import SparseTransformer

import numpy as np
import random
from scipy.special import log_softmax


VOCAB_PATH = "/mnt/nfs/work1/miyyer/simengsun/in-book-retrieval/RT-data/vocab.pg19_length8k.32768.subwords"
HPARAMS_PATH = "/mnt/nfs/work1/miyyer/simengsun/in-book-retrieval/RT-models/rt-checkpoint/hparams.json"
CKPT_PATH = "/mnt/nfs/work1/miyyer/simengsun/in-book-retrieval/RT-models/rt-checkpoint/ckpt-3530000"
MAX_SEQUENCE_LENGTH = 8192


class SparseTransformerWrapper(object):
    def __init__(self, max_seq_length=None):
        # Load hyperparameters
        self.max_seq_length = max_seq_length or MAX_SEQUENCE_LENGTH
        # Needed since RT uses blocks of size 256
        assert self.max_seq_length % 256 == 0

        hparams = hparams_lib.create_hparams_from_json(HPARAMS_PATH)
        hparams.use_tpu = False
        hparams = zero_dropout(hparams)
        # Build TF1 graph of model
        sptf_model = SparseTransformer(hparams, tf.estimator.ModeKeys.EVAL)
        self.input_nodes = {
            "targets": tf.placeholder(tf.int32, [None, self.max_seq_length])
        }
        self.output_nodes = sptf_model.body(self.input_nodes)
        # Map the checkpoint variables to the graph
        init_from_checkpoint(CKPT_PATH, variable_prefix="sparse_transformer/body")
        # create a session object, and actually initialize the graph
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.encoder = text_encoder.SubwordTextEncoder(VOCAB_PATH)

    def forward(self, sentences, encode_sentences=True, relevant_subsequences=None):
        encoded_sents = []
        encoded_seqs_no_pad = []
        if encode_sentences:
            for sent in sentences:
                encoded = []
                for line in sent.split("\n"):
                    new_tokens = self.encoder.encode(line.strip())
                    if len(encoded) + len(new_tokens) >= self.max_seq_length:
                        break
                    encoded.extend(new_tokens)
                encoded.append(text_encoder.EOS_ID)
                encoded_seqs_no_pad.append(encoded)
                # pad shorter sequences to the full length
                encoded = encoded + [text_encoder.PAD_ID for _ in range(self.max_seq_length - len(encoded))]
                assert len(encoded) == self.max_seq_length
                encoded_sents.append(encoded)
        else:
            # assume sentences are encoded, pad/truncate them
            for sent in sentences:
                sent = sent[:self.max_seq_length]
                encoded_seqs_no_pad.append(sent)
                sent = sent + [text_encoder.PAD_ID for _ in range(self.max_seq_length - len(sent))]
                encoded_sents.append(sent)

        feed_dict = {
            self.input_nodes["targets"]: np.array(encoded_sents)
        }
        outputs = self.sess.run(self.output_nodes, feed_dict=feed_dict)

        return_outputs = {
            "logits": np.squeeze(outputs[0], axis=(2, 3)),
            "loss": outputs[1]["training"],
            "encoded_seqs_no_pad": encoded_seqs_no_pad
        }

        if relevant_subsequences is not None:
            for i, rss in enumerate(relevant_subsequences):
                encoded_subseq = self.encoder.encode(rss)

                positions = find_sub_list(encoded_subseq, encoded_sents[i])
                misaligned_prefix_length = 0
                while positions is None:
                    misaligned_prefix_length += 1
                    encoded_subseq = encoded_subseq[1:]
                    positions = find_sub_list(encoded_subseq, encoded_sents[i])
                start, end = positions[-1]

                relevant_logits = return_outputs["logits"][i][start:end]
                log_probs = log_softmax(relevant_logits, axis=1)
                gold_log_probs = [lp[index] for index, lp in zip(encoded_subseq, log_probs)]
                return_outputs["subseq_log_loss"] = -1 * np.mean(gold_log_probs)
                return_outputs["misaligned_prefix_length"] = misaligned_prefix_length

        return return_outputs

    def close(self):
        self.sess.close()


def find_sub_list(sl, l):
    """Find sub-string, so as to be able to compute ppl of a sub-string."""
    sll=len(sl)
    matches = []
    for ind in (i for i,e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            matches.append(
                (ind, ind + sll)
            )
    if matches:
        return matches


def zero_dropout(hparams):
    hparams.input_dropout = 0.0
    hparams.dropout = 0.0
    hparams.relu_dropout = 0.0
    hparams.attention_dropout = 0.0
    hparams.layer_prepostprocess_dropout = 0.0
    return hparams


def log_variables(name, var_names):
    tf.logging.info("%s (%d total): %s", name, len(var_names),
                    random.sample(var_names, min(len(var_names), 5)))


def init_from_checkpoint(checkpoint_path,
                         checkpoint_prefix=None,
                         variable_prefix=None,
                         target_variables=None):
    """Initializes all of the variables using `init_checkpoint."""
    tf.logging.info("Loading variables from %s", checkpoint_path)
    checkpoint_variables = {
        name: name for name, _ in tf.train.list_variables(checkpoint_path) if "Adafactor" not in name
    }
    if target_variables is None:
        target_variables = tf.trainable_variables()
    target_variables = {var.name.split(":")[0]: var for var in target_variables}

    if checkpoint_prefix is not None:
        checkpoint_variables = {
            checkpoint_prefix + "/" + name: varname
            for name, varname in checkpoint_variables.items()
        }
    if variable_prefix is not None:
        target_variables = {
            variable_prefix + "/" + name: var
            for name, var in target_variables.items()
        }

    checkpoint_var_names = set(checkpoint_variables.keys())
    target_var_names = set(target_variables.keys())
    intersected_var_names = target_var_names & checkpoint_var_names

    assignment_map = {
        checkpoint_variables[name]: target_variables[name]
        for name in intersected_var_names
    }
    tf.train.init_from_checkpoint(checkpoint_path, assignment_map)

    log_variables("Loaded variables", intersected_var_names)
    log_variables("Uninitialized variables", target_var_names - checkpoint_var_names)
    log_variables("Unused variables", checkpoint_var_names - target_var_names)
