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

"""Library to preprocess text data into SMITH dual encoder model inputs."""

import collections
import random

import nltk
import tensorflow.compat.v1 as tf
import tqdm

from smith import utils
from smith import wiki_doc_pair_pb2
from smith.bert import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "Input data path.")

flags.DEFINE_string(
    "output_file", None,
    "Output TF examples (or comma-separated list of files) in TFRecord "
    "files.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the SMITH model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool("add_masks_lm", True,
                  "If true, add masks for word prediction LM pre-training.")

flags.DEFINE_integer(
    "max_sent_length_by_word", 32, "The maximum length of a sentence by tokens."
    "A sentence will be cut off if longer than this length, and will be padded "
    "if shorter than it. The sentence can also be a sentence block.")

flags.DEFINE_integer(
    "max_doc_length_by_sentence", 64,
    "The maximum length of a document by sentences. A "
    "document will be cut off if longer than this length, and"
    "will be padded if shorter than it.")

flags.DEFINE_bool(
    "greedy_sentence_filling", True,
    "If true, apply the greedy sentence filling trick to reduce the "
    "number of padded tokens.")

flags.DEFINE_integer("max_predictions_per_seq", 5,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")


class TrainingInstance(object):
  """A single training instance (sentence pair as dual encoder model inputs)."""

  def __init__(self,
               tokens_1,
               segment_ids_1,
               masked_lm_positions_1,
               masked_lm_labels_1,
               input_mask_1,
               masked_lm_weights_1,
               tokens_2,
               segment_ids_2,
               masked_lm_positions_2,
               masked_lm_labels_2,
               input_mask_2,
               masked_lm_weights_2,
               instance_id,
               documents_match_labels=-1.0):
    self.tokens_1 = tokens_1
    self.segment_ids_1 = segment_ids_1
    self.masked_lm_positions_1 = masked_lm_positions_1
    self.masked_lm_labels_1 = masked_lm_labels_1
    self.input_mask_1 = input_mask_1
    self.masked_lm_weights_1 = masked_lm_weights_1
    self.tokens_2 = tokens_2
    self.segment_ids_2 = segment_ids_2
    self.masked_lm_positions_2 = masked_lm_positions_2
    self.masked_lm_labels_2 = masked_lm_labels_2
    self.input_mask_2 = input_mask_2
    self.masked_lm_weights_2 = masked_lm_weights_2
    self.instance_id = instance_id
    self.documents_match_labels = documents_match_labels

  def __str__(self):
    s = ""
    s += "instance_id: %s\n" % self.instance_id
    s += "documents_match_labels: %s\n" % (str(self.documents_match_labels))
    s += "tokens_1: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens_1]))
    s += "segment_ids_1: %s\n" % (" ".join([str(x) for x in self.segment_ids_1
                                           ]))
    s += "masked_lm_positions_1: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions_1]))
    s += "masked_lm_labels_1: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels_1]))
    s += "input_mask_1: %s\n" % (" ".join([str(x) for x in self.input_mask_1]))
    s += "masked_lm_weights_1: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_weights_1]))
    s += "tokens_2: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens_2]))
    s += "segment_ids_2: %s\n" % (" ".join([str(x) for x in self.segment_ids_2
                                           ]))
    s += "masked_lm_positions_2: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions_2]))
    s += "masked_lm_labels_2: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels_2]))
    s += "input_mask_2: %s\n" % (" ".join([str(x) for x in self.input_mask_2]))
    s += "masked_lm_weights_2: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_weights_2]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def add_features_for_one_doc(features, tokens, segment_ids, input_mask,
                             masked_lm_positions, masked_lm_labels,
                             masked_lm_weights, tokenizer, doc_index):
  """Add features for one document in a WikiDocPair example."""
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  features["input_ids_" + doc_index] = utils.create_int_feature(input_ids)
  features["input_mask_" + doc_index] = utils.create_int_feature(input_mask)
  features["segment_ids_" + doc_index] = utils.create_int_feature(segment_ids)

  if masked_lm_labels:
    masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    features["masked_lm_positions_" +
             doc_index] = utils.create_int_feature(masked_lm_positions)
    features["masked_lm_ids_" +
             doc_index] = utils.create_int_feature(masked_lm_ids)
    features["masked_lm_weights_" +
             doc_index] = utils.create_float_feature(masked_lm_weights)


def write_instance_to_example_files(instances, tokenizer, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))
  writer_index = 0
  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    features = collections.OrderedDict()
    add_features_for_one_doc(
        features=features,
        tokens=instance.tokens_1,
        segment_ids=instance.segment_ids_1,
        input_mask=instance.input_mask_1,
        masked_lm_positions=instance.masked_lm_positions_1,
        masked_lm_labels=instance.masked_lm_labels_1,
        masked_lm_weights=instance.masked_lm_weights_1,
        tokenizer=tokenizer,
        doc_index="1")
    add_features_for_one_doc(
        features=features,
        tokens=instance.tokens_2,
        segment_ids=instance.segment_ids_2,
        input_mask=instance.input_mask_2,
        masked_lm_positions=instance.masked_lm_positions_2,
        masked_lm_labels=instance.masked_lm_labels_2,
        masked_lm_weights=instance.masked_lm_weights_2,
        tokenizer=tokenizer,
        doc_index="2")
    # Adds fields on more content/id information of the current example.
    features["instance_id"] = utils.create_bytes_feature(
        [bytes(instance.instance_id, "utf-8")])
    features["tokens_1"] = utils.create_bytes_feature(
        [bytes(t, "utf-8") for t in instance.tokens_1])
    features["tokens_2"] = utils.create_bytes_feature(
        [bytes(t, "utf-8") for t in instance.tokens_2])
    # Adds the documents matching labels.
    features["documents_match_labels"] = utils.create_float_feature(
        [float(instance.documents_match_labels)])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info(
          "tokens_1: %s" %
          " ".join([tokenization.printable_text(x) for x in instance.tokens_1]))
      tf.logging.info(
          "tokens_2: %s" %
          " ".join([tokenization.printable_text(x) for x in instance.tokens_2]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        elif feature.bytes_list.value:
          values = feature.bytes_list.value
        tf.logging.info("%s: %s" %
                        (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def get_smith_model_tokens(input_text, tokenizer, sent_token_counter):
  """Generate tokens given an input text for the SMITH model."""
  res_tokens = []
  for sent in nltk.tokenize.sent_tokenize(input_text):
    # The returned res_tokens is a 2D list to maintain the sentence boundary
    # information. We removed all the empty tokens in this step.
    if not sent:
      continue
    tokens = [w for w in tokenizer.tokenize(sent) if w]
    sent_token_counter[0] += 1  # Track number of sentences.
    sent_token_counter[1] += len(tokens)  # Track number of tokens.
    res_tokens.append(tokens)
  return (res_tokens, sent_token_counter)


def create_training_instances_wiki_doc_pair(
    input_file, tokenizer, max_sent_length_by_word, max_doc_length_by_sentence,
    masked_lm_prob, max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from WikiDocPair proto data."""
  # The input data is in the WikiDocPair proto format in tfrecord.
  wiki_doc_pair = wiki_doc_pair_pb2.WikiDocPair()
  instances = []
  # Add some counters to track some data statistics.
  sent_token_counter = [0, 0]
  for example in tqdm.tqdm(tf.python_io.tf_record_iterator(input_file)):
    doc_pair = wiki_doc_pair.FromString(example)
    # If model_name = smith_dual_encoder, we firstly use a sentence tokenizer
    # to split doc_one/doc_two texts into different sentences and use [SEN] to
    # label the sentence boundary information. So in the masking and padding
    # step, we know the boundary between different sentences and we can do the
    # masking and padding according to the actual length of each sentence.
    doc_one_text = " \n\n\n\n\n\n ".join(
        [a.text for a in doc_pair.doc_one.section_contents])
    doc_two_text = " \n\n\n\n\n\n ".join(
        [a.text for a in doc_pair.doc_two.section_contents])
    doc_one_text = tokenization.convert_to_unicode(doc_one_text).strip()
    doc_two_text = tokenization.convert_to_unicode(doc_two_text).strip()
    doc_one_tokens, sent_token_counter = get_smith_model_tokens(
        doc_one_text, tokenizer, sent_token_counter)
    doc_two_tokens, sent_token_counter = get_smith_model_tokens(
        doc_two_text, tokenizer, sent_token_counter)
    # Skip the document pairs if any document is empty.
    if not doc_one_tokens or not doc_two_tokens:
      continue
    vocab_words = list(tokenizer.vocab.keys())
    instance_id = doc_pair.id
    if doc_pair.human_label_for_classification:
      doc_match_label = doc_pair.human_label_for_classification
    else:
      # Set the label as 0.0 if there are no available labels.
      doc_match_label = 0.0
    instances.append(
        create_instance_from_wiki_doc_pair(
            instance_id, doc_match_label, doc_one_tokens, doc_two_tokens,
            max_sent_length_by_word, max_doc_length_by_sentence, masked_lm_prob,
            max_predictions_per_seq, vocab_words, rng))
  rng.shuffle(instances)
  return (instances, sent_token_counter)


def create_instance_from_wiki_doc_pair(instance_id, doc_match_label,
                                       doc_one_tokens, doc_two_tokens,
                                       max_sent_length_by_word,
                                       max_doc_length_by_sentence,
                                       masked_lm_prob, max_predictions_per_seq,
                                       vocab_words, rng):
  """Creates `TrainingInstance`s for a WikiDocPair input data."""
  (tokens_1, segment_ids_1, masked_lm_positions_1, masked_lm_labels_1, \
   input_mask_1, masked_lm_weights_1) = \
      get_tokens_segment_ids_masks(max_sent_length_by_word, max_doc_length_by_sentence, doc_one_tokens, masked_lm_prob,
                                   max_predictions_per_seq, vocab_words, rng)
  (tokens_2, segment_ids_2, masked_lm_positions_2, masked_lm_labels_2, \
   input_mask_2, masked_lm_weights_2) = \
      get_tokens_segment_ids_masks(max_sent_length_by_word, max_doc_length_by_sentence, doc_two_tokens, masked_lm_prob,
                                   max_predictions_per_seq, vocab_words, rng)
  instance = TrainingInstance(
      tokens_1=tokens_1,
      segment_ids_1=segment_ids_1,
      masked_lm_positions_1=masked_lm_positions_1,
      masked_lm_labels_1=masked_lm_labels_1,
      input_mask_1=input_mask_1,
      masked_lm_weights_1=masked_lm_weights_1,
      tokens_2=tokens_2,
      segment_ids_2=segment_ids_2,
      masked_lm_positions_2=masked_lm_positions_2,
      masked_lm_labels_2=masked_lm_labels_2,
      input_mask_2=input_mask_2,
      masked_lm_weights_2=masked_lm_weights_2,
      instance_id=instance_id,
      documents_match_labels=doc_match_label)
  return instance


def get_tokens_segment_ids_masks(max_sent_length_by_word,
                                 max_doc_length_by_sentence, doc_one_tokens,
                                 masked_lm_prob, max_predictions_per_seq,
                                 vocab_words, rng):
  """Get the tokens, segment ids and masks of an input sequence."""
  # The format of tokens for SMITH dual encoder models is like:
  # [CLS] block1_token1 block1_token2 block1_token3 ... [SEP] [SEP] [PAD] ...
  # [CLS] block2_token1 block2_token2 block2_token3 ... [SEP] [SEP] [PAD] ...
  # [CLS] block3_token1 block3_token2 block3_token3 ... [SEP] [SEP] [PAD] ...
  # If max_sent_length_by_word is large, then there will be many padded
  # words in the sentence. Here we added an optional "greedy sentence filling"
  # trick in order to reduce the number of padded words and maintain all
  # content in the document. We allow a "sentence" block to contain more than
  # one natural sentence and try to fill as many as sentences into the
  # "sentence" block. If a sentence will be cut off and the current sentence
  # block is not empty, we will put the sentence into the next "sentence" block.
  # According to ALBERT paper and RoBERTa paper, a segment is usually comprised
  # of more than one natural sentence, which has been shown to benefit
  # performance. doc_one_tokens is a 2D list which contains the sentence
  # boundary information.
  sentence_num = len(doc_one_tokens)
  # sent_block_token_list is a 2D list to maintain sentence block tokens.
  sent_block_token_list = []
  natural_sentence_index = -1
  while natural_sentence_index + 1 < sentence_num:
    natural_sentence_index += 1
    sent_tokens = doc_one_tokens[natural_sentence_index]
    if not sent_tokens:
      continue
    if FLAGS.greedy_sentence_filling:
      cur_sent_block_length = 0
      cur_sent_block = []
      # Fill as many senteces as possible in the current sentence block in a
      # greedy way.
      while natural_sentence_index < sentence_num:
        cur_natural_sent_tokens = doc_one_tokens[natural_sentence_index]
        if not cur_natural_sent_tokens:
          natural_sentence_index += 1
          continue
        cur_sent_len = len(cur_natural_sent_tokens)
        if ((cur_sent_block_length + cur_sent_len) <=
            (max_sent_length_by_word - 3)) or cur_sent_block_length == 0:
          # One exceptional case here is that if the 1st sentence of a sentence
          # block is already going across the boundary, then the current
          # sentence block will be empty. So when cur_sent_block_length is 0
          # and we meet a natural sentence with length longer than
          # (max_sent_length_by_word - 3), we still put this natural sentence
          # in the current sentence block. In this case, this long natural
          # sentence will be cut off with the final length up to
          # (max_sent_length_by_word - 3).
          cur_sent_block.extend(cur_natural_sent_tokens)
          cur_sent_block_length += cur_sent_len
          natural_sentence_index += 1
        else:
          # If cur_sent_block_length + cur_sent_len > max_sent_length_by_word-3
          # and the current sentence block is not empty, the sentence which
          # goes across the boundary will be put into the next sentence block.
          natural_sentence_index -= 1
          break
    sent_tokens = cur_sent_block
    sent_block_token_list.append(sent_tokens)
    if len(sent_block_token_list) >= max_doc_length_by_sentence:
      break  # Skip more sentence blocks if the document is too long.
  # For each sentence block, generate the token sequences, masks and paddings.
  tokens_doc = []
  segment_ids_doc = []
  masked_lm_positions_doc = []
  masked_lm_labels_doc = []
  input_mask_doc = []
  masked_lm_weights_doc = []
  for block_index in range(len(sent_block_token_list)):
    tokens_block, segment_ids_block, masked_lm_positions_block, \
    masked_lm_labels_block, input_mask_block, masked_lm_weights_block = \
        get_token_masks_paddings(
            sent_block_token_list[block_index],
            max_sent_length_by_word,
            masked_lm_prob,
            max_predictions_per_seq,
            vocab_words,
            rng,
            block_index)
    tokens_doc.extend(tokens_block)
    segment_ids_doc.extend(segment_ids_block)
    masked_lm_positions_doc.extend(masked_lm_positions_block)
    masked_lm_labels_doc.extend(masked_lm_labels_block)
    input_mask_doc.extend(input_mask_block)
    masked_lm_weights_doc.extend(masked_lm_weights_block)

  # Pad sentence blocks if the actual number of sentence blocks is less than
  # max_doc_length_by_sentence.
  sentence_block_index = len(sent_block_token_list)
  while sentence_block_index < max_doc_length_by_sentence:
    for _ in range(max_sent_length_by_word):
      tokens_doc.append("[PAD]")
      segment_ids_doc.append(0)
      input_mask_doc.append(0)
    for _ in range(max_predictions_per_seq):
      masked_lm_positions_doc.append(0)
      masked_lm_labels_doc.append("[PAD]")
      masked_lm_weights_doc.append(0.0)
    sentence_block_index += 1
  assert len(tokens_doc) == max_sent_length_by_word * max_doc_length_by_sentence
  assert len(masked_lm_labels_doc
            ) == max_predictions_per_seq * max_doc_length_by_sentence
  return (tokens_doc, segment_ids_doc, masked_lm_positions_doc,
          masked_lm_labels_doc, input_mask_doc, masked_lm_weights_doc)


def get_token_masks_paddings(block_tokens, max_sent_length_by_word,
                             masked_lm_prob, max_predictions_per_seq,
                             vocab_words, rng, block_index):
  """Generates tokens, masks and paddings for the input block tokens."""
  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_sent_length_by_word - 3
  # Truncates the sequence if sequence length is longer than max_num_tokens.
  tokens = []
  segment_ids = []
  if len(block_tokens) > max_num_tokens:
    block_tokens = block_tokens[0:max_num_tokens]
  tokens_a = block_tokens
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  masked_lm_positions = []
  masked_lm_labels = []
  masked_lm_weights = []
  if max_predictions_per_seq > 0:
    (tokens, masked_lm_positions,
     masked_lm_labels) = utils.create_masked_lm_predictions(
         tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
  # Add [PAD] to tokens and masked LM related lists.
  input_mask = [1] * len(tokens)
  while len(tokens) < max_sent_length_by_word:
    tokens.append("[PAD]")
    input_mask.append(0)
    segment_ids.append(0)

  assert len(tokens) == max_sent_length_by_word
  assert len(input_mask) == max_sent_length_by_word
  assert len(segment_ids) == max_sent_length_by_word

  if max_predictions_per_seq > 0:
    # Transfer local positions in masked_lm_positions to global positions in the
    # whole document to be consistent with the model training pipeline.
    masked_lm_positions = [
        (i + max_sent_length_by_word * block_index) for i in masked_lm_positions
    ]
    masked_lm_weights = [1.0] * len(masked_lm_labels)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_labels.append("[PAD]")
      masked_lm_weights.append(0.0)
  return (tokens, segment_ids, masked_lm_positions, masked_lm_labels,
          input_mask, masked_lm_weights)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)
  rng = random.Random(FLAGS.random_seed)
  # Creates training instances.
  max_predictions_per_seq = FLAGS.max_predictions_per_seq if FLAGS.add_masks_lm else 0
  masked_lm_prob = FLAGS.masked_lm_prob if FLAGS.add_masks_lm else 0
  instances, sent_token_counter = create_training_instances_wiki_doc_pair(
      input_file=FLAGS.input_file,
      tokenizer=tokenizer,
      max_sent_length_by_word=FLAGS.max_sent_length_by_word,
      max_doc_length_by_sentence=FLAGS.max_doc_length_by_sentence,
      masked_lm_prob=masked_lm_prob,
      max_predictions_per_seq=max_predictions_per_seq,
      rng=rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  # Transfers training instances into tensorflow examples and write the results.
  write_instance_to_example_files(instances, tokenizer, output_files)

  # Finally outputs some data statistics.
  tf.logging.info("sent_count, token_count, doc_pair_count: %d %d %d",
                  sent_token_counter[0], sent_token_counter[1], len(instances))


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
