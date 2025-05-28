# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Builds pair n-gram models from a TSV file.

The training data consists of a two-column TSV file where the first column
gives the input side and the second column gives the corresponding output side.

The outputs are WFSTs representing a joint model over the input and output
strings.

Requirements
------------

This requires the Pynini library and several command-line tools from OpenFst
and OpenGrm. See the included README for instructions for creating a
reproducible build environment.

Synopsis
--------

In the FAR compilation stage, we build FARs (i.e., ordered collections of FSTs)
that contain the input and output strings, respectively.

In the covering grammar compilation stage, we build a covering grammar FST.

In the aligner training stage, we randomize the covering grammar weights and
use expectation maximization (or more precisely, an online variant with Viterbi
training) to set the weights of the aligner WFST to maximize the joint
probability of the data. This is done repeatedly and in parallel with several
different random seeds, and we retain the aligner WFST which obtains the best
likelihood.

In the data alignment stage, we compute best alignments using the best aligner
from the previous stage.

In the encoding phase, we encode these alignments as FSAs.

In the n-gram compilation phase, we build a n-gram model over these alignments,
applying smoothing and shrinking, then decode the n-gram model to produce the
final pair n-gram FST.

Nota bene
---------

This program uses numerous temporary files created using the `tempfile` module.
If you wish to generate these in a different directory than the OS's default
location, set the $TMPDIR environmental variable.

All subprocess calls are logged at DEBUG.
"""

import argparse
import csv
import dataclasses
import itertools
import logging
import math
import multiprocessing
import operator
import os
import random
import re
import subprocess
import tempfile
import time
from typing import Iterable, List, Optional, Set, Tuple

import pywrapfst


RAND_MAX = 32767


# Helpers.


def _log_check_call(cmd: List[str]) -> None:
  """Calls a subprocess and logs the call at DEBUG."""
  logging.debug("Subprocess call:\t%s", " ".join(cmd))
  subprocess.check_call(cmd)


def _mktemp(suffix: str) -> str:
  """Creates temporary file with desired suffix.

  Args:
    suffix: the desired suffix for the temporary file.

  Returns:
    Path to a temporary file.
  """
  path = tempfile.mkstemp(suffix=f".{suffix}")[1]
  logging.debug("New temporary file:\t%s", path)
  return path


def _rmtemp(path: str) -> None:
  """Removes temporary file.

  Args:
    path: path to temporary file to be removed.
  """
  logging.debug("Removing temporary file:\t%s", path)
  os.remove(path)


# FAR compilation.


def _compile_fars(
    tsv: str, input_token_type: str, output_token_type: str
) -> Tuple[str, str]:
  """Compiles FAR files and returns their paths.

  Args:
    tsv: path to the data TSV file.
    input_token_type: input token type (one of: "byte", "utf8", or a symbol
      table).
    output_token_type: output token_type (one of: "byte", "utf8", or a symbol
      table).

  Returns:
    A tuple containing the input FAR path and output FAR path.
  """
  with open(tsv, "r") as source, tempfile.NamedTemporaryFile(
      suffix=".i.txt", mode="w"
  ) as itxt, tempfile.NamedTemporaryFile(suffix=".o.txt", mode="w") as otxt:
    for col1, col2 in csv.reader(source, delimiter="\t"):
      print(col1, file=itxt)
      print(col2, file=otxt)
    ifar_path = _mktemp("i.far")
    args = ["farcompilestrings", "--fst_type=compact"]
    if input_token_type in ["byte", "utf8"]:
      args.append(f"--token_type={input_token_type}")
    else:
      args.append("--token_type=symbol")
      args.append(f"--symbols={input_token_type}")
    args.append(itxt.name)
    args.append(ifar_path)
    _log_check_call(args)
    ofar_path = _mktemp("o.far")
    args = ["farcompilestrings", "--fst_type=compact"]
    if output_token_type in ["byte", "utf8"]:
      args.append(f"--token_type={output_token_type}")
    else:
      args.append("--token_type=symbol")
      args.append(f"--symbols={output_token_type}")
    args.append(otxt.name)
    args.append(ofar_path)
    _log_check_call(args)
  # Temporary text files are now deleted.
  return ifar_path, ofar_path


def get_labels(reader: pywrapfst.FarReader) -> Set[int]:
  """Gets all labels seen in an FAR.

  This assumes all are FSAs. The reader is rewound before we return.

  Args:
      reader: input FAR reader.

  Returns:
      The set of all arcs seen.
  """
  # Collects labels.
  labels: Set[int] = set()
  while not reader.done():
    fst = reader.get_fst()
    for state in fst.states():
      for arc in fst.arcs(state):
        labels.add(arc.olabel)  # Only looks at output label.
    reader.next()
  reader.reset()
  return labels


class CoveringGrammarCompiler:
  """Covering grammar compiler.

  The user specifies iterables of integer input and output labels, and the
  number of insertions and deletions to permit between input/output matches.
  These arguments impose an upper bound on the number of input vs. output
  labels, which is needed for certain forms of decoding.
  """

  ilabels: List[int]
  olabels: List[int]
  insertions: int
  deletions: int
  arc_type: str

  def __init__(
      self,
      ilabels: Iterable[int],
      olabels: Iterable[int],
      insertions=1,
      deletions=1,
      arc_type="standard",
  ):
    """Initializes the covering grammar compiler.

    Args:
        ilabels: iterable of integer input labels.
        olabels: iterable of integer output labels.
        insertions: number of consecutive insertions to permit.
        deletions: number of consecutive deletions to permit.
        arc_type: desired arc type.
    """
    # Sorting for stability.
    self.ilabels = sorted(ilabels)
    self.olabels = sorted(olabels)
    self.insertions = insertions
    self.deletions = deletions
    self.arc_type = arc_type

  def compile(self) -> pywrapfst.VectorFst:
    """Actually compiles the CG.

    Returns:
        The covering grammar FST.
    """
    # Builds the skeleton FSA.
    cg = pywrapfst.VectorFst(arc_type=self.arc_type)
    one = pywrapfst.Weight.one(cg.weight_type())
    if not self.insertions and not self.deletions:  # Special case.
      state = cg.add_state()
      cg.set_start(state)
      for ilabel, olabel in itertools.product(self.ilabels, self.olabels):
        cg.add_arc(state, pywrapfst.Arc(ilabel, olabel, one, state))
      cg.set_final(state)
      assert cg.verify(), "construction error"
      return cg
    cg.reserve_states(2 + self.insertions + self.deletions)
    start_state = cg.add_state()
    cg.set_start(start_state)
    cg.set_final(start_state)
    match_state = cg.add_state()
    # Implements match.
    cg.add_arc(start_state, pywrapfst.Arc(0, 0, one, match_state))
    for ilabel, olabel in itertools.product(self.ilabels, self.olabels):
      cg.add_arc(match_state, pywrapfst.Arc(ilabel, olabel, one, start_state))
    cg.set_final(match_state)
    # Implements insertion.
    if self.insertions:
      state = cg.add_state()
      cg.add_arc(start_state, pywrapfst.Arc(0, 0, one, state))
      for _ in range(self.insertions - 1):
        next_state = cg.add_state()
        for olabel in self.olabels:
          cg.add_arc(state, pywrapfst.Arc(0, olabel, one, next_state))
          cg.add_arc(state, pywrapfst.Arc(0, olabel, one, match_state))
        cg.set_final(next_state)
        state = next_state
      for olabel in self.olabels:
        cg.add_arc(state, pywrapfst.Arc(0, olabel, one, match_state))
    # Implements deletion.
    if self.deletions:
      state = cg.add_state()
      cg.add_arc(start_state, pywrapfst.Arc(0, 0, one, state))
      for _ in range(self.deletions - 1):
        next_state = cg.add_state()
        for ilabel in self.ilabels:
          cg.add_arc(state, pywrapfst.Arc(ilabel, 0, one, next_state))
          cg.add_arc(state, pywrapfst.Arc(ilabel, 0, one, match_state))
        cg.set_final(next_state)
        state = next_state
      for ilabel in self.ilabels:
        cg.add_arc(state, pywrapfst.Arc(ilabel, 0, one, match_state))
    assert cg.verify(), "construction error"
    return cg


@dataclasses.dataclass
class RandomStart:
  """Struct representing a random start."""

  idx: int
  seed: int
  ifar_path: str
  ofar_path: str
  cg_path: str
  train_opts: List[str]

  def train(self) -> Tuple[str, float]:
    """Trains a single random start.

    Returns:
      A tuple containing the aligner FST path and the negative log
      likelihood.
    """
    start = time.time()
    # Randomizes the channel model weights.
    rfst_path = _mktemp(f"random-{self.seed:05d}.fst")
    _log_check_call([
        "baumwelchrandomize",
        f"--seed={self.seed}",
        self.cg_path,
        rfst_path,
    ])
    # Trains model and reads likelihood.
    afst_path = _mktemp(f"aligner-{self.seed:05d}.fst")
    likelihood = math.inf
    cmd = [
        "baumwelchtrain",
        *self.train_opts,
        self.ifar_path,
        self.ofar_path,
        rfst_path,
        afst_path,
    ]
    logging.debug("Subprocess call:\t%s", " ".join(cmd))
    with subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True) as proc:
      for line in proc.stderr:  # type: ignore
        match = re.fullmatch(
            r"INFO: Iteration (\d+) total likelihood:\s+(-?\d*(\.\d*)?)",
            line.rstrip(),
        )
        if not match:
          # Usually a log of the step's likelihood.
          continue
        iteration = int(match.group(1))
        likelihood = float(match.group(2))
        logging.debug(
            "start:\t%3d;\titer:\t%3d;\tLL:\t%.4f;\telapsed:\t%3ds",
            self.idx,
            iteration,
            likelihood,
            time.time() - start,
        )
    _rmtemp(rfst_path)
    return afst_path, likelihood


def _train_aligner(
    ifar_path: str,
    ofar_path: str,
    cg_path: str,
    seed: int,
    random_starts: int,
    processes: int,
    batch_size: Optional[int] = None,
    delta: Optional[float] = None,
    alpha: Optional[float] = None,
    max_iters: Optional[int] = None,
) -> str:
  """Trains the aligner.

  NB: many arguments inherit default values from the `baumwelchtrain` tool.

  Args:
    ifar_path: path to the input FAR.
    ofar_path: path to the output FAR.
    cg_path: path to the convering grammar FST.
    seed: integer random seed.
    random_starts: number of random starts.
    processes: maximum number of processes running concurrently.
    batch_size: batch size (default: from `baumwelchtrain`).
    delta: comparison/quantization delta (default: from `baumwelchtrain`).
    alpha: learning rate (default: from `baumwelchtrain`).
    max_iters: maximum number of iterations (default: from `baumwelchtrain`).

  Returns:
    The path to the aligner FST.
  """
  train_opts: List[str] = []
  if batch_size:
    train_opts.append(f"--batch_size={batch_size}")
  if delta:
    train_opts.append(f"--delta={delta}")
  if alpha:
    train_opts.append(f"--alpha={alpha}")
  if max_iters:
    train_opts.append(f"--max_iters={max_iters}")
  random.seed(seed)
  # Each random start is associated with a randomly chosen unique integer in
  # the range [1, RAND_MAX).
  starts = [
      RandomStart(idx, seed, ifar_path, ofar_path, cg_path, train_opts)
      for idx, seed in enumerate(
          random.sample(range(1, RAND_MAX), random_starts), 1
      )
  ]
  with multiprocessing.Pool(processes) as pool:
    # Setting chunksize to 1 means that random starts are processed
    # in roughly the order you'd expect.
    pairs = pool.map(RandomStart.train, starts, chunksize=1)
  # Finds best aligner; we `min` because this is in negative log space.
  best_aligner_path, best_likelihood = min(pairs, key=operator.itemgetter(1))
  logging.debug("Best aligner:\t%s", best_aligner_path)
  logging.debug("Best likelihood:\t%.4f", best_likelihood)
  # Deletes suboptimal aligner FSTs.
  for aligner_path, _ in pairs:
    if aligner_path == best_aligner_path:
      continue
    _rmtemp(aligner_path)
  _rmtemp(cg_path)
  return best_aligner_path


def _align_encode(
    ifar_path: str, ofar_path: str, afst_path: str
) -> Tuple[str, str]:
  """Computes the unweighted alignments FAR.

  Args:
    ifar_path: path to the input FAR.
    ofar_path: path to the output FAR.
    afst_path: path to the aligner FST.

  Returns:
     A (path to the encoded FAR, path to the encoder) tuple.
  """
  afar_path = _mktemp("a.far")
  _log_check_call(
      ["baumwelchdecode", ifar_path, ofar_path, afst_path, afar_path]
  )
  _rmtemp(ifar_path)
  _rmtemp(ofar_path)
  efar_path = _mktemp("e.far")
  reader = pywrapfst.FarReader.open(afar_path)
  writer = pywrapfst.FarWriter.create(efar_path)
  mapper = pywrapfst.EncodeMapper(reader.arc_type(), encode_labels=True)
  while not reader.done():
    # Removes weights, encodes labels, and compacts. One can convert to the
    # most compact representation possible by doing this all at once.
    fst = pywrapfst.arcmap(reader.get_fst(), map_type="rmweight")
    fst.encode(mapper)
    fst = pywrapfst.convert(fst, "compact_string")
    writer.add(reader.get_key(), fst)
    reader.next()
  assert not reader.error()
  del reader
  assert not writer.error()
  del writer
  _rmtemp(afar_path)
  encoder_path = _mktemp("encoder")
  mapper.write(encoder_path)
  return efar_path, encoder_path


def _compile_pair_ngram(
    efar_path: str,
    encoder_path: str,
    ofst_path: str,
    order: Optional[int] = None,
    size: Optional[int] = None,
) -> None:
  """Compiles the pair n-gram model.

  Args:
    efar_path: path to the encoded FAR.
    encoder_path: path to the encoder.
    ofst_path: path for the pair n-gram FST.
    order: n-gram model order (default: from `ngramcount`).
    size: n-gram model size to prune to (default: no pruning).
  """
  cfst_path = _mktemp("c.fst")
  cmd = ["ngramcount", "--require_symbols=false"]
  if order:
    cmd.append(f"--order={order}")
  cmd.append(efar_path)
  cmd.append(cfst_path)
  _log_check_call(cmd)
  mfst_path = _mktemp("m.fst")
  _log_check_call(["ngrammake", "--method=kneser_ney", cfst_path, mfst_path])
  _rmtemp(cfst_path)
  if size:
    sfst_path = _mktemp("s.fst")
    _log_check_call([
        "ngramshrink",
        "--method=relative_entropy",
        f"--target_number_of_ngrams={size}",
        mfst_path,
        sfst_path,
    ])
    _rmtemp(mfst_path)
  else:
    sfst_path = mfst_path
  _log_check_call(["fstencode", "--decode", sfst_path, encoder_path, ofst_path])
  _rmtemp(encoder_path)
  _rmtemp(sfst_path)


def main(args: argparse.Namespace) -> None:
  logging.info("Compiling FARs")
  ifar_path, ofar_path = _compile_fars(
      args.tsv, args.input_token_type, args.output_token_type
  )
  logging.info("Compiling covering grammar")
  compiler = CoveringGrammarCompiler(
      ilabels=get_labels(pywrapfst.FarReader.open(ifar_path)),
      olabels=get_labels(pywrapfst.FarReader.open(ofar_path)),
      insertions=args.insertions,
      deletions=args.deletions,
  )
  cg_path = _mktemp("cg.fst")
  compiler.compile().write(cg_path)
  logging.info("Training aligner")
  aligner_path = _train_aligner(
      ifar_path,
      ofar_path,
      cg_path,
      args.seed,
      args.random_starts,
      args.processes,
      args.batch_size,
      args.delta,
      args.alpha,
      args.max_iters,
  )
  logging.info("Aligning and encoding data")
  efar_path, encoder_path = _align_encode(ifar_path, ofar_path, aligner_path)
  logging.info("Compiling pair n-gram model")
  _compile_pair_ngram(efar_path, encoder_path, args.fst, args.order, args.size)


if __name__ == "__main__":
  logging.basicConfig(level="INFO", format="%(levelname)s:\t%(message)s")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--tsv",
      required=True,
      help="path to source TSV file",
  )
  parser.add_argument(
      "--input_token_type",
      default="utf8",
      help=(
          "input side token type, or the path to the input symbol table "
          "(default: %(default)s)"
      ),
  )
  parser.add_argument(
      "--output_token_type",
      default="utf8",
      help=(
          "output side token type, or the path to the output symbol table "
          "(default: %(default)s)"
      ),
  )
  parser.add_argument(
      "--insertions",
      default=1,
      type=int,
      help="number of consecutive insertions (default: %(default)s)",
  )
  parser.add_argument(
      "--deletions",
      default=1,
      type=int,
      help="number of consecutive deletions (default: %(default)s)",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=time.time_ns(),
      help="random seed (default: current time)",
  )
  parser.add_argument(
      "--random_starts",
      type=int,
      default=10,
      help="number of random starts (default: %(default)s)",
  )
  parser.add_argument(
      "--processes",
      type=int,
      default=multiprocessing.cpu_count(),
      help="maximum number of concurrent processes (default: %(default)s)",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      help="EM batch size (default: from `baumwelchtrain`)",
  )
  parser.add_argument(
      "--delta",
      type=float,
      help="EM comparison delta (default: from `baumwelchtrain`)",
  )
  parser.add_argument(
      "--alpha",
      type=float,
      help="EM learning rate (default: from `baumwelchtrain`)",
  )
  parser.add_argument(
      "--max_iters",
      type=int,
      help="EM maximum number of iterations (default: from `baumwelchtrain`)",
  )
  # Pair n-gram options.
  parser.add_argument(
      "--order",
      type=int,
      help="n-gram model order (default: from `ngramcount`)",
  )
  parser.add_argument(
      "--size",
      type=int,
      help="n-gram model size to prune to (default: no pruning)",
  )
  parser.add_argument(
      "--fst",
      required=True,
      help="path for sink covering grammar FST file",
  )
  main(parser.parse_args())
