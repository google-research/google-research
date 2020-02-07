// Copyright 2020 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Simple driver for the language model.
//
// Please note, the trie is memory hungry, but large training corpora it's not
// unusual to run out of memory using NodeJS. For example, for the training set
// of about 160M characters, the 7-gram trie consists of about 50M nodes. In
// order for the training to succeed, increase the RAM size by tweaking the
// --max-old-space-size parameter.
//
// Example:
// --------
// node --max-old-space-size=4096 language_model_driver.js \
//   7 train.txt test.txt

const assert = require("assert");
const fs = require("fs");
const v = require("./vocabulary");
const ppm = require("./ppm_language_model");

/**
 * Train from file.
 */
let myArgs = process.argv.slice(2);
assert(myArgs.length >= 3, "Max order (int), Training file (string) and test " +
       "files (string) are required!");
trainFile = myArgs[1];

// Initialize vocabulary.
console.log("Initializing vocabulary from " + trainFile + " ...");
let contents = fs.readFileSync(trainFile, 'utf8');
let vocab = new v.Vocabulary();
for (let i = 0; i < contents.length; ++i) {
  vocab.addSymbol(contents[i]);
}
console.log("Created vocabulary with " + vocab.size() + " symbols.");

// Train the model.
maxOrder = myArgs[0];
console.log("Constructing " + maxOrder + "-gram LM ...");
let lm = new ppm.PPMLanguageModel(vocab, maxOrder);
c = lm.createContext();
for (let i = 0; i < contents.length; ++i) {
  lm.addSymbolAndUpdate(c, vocab.symbols_.indexOf(contents[i]));
}
console.log("Created trie with " + lm.numNodes_ + " nodes.");

/**
 * Accumulators for statistics.
 */
class SequenceStats {
  constructor() {
    // Log-probability for the sequence computed using logarithm base 10 function.
    this.log10Prob_ = 0.0;

    // Log-probability for the sequence computed using binary logarithm.
    this.log2Prob_ = 0.0;

    // Entropy of the sequence: Estimated number of bits used to encode each
    // character.
    this.entropy_ = 0.0;

    // Sequence perplexity computed using base 10 logarithm in order to be
    // compatible with SRILM and the results reported in Daniel Rough, Keith
    // Vertanen, Per Ola Kristensson: "An evaluation of Dasher with a
    // high-performance language model as a gaze communication method",
    // Proc. 2014 International Working Conference on Advanced Visual Interfaces
    // (AVI), pp. 169--176, Italy.
    this.perplexity_ = 0.0;

    // Number of OOV symbols.
    this.numUnknownSymbols_ = 0;
  }
}

/**
 * Test using held-out data.
 */
const adaptiveMode = false;
testFile = myArgs[2];
console.log("Running over " + testFile + " ...");
test_lines = fs.readFileSync(testFile, "utf8").split("\n");
let corpusStats = new SequenceStats();
let numSymbols = 0;
for (let i = 0; i < test_lines.length; ++i) {
  if (test_lines[i].length <= 1) {
    continue;
  }
  c = lm.createContext();
  let sentStats = new SequenceStats();
  for (let j = 0; j < test_lines[i].length; ++j) {
    // Fetch symbol from the vocabulary.
    const symbol = vocab.getSymbolOrOOV(test_lines[i][j]);
    if (symbol == vocab.oovSymbol_) {
      sentStats.numUnknownSymbols_++;
    }

    // Obtain probabilities for all the symbols. The context is empty initially.
    let probs = lm.getProbs(c);
    let prob = probs[symbol];
    assert(prob > 0.0, "Invalid symbol probability: " + prob);
    sentStats.log10Prob_ += Math.log10(prob);
    sentStats.log2Prob_ += Math.log2(prob);

    // Update the context. In adaptive mode, the model is also updated in the
    // process.
    if (adaptiveMode) {
      lm.addSymbolAndUpdate(c, symbol);
    } else {
      lm.addSymbolToContext(c, symbol);
    }
  }

  // Update global statistics.
  corpusStats.log10Prob_ += sentStats.log10Prob_;
  corpusStats.log2Prob_ += sentStats.log2Prob_;
  corpusStats.numUnknownSymbols_ += sentStats.numUnknownSymbols_;
  numSymbols += test_lines[i].length;
}

// Compute the results for the entire test corpus:
//
// [1] Compute entropy (bits/character) via cross-entropy:
//       H = -\sum_{i=1}^N q(w_i) \log_2 p(w_i), where $p$ is the distribution
//     estimated by the model and $q$ is the "real" distribution. Assuming that
//     all words in the real distribution $q$ are equally likely, we end up
//     with:
//       H = -\frac{1}{N} \sum_{i=1}^N \log_2 p(w_i).
// [2] Compute the sequence perplexity via word entropy as
//       H = -\frac{1}{N} \log_{10} P(w_1,\ldots,w_N), PP = {10}^H.
corpusStats.entropy_ = -corpusStats.log2Prob_ / numSymbols;
corpusStats.perplexity_ = Math.pow(10.0, -corpusStats.log10Prob_ / numSymbols);
console.log("Results: numSymbols = " + numSymbols +
            ", ppl = " + corpusStats.perplexity_ +
            ", entropy = " + corpusStats.entropy_ + " bits/char" +
            ", OOVs = " + corpusStats.numUnknownSymbols_ +
            " (" + 100.0 * corpusStats.numUnknownSymbols_ / numSymbols + "%).");
