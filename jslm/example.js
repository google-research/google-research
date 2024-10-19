// Copyright 2024 The Google Research Authors.
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

/**
 * @fileoverview Simple example on how to use the language modeling API.
 */

const assert = require("assert");
const hist = require("./histogram_language_model");
const polya = require("./polya_tree_language_model");
const ppm = require("./ppm_language_model");
const vocab = require("./vocabulary");

/*
 * Create a small vocabulary.
 */
let v = new vocab.Vocabulary();
v.addSymbol("a");
v.addSymbol("b");

const a_id = v.symbols_.indexOf("a");
const b_id = v.symbols_.indexOf("b");

/*
 * --------------------------------------------------------
 * Build the PPM language model trie and update the counts.
 * --------------------------------------------------------
 */
// Build the trie from a single string "ab".
const maxOrder = 5;
let lm = new ppm.PPMLanguageModel(v, maxOrder);
c = lm.createContext();
lm.addSymbolAndUpdate(c, a_id);
lm.addSymbolAndUpdate(c, b_id);
console.log("Initial count trie:");
lm.printToConsole();

/*
 * ---------------------------------
 * Check static (non-adaptive) mode.
 * ---------------------------------
 */
// In the example below we always ignore the 0th symbol. It is a special symbol
// corresponding to the root of the trie.
c = lm.createContext();
let probs = lm.getProbs(c);
assert(probs.length == 3, "Expected \"a\", \"b\" and root");

// Nothing has been entered yet. Since we've observed both "a" and "b", there is
// an equal likelihood of getting either.
assert(probs[1] > 0 && probs[1] == probs[2],
       "Probabilities for both symbols should be equal");
console.log(probs);

// Enter "a" and check the probability estimates. Since we've seen the sequence
// "ab" during the training, the "b" should be more likely than "a".
lm.addSymbolToContext(c, a_id);
probs = lm.getProbs(c);
assert(probs[1] > 0 && probs[1] < probs[2],
       "Probability for \"b\" should be more likely");
console.log(probs);

// Enter "b". The context becomes "ab". Now it's back to square one: Any symbol
// is likely again.
lm.addSymbolToContext(c, b_id);
probs = lm.getProbs(c);
assert(probs[1] > 0 && probs[1] == probs[2],
       "Probabilities for both symbols should be equal");
console.log(probs);

// Try to enter "ba". Since the model has only observed "ab" sequence, it is
// expecting the next most probable symbol to be "b".
c = lm.createContext();
lm.addSymbolToContext(c, b_id);
lm.addSymbolToContext(c, a_id);
probs = lm.getProbs(c);
assert(probs[1] > 0 && probs[2] > probs[1],
       "Probability for \"b\" should be more likely");
console.log(probs);

/*
 * -------------------------------------------------------------------------
 * Check adaptive mode in which the model is updated as symbols are entered.
 * -------------------------------------------------------------------------
 */
lm = new ppm.PPMLanguageModel(v, maxOrder);  // Re-create.

// Enter "a" and update the model. At this point the frequency for "a" is
// higher, so it's more probable.
c = lm.createContext();
lm.addSymbolAndUpdate(c, a_id);
probs = lm.getProbs(c);
assert(probs[1] > 0 && probs[1] > probs[2],
       "Probability for \"a\" should be more likely");
console.log(probs);

// Enter "b" and update the model. At this point both symbols should become
// equally likely again.
lm.addSymbolAndUpdate(c, b_id);
probs = lm.getProbs(c);
assert(probs[1] > 0 && probs[1] == probs[2],
       "Probabilities for both symbols should be the same");
console.log(probs);

// Enter "b" and update the model. Current context "abb". Since we've seen
// "ab" and "abb" by now, the "b" becomes more likely.
lm.addSymbolAndUpdate(c, b_id);
probs = lm.getProbs(c);
console.log(probs);
assert(probs[1] > 0 && probs[1] < probs[2],
       "Probability for \"b\" should be more likely");

// Dump the final count trie.
console.log("Final count trie:");
lm.printToConsole();

/*
 * ---------------------------------------------------------
 * Build the histogram language model and update the counts.
 * ---------------------------------------------------------
 * Context here is meaningless.
 */
lm = new hist.HistogramLanguageModel(v);
probs = lm.getProbs(c);
assert(probs[1] == 0.5 && probs[2] == 0.5, "Expected equal probs");

let trainingData = "ababababab";
for (let i = 0; i < trainingData.length; ++i) {
  lm.addSymbolAndUpdate(c, v.symbols_.indexOf(trainingData[i]));
}
lm.printToConsole();
probs = lm.getProbs(c);
assert(probs.length == 3, "Wrong cardinality: " + probs.length);
assert(probs[1] > 0.0 && probs[1] == probs[2], "Expected two equal probs!");

lm.addSymbolAndUpdate(c, a_id);
probs = lm.getProbs(c);
assert(probs[2] > 0.0 && probs[1] > probs[2],
       "Probability for \"a\" should be higher");

lm.addSymbolAndUpdate(c, b_id);
probs = lm.getProbs(c);
assert(probs[1] > 0.0 && probs[1] == probs[2], "Expected two equal probs!");

lm.addSymbolAndUpdate(c, b_id);
probs = lm.getProbs(c);
assert(probs[1] > 0.0 && probs[1] < probs[2],
       "Probability for \"b\" should be higher");

/*
 * ------------------------------------
 * Build and test Polya language model.
 * ------------------------------------
 * Context here is meaningless.
 */

v = new vocab.Vocabulary();
v.addSymbol("a");
v.addSymbol("b");
v.addSymbol("c");
v.addSymbol("d");

const c_id = v.symbols_.indexOf("c");
const d_id = v.symbols_.indexOf("d");

lm = new polya.PolyaTreeLanguageModel(v);
console.log(lm.getPath_(a_id));
console.log(lm.getPath_(b_id));
console.log(lm.getPath_(c_id));
console.log(lm.getPath_(d_id));

c = lm.createContext();
lm.addSymbolAndUpdate(c, a_id);
lm.addSymbolAndUpdate(c, a_id);
lm.addSymbolAndUpdate(c, b_id);
lm.addSymbolAndUpdate(c, c_id);
lm.addSymbolAndUpdate(c, c_id);
lm.addSymbolAndUpdate(c, c_id);
lm.addSymbolAndUpdate(c, c_id);
lm.addSymbolAndUpdate(c, c_id);
lm.addSymbolAndUpdate(c, d_id);
lm.addSymbolAndUpdate(c, d_id);
lm.addSymbolAndUpdate(c, d_id);
lm.printToConsole();

probs = lm.getProbs(c);
console.log(probs);
