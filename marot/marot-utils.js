// Copyright 2023 The Google Research Authors.
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
 * @fileoverview Common utils for use in Marot as well as Anthea.
 */

class MarotUtils {
  /**
   * Splits a segment consisting of a sequence of sentences (with some possibly
   * ending in line-breaks/para-breaks) into paragraph-like subsequences called
   * "paralets." Each paralet is either a paragraph or a sub-paragraph. It
   * has no more sentences than maxParaSentences and no more tokens than
   * maxParaTokens (the tokens constraint may not be met for paralets that only
   * have one sentence).
   *
   * @param {!Array<!Object>} sentenceSplits Each object has num_tokens and
   *   possibly ends_with_para_break/ends_with_line_break.
   * @param {number=} maxParaSentences
   * @param {number=} maxParaTokens
   * @return {!Array<!Array<number>>} Each paragraph is a sequence of
   *   sentence indices.
   */
  static makeParalets(sentenceSplits, maxParaSentences=4, maxParaTokens=400) {
    console.assert(maxParaSentences > 0, maxParaSentences);
    console.assert(maxParaTokens > 0, maxParaTokens);
    /**
     * We start with breaking paragraphs after every sentence that already has
     * ends_with_para_break set. We also create paragraphs out of evey
     * sequence of consecutive sentences that all have ends_with_line_break set.
     */
    const preSplitParas = [];
    let currPara = [];
    let inSeqOfSentsWithLineBreaks = false;
    for (let s = 0; s < sentenceSplits.length; s++) {
      const sentence = sentenceSplits[s];
      const hasLineBreak = sentence.ends_with_line_break ?? false;
      if ((inSeqOfSentsWithLineBreaks != hasLineBreak) &&
          (currPara.length > 0)) {
        preSplitParas.push(currPara);
        currPara = [];
      }
      currPara.push(s);
      if (sentence.ends_with_para_break) {
        preSplitParas.push(currPara);
        currPara = [];
        inSeqOfSentsWithLineBreaks = false;
      } else {
        inSeqOfSentsWithLineBreaks = hasLineBreak;
      }
    }
    if (currPara.length > 0) {
      preSplitParas.push(currPara);
    }

    const tokenCount = (p) => {
      let count = 0;
      for (const s of p) {
        count += sentenceSplits[s].num_tokens;
      }
      return count;
    };

    const needsSplit = (p) => {
      /**
       * Note that the maxParaTokens constraint is only checked if there is
       * more than one sentence.
       */
      return ((p.length > maxParaSentences) ||
              ((p.length > 1) && (tokenCount(p) > maxParaTokens)));
    };

    const splitAsNeeded = (p) => {
      console.assert(p.length > 0, p);
      /**
       * Try to split p (an array of sentence indices forming a para) into
       * numParts "paralets", roughly evenly.
       *
       * Use the smallest value of numParts at which each split passes
       * constraints.  If numParts reaches p.length, then the constraints
       * will be trivially met as each paralet will have just one sentence.
       */
      for (let numParts = 1; numParts <= p.length; numParts++) {
        const partLen = Math.floor(p.length / numParts);
        const leftOver = p.length - (partLen * numParts);
        const splits = [];
        let passesConstraints = true;
        let start = 0;
        for (let s = 0; s < numParts; s++) {
          const len = partLen + (s < leftOver ? 1 : 0);
          const paraSlice = p.slice(start, start + len);
          if (needsSplit(paraSlice)) {
            /**
             * Note that when numParts == p.length, each para will have
             * length 1 and needsSplit() will fail (as maxParaSentences is at
             * least 1).
             */
            passesConstraints = false;
            break;
          }
          splits.push(paraSlice);
          start += len;
        }
        if (passesConstraints) {
          return splits;
        }
      }
      console.log('Error: code should never have reached here!');
      return [];
    };

    const paragraphs = [];
    for (const para of preSplitParas) {
      const splits = splitAsNeeded(para);
      for (const p of splits) {
        paragraphs.push(p);
      }
    }
    return paragraphs;
  }
}