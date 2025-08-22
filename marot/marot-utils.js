// Copyright 2025 The Google Research Authors.
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
   * ending in line-breaks/para-breaks) into subparas. Each subpara is either a
   * paragraph or a contiguous part of one, consisisting of full sentences. It
   * has no more sentences than maxSubparaSentences and no more tokens than
   * maxSubparaTokens (the tokens constraint may not be met for subparas that
   * only have one sentence).
   *
   * If maxSubparaSentences or maxSubparaTokens is negative, then no segment
   * splitting is performed.
   *
   * @param {!Array<!Object>} sentenceSplits Each object has num_tokens and
   *   possibly ends_with_para_break/ends_with_line_break. Such an object
   *   can be created using MarotUtils.tokenizeText().
   * @param {number=} maxSubparaSentences
   * @param {number=} maxSubparaTokens
   * @return {!Array<!Array<number>>} Each subpara is a sequence of
   *   sentence indices.
   */
  static makeSubparas(
      sentenceSplits, maxSubparaSentences=4, maxSubparaTokens=400) {
    const forceUnbroken = maxSubparaSentences < 0 || maxSubparaTokens < 0;
    /**
     * We start with breaking subparas after every sentence that already has
     * ends_with_para_break set. We also create subparas out of evey
     * sequence of consecutive sentences that all have ends_with_line_break set.
     * If such a sequence does not follow a paragraph break, the the first
     * sentence in the sequence is left at the end of the preceding subpara.
     */
    const preSplitParas = [];
    let currPara = [];
    let inSeqOfSentsWithLineBreaks = false;
    for (let s = 0; s < sentenceSplits.length; s++) {
      const sentence = sentenceSplits[s];
      const hasParaBreak = sentence.ends_with_para_break ?? false;
      const hasLineBreak = !hasParaBreak &&
                           (sentence.ends_with_line_break ?? false);
      const hasBreak = hasParaBreak || hasLineBreak ||
                       (s == sentenceSplits.length - 1);
      let currSentAlreadyPlaced = false;
      if (forceUnbroken) {
        currPara.push(s);
        continue;
      }
      if (((inSeqOfSentsWithLineBreaks && !hasBreak) ||
           (!inSeqOfSentsWithLineBreaks && hasLineBreak)) &&
          (currPara.length > 0)) {
        if (hasLineBreak) {
          /**
           * Make the first sentence in the sequence be a part of the
           * preceding subpara.
           */
          currPara.push(s);
          currSentAlreadyPlaced = true;
        }
        preSplitParas.push(currPara);
        currPara = [];
      }
      if (!currSentAlreadyPlaced) {
        currPara.push(s);
      }
      if (hasParaBreak) {
        console.assert(!currSentAlreadyPlaced);
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
       * Note that the maxSubparaTokens constraint is only checked if there is
       * more than one sentence.
       */
      return (!forceUnbroken && ((p.length > maxSubparaSentences) ||
              ((p.length > 1) && (tokenCount(p) > maxSubparaTokens))));
    };

    /**
     * At the end of some agressive splitting by splitAsNeeded(), it may be
     * possible to combine some consecutive small subparas. A typical case is
     * that of the first few subparas having 2 sentences each, followed by a
     * string of 1-sentence subparas, amongst which some can be combined. We
     * do the merging greedily, extending each subpara to subsume any
     * following subparas without violating the constraints.
     *
     * @param {!Array<!Array<number>>} subparas
     * @return {!Array<!Array<number>>}
     */
    const maybeCombineSome = (subparas) => {
      const newSubparas = [];
      let start = 0;
      while (start < subparas.length) {
        let subpara = subparas[start];
        let end = start;
        while ((end + 1) < subparas.length) {
          const combinedSubpara = subpara.concat(subparas[end + 1]);
          if (needsSplit(combinedSubpara)) {
            break;
          }
          subpara = combinedSubpara;
          end = end + 1;
        }
        newSubparas.push(subpara);
        start = end + 1;
      }
      return newSubparas;
    };

    /**
     * Try to split p (an array of sentence indices forming a para) into
     * "subparas", roughly evenly.
     *
     * At the end, combine any unnecessarily small consecutive subparas
     * greedily.
     *
     * @param {!Array<number>} p
     * @return {!Array<!Array<number>>}
     */
    const splitAsNeeded = (p) => {
      console.assert(p.length > 0, p);
      /**
       * Use the smallest value of numParts at which each split passes
       * constraints. If numParts reaches p.length, then the constraints
       * will be trivially met as each subpara will have just one sentence.
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
             * length 1 and needsSplit() will fail (as maxSubparaSentences is at
             * least 1).
             */
            passesConstraints = false;
            break;
          }
          splits.push(paraSlice);
          start += len;
        }
        if (passesConstraints) {
          return maybeCombineSome(splits);
        }
      }
      console.log('Error: code should never have reached here!');
      return [];
    };

    const subparas = [];
    for (const para of preSplitParas) {
      const splits = splitAsNeeded(para);
      for (const p of splits) {
        subparas.push(p);
      }
    }
    return subparas;
  }

  /**
   * Returns the location of elt in sorted array arr using binary search. if
   * elt is not present in arr, then returns the slot where it belongs in sorted
   * order. To be precise, it returns the smallest x such that arr[x] <= elt.
   * @param {!Array<number>} arr Sorted array of numbers. If arr is empty or
   * if elt < arr[0], then 0 is returned.
   *
   * @param {number} elt
   * @return {number}
   */
  static binSearch(arr, elt) {
    let l = 0;
    let r = arr.length;
    while (l < r) {
      const m = Math.floor((l + r) / 2);
      if (arr[m] < elt) {
        l = m + 1;
      } else {
        r = m;
      }
    }
    while (l > 0 && arr[l - 1] == arr[l]) {
      l--;
    }
    return l;
  }

  /**
   * Returns a string that is a hash of the input and also suitable as an
   * HTML identifier.
   * @param {string} str
   * @return {string}
   */
  static javaHashKey(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      let c = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + c;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(36);
  }

  /**
   * Splits text into sentences. The splitting is done using characters in
   * Unicode's "Sentence_Terminal" class. Some heuristics are used for the
   * period character (see details in the code). Guarantees that joining the
   * returned array on '' would recreate the original text.
   *
   * Useful in Marot for data that's not already sentence-segmented.
   *
   * @param {string} text
   * @return {!Array<string>}
   */
  static splitIntoSentences(text) {
    const splitters = new RegExp('\\p{Sentence_Terminal}', 'u');
    const isUpper = new RegExp('\\p{Uppercase}', 'u');
    const isLetter = new RegExp('\\p{Alphabetic}', 'u');
    const spaces = /\s/;
    const sentences = [];
    let start = 0;
    let i = 0;
    while (i < text.length) {
      const c = text[i];
      i++;
      if (!splitters.test(c)) {
        continue;
      }
      if (c == '.') {
        /**
         * Take some extra care. The period should be followed by a space and
         * the token preceding the period (i.e., the text from after the last
         * space before the period or from the start if there is no intervening
         * space) must satisfy a few heuristics:
         * - It should be non-empty.
         * - It should not contain any periods.
         * - If it begins with an uppercase letter then its length must be
         *   at least 4 (to rule out "Dr." "Mr." "Mrs." "PhD." etc.).
         * This means that we miss out on identifying some
         * period-sentence-endings such as "... in May."
         */
        if (i >= text.length || !spaces.test(text[i])) {
          continue;
        }
        /**
         * i >= 1 and text[i - 1] == '.'  and text[i] == ' '
         */
        const periodIndex = i - 1;
        let prePeriodTokenLen = 0;
        let cLast = '';
        let maybeExtendTokenStartTo;
        while ((maybeExtendTokenStartTo =
                    periodIndex - (prePeriodTokenLen + 1)) >= 0 &&
               !spaces.test((cLast = text[maybeExtendTokenStartTo])) &&
               (cLast != '.')) {
          prePeriodTokenLen++;
        }
        if (cLast == '.' || prePeriodTokenLen == 0 ||
            (isUpper.test(text[periodIndex - prePeriodTokenLen]) &&
             (prePeriodTokenLen <= 3))) {
          continue;
        }
      }
      /** Include trailing punctuation and spaces */
      while (i < text.length && !isLetter.test(text[i])) {
        i++;
      }
      sentences.push(text.substring(start, i));
      start = i;
    }
    if (start < text.length) {
      sentences.push(text.substr(start));
    }
    return sentences;
  }

  /**
   * Splits text into sentences and tokens, also capturing any paragraph breaks
   * (pair of consecutive newlines) and line breaks (single newline) as
   * properties of the sentences (sentence ending is forced at both such
   * breaks). Sentence-splitting is done via splitIntoSentences(), within text
   * spans identified by paragraph breaks and line breaks.
   *
   * The returned tokens guarantee that (1) tokens.join('') == text, (2) no
   * token is the empty string. There can be consecutive space tokens.
   *
   * Useful in Marot for data that's not already tokenized/sentence-segmented.
   *
   * @param {string} text
   * @return {!Object} Contains tokens and sentence_splits array properties.
   *     Each sentence_split is an object containing:
   *         offset: index in tokens[] of the starting token
   *         num_tokens: number of tokens in the sentence
   *         sentence: the full text of the sentence
   *         ends_with_para_break: boolean
   *         ends_with_line_break: boolean
   */
  static tokenizeText(text) {
    const ret = {
      tokens: [],
      sentence_splits: []
    };
    const paragraphs = text.split('\n\n');
    let offset = 0;
    for (let p = 0; p < paragraphs.length; p++) {
      const para = paragraphs[p];
      const lines = para.split('\n');
      if (p < paragraphs.length - 1) {
        if (lines.length == 0) {
          lines.push('\n\n');
        } else {
          lines[lines.length - 1] = lines[lines.length - 1] + '\n\n';
        }
      }
      for (let l = 0; l < lines.length; l++) {
        const line = lines[l];
        const sentences = MarotUtils.splitIntoSentences(line);
        if (l < lines.length - 1) {
          if (sentences.length == 0) {
            sentences.push('\n');
          } else {
            sentences[sentences.length - 1] =
                sentences[sentences.length - 1] + '\n';
          }
        }
        for (const sentence of sentences) {
          const spacedTokens = sentence.split(' ');
          let numSentTokens = 0;
          let lastSpacedToken = '';
          for (let t = 0; t < spacedTokens.length; t++) {
            const token = spacedTokens[t];
            if (token) {
              ret.tokens.push(token);
              numSentTokens++;
              lastSpacedToken = token;
            }
            if (t < spacedTokens.length - 1) {
              ret.tokens.push(' ');
              numSentTokens++;
            }
          }
          const splitInfo = {
            num_tokens: numSentTokens,
            offset: offset,
            sentence: sentence,
          };
          offset += numSentTokens;
          if (lastSpacedToken.endsWith('\n\n')) {
            splitInfo.ends_with_para_break = true;
          } else if (lastSpacedToken.endsWith('\n')) {
            splitInfo.ends_with_line_break = true;
          }
          ret.sentence_splits.push(splitInfo);
        }
      }
    }
    return ret;
  }

}

/**
 * Class for doing a simple token-offset-based alignment between a source text
 * segment and a target text segment, both made up of subparas. To use it,
 * construct a MarotAligner object, passing it the detailed strutures of
 * source/target. The structures should be created using the static function
 * MarotAligner.getAlignmentStructure(), and can be reused (for example, to
 * align a common source segment with multiple translation segments).
 *
 * If the source and target sides have an equal number of "natural paragraphs"
 * (subparas ending in line-break/para-break), then the token-offset-based
 * alignment is done *within* this natural alignment.
 *
 * The token-offset-based alignment algorithms simply maps each target token,
 * using its fractional relative position, to a source subpara that covers the
 * same fractional relative token position on the source side. A target sentence
 * or subpara is mapped to the range of source subparas defined by mapping the
 * first and last target tokens in the target sentence or subpara.
 */
class MarotAligner {
  /**
   * Both parameters should be created using
   *   MarotAligner.getAlignmentStructure().
   * @param {!Object} srcStructure
   * @param {!Object} tgtStructure
   */
  constructor(srcStructure, tgtStructure) {
    this.srcStructure = srcStructure;
    this.tgtStructure = tgtStructure;
    this.hasAlignedParagraphs =
        (this.srcStructure.naturalParas.length > 1) &&
        (this.srcStructure.naturalParas.length ==
         this.tgtStructure.naturalParas.length);
  }

  /**
   * The basis of the alignment: return the mapped source subpara index for a
   * 0-based target token index. In the rare/degenrate case that a target token
   * is not mapped to any source subpara (for example, if the source segment is
   * empty), return -1.
   *
   * @param {number} tgtToken Must be >= 0 and < the number of target tokens.
   * @return {number}
   */
  tgtTokenToSrcSubpara(tgtToken) {
    console.assert(tgtToken >= 0 && tgtToken < this.tgtStructure.num_tokens,
                   tgtToken);
    if (this.srcStructure.subparas.length == 0) {
      return -1;
    }
    let srcRange = [0, this.srcStructure.subparas.length - 1];
    let tgtRange = [0, this.tgtStructure.subparas.length - 1];
    console.assert(this.tgtStructure.subparas.length >= 1);
    const tgtTokenNumber = tgtToken + 1;
    if (this.hasAlignedParagraphs) {
      const naturalParaIndex = MarotUtils.binSearch(
          this.tgtStructure.naturalParaTokens, tgtTokenNumber);
      console.assert(naturalParaIndex < this.tgtStructure.naturalParas.length);
      srcRange = this.srcStructure.naturalParas[naturalParaIndex].range;
      tgtRange = this.tgtStructure.naturalParas[naturalParaIndex].range;
    }
    const srcOffset = this.srcStructure.subparas[srcRange[0]].offset;
    const srcCount = this.srcStructure.subparas[srcRange[1]].offset -
                     srcOffset +
                     this.srcStructure.subparas[srcRange[1]].num_tokens;
    const tgtOffset = this.tgtStructure.subparas[tgtRange[0]].offset;
    const tgtCount = this.tgtStructure.subparas[tgtRange[1]].offset -
                     tgtOffset +
                     this.tgtStructure.subparas[tgtRange[1]].num_tokens;
    console.assert(srcCount > 0, srcCount);
    console.assert(tgtCount > 0, tgtCount);
    const srcTokenNumber = srcOffset +
        Math.round((tgtTokenNumber - tgtOffset) * srcCount / tgtCount);
    console.assert(srcTokenNumber >= 0 &&
                   srcTokenNumber <= srcOffset + srcCount, srcTokenNumber);
    return MarotUtils.binSearch(
        this.srcStructure.subparaTokens, srcTokenNumber);
  }

  /**
   * Return the mapped source subpara index range [first, last] for a 0-based
   * target token index range. If either side maps to -1 (because of degenerate
   * cases), then trim from that end. The only degenerate return value is
   * [-1, -1], when such trimming eats up the whole range.
   *
   * @param {number} tgtTokenFirst Must be >= 0, < the number of target tokens.
   * @param {number} tgtTokenLast Must be >= 0, < the number of target tokens.
   * @return {!Array<number>}
   */
  tgtTokenRangeToSrcSubparaRange(tgtTokenFirst, tgtTokenLast) {
    console.assert(tgtTokenFirst <= tgtTokenLast);
    let rangeStart, rangeEnd;
    while ((rangeStart = this.tgtTokenToSrcSubpara(tgtTokenFirst)) < 0 &&
           tgtTokenFirst < tgtTokenLast) {
      tgtTokenFirst++;
    }
    while ((rangeEnd = this.tgtTokenToSrcSubpara(tgtTokenLast)) < 0 &&
           tgtTokenFirst < tgtTokenLast) {
      tgtTokenLast--;
    }
    return [rangeStart, rangeEnd];
  }

  /**
   * Return the mapped source subpara index range (both ends included) for a
   * 0-based target sentence index. Returns [-1, -1] in the degenerate case when
   * no target token in the sentence maps to any source subpara.
   *
   * @param {number} tgtSentence
   * @return {!Array<number>} The returned value is a pair,
   *    [firstSubparaIndex, lastSubparaIndex]
   */
  tgtSentenceToSrcSubparaRange(tgtSentence) {
    console.assert(tgtSentence >= 0 &&
                   tgtSentence < this.tgtStructure.sentences.length,
                   tgtSentence);
    const tgtSentInfo = this.tgtStructure.sentences[tgtSentence];
    const firstToken = tgtSentInfo.offset;
    const lastToken = tgtSentInfo.offset + tgtSentInfo.num_tokens - 1;
    return this.tgtTokenRangeToSrcSubparaRange(firstToken, lastToken);
  }

  /**
   * Return the mapped source subpara index range (both ends included) for a
   * 0-based target subpara index. Returns [-1, -1] in the degenerate case when
   * no target token in the subpara maps to any source subpara.
   *
   * @param {number} tgtSubpara
   * @return {!Array<number>} The returned value is a pair,
   *    [firstSubparaIndex, lastSubparaIndex]
   */
  tgtSubparaToSrcSubparaRange(tgtSubpara) {
    console.assert(tgtSubpara >= 0 &&
                   tgtSubpara < this.tgtStructure.subparas.length, tgtSubpara);
    const tgtSubparaInfo = this.tgtStructure.subparas[tgtSubpara];
    const firstToken = tgtSubparaInfo.offset;
    const lastToken = tgtSubparaInfo.offset + tgtSubparaInfo.num_tokens - 1;
    return this.tgtTokenRangeToSrcSubparaRange(firstToken, lastToken);
  }

  /**
   * Create a structure containing natural paragraph splits as well as
   * cumulative token offsets for sentences, subparas, and natural paragraphs.
   * This structure is passed to construct MarotAligner objects for aligning
   * a source segment with a target segment.
   *
   * If sentence structures include num_chars, then it is aggregated into
   * the retured structure's subparas members too.
   *
   * @param {!Array<!Object>} sentences Array of objects, each containing
   *     num_tokens and optional booleans starts_with_line/para_break
   * @param {!Array<number>} subparas Array of sentence indices comprising
   *     subparas, as returned by MarotUtils.makeSubparas()
   * @return {!Object} The structure needed to construct a MarotAligner
   */
  static getAlignmentStructure(sentences, subparas) {
    let offset = 0;
    let sIndex = 0;
    const structure = {
      sentences: sentences,
      naturalParas: [],
      subparas: [],
      naturalParaTokens: [],
      subparaTokens: [],
      sentenceTokens: [],
      num_tokens: 0,
    };
    for (let pi = 0; pi < subparas.length; pi++) {
      const p = subparas[pi];
      console.assert(p.length > 0);
      const subpara = {
        index: pi,
        offset: offset,
        start_sentence: sIndex,
        sentences: [],
        num_tokens: 0,
      };
      structure.subparas.push(subpara);
      let num_chars = 0;
      let sentences_with_num_chars = 0;
      for (const s of p) {
        console.assert(sIndex == s, sIndex, s);
        sIndex++;
        const sentence = sentences[s];
        sentence.index = s;
        sentence.subpara = pi;
        sentence.offset = offset;
        subpara.sentences.push(sentence);
        subpara.num_tokens += sentence.num_tokens;
        offset += sentence.num_tokens;
        structure.sentenceTokens.push(offset);
        if (sentence.hasOwnProperty('num_chars')) {
          sentences_with_num_chars++;
          num_chars += sentence.num_chars;
        }
      }
      if (sentences_with_num_chars == p.length) {
        subpara.num_chars = num_chars;
      }
      structure.subparaTokens.push(offset);
      const lastSent = sentences[p[p.length - 1]];
      subpara.ends_with_line_break = lastSent.ends_with_line_break ?? false;
      subpara.ends_with_para_break = lastSent.ends_with_para_break ?? false;
    }
    structure.num_tokens = offset;
    /**
     * Now find the "natural" paragraph, i.e., sequences of subparas with the
     * last one ending in a line/para break. When we align source and
     * translation, we'll align the natural paragraphs first, but only if there
     * are an equal number of them on both sides.
     */
    structure.naturalParas = [];
    structure.naturalParaTokens = [];
    let startSubparaIndex = 0;
    for (let pi = 0; pi < subparas.length; pi++) {
      const subpara = structure.subparas[pi];
      if ((pi == subparas.length - 1) ||
          subpara.ends_with_line_break || subpara.ends_with_para_break) {
        const naturalPara = {
          offset: structure.subparas[startSubparaIndex].offset,
          num_tokens: 0,
        };
        structure.naturalParas.push(naturalPara);
        for (let pj = startSubparaIndex; pj <= pi; pj++) {
          naturalPara.num_tokens += structure.subparas[pj].num_tokens;
        }
        naturalPara.range = [startSubparaIndex, pi];
        startSubparaIndex = pi + 1;
        structure.naturalParaTokens.push(
            naturalPara.offset + naturalPara.num_tokens);
      }
    }
    return structure;
  }
}