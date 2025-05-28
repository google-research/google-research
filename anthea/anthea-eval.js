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
 * All templates are instantiated as properties of the antheaTemplates
 * global object. By convention, template antheaTemplates['Some-TemplateName']
 * is set in template-some-templatename.js.
 */
const /** !Object */ antheaTemplates = {};

/**
 * GoogDOM is simplified and short derivative of goog.dom, used for
 *   convenience methods for creating HTML elements.
 */
class GoogDOM {
  constructor() {
    /** @private @const {!RegExp} */
    this.UNSAFE_RE_ = /<(?:math|script|style|svg|template)[^>]*>/i;
  }

  /**
   * Sets innerHTML of an element, throwing an exception if the passed markup
   *   seems unsafe.
   * @param {!Element} elt
   * @param {string} html
   */
  setInnerHtml(elt, html) {
    if (this.UNSAFE_RE_.test(html)) {
      throw 'SetInnerHtml found unsafe HTML';
    }
    elt.innerHTML = html;
  }

  /**
   * Creates a DOM element, accepting a variable number of args.
   *   The first argument should be the tag name, the second should be
   *   a class name or an array of class names or a dict of attributes.
   *   Any subsequent args are child elements.
   * @return {!Element}
   */
  createDom() {
    const tagName = String(arguments[0]);
    const attributes = arguments[1];
    const element = document.createElement(tagName);
    if (attributes) {
      if (typeof attributes === 'string') {
        element.className = attributes;
      } else if (Array.isArray(attributes)) {
        element.className = attributes.join(' ');
      } else {
        for (let k in attributes) {
          if (k == 'class') {
            element.className = attributes[k];
          } else {
            element.setAttribute(k, attributes[k]);
          }
        }
      }
    }
    if (arguments.length > 2) {
      this.append_(element, arguments, 2);
    }
    return element;
  }

  /**
   * Helper method for createDom(): appends each args[i] as a child to parent,
   *   starting at i = startIndex. args can include text as well as
   *   Elements.
   * @param {!Element} parent
   * @param {!Array} args
   * @param {number} startIndex
   * @return {!Element}
   */
  append_(parent, args, startIndex) {
    for (let i = startIndex; i < args.length; i++) {
      const child = args[i];
      if (child) {
        parent.appendChild(
            typeof child === 'string' ? document.createTextNode(child) : child);
      }
    }
  }
}

/** @const {!GoogDOM} Helper object for DOM utils */
const googdom = new GoogDOM;

/**
 * This is the data format for representing a document and its translation
 * done by a particular system (or human). The "annotations" array stores
 * any extra data provided beyond the mandatory 4 fields;
 */
class AntheaDocSys {
  constructor() {
    /** @public {string} */
    this.doc = '';
    /** @public {string} */
    this.sys = '';
    /** @public @const {!Array<string>} */
    this.srcSegments = [];
    /** @public @const {!Array<string>} */
    this.tgtSegments = [];
    /** @public @const {!Array<string>} */
    this.annotations = [];
    /** @public {number} */
    this.numNonBlankSegments = 0;
  }

  /**
   * Parse the passed contents of a TSV-formatted project file and return
   * an array of AntheaDocSys objects. The input data format should have four
   * tab-separated fields on each line:
   *
   *     source-segment
   *     target-segment
   *     document-name
   *     system-name
   *
   * Any data in an extra field (beyond these four) is stored in the annotations
   * array.
   *
   * Both source-segment and target-segment can together
   * be empty to indicate a paragraph break.
   * For convenience, a completely blank line (without the tabs
   * and without document-name and system-name) can also be used to indicate
   * a paragraph break.
   *
   * This function also unescapes any tabs ('\\t' -> '\t') and newlines
   * ('\\n' -> '\n') in source-segment/target-segment.
   *
   * The first line should contain a JSON object with the source and the target
   * language codes, and potentially other parameters. For instance:
   *   {"source_language": "en", "target_language": "fr"}
   *
   * @param {string} projectFileContents TSV-formatted text data.
   * @return {!Array<!AntheaDocSys>} An array of parsed AntheaDocSys objects.
   *   The parsed parameters from the first line as included as the
   *   "parameters" property of this array.
   */
  static parseProjectFileContents(projectFileContents) {
    const parsed = [];
    const lines = projectFileContents.split('\n');
    let parameters = {};
    let parsingErr = null;
    try {
      parameters = JSON.parse(lines[0]);
    } catch (err) {
      parameters = {};
      parsingErr = err;
    }
    /**
     * Convert hyphenated (legacy) to underscored property names.
     */
    const keys = Object.keys(parameters);
    for (let key of keys) {
      const underscored_key = key.replace(/-/g, '_');
      if (key == underscored_key) {
        continue;
      }
      parameters[underscored_key] = parameters[key];
      delete parameters[key];
    }
    if (!parameters['source_language'] || !parameters['target_language']) {
      throw 'The first line must be a JSON object and contain source and ' +
          'target language codes with keys "source_language" and ' +
          '"target_language".' +
          (parsingErr ? (' Parsing error: ' + parsingErr.toString() +
                         ' — Stack: ' + parsingErr.stack) : '');
    }
    const srcLang = parameters['source_language'];
    const tgtLang = parameters['target_language'];
    let docsys = new AntheaDocSys();
    const unescaper = (s) => s.replace(/\\n/g, '\n').replace(/\\t/g, '\t');
    for (let line of lines.slice(1)) {
      // Remove leading and trailing non-tab whitespace.
      line = line.replace(/^[^\S\t]+|[^\S\t]+$/g, '');
      const parts = line.split('\t', 5);
      if (!line || parts.length == 2) {
        /** The line may be blank or may have just doc+sys */
        docsys.srcSegments.push('');
        docsys.tgtSegments.push('');
        docsys.annotations.push('');
        continue;
      }
      if (parts.length < 4) {
        console.log('Skipping ill-formed text line: [' + line + ']');
        continue;
      }
      /** Note that we do not trim() srcSegment/tgtSegment. */
      const srcSegment = unescaper(parts[0]);
      const tgtSegment = unescaper(parts[1]);
      const doc = parts[2].trim() + ':' + srcLang + ':' + tgtLang;
      const sys = parts[3].trim();
      const annotation = parts.length > 4 ? parts[4].trim() : '';
      if (!doc || !sys) {
        console.log('Skipping text line with empty doc/sys: [' + line + ']');
        continue;
      }
      if (docsys.doc && (doc != docsys.doc || sys != docsys.sys)) {
        if (docsys.numNonBlankSegments > 0) {
          parsed.push(docsys);
        } else {
          console.log(
              'Skipping docsys with no non-empty segments: ' +
              JSON.stringify(docsys));
        }
        docsys = new AntheaDocSys();
      }
      docsys.doc = doc;
      docsys.sys = sys;
      docsys.srcSegments.push(srcSegment);
      docsys.tgtSegments.push(tgtSegment);
      docsys.annotations.push(annotation);
      if (srcSegment || tgtSegment) {
        docsys.numNonBlankSegments++;
      }
    }
    if (docsys.numNonBlankSegments > 0) {
      parsed.push(docsys);
    } else if (docsys.pragmas.length > 0 || docsys.doc) {
      console.log(
          'Skipping docsys with no non-empty segments: ' +
          JSON.stringify(docsys));
    }
    /**
     * Trim blank lines at the end.
     */
    for (const docsys of parsed) {
      console.assert(docsys.numNonBlankSegments > 0, docsys);
      console.assert(
          docsys.srcSegments.length == docsys.tgtSegments.length, docsys);
      console.assert(docsys.srcSegments.length > 0, docsys);
      console.assert(docsys.tgtSegments.length > 0, docsys);
      let l = docsys.srcSegments.length - 1;
      while (l > 0 && !docsys.srcSegments[l] && !docsys.tgtSegments[l]) {
        docsys.srcSegments.pop();
        docsys.tgtSegments.pop();
        l--;
      }
    }
    if (parsed.length == 0) {
      throw 'Did not find any properly formatted text lines in file';
    }
    /* Add project parameters as a property of the parsed array. */
    parsed.parameters = parameters;
    return parsed;
  }
}

/**
 * The AntheaCursor class keeps track of the current location (the "cursor")
 * during an eval. There is a location on the source side and one on the
 * target side. Each location consists of a segment index and a para index
 * within the segment. The para index points to a "subpara", which is a group of
 * sentences that forms the navigation unit for the rater. It is typically a
 * paragraph or a part of a paragraph that is not too big. The cursor further
 * remembers the max para index shown within each segment on each side.
 */
class AntheaCursor {
  /**
   * @param {!Array<!Object>} segments Each segment object should contain a
   *     "doc" field and arrays "srcSubparas" and "tgtSubparas".
   * @param {boolean} tgtOnly Set to true for monolingual evals.
   * @param {boolean} tgtFirst Set to true for target-first evals.
   * @param {boolean} sideBySide Set to true for sideBySide evals.
   * @param {function(number)} segmentDone Called with seg id for each segment.
   * @param {!Array<number>} presentationOrder Order in which to display docs.
   */
  constructor(segments, tgtOnly, tgtFirst, sideBySide, segmentDone,
              presentationOrder) {
    this.segments = segments;
    console.assert(segments.length > 0, segments.length);
    this.tgtOnly = tgtOnly;
    this.tgtFirst = tgtFirst;
    this.sideBySide = sideBySide;
    // numSides = 2 when not SideBySide (src, tgt)
    // numSides = 3 when is SideBySide  (src, tgt, tgt2)
    this.numSides = sideBySide ? 3 : 2;
    this.segmentDone_ = segmentDone;
    this.numSubparas = Array.from(Array(this.numSides), () => []);
    this.numSubparasShown = Array.from(Array(this.numSides), () => []);
    this.presentationOrder = presentationOrder;
    /** number identifying the current index within presentationOrder. */
    this.presentationIndex = 0;
    /** Array<number> identifying the starting seg for each doc. */
    this.docSegStarts = [];
    let doc = -1;
    // Get the number of subparas on each side for each segment.
    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      const sideSubparas = [segment.srcSubparas,
                            segment.tgtSubparas,
                            segment.tgtSubparas2 || []];
      for (let j = 0; j < this.numSides; j++) {
        this.numSubparas[j].push(sideSubparas[j].length);
        this.numSubparasShown[j].push(0);
      }
      if (this.docSegStarts.length == 0 || segment.doc != doc) {
        this.docSegStarts.push(i);
        doc = segment.doc;
      }
    }
    console.assert(doc == this.docSegStarts.length - 1);
    this.doc = 0;
    this.seg = 0;
    /** number that is 0 when the current side is src, and 1 when tgt.
     * In sideBySide mode, it is 2 when the current side is tgt2.
     */
    this.side = 0;
    /** number that is the index of the current subpara. */
    this.para = 0;
    const firstSeg = this.docSegStarts[this.presentationOrder[0]];
    this.updateSideOrder(firstSeg);
    this.goto(firstSeg, this.sideOrder[0], 0);
  }

  /**
   * Builds the order in which to display the columns.
   * Takes as input a segment Idx and uses that to update the sideOrder.
   * @param {!Array<number>} segmentIdx
   * @return {!Array<number>}
   */
  updateSideOrder(segmentIdx) {
    /* Define sideOrder for functions next(), prev(), and cycleSides().
     * Initialize with the target side(s).
     * Add in the source side depending on the mode only once
     * Updating only once is achieved by checking sideOrder length.*/
    this.sideOrder = this.segments[segmentIdx].tgtsOrder;
    if (this.tgtOnly || this.sideOrder.length >= this.numSides) {
      return;
    }
    if (this.tgtFirst) {
      this.sideOrder.push(0);
    } else {
      this.sideOrder.unshift(0);
    }
  }

  /**
   * Returns true if we are at the start of a doc.
   * @return {boolean}
   */
  atDocStart() {
    if (this.seg != this.docSegStarts[this.doc]) {
      return false;
    }
    const startSide = this.sideOrder[0];
    if (this.para != 0 || (this.side != startSide)) {
      return false;
    }
    return true;
  }

  /**
   * Returns true if the passed segment has been fully seen.
   * @param {number} seg
   * @return {boolean}
   */
  segFullySeen(seg) {
  for (let side of this.sideOrder) {
    if (this.numSubparasShown[side][seg] < this.numSubparas[side][seg]) {
      return false;
    }
  }
  return true;
  }

  /**
   * Returns true if we are at the end of a doc.
   * @return {boolean}
   */
  atDocEnd() {
    const endSide = this.sideOrder.at(-1);
    if (this.side != endSide) {
      return false;
    }
    if (this.para + 1 != this.numSubparas[endSide][this.seg]) {
      return false;
    }
    if (!this.segFullySeen(this.seg)) {
      return false;
    }
    if (this.seg != this.segments.length - 1 &&
        this.segments[this.seg + 1].doc == this.doc) {
      return false;
    }
    return true;
  }

  /**
   * Returns true if the cursor has already been at the current doc's end.
   * @return {boolean}
   */
  seenDocEnd() {
    const endSeg = (this.doc + 1 < this.docSegStarts.length) ?
                   this.docSegStarts[this.doc + 1] - 1 :
                   this.segments.length - 1;
    return this.segFullySeen(endSeg);
  }

  /**
   * Moves the cursor to the next subpara. If this results in changing sides,
   * the new side is the next element in this.sideOrder.
   */
  next() {
    if (this.atDocEnd()) {
      return;
    }
    // Check if the current side has unseen subparas.
    // If yes, stay in the side and return.
    if (this.para + 1 < this.numSubparas[this.side][this.seg]) {
      /** Goto: next subpara, same side. */
      this.goto(this.seg, this.side, this.para + 1);
      return;
    }
    const currSideIdx = this.getCurrSideIdx(this.side);
    if (currSideIdx + 1 < this.sideOrder.length) {
      /* currSideIdx is not at the last to-be-visited side,
       * go to the next side in sideOrder. */
      const nextSide = this.sideOrder[currSideIdx + 1];
      const nextSubpara = Math.max(
          0, this.numSubparasShown[nextSide][this.seg] - 1
          );
      this.goto(this.seg, nextSide, nextSubpara);
    } else {
      /* By using Tab to switch sides, it's possible that you
       * haven't yet seen all of side 0 (src). Check:
       */
      if (!this.segFullySeen(this.seg)) {
        this.cycleSides();
      } else if (this.seg + 1 < this.segments.length) {
        /* Goto: start subpara of next seg in the first side in sideOrder. */
        this.goto(this.seg + 1, this.sideOrder[0], 0);
      }
    }
  }

  /**
   * Moves the cursor to the previous subpara. If it results in changing sides,
   * the new side is the previous element in this.sideOrder.
   */
  prev() {
    if (this.atDocStart()) {
      return;
    }
    if (this.para > 0) {
      this.goto(this.seg, this.side, this.para - 1);
      return;
    }
    const currSideIdx = this.getCurrSideIdx(this.side);
    const nextSideIdx = (
        currSideIdx + this.sideOrder.length - 1
        ) % this.sideOrder.length;
    const nextSide = this.sideOrder[nextSideIdx];
    if (currSideIdx === 0) {
      this.goto(this.seg - 1,
                nextSide,
                this.numSubparasShown[nextSide][this.seg - 1] - 1);
    } else {
      this.goto(this.seg,
                nextSide,
                this.numSubparasShown[nextSide][this.seg] - 1);
    }
  }

  /**
   * Returns true if the source text is visible in segment #seg. In tgtFirst
   * mode, until the source side of a segment is revealed, this returns false.
   * In tgtOnly mode, this always returns false. If both tgtOnly and tgtFirst
   * are false, then this function always returns true.
   *
   * @param {number} seg
   * @return {boolean}
   */
  srcVisible(seg) {
    return this.numSubparasShown[0][seg] > 0;
  }

  /**
   * Makes the cursor jump to the next side without going through
   * all the subparas. The next side is determined by this.sideOrder.
   */
  cycleSides() {
    const currSideIdx = this.getCurrSideIdx(this.side);
    const nextSideIdx = (currSideIdx + 1) % this.sideOrder.length;
    const nextSide = this.sideOrder[nextSideIdx];
    /* For tgtFirst, if the target side(s) is/are not fully seen, do not
     * cycle to the source side. */
    if (nextSide === 0 && !this.srcVisible(this.seg)) {
      return;
    }
    const nextSubpara = Math.min(
        this.numSubparasShown[nextSide][this.seg] - 1,
        this.para);
    this.goto(this.seg, nextSide, nextSubpara);
  }

  /**
   * Returns the index of the current side in the sideOrder array.
   * @param {number} currSide
   * @return {number}
   */
  getCurrSideIdx(currSide) {
    return this.sideOrder.indexOf(currSide);
  }

  /**
   * Moves the cursor to the specified segment, side, and para.
   * @param {number} seg
   * @param {number} side
   * @param {number} para
   */
  goto(seg, side, para) {
    console.assert(seg >= 0 && seg < this.segments.length, seg);
    this.seg = seg;
    this.doc = this.segments[seg].doc;
    this.presentationIndex = this.presentationOrder.indexOf(this.doc);
    console.assert(this.sideOrder.includes(side), side);
    this.side = side;
    console.assert(para >= 0 && para < this.numSubparas[side][seg], para);
    this.para = para;
    // For each segment from a previously-presented doc, mark all of its
    // subparas as shown. Also do this for prior segments in the current doc.
    for (let presIdx = 0; presIdx <= this.presentationIndex; presIdx++) {
      const presDoc = this.presentationOrder[presIdx];
      // Last segment for this doc: one before the next doc's first segment, or
      // the last segment overall.
      let endSeg = (presDoc + 1 < this.presentationOrder.length) ?
          this.docSegStarts[presDoc + 1] - 1 :
          this.segments.length - 1;
      // For the currently-presented document, only mark up to the current
      // segment.
      if (presIdx === this.presentationIndex) {
        endSeg = Math.min(endSeg, seg);
      }
      for (let s = this.docSegStarts[presDoc]; s < endSeg; s++) {
        for (let i = 0; i < this.numSides; i++) {
          this.numSubparasShown[i][s] = this.numSubparas[i][s];
        }
      }
    }
    this.numSubparasShown[side][seg] = Math.max(
        this.numSubparasShown[side][seg], para + 1);
    if (!this.tgtFirst || side == 0) {
      /**
       * Make at least 1 subpara visible on all sides.
       */
      for (let i = 0; i < this.numSides; i++) {
        this.numSubparasShown[i][seg] = Math.max(
            this.numSubparasShown[i][seg], 1
            );
      }
    }
    this.maybeMarkSegmentDone(seg);
  }

  /**
   * If all subparas on all sides have been shown for the specified segment,
   * then mark the segment as done.
   * @param {number} seg
   */
  maybeMarkSegmentDone(seg) {
    const sideDone = (s) =>
        this.numSubparasShown[s][seg] === this.numSubparas[s][seg];
    if (this.sideOrder.every((s) => sideDone(s))) {
      this.segmentDone_(seg);
    }
  }

  /**
   * Returns true if the cursor has been through the specified
   * segment/side/para.
   * @param {number} seg
   * @param {number} side
   * @param {number} para
   * @return {boolean}
   */
  hasBeenRead(seg, side, para) {
    return this.segments[seg].doc == this.doc &&
           this.numSubparasShown[side][seg] > para;
  }

  /**
   * Returns true if the cursor is currently at the specified (seg, side, para).
   * @param {number} seg
   * @param {number} side
   * @param {number} para
   * @return {boolean}
   */
  equals(seg, side, para) {
    return this.seg == seg && this.side == side && this.para == para;
  }

  /**
   * Moves the cursor to the start of the specified doc.
   * @param {number} doc
   */
  gotoDoc(doc) {
    console.assert(doc >= 0 && doc < this.docSegStarts.length, doc);
    this.updateSideOrder(this.docSegStarts[doc]);
    this.goto(this.docSegStarts[doc], this.sideOrder[0], 0);
  }
}

/**
 * A class encapsulating one marked error. A marked error is complete once
 * it has three things set: an error span, a severity, and a type.
 */
class AntheaError {
  constructor() {
    this.location = 'translation';
    this.start = -1;
    this.end = -1;
    this.severity = '';
    this.type = '';
    this.subtype = '';
    this.prefix = '';
    this.selected = '';
    this.metadata = {
      timestamp: Date.now(),
      timing: {},
    };
  }

  /**
   * Returns true if this error has its error span start marked.
   * @return {boolean}
   */
  hasSpanStart() {
    return this.start >= 0;
  }

  /**
   * Returns true if this error has its error span end marked.
   * @return {boolean}
   */
  hasSpanEnd() {
    return this.end >= 0;
  }

  /**
   * Returns true if this error has its error span marked.
   * @return {boolean}
   */
  hasSpan() {
    return this.start >= 0 && this.end >= 0;
  }

  /**
   * Returns true if this error has error severity set.
   * @return {boolean}
   */
  hasSeverity() {
    return this.severity != '';
  }

  /**
   * Returns true if this error has error type set.
   * @return {boolean}
   */
  hasType() {
    return this.type != '';
  }

  /**
   * Returns true if this error has all the needed pieces.
   * @return {boolean}
   */
  isComplete() {
    return this.hasSpan() && this.hasSeverity() && this.hasType();
  }

  /**
   * Make an AntheaError from obj, which can be another AntheaError object, or
   * can be a JSON-decoded dictionary that has some or all the fields in
   * AntheaError.
   *
   * @param {!Object} obj
   * @return {!AntheaError}
   */
  static clone(obj) {
    const error = new AntheaError;
    for (const k in obj) {
      error[k] = obj[k];
    }
    error.metadata = {...obj.metadata};
    return error;
  }

  /**
   * Create a new AntheError from a prior error object (taken from a prior
   * rater's evaluation).
   *
   * @param {string} priorRater
   * @param {!Object} priorError
   * @return {!AntheaError}
   */
  static newFromPriorError(priorRater, priorError) {
    const error = AntheaError.clone(priorError);
    error.metadata.timestamp = Date.now();
    error.metadata.timing = {};
    error.metadata.prior_rater = priorRater;
    error.metadata.prior_error = priorError;
    return error;
  }

  /**
   * Add timing events from a timing object to the current error.
   *
   * @param {!Object} timing
   */
  addTiming(timing) {
    const myTiming = this.metadata.timing;
    for (const action in timing) {
      if (!myTiming.hasOwnProperty(action)) {
        myTiming[action] = timing[action];
        continue;
      }
      myTiming[action].count += timing[action].count;
      myTiming[action].timeMS += timing[action].timeMS;
      myTiming[action].log.push.apply(myTiming[action].log, timing[action].log);
    }
  }

  /**
   * Returns the count of errors (not counting those marked_deleted).
   *
   * @param {!Array<!AntheaError>} errors
   * @return {number}
   */
  static count(errors) {
    let ct = 0;
    for (const error of errors) {
      if (!error.marked_deleted) {
        ct++;
      }
    }
    return ct;
  }
}

/**
 * A seeded random number generator based on C++ std::minstd_rand. Note that
 * using seed=0 is a special case that results in always returning 0. Because
 * this class is only used for shuffling (see AntheaEval.pseudoRandomShuffle),
 * using seed=0 effectively disables shuffling.
 */
class AntheaDeterministicRandom {
  /**
   * @param {number} seed
   */
  constructor(seed) {
    this.seed_ = seed;
    this.MULTIPLIER_ = 48271;
    this.MOD_ = 2147483647;  // 2^31 - 1

    // Some seed values (e.g. 1) can result in the first few random numbers
    // having low entropy. To avoid this, we skip the first 30.
    for (let i = 0; i < 30; i++) {
      this.next();
    }
  }

  /**
   * Generates the next random number in [0, 1).
   * @return {number}
   */
  next() {
    const candidate_seed = (this.MULTIPLIER_ * this.seed_) % this.MOD_;
    // If candidate_seed is the wrong sign, add this.MOD_ to wrap it to the
    // correct sign.
    this.seed_ = (candidate_seed * this.MOD_ < 0) ? candidate_seed + this.MOD_
                                                  : candidate_seed;
    return this.seed_ / this.MOD_;
  }
}

/**
 * An object that encapsulates one active evaluation.
 *
 * After constructing, setUpEval() should be called. After finishing one
 *   eval, clear() should be called. A new AntheaEval object should be
 *   used for every new eval.
 */
class AntheaEval {
  /**
   * @param {?AntheaManager} manager Optional AntheaManager controlling object.
   * @param {boolean=} readOnly Set to true when only reviewing an existing eval.
   */
  constructor(manager, readOnly=false) {
    /** const string */
    this.VERSION = 'v1.00-Feb-13-2023';

    /** @private ?AntheaManager */
    this.manager_ = manager;
    /** const boolean */
    this.READ_ONLY = readOnly;

    let scriptUrlPrefix = '';
    const scriptTags = document.getElementsByTagName('script');
    for (let i = 0; i < scriptTags.length; i++) {
      const src = scriptTags[i].src;
      const loc = src.lastIndexOf('/anthea-eval.js');
      if (loc >= 0) {
        scriptUrlPrefix = src.substring(0, loc + 1);
        break;
      }
    }
    /** @private @const {string} */
    this.scriptUrlPrefix_ = scriptUrlPrefix;


    /** ?AntheaCursor Which doc/segment/side/subpara we are at. */
    this.cursor = null;

    /** ?Object */
    this.config = null;

    /** @private {!Array<!Object>} */
    this.evalResults_ = [];

    /**
     * @private @const {!Object} Dictionary mapping error index to the div
     * containing the edit button for that error.
     */
    this.modButtonParents_ = {};

    /** @private {?Element} */
    this.fadingTextSpan_ = null;

    /** @private @const {string} */
    this.beforeColor_ = 'gray';
    /** @private @const {string} */
    this.currColor_ = 'black';
    /** @private @const {string} */
    this.afterColor_ = 'lightgray';
    /** @private @const {string} */
    this.buttonColor_ = 'azure';
    /** @private @const {string} */
    this.highlightColor_ = 'gainsboro';

    /** @private {?AntheaError} Current error getting added or edited. */
    this.error_ = null;
    /**
     * @private {number} When error_ is not null, this is the index of the error
     *     getting edited (-1 for a new error getting annotated).
     */
    this.errorIndex_ = -1;
    /**
     * @private {string} Currently active action for the current error.
     *   Something like 'new-error', 'deletion', etc.
     */
    this.errorAction_ = '';

    /** @private {?AntheaPhraseMarker} */
    this.phraseMarker_ = null;
    /** @private {?Element} */
    this.evalPanel_ = null;
    /** @private {?Element} */
    this.prevButton_ = null;
    /** @private {?Element} */
    this.nextButton_ = null;

    /** @private {?Element} */
    this.contextRow_ = null;

    /** @private {!Array<!Object>} Details for segments */
    this.segments_ = [];
    /** @private {!Array<!Object>} Details for docs */
    this.docs_ = [];

    /** @private {?Element} */
    this.prevDocButton_ = null;
    /** @private {?Element} */
    this.nextDocButton_ = null;
    /** @private {?Element} */
    this.displayedDocNum_ = null;
    /** @private {number} */
    this.numWordsEvaluated_ = 0;
    /** @private {number} */
    this.numTgtWordsTotal_ = 0;
    /** @private {?Element} */
    this.displayedProgress_ = null;

    /** @private {number} */
    this.viewportHeight_ = 500;

    /** @const @private {!Object} Dict of severity/category buttons */
    this.buttons_ = {};

    /** {!Array<!Object>} */
    this.keydownListeners = [];

    /** number */
    this.lastTimestampMS_ = Date.now();

    /** function */
    this.resizeListener_ = null;

    /** @private {string} Source language */
    this.srcLang = '';
    /** @private {string} Target language */
    this.tgtLang = '';
  }
  /**
   * Escapes HTML to safely render as text.
   * @param {string} unescapedHtml
   * @return {string}
   */
  escapeHtml(unescapedHtml) {
    return unescapedHtml.replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
  }

  /**
   * Shuffles an array using the given pseudo-random number generator.
   * If the generator always returns the same value, no shuffling occurs.
   * @param {!Array<!Object>} data
   * @param {!AntheaDeterministicRandom} pseudoRandNumGenerator
   * @return {!Array<!Object>}
   */
  static pseudoRandomShuffle(data, pseudoRandNumGenerator) {
    const dataWithRandoms =
        data.map((x) => ({element: x, random: pseudoRandNumGenerator.next()}));
    dataWithRandoms.sort((a, b) => a.random - b.random);
    return dataWithRandoms.map((x) => x.element);
  }

  /**
   * Removes all window/document-level listeners and sets manager_ to null;
   */
  clear() {
    if (this.resizeListener_) {
      window.removeEventListener('resize', this.resizeListener_);
      this.resizeListener_ = null;
    }
    for (let listener of this.keydownListeners) {
      document.removeEventListener('keydown', listener);
    }
    this.keydownListeners = [];
    this.manager_ = null;
  }

  /**
   * Saves eval results to the manager_.
   */
  saveResults() {
    if (!this.manager_ || this.READ_ONLY) {
      return;
    }
    const segStartIdxArray = [];
    /** Save any feedback, for each doc */
    for (let docIdx = 0; docIdx < this.docs_.length; docIdx++) {
      const doc = this.docs_[docIdx];
      segStartIdxArray.push(doc.startSG);
      const results = this.evalResults_[doc.startSG];
      results.feedback = {};
      const notes = doc.feedbackNotes ? doc.feedbackNotes.innerText.trim() : '';
      if (notes) {
        results.feedback.notes = notes;
      }
      if (doc.thumbsUp && doc.thumbsUp.checked) {
        results.feedback.thumbs = 'up';
      } else if (doc.thumbsDn && doc.thumbsDn.checked) {
        results.feedback.thumbs = 'down';
      }
    }
    let evalResults = this.evalResults_;
    if (this.config.SIDE_BY_SIDE) {
      // Split the eval results.
      evalResults = this.splitSideBySideEvalResults(segStartIdxArray, evalResults);
    }
    // Remove the location field from hotw_list which was added in for splitting
    // the eval results in the sideBySide mode.
    this.removeLocationFromHotw(evalResults);
    this.manager_.persistActiveResults(evalResults);
  }

  /**
   * Retrieves the error subtype information of an error
   * using the type and subtype information.
   * @param {!AntheaError} error
   * @return {!Array<!Object>}
   */
  getErrorSubtypeInfo(error) {
    let errorInfo = this.config.errors[error.type];
    if (error.subtype &&
      errorInfo.subtypes && errorInfo.subtypes[error.subtype]) {
      errorInfo = errorInfo.subtypes[error.subtype];
    }
    return errorInfo;
  }

  /**
   * Split the eval results for target and target2 in the sideBySide mode.
   * The number of segments will be doubled after splitting and formatted
   * as follows:
   * doc0
   *    seg0 [srcErrors, tgtErrors (loc = 'translation'), tgtHotw (w/o loc info)]
   *    seg1 [srcErrors, tgtErrors (loc = 'translation'), tgtHotw (w/o loc info)]
   * doc1 (source is the same as doc0)
   *    seg0 [srcErrors, tgt2Errors (loc = 'translation'), tgt2Hotw (w/o loc info)]
   *    seg1 [srcErrors, tgt2Errors (loc = 'translation'), tgt2Hotw (w/o loc info)]
   * doc2 (previously doc1) ...
   * doc3 (source is the same as doc2)
   * @param {!Array<number>} segStartIdxArray
   * @param {!Array<!Object>} evalResults
   * @return {!Array<!Object>}
   */
  splitSideBySideEvalResults(segStartIdxArray, evalResults) {
    const splitEvalResults = [];
    for (let i = 0; i < segStartIdxArray.length; i++) {
      const startIdx = segStartIdxArray[i];
      const endIdx = i + 1 < segStartIdxArray.length ?
          segStartIdxArray[i + 1] :
          evalResults.length;
      // Loop through each doc twice.
      // 1st time only keep (src, translation, 1) errors.
      // 2nd time only keep (src, translation2, 2) errors.
      // number 1 and 2 refer to which_translation_side of omission errors.
      // Other errors marked on the source side are not affected.
      const validErrorLists = [
        ['source', 'translation', 1],
        ['source', 'translation2', 2]
      ];
      for (let j = 0; j < validErrorLists.length; j++) {
        // Loop through each segment in a doc.
        for (let s = startIdx; s < endIdx; s++) {
          const evalResultCopy = JSON.parse(JSON.stringify(evalResults[s]));
          evalResultCopy.errors = evalResults[s].errors.filter((error) => {
            // For omission errors, split based on which_translation_side
            // in the error subtype information.
            const errorSubtypeInfo = this.getErrorSubtypeInfo(error);
            if (errorSubtypeInfo &&
                errorSubtypeInfo.hasOwnProperty('which_translation_side')) {
              const translationSide = errorSubtypeInfo.which_translation_side;
              return this.cursor.sideOrder[translationSide] ===
                  validErrorLists[j][2];
            }
            // For other errors, split based on error.location.
            return validErrorLists[j].includes(error.location);
          });
          // Split the hotw_list based on their injection location.
          evalResultCopy.hotw_list = evalResultCopy.hotw_list.filter(
              (hotwError) => validErrorLists[j].includes(hotwError.location)
              );
          // Change the doc value based on the side which matters for Marot's
          // error result parsing.
          evalResultCopy.doc = evalResults[s].doc * 2 + j;
          // Save the single quality score corresponding to this result.
          if (this.config.COLLECT_QUALITY_SCORE) {
            evalResultCopy.quality_scores = [evalResults[s].quality_scores[j]];
          }
          splitEvalResults.push(evalResultCopy);
        }
      }
    }
    // Change the location of translation2 errors back to translation.
    for (let i = 0; i < splitEvalResults.length; i++) {
      splitEvalResults[i].errors = splitEvalResults[i].errors.map((error) => {
        const errorCopy = { ...error };
        if (errorCopy.location === 'translation2') {
          errorCopy.location = 'translation';
        }
        return errorCopy;
      });
    }
    return splitEvalResults;
  }

  /**
   * Restores eval results from the previously persisted value.
   *
   * @param {?Array<!Object>} projectResults
   */
  restoreEvalResults(projectResults) {
    if (!this.manager_) {
      return;
    }
    if (!projectResults || projectResults.length == 0) {
      if (this.READ_ONLY) {
        this.manager_.log(this.manager_.ERROR,
                          'Cannot have a read-only eval when there are no ' +
                          'previous results to use');
      } else {
        this.manager_.log(this.manager_.INFO, 'No previous results to restore');
      }
      return;
    }
    if (this.config.SIDE_BY_SIDE) {
      projectResults = this.mergeEvalResults(projectResults);
    }
    if (projectResults.length != this.evalResults_.length) {
      this.manager_.log(
          this.manager_.ERROR,
          'Not restoring previous results as they are for ' +
              projectResults.length +
              ' segments, but the current project has ' +
              this.evalResults_.length);
      return;
    }
    this.docs_[this.cursor.doc].row.style.display = 'none';

    const replacedResults = this.evalResults_;
    this.evalResults_ = projectResults;
    for (let seg = 0; seg < this.evalResults_.length; seg++) {
      const segment = this.segments_[seg];
      const result = this.evalResults_[seg];
      /** Convert parsed error dicts to AntheaError objects */
      for (let x = 0; x < result.errors.length; x++) {
        result.errors[x] = AntheaError.clone(result.errors[x]);
      }
      if (this.READ_ONLY || result.visited) {
        /** Clear any new HOTW injections in this segment */
        for (let p = 0; p < segment.tgtSubparas.length; p++) {
          /* Loop through each side of the segment. */
          for (let side = 1; side < this.cursor.numSides; side ++) {
            const subpara = this.getSubpara(seg, side, p);
            delete subpara.hotw;
            subpara.hotwSpanHTML = '';
            subpara.hotwError = '';
            subpara.hotwType = '';
          }
        }
        const result = this.evalResults_[seg];
        for (let hotw of result.hotw_list || []) {
          const side = (hotw.hasOwnProperty('location') &&
                        hotw.location !== 'translation') ? 2 : 1;
          const subpara = this.getSubpara(seg, side, hotw.para);
          subpara.hotw = hotw;
          subpara.hotwSpanHTML = hotw.hotw_html;
          subpara.hotwError = hotw.injected_error;
          subpara.hotwType = hotw.hotw_type;
        }
        this.cursor.goto(seg, this.cursor.sideOrder[0], 0);
      } else {
        result.hotw_list = replacedResults[seg].hotw_list;
      }
    }
    if (this.READ_ONLY) {
      this.cursor.goto(0, this.config.TARGET_SIDE_ONLY ? 1 : 0, 0);
    }

    /** Restore any feedback, for each doc */
    for (let docIdx = 0; docIdx < this.docs_.length; docIdx++) {
      const doc = this.docs_[docIdx];
      const results = this.evalResults_[doc.startSG];
      if (!results.feedback) {
        continue;
      }
      const feedback = results.feedback;
      if (feedback.notes && doc.feedbackNotes) {
        googdom.setInnerHtml(doc.feedbackNotes, feedback.notes);
      }
      if (feedback.thumbs) {
        if (feedback.thumbs == 'up' && doc.thumbsUp) {
          doc.thumbsUp.checked = true;
        } else if (feedback.thumbs == 'down' && doc.thumbsDn) {
          doc.thumbsDn.checked = true;
        }
      }
    }

    this.docs_[this.cursor.doc].row.style.display = '';
    this.displayedDocNum_.innerHTML = `${(this.cursor.presentationIndex + 1)}`;
    const docEvalCell = this.docs_[this.cursor.doc].eval;
    docEvalCell.appendChild(this.evalPanel_);
    this.displayedProgress_.innerHTML = this.getPercentEvaluated();
    this.showContextAndMediaIfPresent();
    this.redrawAllSegments();
    this.recomputeTops();
    this.refreshCurrSubpara();

    this.manager_.log(this.manager_.INFO,
                      'Restored previous evaluation results');
  }

  /**
   * Merge the eval results from the sideBySide mode.
   * The number of segments will be reduced by half after merging.
   * The evalResults are formatted as follows:
   * doc0
   *    seg0 [srcErrors,
   *          tgtErrors (loc = 'translation'),
   *          tgt2Errors (loc = 'translation2'),
   *          tgtHotw (w/ loc info being 'translation' or 'translation2')]
   *    ...
   * doc1 ...
   *
   * @param {!Array<!Object>} projectResults
   * @return {!Array<!Object>}
   */
  mergeEvalResults(projectResults) {
    const mergedEvalResults = [];
    const segStartIdxArray = this.docs_.map(doc => doc.startSG);

    // Loop through each doc.
    for (let i = 0; i < segStartIdxArray.length; i++) {
      const startIdx = segStartIdxArray[i];
      const endIdx = i + 1 < segStartIdxArray.length ?
          segStartIdxArray[i + 1] :
          this.evalResults_.length;
      const numSegs =  endIdx - startIdx;

      // Loop through each segment in a doc.
      for (let j = 0; j < numSegs; j++) {
        const splitOne = projectResults[2*startIdx + j];
        const splitTwo = projectResults[2*startIdx + numSegs + j];
        // Change the doc value back to non-doubled one.
        // See splitSideBySideEvalResults().
        splitOne.doc = splitOne.doc / 2;
        splitTwo.doc = (splitTwo.doc - 1) / 2;
        // Merge the errors.
        for (let error of splitTwo.errors) {
          if (error.location === 'source') {
            // Only add the omission error which is split
            // based on which_translation_side.
            const errorSubtypeInfo = this.getErrorSubtypeInfo(error);
            if (errorSubtypeInfo &&
                errorSubtypeInfo.hasOwnProperty('which_translation_side')) {
              splitOne.errors.push(error);
            }
          } else {
            error.location = 'translation2';
            splitOne.errors.push(error);
          }
        }
        this.addLocationToHotw(splitOne, "translation");
        this.addLocationToHotw(splitTwo, "translation2");
        splitOne.hotw_list.push(...splitTwo.hotw_list);

        if (this.config.COLLECT_QUALITY_SCORE) {
          splitOne.quality_scores = [
            splitOne.quality_scores[0],
            splitTwo.quality_scores[0]
          ];
        }
        mergedEvalResults.push(splitOne);
      }
    }
    return mergedEvalResults;
  }

  /**
   * Function to add location of injected hotw to the hotw_list when merging.
   * @param {!Object} splitObject of either translation or translation2.
   * @param {string} location of the injected hotw.
   */
  addLocationToHotw(splitObject, location) {
      splitObject.hotw_list.forEach((hotwError) => {
        hotwError.location = location;
      });
  }

  /**
   * Function to remove location of injected hotw from the hotw_list when
   * sideBySide mode is off.
   * @param {!Array<!Object>} evalResults
   */
  removeLocationFromHotw(evalResults) {
    for (const evalResult of evalResults) {
      for (const hotw of evalResult.hotw_list) {
        delete hotw.location;
      }
    }
  }

  /**
   * Copies previous evaluation results as the starting point.
   *
   * @param {!Array<string>} priorRaters
   * @param {!Array<!Object>} priorResults
   */
  startFromPriorResults(priorRaters, priorResults) {
    if (!this.manager_ || this.READ_ONLY) {
      return;
    }
    if (!priorResults || priorResults.length == 0) {
      this.manager_.log(this.manager_.ERROR,
                        'Cannot start from empty prior eval results');
      return;
    }
    if (!priorRaters || priorRaters.length === 0) {
      this.manager_.log(
          this.manager_.ERROR,
          'Cannot start from prior eval results with empty prior raters');
      return;
    }
    if (priorResults.length != this.evalResults_.length) {
      this.manager_.log(
          this.manager_.ERROR,
          'Cannot start from previous results as they are for ' +
              priorResults.length +
              ' segments, but the current project has ' +
              this.evalResults_.length);
      return;
    }
    if (priorRaters.length !== priorResults.length) {
      this.manager_.log(
          this.manager_.ERROR,
          'Cannot start from previous results: found ' + priorResults.length +
              ' prior results vs. ' + priorRaters.length + ' prior raters');
      return;
    }

    for (let seg = 0; seg < this.evalResults_.length; seg++) {
      const segment = this.segments_[seg];
      const result = this.evalResults_[seg];
      result.prior_rater = priorRaters[seg];
      const priorResult = priorResults[seg];
      for (const priorError of priorResult.errors) {
        const newError = AntheaError.newFromPriorError(priorRaters[seg], priorError);
        result.errors.push(newError);
      }
      if (result.errors.length > 0) {
        /** Clear any HOTW injections in this segment */
        for (let p = 0; p < segment.tgtSubparas.length; p++) {
          for (let side = 1; side < this.cursor.numSides; side++) {
            const subpara = this.getSubpara(seg, side, p);
            delete subpara.hotw;
            subpara.hotwSpanHTML = '';
            subpara.hotwError = '';
            subpara.hotwType = '';
          }
        }
        result.hotw_list = [];
      }
    }
    this.manager_.log(this.manager_.INFO,
                      'Starting from previous evaluation results');
  }

  /**
   * Returns the evalResults_[] entry for the current segment.
   * @return {!Object}
   */
  currSegmentEval() {
    return this.evalResults_[this.cursor.seg];
  }

  /**
   * Records time used for the given action type, in the current evalResults_.
   */
  noteTiming(action) {
    if (this.READ_ONLY) {
      return;
    }
    const currEval = this.currSegmentEval();
    if (!currEval) {
      return;
    }
    const timing = currEval.timing;
    if (!timing) {
      return;
    }
    if (!timing[action]) {
      timing[action] = {
        count: 0,
        timeMS: 0,
        /**
         * This is a log of event details, including timestamps, for
         * reconstructing the rater behavior fully when needed.
         */
        log: [],
      };
    }
    const tinfo = timing[action];
    tinfo.count++;
    currEval.timestamp = Date.now();
    tinfo.timeMS += (currEval.timestamp - this.lastTimestampMS_);
    const details = {
      ts: currEval.timestamp,
      side: this.cursor.side,
      para: this.cursor.para,
    };
    if (!this.cursor.srcVisible(this.cursor.seg)) {
      details.source_not_seen = true;
    }
    tinfo.log.push(details);
    this.lastTimestampMS_ = currEval.timestamp;
    this.saveResults();
  }

  /**
   * Note timing for the given action. The action string is saved so that the
   * subsequent concludeError() call can note the timing for cancelled-<action>
   * or finished-<action> appropriately.
   * @param {string} action
   */
  initiateErrorAction(action) {
    this.errorAction_ = action;
    this.noteTiming('started-' + action);
  }

  /**
   * Redraws the current subpara.
   */
  redrawCurrSubpara() {
    this.redrawSubpara(this.cursor.seg, this.cursor.side, this.cursor.para);
  }

  /**
   * Redraws the current subpara and refreshes buttons.
   */
  refreshCurrSubpara() {
    this.redrawCurrSubpara();
    this.setEvalButtonsAvailability();
  }

  /**
   * Returns true iff n is the current document's segment range.
   * @param {number} n The segment index to be tested.
   * @return {boolean}
   */
  inCurrDoc(n) {
    const start = this.docs_[this.cursor.doc].startSG;
    const num = this.docs_[this.cursor.doc].numSG;
    return n >= start && n < start + num;
  }

  /**
   * Returns the intersection of the ranges [a[0], a[1]] and [b[0], b[1]].
   * Returns null if the ranges do not overlap.
   * @param {!Array<number>} a
   * @param {!Array<number>} b
   * @return {?Array<number>}
   */
  intersectRanges(a, b) {
    if (b[0] > a[1] || a[0] > b[1] ) {
      return null;
    }
    const xs = Math.max(a[0] ,b[0]);
    const xe = Math.min(a[1], b[1]);
    return [xs, xe];
  }

  /**
   * Returns the subpara object for the specified segment, side, para.
   * @param {number} seg
   * @param {number} side
   * @param {number} para
   * @return {!Object}
   */
  getSubpara(seg, side, para) {
    const segment = this.segments_[seg];
    if (side === 0) {
      return segment.srcSubparas[para];
    } else if (side == 1) {
      return segment.tgtSubparas[para];
    } else if (side == 2) {
      return segment.tgtSubparas2[para];
    }
  }

  /**
   * Returns the subpara object for the current subpara where the cursor is.
   * @return {!Object}
   */
  getCurrSubpara() {
    return this.getSubpara(
        this.cursor.seg, this.cursor.side, this.cursor.para);
  }

  /**
   * Returns the SPAN elements for the current subpara.
   * @return {!HTMLCollection}
   */
  getCurrTokenSpans() {
    const subpara = this.getCurrSubpara();
    return subpara.subparaSpan.getElementsByTagName('span');
  }

  /**
   * Returns the sentence index within the subpara identified by
   * (seg, side, para) of the token whose index (within all source/target tokens
   * for the segment) * is tokenIndex.
   *
   * @param {number} seg
   * @param {number} side
   * @param {number} para
   * @param {number} tokenIndex The tokenIndex within the whole segment on the
   *     appropriate side (source/target).
   * @return {number} The sentence index for the token, in the subpara.
   */
  sentenceForToken(seg, side, para, tokenIndex) {
    const subpara = this.getSubpara(seg, side, para);
    console.assert(tokenIndex >= subpara.token_offset &&
                   tokenIndex < subpara.token_offset + subpara.num_tokens,
                   tokenIndex);
    let offset = subpara.token_offset;
    for (let s = 0; s < subpara.sentInfos.length; s++) {
      const sentInfo = subpara.sentInfos[s];
      if (tokenIndex < offset + sentInfo.num_tokens) {
        return s;
      }
      offset += sentInfo.num_tokens;
    }
    return -1;
  }

  /**
   * Returns a set of all sentence indices within the current subpara that
   * are markable. A sentence is not markable if it already has an error
   * marked with the property override_all_errors (typical example: a
   * non-translation error).
   *
   * If the optional startSpanIndex parameter is non-negative, then we are
   * limiting to just the sentence index where the already-started marked span
   * lies.
   *
   * @param {number=} startSpanIndex optional index of the starting span of
   *     an already begun marking.
   * @return {!Set} of 0-based indices into the sentInfos in the current
   *     subpara.
   */
  getMarkableSentences(startSpanIndex=-1) {
    const subpara = this.getCurrSubpara();
    const evalResult = this.currSegmentEval();
    const markableSentences = new Set;
    const numSents = subpara.sentInfos.length;
    for (let i = 0; i < numSents; i++) {
      markableSentences.add(i);
    }
    const subparaErrorIndices = this.getSubparaErrorIndices(
        evalResult.errors, this.cursor.seg, this.cursor.side, this.cursor.para);
    for (let e = 0; e < evalResult.errors.length; e++) {
      if (!subparaErrorIndices.has(e)) continue;
      const error = evalResult.errors[e];
      if (error.marked_deleted) {
        continue;
      }
      const errorInfo = this.config.errors[error.type];
      if (errorInfo.override_all_errors) {
        const sent = this.sentenceForToken(
            this.cursor.seg, this.cursor.side, this.cursor.para, error.start);
        console.assert(sent >= 0 && sent < numSents, sent);
        markableSentences.delete(sent);
      }
    }
    if (startSpanIndex >= 0) {
      const startSent = this.sentenceForToken(
          this.cursor.seg, this.cursor.side, this.cursor.para,
          subpara.token_offset + startSpanIndex);
      console.assert(startSent >= 0 && startSent < numSents);
      const hasSent = markableSentences.has(startSent);
      markableSentences.clear();
      if (hasSent) {
        markableSentences.add(startSent);
      }
    }
    return markableSentences;
  }

  /**
   * Returns a set of all token span indices within the current subpara that
   * are markable. See the documentation of getMarkableSentences() above for
   * a description of which tokens are not markable.
   *
   * If the optional startSpanIndex parameter is non-negative, then we are
   * limiting to token span indices within the same sentence.
   *
   * @param {number=} startSpanIndex optional index of the starting span of
   *     an already begun marking.
   * @return {!Set} of 0-based indices into the token spans in the current
   *     subpara.
   */
  getMarkableSpanIndices(startSpanIndex=-1) {
    const subpara = this.getCurrSubpara();
    const markableSentences = this.getMarkableSentences(startSpanIndex);
    const markableSpans = new Set;
    let offset = 0;
    for (let s = 0; s < subpara.sentInfos.length; s++) {
      const sentInfo = subpara.sentInfos[s];
      if (markableSentences.has(s)) {
        for (let i = 0; i < sentInfo.num_tokens; i++) {
          markableSpans.add(offset + i);
        }
      }
      offset += sentInfo.num_tokens;
    }
    return markableSpans;
  }

  /**
   * Returns the range [start, end] of token indices (within the whole segment)
   * of the sentence to which the given tokenIndex belongs.
   * @param {number} seg
   * @param {number} side
   * @param {number} para
   * @param {number} tokenIndex The tokenIndex within the whole segment.
   * @return {!Array<number>} The range of tokens for the sentence in which
   *     tokenIndex falls.
   */
  sentenceTokensRange(seg, side, para, tokenIndex) {
    const subpara = this.getSubpara(seg, side, para);
    console.assert(tokenIndex >= subpara.token_offset &&
                   tokenIndex < subpara.token_offset + subpara.num_tokens,
                   tokenIndex, subpara);
    let offset = subpara.token_offset;
    for (const sentInfo of subpara.sentInfos) {
      if (tokenIndex < offset + sentInfo.num_tokens) {
        return [offset, offset + sentInfo.num_tokens - 1];
      }
      offset += sentInfo.num_tokens;
    }
    return [-1, -1];
  }

  /**
   * Returns indices into the errors array that contain errors for the given
   * seg, side, para, and optionally sentence containing a tokenIndex.
   * If tokenIndex is negative, then error indices for all sentences in the
   * subpara are returned.
   * The indices are returned as a Set.
   *
   * @param {!Array<!AntheaError>} errors
   * @param {number} seg
   * @param {number} side
   * @param {number} para
   * @param {number=} tokenIndex
   * @return {!Set<number>}
   */
  getSubparaErrorIndices(errors, seg, side, para, tokenIndex=-1) {
    const ret = new Set;
    const subpara = this.getSubpara(seg, side, para);
    const tokenRangeInSeg = [subpara.token_offset,
                             subpara.token_offset + subpara.num_tokens - 1];
    if (tokenIndex >= 0) {
      let offset = subpara.token_offset;
      for (const sentInfo of subpara.sentInfos) {
        if (tokenIndex < offset + sentInfo.num_tokens) {
          tokenRangeInSeg[0] = offset;
          tokenRangeInSeg[1] = offset + sentInfo.num_tokens - 1;
          break;
        }
        offset += sentInfo.num_tokens;
      }
    }
    for (let e = 0; e < errors.length; e++) {
      const error = errors[e];
      if (side === 0 && error.location !== 'source') continue;
      if (side === 1 && error.location !== 'translation') continue;
      if (side === 2 && error.location !== 'translation2') continue;
      const range = this.intersectRanges(
        [error.start, error.end], tokenRangeInSeg);
      if (!range) continue;
      ret.add(e);
    }
    return ret;
  }

  /**
   * Resets the quality score and display for the current segment.
   */
  resetQualityScore() {
    if (this.config.COLLECT_QUALITY_SCORE) {
      this.currSegmentEval().quality_scores[this.cursor.side - 1] = -1;
      this.undoProgressForCurrentSegment();
      this.updateQualityScoreDisplay();
    }
  }

  /**
   * Updates the quality score slider display based on the current segment.
   */
  updateQualityScoreDisplay() {
    if (this.config.COLLECT_QUALITY_SCORE) {
      if (this.cursor.side === 0) {
        this.qualityScorePanel_.style.display = 'none';
      } else {
        this.qualityScorePanel_.style.display = '';
        const qualityScore = this.currSegmentEval().quality_scores[this.cursor.side - 1];
        if (qualityScore >= 0) {
          this.qualityScoreSlider_.value = qualityScore;
          this.qualityScoreSlider_.style.setProperty('--slider-thumb-color',
              'var(--slider-set-thumb-color)');
          this.qualityScoreText_.innerHTML = qualityScore;
        } else {
          this.qualityScoreSlider_.value = 50;
          this.qualityScoreSlider_.style.setProperty('--slider-thumb-color',
              'var(--slider-unset-thumb-color)');
          this.qualityScoreText_.innerHTML = 'Unset';
        }
      }
    }
  }

  /**
   * Shows the subpara at index (seg, side, para). How the subpara gets shown
   *     depends on whether it is before, at, or after this.cursor.
   * @param {number} seg
   * @param {number} side
   * @param {number} para
   */
  redrawSubpara(seg, side, para) {
    if (!this.inCurrDoc(seg)) {
      return;
    }

    const subpara = this.getSubpara(seg, side, para);

    /* Get rid of any old highlighting or listeners */
    if (subpara.clickListener) {
      subpara.subparaSpan.removeEventListener('click', subpara.clickListener);
    }
    subpara.clickListener = null;

    const evalResult = this.evalResults_[seg];

    let spanHTML = subpara.spanHTML;
    if (!this.READ_ONLY && subpara.hotw && !subpara.hotw.done) {
      spanHTML = subpara.hotwSpanHTML;
    }
    googdom.setInnerHtml(subpara.subparaSpan, spanHTML);
    subpara.subparaSpan.classList.remove('anthea-subpara-nav');

    const isCurr = this.cursor.equals(seg, side, para);

    /* Highlight errors in subpara */
    const tokenSpans = subpara.subparaSpan.getElementsByTagName('span');
    console.assert(tokenSpans.length == subpara.num_tokens);
    const subparaErrorIndices = this.getSubparaErrorIndices(evalResult.errors,
                                                            seg, side, para);
    const tokenRangeInSeg = [subpara.token_offset,
                             subpara.token_offset + subpara.num_tokens - 1];
    for (let e = 0; e < evalResult.errors.length; e++) {
      if (!subparaErrorIndices.has(e)) continue;
      const error = evalResult.errors[e];
      if (error.marked_deleted) {
        continue;
      }
      /** Code to highlight the span in the subpara: */
      const range = this.intersectRanges(
        [error.start, error.end], tokenRangeInSeg);
      const isBeingEdited = isCurr && this.error_ && (this.errorIndex_ == e);
      const severity = this.config.severities[error.severity];
      const color = severity.color;
      for (let x = range[0]; x <= range[1]; x++) {
        const tokenSpan = tokenSpans[x - subpara.token_offset];
        tokenSpan.style.backgroundColor = color;
        if (isBeingEdited) {
          tokenSpan.classList.add('anthea-being-edited');
        }
      }
    }

    if (isCurr) {
      this.updateQualityScoreDisplay();
      subpara.subparaSpan.classList.remove('anthea-fading-text');
      this.evalPanel_.style.top = subpara.top;
      this.evalPanelErrors_.innerHTML = '';
      if (subpara.hotw && subpara.hotw.done) {
        this.displayHOTWMessage(subpara.hotw.found,
                                subpara.hotw.injected_error);
      }
      for (let e = 0; e < evalResult.errors.length; e++) {
        if (subparaErrorIndices.has(e)) {
          if (!this.cursor.srcVisible(this.cursor.seg) &&
              !evalResult.errors[e].metadata.source_not_seen) {
            /* This error will be shown only after the source is shown */
            continue;
          }
          this.displayError(evalResult, e);
        }
      }
      if (this.error_ && this.errorIndex_ < 0 && this.error_.hasSpan()) {
        /* New error still getting annotated */
        let color = this.highlightColor_;
        if (this.error_.severity) {
          const severity = this.config.severities[this.error_.severity];
          color = severity.color;
        }
        for (let x = this.error_.start; x <= this.error_.end; x++) {
          tokenSpans[x - subpara.token_offset].style.backgroundColor = color;
        }
      }
    }

    const hasBeenRead = this.cursor.hasBeenRead(seg, side, para);

    if (!isCurr && hasBeenRead && !this.error_)  {
      /* anthea-subpara-nav class makes the mouse a pointer on hover. */
      subpara.subparaSpan.classList.add('anthea-subpara-nav');
      subpara.clickListener = () => {
        this.revisitSubpara(seg, side, para);
      };
      subpara.subparaSpan.addEventListener('click', subpara.clickListener);
    }

    const afterColor = (side == 0 && this.cursor.tgtFirst &&
                        !this.cursor.srcVisible(seg)) ?
                       'transparent' : this.afterColor_;
    subpara.subparaSpan.style.color = isCurr ? this.currColor_ :
        (hasBeenRead ? this.beforeColor_ : afterColor);
    subpara.subparaSpan.style.fontWeight = isCurr ? 'bold' : 'normal';
  }

  /**
   * Redraws all segments and calls setEvalButtonsAvailability().
   */
  redrawAllSegments() {
    for (let n = 0; n < this.segments_.length; n++) {
      const segment = this.segments_[n];
      for (let s = 0; s < segment.srcSubparas.length; s++) {
        this.redrawSubpara(n, 0, s);
      }
      for (let t = 0; t < segment.tgtSubparas.length; t++) {
        this.redrawSubpara(n, 1, t);
      }
      if (this.config.SIDE_BY_SIDE) {
        for (let u = 0; u < segment.tgtSubparas2.length; u++) {
          this.redrawSubpara(n, 2, u);
        }
      }
    }
    this.setEvalButtonsAvailability();
    this.lastTimestampMS_ = Date.now();
  }

  /**
   * Displays a "hands-on-the-wheel message", telling the rater whether or not
   *     they successfully found a deliberately injected HOTW error.
   * @param {boolean} found Whether the error was detected
   * @param {string} span The injected error phrase
   */
  displayHOTWMessage(found, span) {
    const tr = document.createElement('tr');
    this.evalPanelErrors_.appendChild(tr);
    let html = '<td class="anthea-eval-panel-text" dir="auto" colspan="2">';
    if (!found) {
      html += '<p class="anthea-hotw-missed">You missed some injected ' +
              'error(s) in this ';
    } else {
      html += '<p class="anthea-hotw-found">You successfully found an error ' +
              'in this ';
    }
    html += '<span class="anthea-hotw-def" title="A small fraction of ' +
        'sentences that are initially shown with some deliberately injected ' +
        'error(s). Evaluating translation quality is a difficult and ' +
        'demanding task, and these test sentences are simply meant to ' +
        'help you maintain the high level of attention the task needs.">' +
        'test sentence that had been artificially altered</span>.</p> ';
    html += '<p>The injected error (now <b>reverted</b>) was: ' + span + '</p>';
    html += '<p><b>Please continue to rate the translation as now shown ' +
            'without alterations, thanks!</b></p></td>';
    googdom.setInnerHtml(tr, html);
  }

  /**
   * Displays the previously marked error in evalResult.errors[index], alongside
   *     the current segment. The displayed error also includes a hamburger
   *     menu button for deleting or modifying it.
   * @param {!Object} evalResult
   * @param {number} index
   */
  displayError(evalResult, index) {
    const errors = evalResult.errors;
    const error = errors[index];
    const tr = googdom.createDom('tr', 'anthea-eval-panel-row');
    this.evalPanelErrors_.appendChild(tr);

    let color = this.highlightColor_;
    let desc = '';
    if (error.type) {
      const errorInfo = this.config.errors[error.type];
      desc = errorInfo.display;
      if (error.subtype &&
          errorInfo.subtypes && errorInfo.subtypes[error.subtype]) {
        desc = desc + ': ' + errorInfo.subtypes[error.subtype].display;
      }
    }
    if (error.severity) {
      const severity = this.config.severities[error.severity];
      color = severity.color;
      if (!desc) {
        desc = severity.display;
      }
    }
    if (error.metadata && error.metadata.note) {
      desc = desc + ' [' + error.metadata.note + ']';
    }
    desc += ': ';

    let textCls = 'anthea-eval-panel-text';
    if (error.marked_deleted) {
      textCls += ' anthea-deleted-error';
    }
    if (this.error_ && (this.errorIndex_ == index)) {
      textCls += ' anthea-being-edited';
    }
    const lang = error.location == 'source' ? this.srcLang : this.tgtLang;
    /**
     * Use 0-width spaces to ensure leading/trailing spaces get shown.
     */
    tr.appendChild(googdom.createDom(
        'td', {class: textCls}, desc,
        googdom.createDom(
            'span', {
              dir: 'auto',
              lang: lang,
              style: 'background-color:' + color,
            },
            '\u200b' + error.selected + '\u200b')));

    const modButton = googdom.createDom(
        'button',
        {class: 'anthea-stretchy-button anthea-eval-panel-mod'});
    modButton.innerHTML = '&#9776;';
    const modButtonParent = googdom.createDom(
        'div', 'anthea-modifier-menu-parent', modButton);
    tr.appendChild(googdom.createDom(
        'td', 'anthea-eval-panel-nav', modButtonParent));
    this.modButtonParents_[index] = modButtonParent;

    if (this.READ_ONLY) {
      modButton.disabled = true;
    } else {
      modButton.addEventListener(
          'mouseover', this.showModifierMenu.bind(this, index));
      modButton.addEventListener('click', this.toggleModifierMenu.bind(this));
    }
  }

  /**
   * Called from the AntheaPhraseMarker object, this is set when a phrase-start
   *     has been marked.
   */
  setStartedMarkingSpan() {
    if (!this.error_) {
      this.error_ = new AntheaError;
      this.errorIndex_ = -1;
      this.initiateErrorAction('new-error');
    }
    this.setEvalButtonsAvailability();
  }

  /**
   * Displays the passed guidance message.
   * @param {string} text The guidance message.
   */
  showGuidance(text) {
    if (!this.guidance_) {
      return;
    }
    this.guidance_.innerHTML = text;
  }

  /**
   * Returns a distinctive key in this.buttons_[] for the button corresponding
   * to this error type and subtype.
   * @param {string} type
   * @param {string=} subtype
   * @return {string}
   */
  errorButtonKey(type, subtype='') {
    let key = 'error:' + type;
    if (subtype) key += '/' + subtype;
    return key;
  }

  /**
   * Sets the disabled/display state of all evaluation buttons appropriately.
   *    This is a critical function, as it determines, based upon the current
   *    state, which UI controls/buttons get shown and enabled.
   */
  setEvalButtonsAvailability() {
    const evalResult = this.currSegmentEval();
    const subparaErrorIndices = this.getSubparaErrorIndices(
        evalResult.errors, this.cursor.seg, this.cursor.side, this.cursor.para);
    const markableSentences = this.getMarkableSentences();
    const noNewErrors =
        evalResult.errors && evalResult.errors.length > 0 &&
        ((this.config.MAX_ERRORS > 0 &&
          AntheaError.count(evalResult.errors) >= this.config.MAX_ERRORS) ||
         (markableSentences.size == 0));
    const disableMarking = this.READ_ONLY ||
        (this.error_ && this.error_.isComplete()) ||
        (this.errorIndex_ < 0 && noNewErrors);
    const disableSevCat = disableMarking ||
                          !this.error_ || !this.error_.hasSpan();
    const disableSeverity = disableSevCat ||
        (this.error_ && this.error_.hasSeverity());
    let forcedSeverity = '';
    if (!disableSeverity && this.error_ && this.error_.hasType()) {
      const errorInfo = this.config.errors[this.error_.type];
      if (errorInfo.forced_severity) {
        forcedSeverity = errorInfo.forced_severity;
      }
    }
    for (const s in this.config.severities) {
      this.buttons_[s].disabled = disableSeverity;
      if (forcedSeverity && s != forcedSeverity) {
        this.buttons_[s].disabled = true;
      }
    }

    const disableErrors = disableSevCat ||
        (this.error_ && this.error_.hasType());
    for (let type in this.config.errors) {
      const errorInfo = this.config.errors[type];
      const errorButton = this.buttons_[this.errorButtonKey(type)];
      errorButton.disabled = disableErrors;
      if (!disableErrors) {
        if (errorInfo.source_side_only && this.cursor.side == 1) {
          errorButton.disabled = true;
        }
        if (!errorInfo.source_side_ok && this.cursor.side == 0) {
          errorButton.disabled = true;
        }
        if (errorInfo.forced_severity && this.error_.severity &&
            this.error_.severity != errorInfo.forced_severity) {
          errorButton.disabled = true;
        }
        if (errorInfo.needs_source &&
            !this.cursor.srcVisible(this.cursor.seg)) {
          errorButton.disabled = true;
        }
      }
      for (let subtype in errorInfo.subtypes) {
        const subtypeInfo = errorInfo.subtypes[subtype];
        const subtypeButton = this.buttons_[this.errorButtonKey(type, subtype)];
        subtypeButton.disabled = errorButton.disabled;
        if (!disableErrors) {
          if (subtypeInfo.source_side_only && this.cursor.side != 0) {
            subtypeButton.disabled = true;
          }
          if (!subtypeInfo.source_side_ok && this.cursor.side == 0) {
            subtypeButton.disabled = true;
          }
        }
      }
    }
    this.prevButton_.disabled = this.cursor.atDocStart();
    this.nextButton_.disabled = this.cursor.atDocEnd();
    this.prevDocButton_.style.display =
        (this.cursor.presentationIndex === 0) ? 'none' : '';
    this.prevDocButton_.disabled = false;
    if (this.cursor.presentationIndex === this.docs_.length - 1) {
      this.nextDocButton_.style.display = 'none';
    } else {
      this.nextDocButton_.style.display = '';
      this.nextDocButton_.disabled = !this.READ_ONLY &&
                                     !this.cursor.seenDocEnd();
    }

    for (let e = 0; e < evalResult.errors.length; e++) {
      if (!subparaErrorIndices.has(e)) continue;
      const modButtonParent = this.modButtonParents_[e];
      if (!modButtonParent) continue;
      const modButton = modButtonParent.firstElementChild ?? null;
      if (!modButton) continue;
      modButton.disabled = true;
      if (this.READ_ONLY) {
        continue;
      }
      const error = evalResult.errors[e];
      if (error.marked_deleted && noNewErrors) {
        /* cannot undelete when no more errors are allowed */
        continue;
      }
      if (this.error_) {
        /* cannot edit an error while another error is getting annotated */
        continue;
      }
      modButton.disabled = false;
    }
    this.guidancePanel_.style.backgroundColor = 'whitesmoke';
    this.evalPanelBody_.style.display = 'none';
    this.cancel_.style.display = 'none';
    if (this.READ_ONLY) {
      this.showGuidance('Read-only mode, no editing possible');
      return;
    }
    if (!this.error_)  {
      if (noNewErrors) {
        this.showGuidance('Cannot mark more errors here');
      } else {
        this.showGuidance('Click on the start of an error span not yet marked');
        this.phraseMarker_.getMarkedPhrase();
      }
      return;
    }
    if (this.error_.isComplete()) {
      return;
    }

    // Already in the middle of editing a new or existing error.
    this.cancel_.style.display = '';
    this.openSubtypes(null);
    this.prevButton_.disabled = true;
    this.nextButton_.disabled = true;
    this.prevDocButton_.disabled = true;
    this.nextDocButton_.disabled = true;

    if (!this.error_.hasType()) {
      this.evalPanelBody_.style.display = '';
    }
    if (this.error_.hasSeverity() && this.error_.hasType()) {
      if (!this.phraseMarker_.startAlreadyMarked()) {
        this.showGuidance('Click on the corrected start of the error span');
        this.phraseMarker_.getMarkedPhrase();
      }
    } else if (this.error_.hasSeverity()) {
      const severity = this.config.severities[this.error_.severity];
      this.guidancePanel_.style.backgroundColor = severity.color;
      this.showGuidance('Choose error type / sybtype');
    } else if (this.error_.hasType()) {
      this.showGuidance('Choose error severity');
    } else {
      this.showGuidance('Choose error severity, type / subtype');
    }
  }

  /**
   * Updates displayed progress when segment seg is fully visited for the first
   * time.
   * @param {number} seg
   */
  updateProgressForSegment(seg) {
    if (this.evalResults_[seg].visited) {
      return;
    }
    if (this.config.COLLECT_QUALITY_SCORE &&
        this.evalResults_[seg].quality_scores.some(value => value === -1)) {
      return;
    }
    this.evalResults_[seg].visited = true;
    this.saveResults();
    this.numWordsEvaluated_ += this.segments_[seg].numTgtWords;
    if (this.displayedProgress_) {
      this.displayedProgress_.innerHTML = this.getPercentEvaluated();
    }
  }

  /**
   * Undoes the progress update for the current segment. This can happen if the
   * quality score gets reset because of a HOTW error.
   */
  undoProgressForCurrentSegment() {
    const seg = this.cursor.seg;
    if (!this.evalResults_[seg].visited) {
      return;
    }
    this.evalResults_[seg].visited = false;
    this.saveResults();
    this.numWordsEvaluated_ -= this.segments_[seg].numTgtWords;
    if (this.displayedProgress_) {
      this.displayedProgress_.innerHTML = this.getPercentEvaluated();
    }
  }

  /**
   * Called after a subpara should be done with. Returns false in
   *     the (rare) case that the subpara was a HOTW subpara with
   *     injected errors shown, which leads to the end of the HOTW check
   *     but makes the rater continue to rate the subpara.
   * @return {boolean}
   */
  finishCurrSubpara() {
    const subpara = this.getCurrSubpara();
    if (!this.READ_ONLY && subpara.hotw && !subpara.hotw.done) {
      this.resetQualityScore();
      const evalResult = this.currSegmentEval();
      this.noteTiming('missed-hands-on-the-wheel-error');
      subpara.hotw.done = true;
      subpara.hotw.timestamp = evalResult.timestamp;
      subpara.hotw.timing = evalResult.timing;
      evalResult.timing = {};
      this.redrawAllSegments();
      return false;
    }
    return true;
  }

  /**
   * Moves the current subpara into view, if off-screen.
   */
  ensureCurrSubparaVisible() {
    const subparaSpan = this.getCurrSubpara().subparaSpan;
    const subparaRect = subparaSpan.getBoundingClientRect();
    if (subparaRect.top >= 0 && subparaRect.bottom < this.viewportHeight_) {
      return;
    }
    subparaSpan.scrollIntoView({block: "center"});
  }

  /**
   * Navigates to the other side.
   */
  handleSwitch() {
    if (this.error_ || this.config.TARGET_SIDE_ONLY) {
      return;
    }
    this.noteTiming('switch-sides');
    this.cursor.cycleSides();
    this.redrawAllSegments();
    this.ensureCurrSubparaVisible();
  }

  /**
   * Navigates to the previous subpara.
   */
  handlePrev() {
    this.noteTiming('back-arrow');
    this.cursor.prev();
    this.redrawAllSegments();
    this.ensureCurrSubparaVisible();
  }

  /**
   * Navigates to the next subpara.
   */
  handleNext() {
    this.noteTiming('next-arrow');
    if (!this.finishCurrSubpara()) {
      return;
    }
    this.cursor.next();
    this.redrawAllSegments();
    this.ensureCurrSubparaVisible();
  }

  /**
   * Gets the percentage of words evaluated.
   * @return {string}
   */
  getPercentEvaluated() {
    return '' +
        Math.min(
            100,
            Math.floor(
                100 * this.numWordsEvaluated_ / this.numTgtWordsTotal_)) +
        '%';
  }

  /**
   * If the current error requires additional user input, this function
   *     augments it with the needed info. Returns false if the user cancels.
   * @return {boolean} Whether to continue with marking the error.
   */
  maybeAugmentError() {
    if (!this.error_) {
      return false;
    }
    const errorInfo = this.config.errors[this.error_.type];
    if (errorInfo.override_all_errors) {
      if (!confirm(
              'This error category will remove all other marked errors ' +
              'from this sentence and will set the error span to ' +
              'be the whole sentence. Please confirm!')) {
        this.noteTiming('cancelled-override-all-errors');
        return false;
      }
      this.noteTiming('confirmed-override-all-errors');
      this.error_.prefix = '';
      const subpara = this.getCurrSubpara();
      const range = this.sentenceTokensRange(
          this.cursor.seg, this.cursor.side, this.cursor.para,
          this.error_.start);
      const spanArray = this.getCurrTokenSpans();
      this.error_.selected = '';
      for (let x = range[0]; x <= range[1]; x++) {
        this.error_.selected += spanArray[x - subpara.token_offset].innerText;
      }
      this.error_.start = range[0];
      this.error_.end = range[1];
    }

    this.error_.metadata.para = this.cursor.para;
    this.error_.metadata.side = this.cursor.side;

    if (errorInfo.needs_note && !this.error_.metadata.note) {
      this.error_.metadata.note = prompt(
          "Please enter a short error description", "");
      if (!this.error_.metadata.note) {
        this.noteTiming('cancelled-required-error-note');
        return false;
      }
      this.noteTiming('added-required-error-note');
    }
    return true;
  }

  /**
   * Calling this marks the end of an error-marking or editing in the current
   *     subpara.
   * @param {boolean=} cancel
   */
  concludeError(cancel = false) {
    let actionPrefix = 'cancelled-';
    if (!cancel && this.error_ && this.error_.isComplete()) {
      const evalResult = this.currSegmentEval();
      const errorInfo = this.config.errors[this.error_.type];
      if (errorInfo.override_all_errors) {
        const subparaErrorIndices = this.getSubparaErrorIndices(
            evalResult.errors, this.cursor.seg, this.cursor.side,
            this.cursor.para, this.error_.start);
        for (let x = 0; x < evalResult.errors.length; x++) {
          if (!subparaErrorIndices.has(x) || x == this.errorIndex_) {
            continue;
          }
          evalResult.errors[x].marked_deleted = true;
        }
      }
      this.error_.metadata.timestamp = evalResult.timestamp;
      this.error_.addTiming(evalResult.timing);
      if (!this.cursor.srcVisible(this.cursor.seg)) {
        this.error_.metadata.source_not_seen = true;
      }
      evalResult.timing = {};
      if (this.errorIndex_ >= 0) {
        evalResult.errors[this.errorIndex_] = this.error_;
      } else {
        evalResult.errors.push(this.error_);
      }
      actionPrefix = 'finished-';
    }
    if (this.errorAction_) {
      this.noteTiming(actionPrefix + this.errorAction_);
    }
    this.saveResults();
    this.error_ = null;
    this.errorIndex_ = -1;
    this.errorAction_ = '';
    this.redrawAllSegments();
  }

  /**
   * Opens the visibility of the list of subtypes for the current error type,
   *     closing all others.
   * @param {?Element} subtypes If not-null, open this subtypes panel.
   */
  openSubtypes(subtypes) {
    if (this.openSubtypes_) {
      this.openSubtypes_.style.display = 'none';
      if (subtypes == this.openSubtypes_) {
        this.openSubtypes_ = null;
        return;
      }
    }
    if (subtypes) {
      subtypes.style.display = '';
      this.openSubtypes_ = subtypes;
    }
  }

  /**
   * Creates the MQM eval panel shown in the evaluation column.
   */
  makeEvalPanel() {
    this.guidancePanel_ = googdom.createDom('div', 'anthea-guidance-panel');
    this.evalPanelHead_.insertAdjacentElement(
        'afterbegin', this.guidancePanel_);

    this.guidance_ = googdom.createDom('div', 'anthea-eval-guidance');
    this.guidancePanel_.appendChild(this.guidance_);

    if (this.config.COLLECT_QUALITY_SCORE) {
      this.qualityScoreSlider_ = googdom.createDom('input', {
        'class': 'anthea-slider',
        'type': 'range',
        'min': 0,
        'max': 100,
        'value': 50,
        'onkeydown': 'return false;' // Prevent arrow keys from affecting score.
      });
      this.qualityScoreText_ =
          googdom.createDom('div', 'anthea-quality-score-text', 'Unset');
      this.qualityScorePanel_ = googdom.createDom(
          'div', 'anthea-quality-score-panel',
          'Quality Score: ', this.qualityScoreText_, this.qualityScoreSlider_);
      const qualityScoreLandmarks = [
        '0: No meaning preserved', '33: Some meaning preserved',
        '66: Most meaning preserved, few grammar mistakes',
        '100: Perfect meaning/grammar'
      ];
      qualityScoreLandmarks.forEach(
          landmark => this.qualityScorePanel_.appendChild(googdom.createDom(
              'div', 'anthea-quality-score-landmark', landmark)));

      this.guidancePanel_.appendChild(this.qualityScorePanel_);
      const qualityScoreListener = (e) => {
        this.currSegmentEval().quality_scores[this.cursor.side - 1] =
            Number(this.qualityScoreSlider_.value);
        this.updateQualityScoreDisplay();
        this.cursor.maybeMarkSegmentDone(this.cursor.seg);
      };
      this.qualityScoreSlider_.addEventListener('input', qualityScoreListener);
  }

    this.cancel_ = googdom.createDom(
        'button', 'anthea-stretchy-button anthea-eval-cancel', 'Cancel (Esc)');
    this.cancel_.style.display = 'none';
    const cancelListener = (e) => {
      if (e.type == 'click' || (e.key && e.key === 'Escape')) {
        e.preventDefault();
        this.handleCancel();
      }
    };
    this.cancel_.addEventListener('click', cancelListener);
    this.keydownListeners.push(cancelListener);
    document.addEventListener('keydown', cancelListener);
    this.guidancePanel_.appendChild(this.cancel_);

    this.evalPanelErrorTypes_ = googdom.createDom(
        'table', 'anthea-eval-panel-table');
    this.evalPanelBody_.appendChild(this.evalPanelErrorTypes_);
    this.openSubtypes_ = null;

    for (let type in this.config.errors) {
      const errorInfo = this.config.errors[type];
      const errorButton = googdom.createDom(
          'button', 'anthea-error-button',
          errorInfo.display + (errorInfo.subtypes ? ' ▶' : ''));
      if (errorInfo.description) {
        errorButton.title = errorInfo.description;
      }
      const errorCell = googdom.createDom(
          'td', 'anthea-eval-panel-cell', errorButton);
      if (!errorInfo.hidden) {
        // We add the button to the DOM only if not hidden.
        this.evalPanelErrorTypes_.appendChild(
            googdom.createDom('tr', null, errorCell));
      }
      if (errorInfo.subtypes) {
        const subtypesDiv = googdom.createDom('div',
                                              'anthea-eval-panel-subtypes');
        errorCell.appendChild(subtypesDiv);
        const subtypes = googdom.createDom('table',
                                           'anthea-eval-panel-table');
        subtypesDiv.appendChild(subtypes);
        for (let subtype in errorInfo.subtypes) {
          let subtypeInfo = errorInfo.subtypes[subtype];
          const display = this.config.FLATTEN_SUBTYPES ?
              errorInfo.display + ' / ' + subtypeInfo.display :
              subtypeInfo.display;
          const subtypeButton =
              googdom.createDom('button', 'anthea-error-button', display);
          if (subtypeInfo.description) {
            subtypeButton.title = subtypeInfo.description;
          }
          subtypeButton.addEventListener('click', (e) => {
            e.preventDefault();
            this.handleErrorTypeClick(type, subtype);
          });
          this.buttons_[this.errorButtonKey(type, subtype)] = subtypeButton;
          if (!subtypeInfo.hidden) {
            // We add the button to the DOM only if not hidden.
            subtypes.appendChild(googdom.createDom(
                'tr', null,
                googdom.createDom(
                    'td', 'anthea-eval-panel-cell', subtypeButton)));
          }
        }
        if (this.config.FLATTEN_SUBTYPES) {
          errorButton.style.display = 'none';
        } else {
          subtypesDiv.classList.add('anthea-eval-panel-unflattened');
          subtypes.style.display = 'none';
          errorButton.addEventListener('click', (e) => {
            e.preventDefault();
            this.openSubtypes(subtypes);
          });
        }
      } else {
        errorButton.addEventListener('click', (e) => {
          e.preventDefault();
          this.handleErrorTypeClick(type);
        });
      }
      errorButton.disabled = true;
      this.buttons_[this.errorButtonKey(type)] = errorButton;
    }
  }

  /**
   * Called after any of the three parts of an MQM rating (error span,
   *     severity, error type) is added, this finishes the error rating when
   *     all three parts have been received.
   */
  maybeConcludeError() {
    if (!this.error_ || !this.error_.isComplete()) {
      this.refreshCurrSubpara();
      return;
    }
    if (!this.error_.marked_deleted && !this.maybeAugmentError()) {
      this.concludeError(true);
      return;
    }
    this.concludeError();
  }

  /**
   * Sets the severity level for the current MQM rating.
   * @param {string} severityId The severity of the error.
   */
  setMQMSeverity(severityId) {
    this.error_.severity = severityId;
    this.maybeConcludeError();
  }

  /**
   * Sets the MQM error type and subtype for the current MQM rating.
   * @param {string} type
   * @param {string=} subtype
   */
  setMQMType(type, subtype = '') {
    this.error_.type = type;
    this.error_.subtype = subtype;
    const errorInfo = this.config.errors[type];
    if (errorInfo.forced_severity) {
      this.setMQMSeverity(errorInfo.forced_severity);
      return;
    }
    this.maybeConcludeError();
  }

  /**
   * Sets the MQM error span for the current MQM rating.
   * @param {number} start
   * @param {number} end
   * @param {string} prefix
   * @param {string} selected
   */
  setMQMSpan(start, end, prefix, selected) {
    const subpara = this.getCurrSubpara();
    if (subpara.hotw && !subpara.hotw.done) {
      this.resetQualityScore();
      const evalResult = this.currSegmentEval();
      this.noteTiming('found-hands-on-the-wheel-error');
      subpara.hotw.done = true;
      subpara.hotw.found = true;
      subpara.hotw.timestamp = evalResult.timestamp;
      subpara.hotw.timing = evalResult.timing;
      evalResult.timing = {};
      this.errorAction_ = '';  /** concludeError() need not call noteTiming() */
      this.concludeError(true);
      return;
    }
    this.error_.start = start + subpara.token_offset;
    this.error_.end = end + subpara.token_offset;
    this.error_.location = ['source', 'translation', 'translation2'][this.cursor.side];
    this.error_.prefix = prefix;
    this.error_.selected = selected;
    this.maybeConcludeError();
  }

  /**
   * Handles cancellation of the current error-marking.
   */
  handleCancel() {
    this.noteTiming('pressed-cancel');
    if (!this.error_) {
      return;
    }
    this.concludeError(true);
  }

  /**
   * Handles a click on a "<severity>" button or its hotkey.
   * @param {string} severityId
   */
  handleSeverityClick(severityId) {
    if (this.buttons_[severityId].disabled) {
      /** We could have come here from a keyboard shortcut. */
      return;
    }
    this.noteTiming('chose-severity-' + severityId);
    this.setMQMSeverity(severityId);
  }

  /**
   * Handles a click on an error "<type>/<subsubtype>" button.
   * @param {string} type
   * @param {string=} subtype
   */
  handleErrorTypeClick(type, subtype='') {
    this.noteTiming('chose-error-' + type + (subtype ? '-' + subtype : ''));
    this.setMQMType(type, subtype);
  }

  /**
   * Marks the passed span as a "fading text" span (its text will slowly fade).
   * @param {!Element} span
   */
  fadeTextSpan(span) {
    if (this.fadingTextSpan_) {
      this.fadingTextSpan_.classList.remove('anthea-fading-text');
    }
    span.classList.add('anthea-fading-text');
    this.fadingTextSpan_ = span;
  }

  /**
   * Navigates to the specified segment and opens highlighting UI.
   * @param {number} n
   * @param {number} side
   * @param {number} s
   */
  revisitSubpara(n, side, s) {
    if (!this.inCurrDoc(n) ||
        !this.cursor.hasBeenRead(n, side, s) ||
        this.cursor.equals(n, side, s) ||
        this.error_)  {
      return;
    }
    const currSubpara = this.getCurrSubpara();
    this.noteTiming('revisited');
    this.cursor.goto(n, side, s);
    this.redrawAllSegments();
    this.fadeTextSpan(currSubpara.subparaSpan);
  }

  /**
   * Returns a random integer in the range [0, max).
   * @param {number} max
   * @return {number}
   */
  static getRandomInt(max) {
    return Math.floor(Math.random() * max);
  }

  /**
   * If the text is sufficiently long, then this function injects a deliberate
   * translation error in some part of the text by reversing a long-enough
   * sub-phrase, within a sentence. See return format in injectHotw()
   * documentation.
   * @param {!Array<string>} tokens
   * @param {!Object} subparaInfo
   * @param {string} tgtLang
   * @return {?Object}
   */
  static injectHotwWordsFlipped(tokens, subparaInfo, tgtLang) {
    /**
     * Error injection is done by reversing a sequence from tokens that starts
     * and ends within spaces/punctuation (or any single-char tokens) and is
     * within a sentence.
     */
    const sent = this.getRandomInt(subparaInfo.sentInfos.length);
    const sentInfo = subparaInfo.sentInfos[sent];
    const tokenStart = sentInfo.token_offset - subparaInfo.token_offset;
    const tokenLimit = tokenStart + sentInfo.num_tokens;
    const seps = [];
    for (let t = tokenStart; t < tokenLimit; t++) {
      if (tokens[t].length == 1) {
        seps.push(t);
      }
    }
    if (seps.length <= 6) {
      // Too short.
      return null;
    }
    // Start within the first half.
    const start = this.getRandomInt(seps.length / 2);
    const starti = seps[start];
    const end = Math.min(seps.length - 1, start + 4 + this.getRandomInt(4));
    const endi = seps[end];
    // Reverse tokens in the range (starti, endi)
    const reversed = tokens.slice(starti + 1, endi).reverse();
    return {
      tokens: tokens.slice(0, starti + 1)
                  .concat(reversed)
                  .concat(tokens.slice(endi)),
      hotwError: `<span class="anthea-hotw-revealed" lang="${tgtLang}">` +
          reversed.join('') + '</span>',
      hotwType: 'words-flipped',
    };
  }

  /**
   * If the text has a sufficiently long word, then this function injects a
   * deliberate translation error in some part of the text by reversing a
   * long-enough sub-string in a word. See return format in injectHotw()
   * documentation.
   * @param {!Array<string>} tokens
   * @param {!Object} subparaInfo
   * @param {string} tgtLang
   * @return {?Object}
   */
  static injectHotwLettersFlipped(tokens, subparaInfo, tgtLang) {
    /**
     * Error injection is done by reversing a long word.
     */
    const longTokenIndices = [];
    const MIN_LETTERS_FLIPPED = 4;
    const MIN_LETTERS = 5;
    for (let t = 0; t < tokens.length; t++) {
      const token = tokens[t];
      if (token == ' ') {
        continue;
      }
      if (token.length >= MIN_LETTERS) {
        longTokenIndices.push(t);
      }
    }

    if (longTokenIndices.length == 0) {
      return null;
    }
    const index = longTokenIndices[this.getRandomInt(longTokenIndices.length)];
    const tokenLetters = tokens[index].split('');
    const offsetLimit = tokenLetters.length - MIN_LETTERS_FLIPPED + 1;
    const startOffset = this.getRandomInt(offsetLimit);
    const sliceLengthLimit = tokenLetters.length - startOffset + 1;
    const sliceLength = MIN_LETTERS_FLIPPED + this.getRandomInt(
        sliceLengthLimit - MIN_LETTERS_FLIPPED);
    const rev = tokenLetters.slice(
        startOffset, startOffset + sliceLength).reverse().join('');
    if (rev == tokenLetters.slice(
                   startOffset, startOffset + sliceLength).join('')) {
      /* We found a palindrome. Give up on this one. */
      return null;
    }
    const ret = {
      tokens: tokens.slice(),
      hotwType: 'letters-flipped',
    };
    ret.tokens[index] = tokenLetters.slice(0, startOffset).join('') + rev +
                        tokenLetters.slice(startOffset + sliceLength).join('');
    ret.hotwError =
        `<span class="anthea-hotw-revealed" lang="${tgtLang}">` +
        ret.tokens[index] + '</span>';
    return ret;
  }

  /**
   * Pretend to inject an HOTW error, but don't actually do it. Only used
   * for training demos. See return format in injectHotw() documentation.
   * @param {!Array<string>} tokens
   * @param {!Object} subparaInfo
   * @return {?Object}
   */
  static injectHotwPretend(tokens, subparaInfo) {
    return {
      tokens: tokens.slice(),
      hotwError: '[Not a real injected error: only a training demo]',
      hotwType: 'pretend-hotw',
    };
  }

  /**
   * If possible, inject an HOTW error in the tokenized text. The returned
   * object is null if no injection could be done. Otherwise it is an object
   * consisting of modified "tokens" (guaranteed to be the same length as the
   * input "tokens"), and the fields "hotwError" (that is an HTML snippet
   * that shows the modification clearly) and "hotwType".
   *
   * @param {!Array<string>} tokens
   * @param {!Object} subparaInfo
   * @param {boolean} hotwPretend Only pretend to insert error, for training.
   * @param {string} tgtLang The target language, for text direction.
   * @return {?Object}
   */
  static injectHotw(tokens, subparaInfo, hotwPretend, tgtLang) {
    if (hotwPretend) {
      return AntheaEval.injectHotwPretend(tokens, subparaInfo);
    }
    /* 60% chance for words-flipped, 40% for letter-flipped */
    const tryWordsFlipped = this.getRandomInt(100) < 60;
    if (tryWordsFlipped) {
      const ret =
          AntheaEval.injectHotwWordsFlipped(tokens, subparaInfo, tgtLang);
      if (ret) {
        return ret;
      }
    }
    return AntheaEval.injectHotwLettersFlipped(tokens, subparaInfo, tgtLang);
  }

  /**
   * Tokenizes text, splitting on space and on zero-width space. The zero-width
   * space character is not included in the returned tokens, but space is. Empty
   * strings are not emitted as tokens. Consecutive whitespaces are first
   * normalized to single spaces.
   * @param {string} text
   * @return {!Array<string>}
   */
  static tokenize(text) {
    const normText = text.replace(/[\s]+/g, ' ');
    let tokens = [];
    const textParts = normText.split(' ');
    for (let index = 0; index < textParts.length; index++) {
      const part = textParts[index];
      // Segment further by any 0-width space characters present.
      const subParts = part.split('\u200b');
      for (let subPart of subParts) {
        if (subPart) {
          tokens.push(subPart);
        }
      }
      if (index < textParts.length - 1) {
        tokens.push(' ');
      }
    }
    return tokens;
  }

  /**
   * Wraps each non-space token in text in a SPAN of class "anthea-word"
   * and each space token in a SPAN of class "anthea-space". Appends a <br>
   * tag at the ends of sentences that have ends_with_line_break set.
   * @param {!Array<string>} tokens
   * @param {!Array<!Object>} sentInfos Each one has num_tokens
   *     and optionally ends_with_line_break set.
   * @return {string}
   */
  static spannifyTokens(tokens, sentInfos) {
    let spannified = '';
    let offset = 0;
    for (const sentInfo of sentInfos) {
      for (let i = 0; i < sentInfo.num_tokens; i++) {
        const token = tokens[offset + i];
        const cls = (token == ' ') ? 'anthea-space' : 'anthea-word';
        spannified += `<span class="${cls}">${token}</span>`;
      }
      offset += sentInfo.num_tokens;
      if (sentInfo.ends_with_line_break) {
        spannified += '<br>';
      }
    }
    return spannified;
  }

  /**
   * Splits text into sentences (marked by two zero-width spaces) and tokens
   * (marked by spaces and zero-width spaces), groups them into "subparas",
   * and creates display-ready HTML (including possibly adding HOTW errors).
   * Returns an array of subpara-wise objects. Each object includes the
   * following fields:
   *   spanHTML: The HTML version of the subpara, with tokens wrapped in spans.
   *   hotwSpanHTML: Empty, or spanHTML variant with HOTW error-injected.
   *   hotwType: Empty, or a description of the injected HOTW error.
   *   hotwError: Empty, or the HOTW error.
   *   num_words: number of tokens that are not spaces/line-breaks.
   *   ends_with_para_break: If the text (before space-normalization) ended in
   *         two or more newlines, then ends_with_para_break is set to true.
   *   token_offset: The index of the first token.
   *   num_tokens: The number of tokens.
   *   sentInfos[]: Array of sentence-wise objects containing:
   *       num_tokens
   *       num_words
   *       ends_with_line_break
   *       token_offset
   * In the HTML, each inter-word space is wrapped in a SPAN of class
   * "anthea-space" and each word is wrapped in a SPAN of class "anthea-word".
   * Adds a trailing space token to the last subpara unless addEndSpaces
   * is false or there already is a space there. Sentence endings in line-breaks
   * get a <br> tag inserted into the HTML.
   *
   * @param {string} text
   * @param {boolean} addEndSpaces
   * @param {number} subparaSentences,
   * @param {number} subparaTokens,
   * @param {number} hotwPercent
   * @param {boolean=} hotwPretend
   * @param {string=} tgtLang
   * @return {!Array<!Object>}
   */
  static splitAndSpannify(text, addEndSpaces,
                          subparaSentences, subparaTokens,
                          hotwPercent, hotwPretend=false, tgtLang='') {
    const sentences = text.split('\u200b\u200b');
    const spacesNormalizer = (s) => s.replace(/[\s]+/g, ' ');
    const sentInfos = [];
    let totalNumTokens = 0;
    for (let s = 0; s < sentences.length; s++) {
      const paraBreak = sentences[s].endsWith('\n\n');
      const lineBreak = !paraBreak && sentences[s].endsWith('\n');
      const sentence = spacesNormalizer(sentences[s]);
      const tokens = AntheaEval.tokenize(sentence);
      if (addEndSpaces && (s == sentences.length - 1) &&
          (tokens.length == 0 || tokens[tokens.length - 1] != ' ')) {
        tokens.push(' ');
      }
      let numWords = 0;
      for (const token of tokens) {
        if (token != ' ') {
          numWords++;
        }
      }
      const sentInfo = {
        index: s,
        num_words: numWords,
        ends_with_line_break: lineBreak,
        ends_with_para_break: paraBreak,
        token_offset: totalNumTokens,
        num_tokens: tokens.length,
        tokens: tokens,
      };
      totalNumTokens += tokens.length;
      sentInfos.push(sentInfo);
    }
    const subparas = MarotUtils.makeSubparas(
        sentInfos, subparaSentences, subparaTokens);
    const subparaInfos = [];
    for (const subpara of subparas) {
      console.assert(subpara.length > 0, subparas);
      const subparaInfo = {
        sentInfos: [],
        ends_with_para_break:
            sentInfos[subpara[subpara.length - 1]].ends_with_para_break,
        token_offset: sentInfos[subpara[0]].token_offset,
        hotwSpanHTML: '',
        hotwType: '',
        hotwError: '',
        num_words: 0,
        num_tokens: 0,
      };
      let tokens = [];
      for (const s of subpara) {
        const sentInfo = sentInfos[s];
        tokens = tokens.concat(sentInfo.tokens);
        subparaInfo.sentInfos.push(sentInfo);
        subparaInfo.num_words += sentInfo.num_words;
      }
      subparaInfo.num_tokens = tokens.length;
      subparaInfo.spanHTML = AntheaEval.spannifyTokens(
          tokens, subparaInfo.sentInfos);
      if (hotwPercent > 0 && (100 * Math.random()) < hotwPercent) {
        const hotw =
            AntheaEval.injectHotw(tokens, subparaInfo, hotwPretend, tgtLang);
        if (hotw) {
          subparaInfo.hotwSpanHTML = AntheaEval.spannifyTokens(
              hotw.tokens, subparaInfo.sentInfos);
          subparaInfo.hotwType = hotw.hotwType;
          subparaInfo.hotwError = hotw.hotwError;
        }
      }
      subparaInfos.push(subparaInfo);
    }
    return subparaInfos;
  }

  /**
   * Computes the height of the viewport, useful for making subparas visible.
   */
  setViewportHeight() {
    /** From an iframe do not rely on document.documentElement.clientHeight */
    const ch = (window.location != window.parent.location) ? 0 :
        document.documentElement.clientHeight;
    this.viewportHeight_ =  window.innerHeight && ch ?
        Math.min(window.innerHeight, ch) :
        window.innerHeight || ch ||
        document.getElementsByTagName('body')[0].clientHeight;
  }

  /**
   * This function recomputes the tops of subparas in the current doc.
   */
  recomputeTops() {
    this.setViewportHeight();
    const start = this.docs_[this.cursor.doc].startSG;
    const num = this.docs_[this.cursor.doc].numSG;
    const docRowRect = this.docs_[this.cursor.doc].row.getBoundingClientRect();
    let maxTopPos = 0;
    for (let s = start; s < start + num; s++) {
      const segment = this.segments_[s];
      const allSubparas = segment.srcSubparas.concat(segment.tgtSubparas);
      for (let subpara of allSubparas) {
        const subparaRect = subpara.subparaSpan.getBoundingClientRect();
        subpara.topPos = subparaRect.top - docRowRect.top;
        subpara.top = '' + subpara.topPos + 'px';
        maxTopPos = Math.max(subpara.topPos, maxTopPos);
      }
    }
    if (this.evalPanel_) {
      this.evalPanel_.style.top = this.getCurrSubpara().top;
    }
    // Make sure the table height is sufficient.
    const docEvalCell = this.docs_[this.cursor.doc].eval;
    docEvalCell.style.height = '' + (maxTopPos + 600) + 'px';
  }

  /**
   * Returns to the previous document.
   */
  prevDocument() {
    if (!this.READ_ONLY &&
        (this.error_ || this.cursor.presentationIndex === 0)) {
      return;
    }
    this.noteTiming('prev-document');
    if (!this.finishCurrSubpara()) {
      return;
    }
    this.docs_[this.cursor.doc].row.style.display = 'none';
    this.cursor.gotoDoc(
        this.cursor.presentationOrder[this.cursor.presentationIndex - 1]);
    this.displayedDocNum_.innerHTML = `${(this.cursor.presentationIndex + 1)}`;
    this.docs_[this.cursor.doc].row.style.display = '';
    const docEvalCell = this.docs_[this.cursor.doc].eval;
    docEvalCell.appendChild(this.evalPanel_);
    this.showContextAndMediaIfPresent();
    this.redrawAllSegments();
    this.recomputeTops();
    this.refreshCurrSubpara();
  }

  /**
   * Proceeds to the next document.
   */
  nextDocument() {
    if (!this.READ_ONLY &&
        (this.error_ ||
         this.cursor.presentationIndex === this.docs_.length - 1 ||
         !this.cursor.seenDocEnd())) {
      return;
    }
    this.noteTiming('next-document');
    if (!this.finishCurrSubpara()) {
      return;
    }
    this.docs_[this.cursor.doc].row.style.display = 'none';
    this.cursor.gotoDoc(
        this.cursor.presentationOrder[this.cursor.presentationIndex + 1]);
    this.displayedDocNum_.innerHTML = `${(this.cursor.presentationIndex + 1)}`;
    this.docs_[this.cursor.doc].row.style.display = '';
    const docEvalCell = this.docs_[this.cursor.doc].eval;
    docEvalCell.appendChild(this.evalPanel_);
    this.showContextAndMediaIfPresent();
    this.redrawAllSegments();
    this.recomputeTops();
    this.refreshCurrSubpara();
  }

  /**
   * Builds instructions from section contents and section order. Section order
   * and individual section contents (distinguished by name) can be specified in
   * the template; if left unspecified, defaults are taken from
   * template-base.js.
   *
   * @return {string}
   */
  buildInstructions() {
    const order = this.config.instructions_section_order ||
                  antheaTemplateBase.instructions_section_order;
    let sections = antheaTemplateBase.instructions_section_contents;
    /**
     * Add or override each custom section content defined in the template, if
     * any.
     */
    if (this.config.instructions_section_contents) {
      for (let section_name in this.config.instructions_section_contents) {
        sections[section_name] =
            this.config.instructions_section_contents[section_name];
      }
    }
    let builtInstructions = '';
    for (let section_name of order) {
      builtInstructions += sections[section_name];
    }
    return builtInstructions;
  }

  /**
   * Populates an instructions panel with instructions and lists of severities
   * and error types for MQM.
   * @param {!Element} panel The DIV element to populate.
   */
  populateMQMInstructions(panel) {
    // Use hard-coded instructions if present, otherwise build from
    // (possibly default) section order and contents.
    panel.innerHTML = this.config.instructions +
        (!this.config.SKIP_RATINGS_TABLES ? `
      <p>
        <details open>
          <summary>
            <b>List of severities:</b>
          </summary>
          <ul id="anthea-mqm-list-of-severities"></ul>
        </details>
      </p>
      <p>
        <details>
          <summary>
            <b>Table of error types:</b>
          </summary>
          <table id="anthea-mqm-errors-table" class="anthea-errors-table">
          </table>
        </details>
      </p>` : '');

    if (this.config.SKIP_RATINGS_TABLES) {
      return;
    }
    const listOfSeverities = document.getElementById(
        'anthea-mqm-list-of-severities');
    for (let s in this.config.severities) {
      const severity = this.config.severities[s];
      if (severity.hidden) {
        continue;
      }
      listOfSeverities.appendChild(googdom.createDom(
          'li', null,
          googdom.createDom(
            'span',
            { style: 'font-weight:bold; background-color:' + severity.color },
            severity.display),
          ': ' + severity.description));
    }
    const errorsTable = document.getElementById('anthea-mqm-errors-table');
    for (let type in this.config.errors) {
      const errorInfo = this.config.errors[type];
      if (errorInfo.hidden) {
        continue;
      }
      const errorLabel = googdom.createDom(
          'td', 'anthea-error-label', errorInfo.display);
      const errorDesc = googdom.createDom(
          'td', {colspan: 2}, errorInfo.description);
      errorsTable.appendChild(googdom.createDom(
          'tr', null, errorLabel, errorDesc));
      if (!errorInfo.subtypes) {
        continue;
      }
      for (let subtype in errorInfo.subtypes) {
        const subtypeInfo = errorInfo.subtypes[subtype];
        if (subtypeInfo.hidden) {
          continue;
        }
        const emptyCol = document.createElement('td');
        const subtypeLabel = googdom.createDom(
          'td', 'anthea-error-label', subtypeInfo.display);
        const subtypeDesc = googdom.createDom(
          'td', null, subtypeInfo.description);
        errorsTable.appendChild(googdom.createDom(
          'tr', null, emptyCol, subtypeLabel, subtypeDesc));
      }
    }
  }

  /**
   * Modifies the config object, applying overrides.
   * @param {!Object} config The configuration object.
   * @param {string} overrides
   */
  applyConfigOverrides(config, overrides) {
    const parts = overrides.split(',');
    for (let override of parts) {
      override = override.trim();
      // Each override can look like:
      // +severity:<sev> or -severity:<sev>
      // +error:<type>[:<subtype>] or -error:<type>[:<subtype>]
      let shouldAdd = true;
      if (override.charAt(0) == '+') {
        shouldAdd = true;
      } else if (override.charAt(0) == '-') {
        shouldAdd = false;
      } else {
        continue;
      }
      const subparts = override.substr(1).split(':');
      if (subparts.length < 2) {
        continue;
      }
      if (subparts[0] == 'severity') {
        const severity = subparts[1];
        if (!config.severities[severity]) {
          continue;
        }
        config.severities[severity].hidden = !shouldAdd;
      } else if (subparts[0] == 'error') {
        const type = subparts[1];
        if (!config.errors[type]) {
          continue;
        }
        let errorInfo = config.errors[type];
        if (subparts.length > 2) {
          const subtype = subparts[2];
          if (!errorInfo.subtypes || !errorInfo.subtypes[subtype]) {
            continue;
          }
          errorInfo = errorInfo.subtypes[subtype];
        }
        errorInfo.hidden = !shouldAdd;
      } else {
        continue;
      }
    }
  }

  /**
   * Creates the feedback UI for a doc.
   *
   * @param {number} docIdx The index of the doc.
   */
  createDocFeedbackUI(docIdx) {
    const doc = this.docs_[docIdx];

    const feedback = googdom.createDom('div', 'anthea-feedback');
    feedback.title = 'You can optionally provide feedback on your experience ' +
                     'in rating this document, such as unfamiliarity or ' +
                     'appropriateness of the document, any issues with the ' +
                     'interface, etc.';

    const thumbsUpId = 'anthea-feedback-thumbs-up-' + docIdx;
    doc.thumbsUp = googdom.createDom('input', {
        type: 'checkbox',
        id: thumbsUpId,
        name: thumbsUpId,
        class: 'anthea-feedback-thumbs'
    });
    const thumbsUpLabel = googdom.createDom('label', {for: thumbsUpId});
    googdom.setInnerHtml(thumbsUpLabel, '&#x1F44D;');

    const thumbsDnId = 'anthea-feedback-thumbs-dn-' + docIdx;
    doc.thumbsDn = googdom.createDom('input', {
        type: 'checkbox',
        id: thumbsDnId,
        name: thumbsDnId,
        class: 'anthea-feedback-thumbs'
    });
    const thumbsDnLabel = googdom.createDom('label', {for: thumbsDnId});
    googdom.setInnerHtml(thumbsDnLabel, '&#x1F44E;');
    const thumbsSpan = googdom.createDom(
        'span', 'anthea-feedback-summary',
        doc.thumbsUp, thumbsUpLabel, doc.thumbsDn, thumbsDnLabel);

    doc.thumbsUp.addEventListener('change', (e) => {
      if (doc.thumbsUp.checked) {
        doc.thumbsDn.checked = false;
      }
      this.saveResults();
    });
    doc.thumbsDn.addEventListener('change', (e) => {
      if (doc.thumbsDn.checked) {
        doc.thumbsUp.checked = false;
      }
      this.saveResults();
    });

    if (this.READ_ONLY) {
      doc.thumbsUp.disabled = true;
      doc.thumbsDn.disabled = true;
    }

    const feedbackHeader = googdom.createDom('div', 'anthea-feedback-header');
    googdom.setInnerHtml(feedbackHeader,
                         '<b>Feedback/notes (optional):</b>&nbsp;');
    feedbackHeader.appendChild(thumbsSpan);
    feedback.appendChild(feedbackHeader);

    const feedbackPanel = googdom.createDom(
        'div', 'anthea-feedback-panel');
    feedback.appendChild(feedbackPanel);
    doc.feedbackNotes = googdom.createDom(
        'p', 'anthea-feedback-notes');
    doc.feedbackNotes.addEventListener('input', (e) => {
      this.saveResults();
    });
    if (!this.READ_ONLY) {
      doc.feedbackNotes.contentEditable = 'true';
    }
    feedbackPanel.appendChild(doc.feedbackNotes);

    doc.eval.appendChild(feedback);
  }

  /**
   * Returns true if this keyboard event is not a navigation/UI event.
   *
   * @param {!Event} e
   * @return {boolean}
   */
  nonInterfaceEvent(e) {
    return e.target && e.target.className &&
           e.target.className == 'anthea-feedback-notes';
  }

  /**
   * Common method called to set up the editing of the error at the given
   * index. Returns false if the parameters are not valid for an editable error.
   * @param {number} index
   * @param {!Event} evt
   * @return {boolean}
   */
  setUpErrorEditing(index, evt) {
    evt.preventDefault();
    this.modifierMenu_.innerHTML = '';
    const evalResult = this.currSegmentEval();
    if (index < 0 || index >= evalResult.errors.length) {
      return false;
    }
    const modButtonParent = this.modButtonParents_[index];
    if (!modButtonParent) {
      return false;
    }
    const modButton = modButtonParent.firstElementChild;
    if (!modButton) {
      return false;
    }
    this.errorIndex_ = index;
    this.error_ = AntheaError.clone(evalResult.errors[index]);
    return true;
  }

  /**
   * Handles a click on "Delete" for the error at the given index.
   * @param {number} index
   * @param {!Event} evt
   */
  handleDeleteError(index, evt) {
    if (!this.setUpErrorEditing(index, evt)) {
      return;
    }
    this.error_.marked_deleted = true;
    this.initiateErrorAction('error-deletion');
    this.maybeConcludeError();
  }

  /**
   * Handles a click on "Undelete" for the error at the given index.
   * @param {number} index
   * @param {!Event} evt
   */
  handleUndeleteError(index, evt) {
    if (!this.setUpErrorEditing(index, evt)) {
      return;
    }
    delete this.error_.marked_deleted;
    this.initiateErrorAction('error-undeletion');
    this.maybeConcludeError();
  }

  /**
   * Handles a click on "Edit Severity" for the error at the given index.
   * @param {number} index
   * @param {!Event} evt
   */
  handleEditErrorSeverity(index, evt) {
    if (!this.setUpErrorEditing(index, evt)) {
      return;
    }
    this.error_.severity = '';
    this.initiateErrorAction('editing-severity');
    this.refreshCurrSubpara();
  }

  /**
   * Handles a click on "Edit Type" for the error at the given index.
   * @param {number} index
   * @param {!Event} evt
   */
  handleEditErrorCategory(index, evt) {
    if (!this.setUpErrorEditing(index, evt)) {
      return;
    }
    this.error_.type = '';
    this.error_.subtype = '';
    this.initiateErrorAction('editing-category');
    this.refreshCurrSubpara();
  }

  /**
   * Handles a click on "Edit Span" for the error at the given index.
   * @param {number} index
   * @param {!Event} evt
   */
  handleEditErrorSpan(index, evt) {
    if (!this.setUpErrorEditing(index, evt)) {
      return;
    }
    this.error_.start = -1;
    this.error_.end = -1;
    this.initiateErrorAction('editing-span');
    this.refreshCurrSubpara();
  }

  /**
   * Handles a click on "Edit Note" for the error at the given index.
   * @param {number} index
   * @param {!Event} evt
   */
  handleEditErrorNote(index, evt) {
    if (!this.setUpErrorEditing(index, evt)) {
      return;
    }
    this.initiateErrorAction('editing-note');
    const note = prompt("Please enter a short error description",
                        this.error_.metadata.note ?? '');
    if (!note && note != '') {
      this.concludeError(true);
      return;
    }
    this.error_.metadata.note = note;
    this.concludeError();
  }

  /**
   * Toggles the visibility of the error editing menu.
   * @param {!Event} evt
   */
  toggleModifierMenu(evt) {
    evt.preventDefault();
    if (this.modifierMenu_.style.display == 'none') {
      this.modifierMenu_.style.display = '';
    } else {
      this.modifierMenu_.style.display = 'none';
    }
  }

  /**
   * Creates and shows the error editing menu for the error at the given index.
   * @param {number} index
   * @param {!Event} evt
   */
  showModifierMenu(index, evt) {
    evt.preventDefault();
    const evalResult = this.currSegmentEval();
    if (index < 0 || index >= evalResult.errors.length) {
      return;
    }
    const modButtonParent = this.modButtonParents_[index];
    if (!modButtonParent) {
      return;
    }
    const modButton = modButtonParent.firstElementChild;
    if (!modButton || modButton.disabled) {
      return;
    }

    this.modifierMenu_.innerHTML = '';
    const menuTable = document.createElement('table');
    this.modifierMenu_.appendChild(menuTable);

    const error = evalResult.errors[index];
    if (error.marked_deleted) {
      const modMenuUndelete = googdom.createDom(
          'button',
          {class: 'anthea-stretchy-button anthea-modifier-menu-button'},
          'Undelete');
      modMenuUndelete.addEventListener  (
          'click', this.handleUndeleteError.bind(this, index));
      menuTable.appendChild(
          googdom.createDom(
              'tr', null, googdom.createDom('td', null, modMenuUndelete)));
    } else {
      const modMenuDelete = googdom.createDom(
          'button',
          {class: 'anthea-stretchy-button anthea-modifier-menu-button'},
          'Delete');
      modMenuDelete.addEventListener(
          'click', this.handleDeleteError.bind(this, index));
      menuTable.appendChild(
          googdom.createDom(
              'tr', null, googdom.createDom('td', null, modMenuDelete)));
      const modMenuSeverity = googdom.createDom(
          'button',
          {class: 'anthea-stretchy-button anthea-modifier-menu-button'},
          'Edit Severity');
      modMenuSeverity.addEventListener(
          'click', this.handleEditErrorSeverity.bind(this, index));
      menuTable.appendChild(
          googdom.createDom(
              'tr', null, googdom.createDom('td', null, modMenuSeverity)));
      const modMenuCategory = googdom.createDom(
          'button',
          {class: 'anthea-stretchy-button anthea-modifier-menu-button'},
          'Edit Type');
      modMenuCategory.addEventListener(
          'click', this.handleEditErrorCategory.bind(this, index));
      menuTable.appendChild(
          googdom.createDom(
              'tr', null, googdom.createDom('td', null, modMenuCategory)));
      const modMenuSpan = googdom.createDom(
          'button',
          {class: 'anthea-stretchy-button anthea-modifier-menu-button'},
          'Edit Span');
      modMenuSpan.addEventListener(
          'click', this.handleEditErrorSpan.bind(this, index));
      menuTable.appendChild(
          googdom.createDom(
              'tr', null, googdom.createDom('td', null, modMenuSpan)));
      const modMenuNote = googdom.createDom(
          'button',
          {class: 'anthea-stretchy-button anthea-modifier-menu-button'},
          'Edit Note');
      modMenuNote.addEventListener(
          'click', this.handleEditErrorNote.bind(this, index));
      menuTable.appendChild(
          googdom.createDom(
              'tr', null, googdom.createDom('td', null, modMenuNote)));
    }
    modButtonParent.appendChild(this.modifierMenu_);
    this.modifierMenu_.style.display = '';
  }

  /**
   * Creates the UI.
   *
   * @param {!Element} instructionsPanel The instructions panel to populate.
   * @param {!Element} controlPanel The control panel to populate.
   */
  createUI(instructionsPanel, controlPanel) {
    /* Remove previous keydown listeners if any */
    for (let listener of this.keydownListeners) {
      document.removeEventListener('keydown', listener);
    }
    this.keydownListeners = [];

    const docEvalCell = this.docs_[this.cursor.doc].eval;
    this.evalPanel_ = googdom.createDom(
      'div', {id: 'anthea-eval-panel', class: 'anthea-eval-panel'});
    docEvalCell.appendChild(this.evalPanel_);

    this.populateMQMInstructions(instructionsPanel);

    this.evalPanelHead_ = googdom.createDom('div', 'anthea-eval-panel-head');
    this.evalPanel_.appendChild(this.evalPanelHead_);

    this.evalPanelBody_ = googdom.createDom('div', 'anthea-eval-panel-body');
    this.evalPanelBody_.style.display = 'none';
    this.evalPanel_.appendChild(this.evalPanelBody_);

    this.evalPanelList_ = googdom.createDom('div', 'anthea-eval-panel-list');
    this.evalPanel_.appendChild(this.evalPanelList_);
    this.evalPanelErrors_ = googdom.createDom(
        'table', 'anthea-eval-panel-table');
    this.evalPanelList_.appendChild(this.evalPanelErrors_);

    const buttonsRow = document.createElement('tr');
    this.evalPanelHead_.appendChild(
      googdom.createDom('table', 'anthea-eval-panel-table', buttonsRow));

    /** Create feedback UI for each doc. */
    for (let docIdx = 0; docIdx < this.docs_.length; docIdx++) {
      this.createDocFeedbackUI(docIdx);
    }

    const switchListener = (e) => {
      if (e.key && e.key === "Tab") {
        if (this.nonInterfaceEvent(e)) {
          return;
        }
        e.preventDefault();
        this.handleSwitch();
      }
    };
    this.keydownListeners.push(switchListener);
    document.addEventListener('keydown', switchListener);

    this.prevButton_ = googdom.createDom(
        'button', {
          id: 'anthea-prev-button',
          class: 'anthea-stretchy-button anthea-eval-panel-tall',
          title: 'Go back to the previous sentence(s) ' +
              '(shortcut: left-arrow key)'
        },
        '←');
    const prevListener = (e) => {
      if (e.type == 'click' ||
          (!this.prevButton_.disabled && e.key && e.key === "ArrowLeft")) {
        if (this.nonInterfaceEvent(e)) {
          return;
        }
        e.preventDefault();
        this.handlePrev();
      }
    };
    this.prevButton_.addEventListener('click', prevListener);
    this.keydownListeners.push(prevListener);
    document.addEventListener('keydown', prevListener);
    buttonsRow.appendChild(googdom.createDom(
      'td', 'anthea-eval-panel-nav',
      googdom.createDom('div', 'anthea-eval-panel-nav', this.prevButton_)));

    for (let s in this.config.severities) {
      const severity = this.config.severities[s];
      const buttonText =
          severity.display + (severity.shortcut ?
                              ' [' + severity.shortcut + ']' : '');
      const severityButton = googdom.createDom(
          'button', {
            class: 'anthea-stretchy-button anthea-eval-panel-tall',
            style: 'background-color:' + severity.color,
            title: severity.description
          },
          buttonText);
      const listener = (e) => {
        if (e.type == 'click' || (e.key && e.key == severity.shortcut)) {
          if (this.nonInterfaceEvent(e)) {
            return;
          }
          e.preventDefault();
          this.handleSeverityClick(s);
        }
      };
      if (severity.shortcut) {
        this.keydownListeners.push(listener);
        document.addEventListener('keydown', listener);
      }
      severityButton.addEventListener('click', listener);
      if (!severity.hidden) {
        buttonsRow.appendChild(googdom.createDom(
            'td', 'anthea-eval-panel-cell', severityButton));
      }
      this.buttons_[s] = severityButton;
    }

    /**
     * Create the div that will contain the menu for modification of an
     * existing annotation.
     */
    this.modifierMenu_ = googdom.createDom('div', 'anthea-modifier-menu');

    this.nextButton_ = googdom.createDom(
        'button', {
          id: 'anthea-next-button',
          class: 'anthea-stretchy-button anthea-eval-panel-tall',
          title: 'Go to the next sentence(s) ' +
              '(shortcut: right-arrow key)'
        },
        '→');
    const nextListener = (e) => {
      if (e.type == 'click' ||
          (!this.nextButton_.disabled && e.key && e.key === "ArrowRight")) {
        if (this.nonInterfaceEvent(e)) {
          return;
        }
        e.preventDefault();
        this.handleNext();
      }
    };
    this.nextButton_.addEventListener('click', nextListener);
    this.keydownListeners.push(nextListener);
    document.addEventListener('keydown', nextListener);
    buttonsRow.appendChild(googdom.createDom(
      'td', 'anthea-eval-panel-nav',
      googdom.createDom('div', 'anthea-eval-panel-nav', this.nextButton_)));

    this.displayedDocNum_ = googdom.createDom(
      'span', null, `${(this.cursor.presentationIndex + 1)}`);
    this.displayedProgress_ = googdom.createDom('span', 'anthea-bold',
                                                 this.getPercentEvaluated());
    const progressMessage = googdom.createDom(
        'span', 'anthea-status-text', 'Document no. ',
        this.displayedDocNum_, ' of ' + this.docs_.length);
    if (!this.READ_ONLY) {
      progressMessage.appendChild(googdom.createDom(
          'span', null, ' (across all documents, ', this.displayedProgress_,
          ' of translation segments have been evaluated so far)'));
    }
    const documentDisplayTerm =
        this.config.SHARED_SOURCE ? 'Translation' : 'Document';
    this.prevDocButton_ = googdom.createDom(
      'button', {
        id: 'anthea-prev-doc-button', class: 'anthea-docnav-eval-button',
        title: 'Revisit the previous document' },
      `Prev ${documentDisplayTerm}`);
    this.prevDocButton_.style.backgroundColor = this.buttonColor_;
    this.prevDocButton_.addEventListener(
      'click', (e) => {
        e.preventDefault();
        this.prevDocument();
    });
    this.nextDocButton_ = googdom.createDom(
      'button', {
        id: 'anthea-next-doc-button', class: 'anthea-docnav-eval-button',
        title: 'Proceed with evaluating the next document' },
      `Next ${documentDisplayTerm}`);
    this.nextDocButton_.style.backgroundColor = this.buttonColor_;
    this.nextDocButton_.disabled = true;
    this.nextDocButton_.addEventListener(
      'click', (e) => {
        e.preventDefault();
        this.nextDocument();
    });
    controlPanel.appendChild(
      googdom.createDom(
          'div', null,
          this.prevDocButton_, this.nextDocButton_, progressMessage));

    this.makeEvalPanel();
    this.phraseMarker_ = new AntheaPhraseMarker(this, this.highlightColor_);
  }

  /**
   * Returns the approximate number of Lines in descendants with class cls.
   * @param {!Element} elt The parent element,
   * @param {string} cls The class name of descendants.
   * @return {number} The approximate # of lines in descendants with class cls.
   */
  getApproxNumLines(elt, cls) {
    const desc = elt.getElementsByClassName(cls);
    let height = 0;
    for (let i = 0; i < desc.length; i++) {
      height += desc[i].getBoundingClientRect().height;
    }
    // 1.3 is line-height, 150% of 13 is the font-size.
    return Math.ceil(height / (1.3 * 13 * 1.5));
  }

  /**
   * Adjusts the line-height of the smaller column to compensate.
   * @param {!Element} srcTD The source TD cell.
   * @param {!Element} tgtTD The target TD cell.
   * @param {?Element} tgtTD2 The second target TD cell for sideBySide.
   */
  adjustHeight(srcTD, tgtTD, tgtTD2) {
    if (!srcTD || !tgtTD) {
      return;
    }
    // Set default line-height to 1.3 and get the line count in each column.
    srcTD.style.lineHeight = 1.3;
    tgtTD.style.lineHeight = 1.3;
    const srcLines = this.getApproxNumLines(srcTD, 'anthea-source-para');
    const tgtLines = this.getApproxNumLines(tgtTD, 'anthea-target-para');
    const colTDs = [[srcLines, srcTD], [tgtLines, tgtTD]];
    if (tgtTD2) {
      tgtTD2.style.lineHeight = 1.3;
      const tgtLines2 = this.getApproxNumLines(tgtTD2, 'anthea-target-para');
      colTDs.push([tgtLines2, tgtTD2]);
    }
    // Sort the columns by line count to decide which one to adjust.
    colTDs.sort((a, b) => a[0] - b[0]);
    // Adjust the line-height of the shorter columns to match the longest one.
    for (let i = 0; i < colTDs.length; i++) {
      const perc = Math.floor(100 * colTDs[colTDs.length - 1][0] / colTDs[i][0]);
      if (perc >= 101) {
        colTDs[i][1].style.lineHeight = 1.3 * perc / 100;
      }
    }
  }

  /**
   * Set the rectangle's coordinates from the box, after scaling.
   */
  setRectStyle(rect, box, scale) {
    rect.style.left = '' + (box.x * scale) + 'px';
    rect.style.top = '' + (box.y * scale) + 'px';
    rect.style.width = '' + (box.w * scale) + 'px';
    rect.style.height = '' + (box.h * scale) + 'px';
  }

  /**
   * Set up zoom-on-hover on the image shown in imgCell.
   */
  setUpImageZooming(imgWrapper, imgCell, zoom, url, w, h, selBox, scale) {
    const zoomW = 600;
    const zoomH = 200;
    const box = {x: 0, y: 0, w: zoomW, h: zoomH};
    this.setRectStyle(zoom, box, 1);

    const zoomImg = googdom.createDom(
      'img', {'class': 'anthea-context-image-zoom-bg', 'src': url,
              'width': w, 'height': h});
    zoom.appendChild(zoomImg);

    const zoomSel = googdom.createDom(
        'div',
        {'class':
             'anthea-context-image-zoom-bg anthea-context-image-selection'});
    this.setRectStyle(zoomSel, selBox, 1);
    zoom.appendChild(zoomSel);

    imgWrapper.addEventListener('mousemove', (e) => {
      zoom.style.display = 'none';
      const bodyRect = document.body.getBoundingClientRect();
      const cellRect = imgCell.getBoundingClientRect();

      const scaledX = (e.pageX - cellRect.left + bodyRect.left) / scale;
      if (scaledX < 0 || scaledX >= w) {
        return;
      }
      const scaledY = (e.pageY - cellRect.top + bodyRect.top) / scale;
      if (scaledY < 0 || scaledY >= h) {
        return;
      }
      zoomImg.style.top = '' + (0 - scaledY) + 'px';
      zoomImg.style.left = '' + (0 - scaledX) + 'px';
      zoomSel.style.top = '' + (selBox.y - scaledY) + 'px';
      zoomSel.style.left = '' + (selBox.x - scaledX) + 'px';

      const wrapperRect = imgWrapper.getBoundingClientRect();
      const x = e.pageX - wrapperRect.left + bodyRect.left;
      const y = e.pageY - wrapperRect.top + bodyRect.top;
      zoom.style.left = '' + x + 'px';
      zoom.style.top = '' + y + 'px';

      zoom.style.display = '';
    });
    imgWrapper.addEventListener('mouseleave', (e) => {
      zoom.style.display = 'none';
    });
  }

  /**
   * Extract source media, if provided via an annotation on the first segment.
   */
  extractSourceMedia() {
    this.manager_.log(this.manager_.INFO,
                      'Extracting source media from annotations');
    for (let i = 0; i < this.docs_.length; i++) {
      const thisDoc = this.docs_[i];
      // Add an empty image as a default, so that page layout is not broken for
      // documents that don't have any source media.
      thisDoc.srcMedia = {url: 'data:,', type: 'image'};
      if (!thisDoc.docsys.annotations ||
          thisDoc.docsys.annotations.length === 0 ||
          !thisDoc.docsys.annotations[0]) {
        this.manager_.log(this.manager_.WARNING,
                          `No annotation (hence no source media) for doc ${i}`);
        continue;
      }
      try {
        const annotation = JSON.parse(thisDoc.docsys.annotations[0]);
        if (!annotation.source_media) {
          this.manager_.log(
              this.manager_.ERROR,
              `Incomplete/missing source media in the annotation for doc ${i}`);
          continue;
        }
        // Overwrite the default source media.
        thisDoc.srcMedia = annotation.source_media;
      } catch (err) {
        this.manager_.log(
            this.manager_.ERROR,
            `Unparseable source media in the annotation for doc ${i}`);
        continue;
      }
    }
  }

  /**
   * Extract page contexts, if provided via an annotation on the first segment.
   */
  extractPageContexts() {
    this.manager_.log(this.manager_.INFO,
                      'Extracting page contexts from annotations');
    for (let i = 0; i < this.docs_.length; i++) {
      const thisDoc = this.docs_[i];
      if (!thisDoc.docsys.annotations ||
          thisDoc.docsys.annotations.length == 0 ||
          !thisDoc.docsys.annotations[0]) {
        this.manager_.log(this.manager_.WARNING,
                          'No annotation (hence no page context) for doc ' + i);
        continue;
      }
      try {
        const pageContext = JSON.parse(thisDoc.docsys.annotations[0]);
        if (!pageContext.source_context || !pageContext.target_context) {
          this.manager_.log(
              this.manager_.ERROR,
              'Incomplete/missing page context in the annotation for doc ' + i);
          continue;
        }
        thisDoc.srcContext = pageContext.source_context;
        thisDoc.tgtContext = pageContext.target_context;
      } catch (err) {
        this.manager_.log(
            this.manager_.ERROR,
            'Unparseable page context in the annotation for doc ' + i);
        continue;
      }
    }
  }

  /**
   * If the current document uses media as the source instead of text (e.g. a
   * video), then show it in the source panel.
   */
  showSourceMediaIfPresent() {
    const doc = this.docs_[this.cursor.doc];
    if (!doc.srcMedia) {
      return;
    }
    if (!["video", "image"].includes(doc.srcMedia.type)) {
      this.manager_.log(
          this.manager_.ERROR,
          `Source media is not a video or image (or type not specified): "${
              doc.srcMedia.type}"`);
      return;
    }
    const driveRegex = /https:\/\/drive.google.com\/file\/d\/[^\/]+\/preview/;
    const ytRegex = /https:\/\/www.youtube.com\/embed\/.*/;
    if (doc.srcMedia.type === 'video' &&
        ![driveRegex, ytRegex].some((re) => re.test(doc.srcMedia.url))) {
      this.manager_.log(
          this.manager_.ERROR,
          `Source media is a video but not from YouTube or Drive (or is misformatted): ${
              doc.srcMedia.url}`);
      return;
    }
    const mediaCell = googdom.createDom(
        'div', 'anthea-source-media-cell');
    switch (doc.srcMedia.type) {
      case 'image':
        mediaCell.appendChild(googdom.createDom(
          'img', {src: doc.srcMedia.url, title: 'Source media image'}));
        break;
      case 'video':
        mediaCell.appendChild(googdom.createDom('iframe', {
          src: doc.srcMedia.url,
          title: 'Source media video',
          allow: 'autoplay; encrypted-media;',
          allowfullscreen: true
        }));
        break;
      default:
        this.manager_.log(
            this.manager_.ERROR,
            `Source media is not a video or image (or type not specified): "${
                doc.srcMedia.type}"`);
        return;
    }
    // The source text cell is the first child of the row. Replace it with the
    // media cell.
    doc.row.replaceChild(
        googdom.createDom('td', null, mediaCell), doc.row.firstChild);
  }

  /**
   * If the current document has available page context or source media, then
   * display them. See showPageContextIfPresent() and showSourceMediaIfPresent()
   * for more details.
   */
  showContextAndMediaIfPresent() {
    this.showPageContextIfPresent();
    this.showSourceMediaIfPresent();
  }

  /**
   * If the current document has available page context (screenshots of source
   * and translation, and bounding boxes for the text getting evaluated), then
   * show it in the context row.
   */
  showPageContextIfPresent() {
    const doc = this.docs_[this.cursor.doc];
    this.contextRow_.innerHTML = '';
    this.contextRow_.style.display = 'none';
    if (!doc.srcContext || !doc.tgtContext) {
      return;
    }

    /**
     * Keep image width down to fit more content vertically. But if
     * the height is very large then use a larger width.
     */
    const width = Math.max(doc.srcContext.h,
                           doc.tgtContext.h) > 2000 ? 450 : 320;

    /**
     * Slightly complex layout, to allow scrolling images vertically, and
     * yet let the zoomed view spill outside. The zoomed view also shows
     * the selection outline.
     * td
     *   anthea-context-image-wrapper
     *     anthea-context-image-port (scrollable)
     *       anthea-context-image-cell
     *         img
     *         anthea-context-image-selection
     *     anthea-context-image-zoom
     *       full-img
     *       full-selection
     */
    const srcImg = googdom.createDom(
        'img', {src: doc.srcContext.url,
                class: 'anthea-context-image', width: width});
    const srcScale = width / doc.srcContext.w;
    const srcSelection = googdom.createDom('div',
                                           'anthea-context-image-selection');
    this.setRectStyle(srcSelection, doc.srcContext.box, srcScale);
    const srcCell = googdom.createDom(
        'div', 'anthea-context-image-cell', srcImg, srcSelection);
    const srcPort = googdom.createDom('div',
                                      'anthea-context-image-port', srcCell);
    const srcZoom = googdom.createDom('div', 'anthea-context-image-zoom');
    srcZoom.style.display = 'none';
    const srcWrapper = googdom.createDom(
        'div', 'anthea-context-image-wrapper', srcPort, srcZoom);
    this.contextRow_.appendChild(googdom.createDom('td', null, srcWrapper));
    this.setUpImageZooming(srcWrapper, srcCell, srcZoom, doc.srcContext.url,
                           doc.srcContext.w, doc.srcContext.h,
                           doc.srcContext.box, srcScale);

    const tgtImg = googdom.createDom(
        'img', {src: doc.tgtContext.url,
                class: 'anthea-context-image', width: width});
    const tgtScale = width / doc.tgtContext.w;
    const tgtSelection = googdom.createDom('div',
                                           'anthea-context-image-selection');
    this.setRectStyle(tgtSelection, doc.tgtContext.box, tgtScale);
    const tgtCell = googdom.createDom(
        'div', 'anthea-context-image-cell', tgtImg, tgtSelection);
    const tgtPort = googdom.createDom('div',
                                      'anthea-context-image-port', tgtCell);
    const tgtZoom = googdom.createDom('div',
                                      'anthea-context-image-zoom');
    tgtZoom.style.display = 'none';
    const tgtWrapper = googdom.createDom(
        'div', 'anthea-context-image-wrapper', tgtPort, tgtZoom);
    this.contextRow_.appendChild(googdom.createDom('td', null, tgtWrapper));
    this.setUpImageZooming(tgtWrapper, tgtCell, tgtZoom, doc.tgtContext.url,
                           doc.tgtContext.w, doc.tgtContext.h,
                           doc.tgtContext.box, tgtScale);

    this.contextRow_.appendChild(googdom.createDom('td',
                                                   'anthea-context-eval-cell'));
    this.contextRow_.style.display = '';

    const sOpt = {block: "center"};
    srcSelection.scrollIntoView(sOpt);
    tgtSelection.scrollIntoView(sOpt);
  }

  /**
   * Sets up the eval. This is the main starting point for the JavaScript code,
   *     and is called when the HTML DOM is loaded.
   *
   * @param {!Element} evalDiv The DIV in which to create the eval.
   * @param {string} templateName The template name.
   * @param {!Array<!Object>} projectData Project data, including src/tgt
   *     text segments. The array also may have a "parameters" property.
   * @param {?Array<!Object>} projectResults Previously saved partial results.
   * @param {number=} hotwPercent Percent rate for HOTW testing.
   */
  setUpEval(evalDiv, templateName, projectData, projectResults, hotwPercent=0) {
    if (typeof antheaTemplateBase == 'object' &&
        typeof MarotUtils == 'function' &&
        antheaTemplates[templateName]) {
      /** We have all the pieces. */
      this.setUpEvalWithConfig(
          evalDiv, templateName, antheaTemplates[templateName], projectData,
          projectResults, hotwPercent);
      return;
    }
    if (this.haveInitiatedLoadingDependencies) {
      /** Need to wait till called again with all deppendencies loaded. */
      return;
    }
    const retrier = this.setUpEval.bind(
        this, evalDiv, templateName, projectData, projectResults, hotwPercent);
    googdom.setInnerHtml(
        evalDiv, 'Loading dependencies & template ' + templateName + '...');
    const filesToLoad = [];
    if (typeof antheaTemplateBase != 'object') {
      filesToLoad.push('template-base.js');
    }
    if (!antheaTemplates[templateName]) {
      filesToLoad.push('template-' + templateName.toLowerCase() + '.js');
    }
    if (typeof MarotUtils != 'function') {
      filesToLoad.push('marot-utils.js');
    }
    for (const f of filesToLoad) {
      const scriptTag = document.createElement('script');
      scriptTag.src = this.scriptUrlPrefix_ + f;
      scriptTag.onload = retrier;
      document.head.append(scriptTag);
    }
    this.haveInitiatedLoadingDependencies = true;
  }

  /**
   * Returns true if the language code (BCP-47) is for a language that
   * generally uses spaces between sentences.
   * @param {string} lang BCP 47 code
   * @return {boolean}
   */
  isSpaceSepLang(lang) {
    const lowerLang = lang.toLowerCase();
    if (lowerLang.startsWith('ja') || lowerLang.startsWith('zh')) {
      return false;
    }
    return true;
  }

  /**
   * Validates the input format of the SIDE_BY_SIDE mode.
   * The adjacent docsys should have the same source segments.
   * @param {!Array<!Object>} projectData The input data.
   * @return {boolean}
   */
  validateSideBySideData(projectData) {
    for (let i = 0; i < projectData.length; i += 2) {
      const docsys = projectData[i];
      const docsys2 = projectData[i + 1];
      if (docsys.srcSegments.length !== docsys2.srcSegments.length) {
        this.manager_.log(this.manager_.ERROR,
                          "Source segment lengths differ between systems!");
        return false;
      }
      for (let j = 0; j < docsys.srcSegments.length; j++) {
        if (docsys.srcSegments[j] !== docsys2.srcSegments[j]) {
          this.manager_.log(this.manager_.ERROR,
                            'Source segments differ at index ' + j);
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Checks whether all expected docsys indices are present. For SIDE_BY_SIDE
   * templates, also checks that docsys pairs in the input are still adjacent
   * in the given order.
   * @param {number} dataLength The length of the input data.
   * @param {!Array<number>} dataPresentationOrder A permutation of input data
   *     indices.
   * @param {number} groupSize The size of chunks that must remain contiguous.
   * @return {boolean}
   */
  validateDataPresentationOrder(dataLength, dataPresentationOrder, groupSize) {
    if (dataLength !== dataPresentationOrder.length) {
      this.manager_.log(
          this.manager_.ERROR,
          `Presentation order length (${
              dataPresentationOrder
                  .length}) does not match input data length (${
              dataLength})!`);
      return false;
    }
    const providedIndices = new Set(dataPresentationOrder);
    const expectedIndices = Array.from({length: dataLength}, (v, i) => i);
    const hasAllExpectedIndices = expectedIndices.every((v) => providedIndices.has(v));
    if (!hasAllExpectedIndices) {
      this.manager_.log(
          this.manager_.ERROR,
          'Presentation order is missing one or more expected indices!');
      return false;
    }
    if (groupSize > 1) {
      if (dataPresentationOrder.length % groupSize !== 0) {
        this.manager_.log(
            this.manager_.ERROR,
            `Input data length prevents forming groups of size ${groupSize}!`);
        return false;
      }
      for (let i = 0; i < dataPresentationOrder.length; i += groupSize) {
        const sortedDataGroup =
            dataPresentationOrder.slice(i, i + groupSize).sort();
        if (sortedDataGroup.at(-1) - sortedDataGroup[0] !== groupSize - 1) {
          this.manager_.log(
              this.manager_.ERROR, 'Presentation order separates some groups!');
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Creates a random presentation order, by shuffling the order of groups and
   * the order of data within each group.
   * @param {number} dataLength The length of the input data.
   * @param {number} groupSize The size of groups that should remain contiguous.
   * @param {!AntheaDeterministicRandom} pseudoRandNumGenerator A random number generator.
   * @return {!Array<number>}
   */
  randomPresentationOrder(dataLength, groupSize, pseudoRandNumGenerator) {
    const dataPresentationOrder = [];
    // First decide the order in which to show each group.
    const groupPresentationOrder = AntheaEval.pseudoRandomShuffle(
        Array.from({length: dataLength / groupSize}, (v, i) => i),
        pseudoRandNumGenerator);
    if (groupSize === 1) {
      return groupPresentationOrder;
    }
    // If groups have multiple items, shuffle within each group.
    const groupOffsets = Array.from({length: groupSize}, (v, i) => i);
    groupPresentationOrder.forEach(
        (groupIdx) => dataPresentationOrder.push(...AntheaEval.pseudoRandomShuffle(
            groupOffsets.map((offset) => offset + groupIdx * groupSize),
            pseudoRandNumGenerator)));
    return dataPresentationOrder;
  }

  /**
   * If the JSON parameters provide a presentation order, returns
   * it. If not but a non-zero shuffle_seed is provided, produces a random
   * presentation order. Otherwise, returns an unshuffled presentation order.
   * @param {!Jsonable} parameters The JSON parameters.
   * @param {number} dataLength The length of the input data.
   * @param {number} groupSize The size of groups that should remain contiguous.
   * @return {!Array<number>}
   */
  getOrCreateDataPresentationOrder(parameters, dataLength, groupSize) {
    // A seed of 0 will result in no shuffling.
    const pseudoRandNumGenerator =
        new AntheaDeterministicRandom(parameters.shuffle_seed || 0);
    // If a presentation order is provided, the random number generator is not
    // used.
    const dataPresentationOrder = parameters.presentation_order ||
        this.randomPresentationOrder(
            dataLength, groupSize, pseudoRandNumGenerator);
    return dataPresentationOrder;
  }
  /**
   * Builds HTML for target subparagraphs.
   * Iterates over the provided target subparagraphs, constructs HTML for each,
   * and accumulates the result in `tgtSpannified`. Also updates the `segment` object
   * with relevant information, including word count and potential hotword data.
   * @param {!Element} segment The segment to add the subpara to.
   * @param {!Array<!Object>} tgtSubparas The target subparas.
   * @param {string} tgtSpannified The target span HTML.
   * @param {string} tgtSegmentClass The segment class.
   * @param {string} tgtParaBreak The target paragraph break.
   * @param {!Object} evalResult The eval result object.
   * @param {number} lastTimestampMS The last timestamp.
   * @param {string} tgtSubparaClassName The target subpara class name.
   * @return {string} The updated tgtSpannified.
   */
  buildTargetSubparaHTML(segment, tgtSubparas, tgtSpannified,
                         tgtSegmentClass, tgtParaBreak, evalResult,
                         lastTimestampMS, tgtSubparaClassName) {
    for (let t = 0; t < tgtSubparas.length; t++) {
      const tgtSubpara = tgtSubparas[t];
      tgtSpannified += tgtSubparaClassName + tgtSegmentClass + '">' +
                      (tgtSubpara.hotwSpanHTML || tgtSubpara.spanHTML) +
                      '</span>';
      if (tgtSubpara.ends_with_para_break) {
        tgtSpannified += tgtParaBreak;
      }
      segment.numTgtWords += tgtSubpara.num_words;
      if (tgtSubpara.hotwError) {
        tgtSubpara.hotw = {
          timestamp: lastTimestampMS,
          injected_error: tgtSubpara.hotwError,
          hotw_html: tgtSubpara.hotwSpanHTML,
          hotw_type: tgtSubpara.hotwType,
          para: t,
          // Add in the location of where hotw is injected.
          // It is removed when merging/splitting the saved results.
          location: tgtSubparaClassName.includes("-target2-") ?
              'translation2' :
              'translation',
          done: false,
          found: false,
        };
        evalResult['hotw_list'].push(tgtSubpara.hotw);
      }
    }
    return tgtSpannified;
  }

  /**
   * Determines whether the source column should be displayed. True if it will
   * be populated with source text or source media.
   *
   * @param {!Object} config The template configuration object.
   * @return {boolean} True if the source column should be displayed.
   */
  shouldDisplaySourceColumn(config) {
    return !config.TARGET_SIDE_ONLY || config.USE_SOURCE_MEDIA;
  }

  /**
   * Sets up the eval. This is the starting point called once the template has
   *     been loaded.
   *
   * @param {!Element} evalDiv The DIV in which to create the eval.
   * @param {string} templateName The template name.
   * @param {!Object} config The template configuration object.
   * @param {!Array<!Object>} projectData Project data, including src/tgt
   *     text segments. The array also may have a "parameters" property.
   * @param {?Array<!Object>} projectResults Previously saved partial results.
   * @param {number} hotwPercent Percent rate for HOTW testing.
   */
  setUpEvalWithConfig(
      evalDiv, templateName, config, projectData, projectResults, hotwPercent) {
    this.config = config;
    evalDiv.innerHTML = '';

    if (!this.config.instructions) {
      this.config.instructions = this.buildInstructions();
    }
    const instructionsPanel = googdom.createDom('div',
                                                'anthea-mqm-instructions');
    instructionsPanel.id = 'anthea-mqm-instructions-panel';
    evalDiv.append(instructionsPanel);

    this.lastTimestampMS_ = Date.now();

    this.evalResults_ = [];

    this.segments_ = [];
    this.docs_ = [];

    this.contextRow_ = googdom.createDom('tr', 'anthea-context-row');
    this.contextRow_.style.display = 'none';

    const parameters = projectData.parameters || {};
    this.srcLang = parameters.source_language || '';
    this.tgtLang = parameters.target_language || '';

    let noteToRaters = parameters.hasOwnProperty('note_to_raters') ?
        parameters.note_to_raters :
        '';
    /**
     * By default, raters navigate in units of sentences. If subpara_*
     * parameters have been passed in, they control the unit size.
     * Also support the old names of these parameters (paralet_*).
     *
     * When collecting quality scores, we want to make it clear that the score
     * applies to the entire segment, so we set the unit sizes to -1 to indicate
     * that no splitting should happen.
     */
    if (config.COLLECT_QUALITY_SCORE) {
      parameters.subpara_sentences = -1;
      parameters.subpara_tokens = -1;
    }
    const subparaSentences = parameters.subpara_sentences ?? (parameters.paralet_sentences ?? 1);
    const subparaTokens = parameters.subpara_tokens ?? (parameters.paralet_tokens ?? 1);

    if (parameters.hasOwnProperty('hotw_percent')) {
      /* Override the passed value */
      hotwPercent = parameters.hotw_percent;
    }
    /** Are we only pretending to add hotw errors, for training? */
    const hotwPretend = parameters.hotw_pretend || false;

    const srcHeading =
        this.srcLang ? ('Source (' + this.srcLang + ')') : 'Source';
    const srcHeadingDiv = googdom.createDom('div', null, srcHeading);

    const targetLabel =
        this.shouldDisplaySourceColumn(config) ? 'Translation' : 'Text';
    const tgtHeading = this.tgtLang ?
        (targetLabel + ' (' + this.tgtLang + ')') : targetLabel;
    const tgtHeadingDiv = googdom.createDom('div', null, tgtHeading);

    // Set up second target column in the sideBySide mode.
    const targetLabel2 = `${targetLabel} 2`;
    const tgtHeading2 =
        this.tgtLang ? `${targetLabel2} (${this.tgtLang})` : targetLabel2;
    const tgtHeadingDiv2 = googdom.createDom('div', null, tgtHeading2);

    const evalHeading = this.READ_ONLY ?
        'Evaluations (view-only)' : 'Evaluations';
    const evalHeadingDiv = googdom.createDom('div', null, evalHeading);

    if (config.subheadings) {
      if (config.subheadings.source) {
        srcHeadingDiv.appendChild(googdom.createDom('br'));
        srcHeadingDiv.appendChild(googdom.createDom(
            'span', 'anthea-subheading', config.subheadings.source));
      }
      if (config.subheadings.target) {
        tgtHeadingDiv.appendChild(googdom.createDom('br'));
        tgtHeadingDiv.appendChild(googdom.createDom(
            'span', 'anthea-subheading', config.subheadings.target));
      }
      if (config.subheadings.target) {
        tgtHeadingDiv2.appendChild(googdom.createDom('br'));
        tgtHeadingDiv2.appendChild(googdom.createDom(
            'span', 'anthea-subheading', config.subheadings.target));
      }
      if (config.subheadings.evaluations) {
        evalHeadingDiv.appendChild(googdom.createDom('br'));
        evalHeadingDiv.appendChild(googdom.createDom(
            'span', 'anthea-subheading', config.subheadings.evaluations));
      }
    }

    const srcHeadingTD = googdom.createDom(
        'td', 'anthea-text-heading', srcHeadingDiv);
    const tgtHeadingTD = googdom.createDom(
        'td', 'anthea-text-heading', tgtHeadingDiv);
    const headerRow = googdom.createDom(
      'tr', null, srcHeadingTD, tgtHeadingTD);
    if (config.SIDE_BY_SIDE) {
      const tgtHeadingTD2 = googdom.createDom(
        'td', 'anthea-text-heading', tgtHeadingDiv2);
      headerRow.appendChild(tgtHeadingTD2);
    }
    const evalHeadingTD = googdom.createDom(
      'td', 'anthea-text-heading', evalHeadingDiv);
    headerRow.appendChild(evalHeadingTD);
    const docTextTable = googdom.createDom(
        'table', 'anthea-document-text-table',
        headerRow,
        this.contextRow_);
    if (!this.shouldDisplaySourceColumn(config)) {
      srcHeadingTD.style.display = 'none';
    }
    evalDiv.appendChild(docTextTable);

    const srcParaBreak =
        `</p><p class="anthea-source-para" dir="auto" lang="${this.srcLang}">`;
    const tgtParaBreak =
        `</p><p class="anthea-target-para" dir="auto" lang="${this.tgtLang}">`;

    let priorResults = [];
    let priorRaters = [];

    // The size of groups meant to be evaluated together. Shuffling will not
    // break up these groups.
    const groupSize = config.SIDE_BY_SIDE ? 2 : 1;
    if (config.SIDE_BY_SIDE &&
        !this.validateSideBySideData(projectData, groupSize)) {
      return;
    }

    const dataPresentationOrder = this.getOrCreateDataPresentationOrder(
        parameters, projectData.length, groupSize);
    if (!this.validateDataPresentationOrder(
            projectData.length, dataPresentationOrder, groupSize)) {
      return;
    }
    // Recover presentation order of docsys groups.
    const groupPresentationOrder = [];
    for (let i = 0; i < dataPresentationOrder.length; i += groupSize) {
      groupPresentationOrder.push(
          Math.floor(dataPresentationOrder[i] / groupSize));
    }

    for (let i = 0; i < projectData.length; i += groupSize) {
      const docsys = projectData[i];
      const doc = {'docsys': docsys};
      const docsys2 = config.SIDE_BY_SIDE ? projectData[i + 1] : null;
      if (config.SIDE_BY_SIDE) {
        doc.docsys2 = docsys2;
      }
      this.docs_.push(doc);
      doc.eval = googdom.createDom('div', 'anthea-document-eval-div');

      const docTextSrcRow = googdom.createDom('td',
                                              'anthea-document-text-cell');
      const docTextTgtRow = googdom.createDom('td',
                                              'anthea-document-text-cell');
      const docTextTgtRow2 = config.SIDE_BY_SIDE ? googdom.createDom('td',
                                              'anthea-document-text-cell') : null;
      const tgtsOrder = [1];
      const tgtRows = [docTextTgtRow];
      if (config.SIDE_BY_SIDE) {
        // Even indices mean that the first docsys should be on the left.
        const tgt2SpliceIndex =
            dataPresentationOrder.indexOf(i) % 2 === 0 ? 1 : 0;
        tgtsOrder.splice(tgt2SpliceIndex, 0, 2);
        tgtRows.splice(tgt2SpliceIndex, 0, docTextTgtRow2);
      }
      doc.row = googdom.createDom('tr',
                                  null, docTextSrcRow,
                                  ...tgtRows,
                                  googdom.createDom('td', 'anthea-document-eval-cell', doc.eval));
      if (config.TARGET_SIDE_ONLY) {
        docTextSrcRow.style.display = 'none';
      }
      doc.startSG = this.evalResults_.length;
      doc.numSG = 0;

      docTextTable.appendChild(doc.row);

      const srcSegments = docsys.srcSegments;
      const tgtSegments = docsys.tgtSegments;
      // Create the second target column for sideBySide templates.
      const tgtSegments2 = config.SIDE_BY_SIDE ? docsys2.tgtSegments : null;
      const annotations = docsys.annotations;
      let srcSpannified =
          `<p class="anthea-source-para" dir="auto" lang="${this.srcLang}">`;
      let tgtSpannified =
          `<p class="anthea-target-para" dir="auto" lang="${this.tgtLang}">`;
      let tgtSpannified2 = config.SIDE_BY_SIDE ?
          `<p class="anthea-target-para" dir="auto" lang="${this.tgtLang}">` :
          '';
      const addEndSpacesSrc = this.isSpaceSepLang(this.srcLang);
      const addEndSpacesTgt = this.isSpaceSepLang(this.tgtLang);
      for (let j = 0; j < srcSegments.length; j++) {
        if (srcSegments[j].length === 0 && tgtSegments[j].length === 0) {
          /* New paragraph. */
          srcSpannified += srcParaBreak;
          tgtSpannified += tgtParaBreak;
          tgtSpannified2 += config.SIDE_BY_SIDE ? tgtParaBreak : '';
          continue;
        }

        const evalResult = {
          'errors': [],
          'doc': this.docs_.length - 1,
          'visited': false,
          'timestamp': this.lastTimestampMS_,
          'timing': {},
          'hotw_list': [],
        };
        if (config.COLLECT_QUALITY_SCORE) {
          evalResult.quality_scores =
              Array.from({'length': config.SIDE_BY_SIDE ? 2 : 1}).fill(-1);
        }
        this.evalResults_.push(evalResult);
        if (j < annotations.length) {
          let parsed_anno = {};
          try {
            parsed_anno = JSON.parse(annotations[j]);
          } catch (err) {
            parsed_anno = {};
          }
          if (parsed_anno.hasOwnProperty('prior_result')) {
            priorResults.push(parsed_anno.prior_result);
            let priorRater = 'unspecified-prior-rater';
            if (parsed_anno.hasOwnProperty('prior_rater')) {
              priorRater = parsed_anno.prior_rater;
            } else if (parameters.hasOwnProperty('prior_rater')) {
              priorRater = parameters.prior_rater;
            }
            priorRaters.push(priorRater);
          }
          if (parsed_anno.hasOwnProperty('note_to_raters')) {
            if (noteToRaters) {
              this.manager_.log(
                  this.manager_.ERROR,
                  'Found note to raters in multiple places (JSON parameters ' +
                  'and/or multiple segment annotations); ignoring ' +
                  'subsequent notes');
              }
            else {
              noteToRaters = parsed_anno.note_to_raters;
            }
          }
        }

        const segment = {
          doc: this.docs_.length - 1,
          srcText: srcSegments[j],
          tgtText: tgtSegments[j],
          tgtText2: config.SIDE_BY_SIDE ? tgtSegments2[j] : '',
          tgtsOrder: tgtsOrder,
          numTgtWords: 0,
          numTgtWords2: 0,
          srcSubparas: AntheaEval.splitAndSpannify(
              srcSegments[j], addEndSpacesSrc,
              subparaSentences, subparaTokens, 0),
          tgtSubparas: AntheaEval.splitAndSpannify(
              tgtSegments[j], addEndSpacesTgt,
              subparaSentences, subparaTokens,
              this.READ_ONLY ? 0 : hotwPercent, hotwPretend, this.tgtLang),
          tgtSubparas2: config.SIDE_BY_SIDE ? AntheaEval.splitAndSpannify(
              tgtSegments2[j], addEndSpacesTgt,
              subparaSentences, subparaTokens,
              this.READ_ONLY ? 0 : hotwPercent, hotwPretend, this.tgtLang) : [],
        };
        const segIndex = this.segments_.length;
        this.segments_.push(segment);

        const srcSegmentClass = 'anthea-source-segment-' + segIndex;
        for (let srcSubpara of segment.srcSubparas) {
          srcSpannified += '<span class="anthea-source-subpara ' +
                          srcSegmentClass + '">' +
                          srcSubpara.spanHTML + '</span>';
          if (srcSubpara.ends_with_para_break) {
            srcSpannified += srcParaBreak;
          }
        }

        const tgtSegmentClass = 'anthea-target-segment-' + segIndex;
        const tgtSegmentClass2 = 'anthea-target2-segment-' + segIndex;
        tgtSpannified = this.buildTargetSubparaHTML(segment,
                                               segment.tgtSubparas,
                                               tgtSpannified,
                                               tgtSegmentClass,
                                               tgtParaBreak,
                                               evalResult,
                                               this.lastTimestampMS_,
                                               '<span class="anthea-target-subpara ');
        if (config.SIDE_BY_SIDE) {
          tgtSpannified2 = this.buildTargetSubparaHTML(segment,
                                                  segment.tgtSubparas2,
                                                  tgtSpannified2,
                                                  tgtSegmentClass2,
                                                  tgtParaBreak,
                                                  evalResult,
                                                  this.lastTimestampMS_,
                                                  '<span class="anthea-target2-subpara ');
        }
        // Increment the total target word count for the progress tracker.
        this.numTgtWordsTotal_ += segment.numTgtWords;
        doc.numSG++;
      }
      googdom.setInnerHtml(docTextSrcRow, srcSpannified + '</p>');
      googdom.setInnerHtml(docTextTgtRow, tgtSpannified + '</p>');
      if (config.SIDE_BY_SIDE) {
        googdom.setInnerHtml(docTextTgtRow2, tgtSpannified2 + '</p>');
      }
      // Adjust line height, then hide all but the first group.
      this.adjustHeight(docTextSrcRow, docTextTgtRow, docTextTgtRow2);
      if (i !== groupSize * groupPresentationOrder[0]) {
        doc.row.style.display = 'none';
      }
    }
    // For shared-source templates, verify that every docsys has the same
    // source segments.
    if (config.SHARED_SOURCE) {
      const sharedSrcSegments = [];
      let sameSrc = true;
      for (const doc of this.docs_) {
        const docsys = doc.docsys;
        if (sharedSrcSegments.length === 0) {
          sharedSrcSegments.push(...docsys.srcSegments);
        } else if (sharedSrcSegments.length === docsys.srcSegments.length) {
          sameSrc &&= docsys.srcSegments.every(
              (val, idx) => val === sharedSrcSegments[idx]);
        } else {
          sameSrc = false;
        }
      }
      if (!sameSrc) {
        this.manager_.log(
            this.manager_.ERROR,
            'This is a shared-source template, but not all documents have' +
                ' the same source segments!');
      }
    }
    for (let i = 0; i < this.segments_.length; i++) {
      const segment = this.segments_[i];
      const srcSubparaSpans = document.getElementsByClassName(
          'anthea-source-segment-' + i);
      const tgtSubparaSpans = document.getElementsByClassName(
        'anthea-target-segment-' + i);
      console.assert(srcSubparaSpans.length == segment.srcSubparas.length);
      console.assert(tgtSubparaSpans.length == segment.tgtSubparas.length);
      for (let s = 0; s < segment.srcSubparas.length; s++) {
        segment.srcSubparas[s].subparaSpan = srcSubparaSpans[s];
      }
      for (let t = 0; t < segment.tgtSubparas.length; t++) {
        segment.tgtSubparas[t].subparaSpan = tgtSubparaSpans[t];
      }
      if (config.SIDE_BY_SIDE) {
        const tgtSubparaSpans2 = document.getElementsByClassName(
          'anthea-target2-segment-' + i);
        console.assert(tgtSubparaSpans2.length === segment.tgtSubparas2.length);
        for (let t = 0; t < segment.tgtSubparas2.length; t++) {
          segment.tgtSubparas2[t].subparaSpan = tgtSubparaSpans2[t];
        }
      }
    }
    const controlPanel = document.createElement('div');
    controlPanel.id = 'anthea-control-panel';
    evalDiv.append(controlPanel);

    // Compute source_side_ok and source_side_only from subtypes.
    for (let type in config.errors) {
      const errorInfo = config.errors[type];
      if (!errorInfo.subtypes || errorInfo.subtypes.length == 0) {
        if (!errorInfo.source_side_only) errorInfo.source_side_only = false;
        if (!errorInfo.source_side_ok) {
          errorInfo.source_side_ok = errorInfo.source_side_only;
        }
        continue;
      }
      errorInfo.source_side_only = true;
      errorInfo.source_side_ok = false;
      for (let subtype in errorInfo.subtypes) {
        const subtypeInfo = errorInfo.subtypes[subtype];
        if (!subtypeInfo.source_side_only) subtypeInfo.source_side_only = false;
        if (!subtypeInfo.source_side_ok) {
          subtypeInfo.source_side_ok = subtypeInfo.source_side_only;
        }
        errorInfo.source_side_only &&= subtypeInfo.source_side_only;
        errorInfo.source_side_ok ||= subtypeInfo.source_side_ok;
      }
    }

    this.cursor = new AntheaCursor(
        this.segments_,
        !!config.TARGET_SIDE_ONLY,
        !!config.TARGET_SIDE_FIRST,
        !!config.SIDE_BY_SIDE,
        this.updateProgressForSegment.bind(this),
        groupPresentationOrder);

    if (noteToRaters) {
      this.config.instructions +=
          '<p class="anthea-note-to-raters">' +
          this.escapeHtml(noteToRaters) + '</p>';
    }
    this.createUI(instructionsPanel, controlPanel);

    if (parameters.hasOwnProperty('prior_results')) {
      if (priorResults.length > 0) {
        this.manager_.log(
            this.manager_.ERROR,
            'Found prior results in both JSON parameters and segment ' +
                'annotations; ignoring JSON parameters');
      } else {
        priorResults = parameters.prior_results;
        let priorRater = 'unspecified-prior-rater';
        if (parameters.hasOwnProperty('prior_rater')) {
          priorRater = parameters.prior_rater;
        }
        priorRaters = Array.from({'length': priorResults.length})
            .fill(priorRater);
      }
    }
    if (priorResults.length > 0) {
      this.startFromPriorResults(priorRaters, priorResults);
    }
    this.restoreEvalResults(projectResults);
    this.saveResults();

    const metadata = {
      template: templateName,
      config: config,
      hotw_percent: hotwPercent,
      anthea_version: this.VERSION,
      ...parameters,
    };
    this.manager_.setMetadata(metadata);

    // Extract page contexts if the config expects them.
    if (config.USE_PAGE_CONTEXT) {
      this.extractPageContexts();
    }
    if (config.USE_SOURCE_MEDIA) {
      this.extractSourceMedia();
    }
    this.showContextAndMediaIfPresent();

    this.redrawAllSegments();
    this.recomputeTops();
    this.resizeListener_ = () => { this.recomputeTops(); };
    window.addEventListener('resize', this.resizeListener_);
  }
}

/**
 * The AntheaPhraseMarker class is used to collect highlighted phrases for the
 *     current subpara.
 * @final
 */
class AntheaPhraseMarker {
  /**
   * @param {!AntheaEval} contextedEval
   * @param {string} color
   */
  constructor(contextedEval, color) {
    /** @private @const {!AntheaEval} */
    this.contextedEval_ = contextedEval;

    /** @private @const {string} */
    this.color_ = color;

    /** @private {number} */
    this.startSpanIndex_ = -1;
    /** @private {number} */
    this.endSpanIndex_ = -1;
    /** @private {?Set} */
    this.markables_ = null;

    /** @private {!Array<!Element>} Token span elements */
    this.tokenSpans_ = [];
    /** @private {!Array<string>} Saved colors of token spans during marking */
    this.tokenSpanColors_ = [];
  }

  /**
   * Returns true if the start of the error span has already been marked.
   * @return {boolean}
   */
  startAlreadyMarked() {
    return this.startSpanIndex_ >= 0;
  }

  /**
   * Resets the word spans in the current subpara, getting rid of any
   *     event listeners from spannification done in the previous state. Sets
   *     element class to 'anthea-word-active' or 'anthea-word-active-begin' or
   *     'anthea-space-active' or 'anthea-space-active-begin'.
   */
  resetWordSpans() {
    const ce = this.contextedEval_;
    ce.redrawCurrSubpara();

    this.tokenSpans_ = ce.getCurrTokenSpans();

    const allowSpaceStart = ce.config.ALLOW_SPANS_STARTING_ON_SPACE || false;

    this.tokenSpanColors_ = [];
    const spanClassSuffix = (this.startSpanIndex_ < 0) ? '-begin' : '';
    const suffix = '-active' + spanClassSuffix;
    const spaceClass = 'anthea-space';
    const wordClass = 'anthea-word';
    const spaceClassActive = spaceClass + suffix;
    const wordClassActive = wordClass + suffix;
    for (let x = 0; x < this.tokenSpans_.length; x++) {
      this.tokenSpanColors_.push(this.tokenSpans_[x].style.backgroundColor);
      if (!this.markables_.has(x)) {
        continue;
      }
      const classList = this.tokenSpans_[x].classList;
      classList.replace(wordClass, wordClassActive);
      if (classList.replace(spaceClass, spaceClassActive)) {
        if (allowSpaceStart) {
          classList.replace(spaceClassActive, wordClassActive);
        }
      }
    }
  }

  /**
   * Colors the background starting from the span starting at startSpanIndex
   *     and ending at spanIndex (which may be < startSpanIndex_)
   * @param {number} spanIndex
   */
  highlightTo(spanIndex) {
    if (spanIndex >= this.tokenSpans_.length || spanIndex < 0) {
      return;
    }
    for (let x = 0; x < this.tokenSpans_.length; x++) {
      const span = this.tokenSpans_[x];
      if ((x >= this.startSpanIndex_ && x <= spanIndex) ||
          (x <= this.startSpanIndex_ && x >= spanIndex)) {
        span.style.backgroundColor = this.color_;
      } else {
        span.style.backgroundColor = this.tokenSpanColors_[x];
      }
    }
  }

  /**
   * Completes the selection of a highlighted phrase, at the span indexed by
   *     spanIndex.
   * @param {number} spanIndex
   */
  pickEnd(spanIndex) {
    const ce = this.contextedEval_;
    ce.noteTiming('marked-error-span-end');
    if (spanIndex < this.startSpanIndex_) {
      this.endSpanIndex_ = this.startSpanIndex_;
      this.startSpanIndex_ = spanIndex;
    } else {
      this.endSpanIndex_ = spanIndex;
    }
    /* Remove anthea-word listeners. */
    this.resetWordSpans();
    /* But re-do the highlighting. */
    this.highlightTo(this.endSpanIndex_);

    let prefix = '';
    for (let x = 0; x < this.startSpanIndex_; x++) {
      prefix = prefix + this.tokenSpans_[x].innerText;
    }
    let selected = '';
    for (let x = this.startSpanIndex_; x <= this.endSpanIndex_; x++) {
      selected = selected + this.tokenSpans_[x].innerText;
    }
    ce.setMQMSpan(this.startSpanIndex_, this.endSpanIndex_, prefix, selected);
  }

  /**
   * Notes that startSpanIndex for the highlighted phrase and sets up the UI for
   *     picking the end of the phrase.
   * @param {number} spanIndex
   */
  prepareToPickEnd(spanIndex) {
    this.startSpanIndex_ = spanIndex;
    const ce = this.contextedEval_;
    this.markables_ = ce.getMarkableSpanIndices(spanIndex);
    /* Remove anthea-word listeners: we'll add new ones. */
    this.resetWordSpans();

    ce.setStartedMarkingSpan();
    ce.noteTiming('marked-error-span-start');
    ce.showGuidance('Click on the end of the error span');

    const span = this.tokenSpans_[spanIndex];
    span.style.backgroundColor = this.color_;

    for (let x = 0; x < this.tokenSpans_.length; x++) {
      if (!this.markables_.has(x)) {
        continue;
      }
      this.tokenSpans_[x].addEventListener(
        'mouseover', () => { this.highlightTo(x); });
      this.tokenSpans_[x].addEventListener('click', () => { this.pickEnd(x); });
    }
  }

  /**
   * The public entrypoint in the AntheaPhraseMarker object. Sets up the UI to
   * collect a highlighted phrase from the current subpara. When phrase-marking
   * is done, the contextedEval_ object's setMQMSpan() function will get called.
   */
  getMarkedPhrase() {
    this.startSpanIndex_ = -1;
    this.endSpanIndex_ = -1;
    const ce = this.contextedEval_;
    this.markables_ = ce.getMarkableSpanIndices();

    this.resetWordSpans();

    const cls = 'anthea-word-active-begin';
    for (let x = 0; x < this.tokenSpans_.length; x++) {
      if (this.tokenSpans_[x].classList.contains(cls)) {
        this.tokenSpans_[x].addEventListener(
          'click',
          () => { this.prepareToPickEnd(x); });
      }
    }
  }
}
