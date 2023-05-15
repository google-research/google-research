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
   * be empty (but not just one of the two), indicating a paragraph break.
   * For convenience, a completely blank line (without the tabs
   * and without document-name and system-name) can also be used to indicate
   * a paragraph break.
   *
   * Lines beginning with # (without any leading spaces) are treated as comment
   * lines. They are ignored. If you have actual source text that begins with #,
   * please prepend a space to it in the file.
   *
   * The first line should contain a JSON object with the source and the target
   * language codes, and potentially other parameters. For instance:
   *   {"source_language": "en", "target_langauge": "fr"}
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
    for (let line of lines.slice(1)) {
      if (line.startsWith('#')) {
        continue;
      }
      line = line.trim();
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
      const srcSegment = parts[0].trim();
      const tgtSegment = parts[1].trim();
      const doc = parts[2].trim() + ':' + srcLang + ':' + tgtLang;
      const sys = parts[3].trim();
      const annotation = parts.length > 4 ? parts[4].trim() : '';
      if ((!srcSegment || !tgtSegment) && (srcSegment || tgtSegment)) {
        console.log('Skipping text line: only one of src/tgt is nonempty: [' +
                    line + ']');
        continue;
      }
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
      if (srcSegment) {
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
    // Add project parameters as a property of the parsed array.
    parsed.parameters = parameters;
    return parsed;
  }
}

/**
 * The AntheaCursor class keeps track of the current location (the "cursor")
 * during an eval. There is a location on the source side and one on the
 * target side. Each location consists of a segment index and a sentence index
 * within the segment. It further remembers the max sentence index shown
 * within each segment.
 */
class AntheaCursor {
  /**
   * @param {!Array<!Object>} segments Each segment object should contain a 
   *     "doc" field and arrays "srcSents" and "tgtSents".
   * @param {boolean} tgtOnly Set to true for monolingual evals.
   * @param {boolean} tgtFirst Set to true for target-first evals.
   * @param {function(number)} segmentDone Called with seg id for each segment.
   */
  constructor(segments, tgtOnly, tgtFirst, segmentDone) {
    this.segments = segments;
    console.assert(segments.length > 0, segments.length);
    this.tgtOnly = tgtOnly;
    this.tgtFirst = tgtFirst;
    this.segmentDone_ = segmentDone;
    this.numSents = [[], []];
    this.numSentsShown = [[], []];
    /** Array<number> identifying the starting seg for each doc. */
    this.docSegStarts = [];
    let doc = -1;
    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      this.numSents[0].push(segment.srcSents.length);
      this.numSents[1].push(segment.tgtSents.length);
      this.numSentsShown[0].push(0);
      this.numSentsShown[1].push(0);
      if (this.docSegStarts.length == 0 || segment.doc != doc) {
        this.docSegStarts.push(i);
        doc = segment.doc;
      }
    }
    console.assert(doc == this.docSegStarts.length - 1);
    this.doc = 0;
    this.seg = 0;
    /** number that is 0 when the current side is src, and 1 when tgt. */
    this.side = 0;
    this.sent = 0;
    this.startAtTgt = this.tgtOnly || this.tgtFirst;
    this.goto(0, this.startAtTgt ? 1 : 0, 0);
  }

  /**
   * Returns true if we are at the start of a doc.
   * @return {boolean}
   */
  atDocStart() {
    if (this.seg != this.docSegStarts[this.doc]) {
      return false;
    }
    const startSide = this.startAtTgt ? 1 : 0;
    if (this.sent != 0 || (this.side != startSide)) {
      return false;
    }
    return true;
  }

  /**
   * Returns true if the passed segment has been full seen.
   * @param {number} seg
   * @return {boolean}
   */
  segFullySeen(seg) {
    if (this.numSentsShown[1][seg] < this.numSents[1][seg]) {
      return false;
    }
    if (this.tgtOnly) {
      return true;
    }
    if (this.numSentsShown[0][seg] < this.numSents[0][seg]) {
      return false;
    }
    return true;
  }

  /**
   * Returns true if we are at the end of a doc.
   * @return {boolean}
   */
  atDocEnd() {
    const endSide = (this.tgtFirst && !this.tgtOnly) ? 0 : 1;
    if (this.side != endSide) {
      return false;
    }
    if (this.sent + 1 != this.numSents[endSide][this.seg]) {
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
   * Moves the cursor to the next sentence. Which sentence that next one is
   * depends on tgtOnly/tgtFirst.
   */
  next() {
    if (this.atDocEnd()) {
      return;
    }
    if (this.sent + 1 < this.numSents[this.side][this.seg]) {
      /** Goto: next sentence, same side. */
      this.goto(this.seg, this.side, this.sent + 1);
    } else {
      if (this.tgtFirst) {
        if (this.side == 1) {
          /** Goto: last-read sentence, src side. */
          const srcSent = Math.max(0, this.numSentsShown[0][this.seg] - 1);
          this.goto(this.seg, 0, srcSent);
        } else {
          if (this.seg + 1 < this.segments.length) {
            /** Goto: start sentence of next seg, tgt side. */
            this.goto(this.seg + 1, 1, 0);
          }
        }
      } else {
        if (this.side == 0) {
          /** Goto: last-read sentence, tgt side. */
          const tgtSent = Math.max(0, this.numSentsShown[1][this.seg] - 1);
          this.goto(this.seg, 1, tgtSent);
        } else {
          /**
           * By using Tab to switch sides, it's possible that you
           * haven't yet seen all of side 0 (src). Check:
           */
          if (!this.segFullySeen(this.seg)) {
            this.switchSides();
          } else if (this.seg + 1 < this.segments.length) {
            /** Goto: start sentence of next seg, src side (tgt for tgtOnly). */
            this.goto(this.seg + 1, this.tgtOnly ? 1 : 0, 0);
          }
        }
      }
    }
  }

  /**
   * Moves the cursor to the previous sentence. Which sentence that previous
   * one is depends on tgtOnly/tgtFirst.
   */
  prev() {
    if (this.atDocStart()) {
      return;
    }
    if (this.sent > 0) {
      this.goto(this.seg, this.side, this.sent - 1);
    } else {
      if (this.tgtFirst) {
        if (this.side == 0) {
          this.goto(this.seg, 1, this.numSents[1][this.seg] - 1);
        } else {
          if (this.seg > 0) {
            this.goto(this.seg - 1, 0, this.numSents[0][this.seg - 1] - 1);
          }
        }
      } else {
        if (this.side == 1 && !this.tgtOnly) {
          this.goto(this.seg, 0, this.numSents[0][this.seg] - 1);
        } else {
          if (this.seg > 0) {
            this.goto(this.seg - 1, 1, this.numSents[1][this.seg - 1] - 1);
          }
        }
      }
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
    return this.numSentsShown[0][seg] > 0;
  }

  /**
   * Makes the cursor jump to the "other" side (src->tgt or tgt->src).
   */
  switchSides() {
    if (this.tgtOnly) {
      return;
    }
    if (!this.srcVisible(this.seg)) {
      return;
    }
    const otherSide = 1 - this.side;
    const otherSent = this.numSentsShown[otherSide][this.seg] - 1;
    this.goto(this.seg, otherSide, otherSent);
  }

  /**
   * Moves the cursor to the specified segment, side, and sentence.
   * @param {number} seg
   * @param {number} side
   * @param {number} sent
   */
  goto(seg, side, sent) {
    console.assert(seg >= 0 && seg < this.segments.length, seg);
    this.seg = seg;
    this.doc = this.segments[seg].doc;
    console.assert(side == 0 || side == 1, side);
    console.assert(!this.tgtOnly || side == 1);
    this.side = side;
    console.assert(sent >= 0 && sent < this.numSents[side][seg], sent);
    this.sent = sent;
    for (let s = 0; s < seg; s++) {
      this.numSentsShown[0][s] = this.numSents[0][s];
      this.numSentsShown[1][s] = this.numSents[1][s];
    }
    this.numSentsShown[side][seg] = Math.max(
        this.numSentsShown[side][seg], sent + 1);
    if (!this.tgtFirst || side == 0) {
      /* At least 1 sent is made visible on the other side, if there is one. */
      const otherSide = 1 - side;
      this.numSentsShown[otherSide][seg] = Math.max(
          this.numSentsShown[otherSide][seg], 1);
    }
    if (this.numSentsShown[1][seg] == this.numSents[1][seg] &&
        (this.tgtOnly ||
         (this.numSentsShown[0][seg] == this.numSents[0][seg]))) {
      this.segmentDone_(seg);
    }
  }

  /**
   * Returns true if the cursor has been through the specified
   * segment/side/sentence.
   * @param {number} seg
   * @param {number} side
   * @param {number} sent
   * @return {boolean}
   */
  hasBeenRead(seg, side, sent) {
    return this.segments[seg].doc == this.doc &&
           this.numSentsShown[side][seg] > sent;
  }

  /**
   * Returns true if the cursor is currently at the specified segment/side/
   * sentence.
   * @param {number} seg
   * @param {number} side
   * @param {number} sent
   * @return {boolean}
   */
  equals(seg, side, sent) {
    return this.seg == seg && this.side == side && this.sent == sent;
  }

  /**
   * Moves the cursor to the start of the specified doc.
   * @param {number} doc
   */
  gotoDoc(doc) {
    console.assert(doc >= 0 && doc < this.docSegStarts.length, doc);
    this.goto(this.docSegStarts[doc], this.startAtTgt ? 1 : 0, 0);
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


    /** ?AntheaCursor Which doc/segment/side/sentence we are at. */
    this.cursor = null;

    /** ?Object */
    this.config = null;

    /** @private {!Array<!Object>} */
    this.evalResults_ = [];

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

    /** @private {?Object} The currently marked phrase. */
    this.markedPhrase_ = null;
    /** @private {boolean} */
    this.startedMarking_ = false;
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
    /** Save any feedback, for each doc */
    for (let docIdx = 0; docIdx < this.docs_.length; docIdx++) {
      const doc = this.docs_[docIdx];
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
    this.manager_.persistActiveResults(this.evalResults_);
  }

  /**
   * Restores eval results from the previously persisted value.
   * 
   * @param {?Array<!Object>} projectResults
   */
  restoreResults(projectResults) {
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
      if (this.READ_ONLY || result.visited) {
        /** Clear any new HOTW injections in this segment */
        for (let sent = 0; sent < segment.tgtSents.length; sent++) {
          const sentence = this.getSentence(seg, 1, sent);
          delete sentence.hotw;
        }
        const result = this.evalResults_[seg];
        for (let hotw of result.hotw_list || []) {
          const sent = hotw.sentence_index;
          const sentence = this.getSentence(seg, 1, sent);
          sentence.hotw = hotw;
        }
        this.cursor.goto(seg, this.config.TARGET_SIDE_ONLY ? 1 : 0, 0);
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
    this.displayedDocNum_.innerHTML = '' + (this.cursor.doc + 1);
    const docEvalCell = this.docs_[this.cursor.doc].eval;
    docEvalCell.appendChild(this.evalPanel_);
    this.displayedProgress_.innerHTML = this.getPercentEvaluated();
    this.showPageContextIfPresent();
    this.redrawAllSegments();
    this.recomputeTops();
    this.redrawCurrSentence();
    this.setEvalButtonsAvailability();

    this.manager_.log(this.manager_.INFO,
                      'Restored previous evaluation results');
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
    // TODO(vratnakar): maybe append cursor.{side,sent} info to action.
    const timing = currEval.timing;
    if (!timing) {
      return;
    }
    if (!timing[action]) {
      timing[action] = {
        count: 0,
        timeMS: 0,
      };
    }
    const tinfo = timing[action];
    tinfo.count++;
    currEval.timestamp = Date.now();
    tinfo.timeMS += (currEval.timestamp - this.lastTimestampMS_);
    this.lastTimestampMS_ = currEval.timestamp;
    this.saveResults();
  }

  /**
   * Redraws the current sentence.
   */
  redrawCurrSentence() {
    this.redrawSentence(this.cursor.seg, this.cursor.side, this.cursor.sent);
  }

  /**
   * Redraws the current sentence and refreshes buttons.
   */
  refreshCurrSentence() {
    this.redrawCurrSentence();
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
   * Returns the sentence object for the specified segment, side, sentence.
   * @param {number} seg
   * @param {number} side
   * @param {number} sent
   * @return {!Object}
   */
  getSentence(seg, side, sent) {
    const segment = this.segments_[seg];
    return side == 0 ? segment.srcSents[sent] : segment.tgtSents[sent];
  }

  /**
   * Returns the sentence object for the current sentence where the cursor is.
   * @return {!Object}
   */
  getCurrSentence() {
    return this.getSentence(
        this.cursor.seg, this.cursor.side, this.cursor.sent);
  }

  /**
   * Returns the SPAN elements for the current sentence.
   * @return {!HTMLCollection}
   */
  getCurrTokenSpans() {
    const sentence = this.getCurrSentence();
    return sentence.sentSpan.getElementsByTagName('span');
  }

  /**
   * Shows the sentence at index seg,side,sent. How the sentence gets shown
   *     depends on whether it is before, at, or after this.cursor.
   * @param {number} seg
   * @param {number} side
   * @param {number} sent
   */
  redrawSentence(seg, side, sent) {
    if (!this.inCurrDoc(seg)) {
      return;
    }

    const sentence = this.getSentence(seg, side, sent);

    /* Get rid of any old highlighting or listeners */
    if (sentence.clickListener) {
      sentence.sentSpan.removeEventListener('click', sentence.clickListener);
    }
    sentence.clickListener = null;

    const evalResult = this.evalResults_[seg];

    let spanHTML = sentence.spanHTML;
    if (!this.READ_ONLY && sentence.hotw && !sentence.hotw.done) {
      spanHTML = sentence.hotwSpanHTML;
    }
    googdom.setInnerHtml(sentence.sentSpan, spanHTML);
    sentence.sentSpan.classList.remove('anthea-sentence-nav');

    /* Highlight errors in sentence */
    const tokenSpans = sentence.sentSpan.getElementsByTagName('span');
    console.assert(tokenSpans.length ==
                   sentence.lastToken - sentence.firstToken + 1);
    const tokenRangeInSeg = [sentence.firstToken, sentence.lastToken];
    const sentErrorIndices = {};
    for (let e = 0; e < evalResult.errors.length; e++) {
      const error = evalResult.errors[e];
      if (side == 0 && error.location != 'source') continue;
      if (side == 1 && error.location == 'source') continue;
      const range = this.intersectRanges(
        [error.start, error.end], tokenRangeInSeg);
      if (!range) continue;
      const severity = this.config.severities[error.severity];
      const color = severity.color;
      for (let x = range[0]; x <= range[1]; x++) {
        tokenSpans[x - sentence.firstToken].style.backgroundColor = color;
      }
      sentErrorIndices[e] = true;
    }

    const isCurr = this.cursor.equals(seg, side, sent);
    if (isCurr) {
      sentence.sentSpan.classList.remove('anthea-fading-text');
      this.evalPanel_.style.top = sentence.top;
      this.evalPanelErrors_.innerHTML = '';
      if (sentence.hotw && sentence.hotw.done) {
        this.displayHOTWMessage(sentence.hotw.found,
                                sentence.hotw.injected_error);
      }
      for (let e = 0; e < evalResult.errors.length; e++) {
        if (sentErrorIndices[e]) {
          this.displayError(evalResult.errors, e);
        }
      }
      if (this.markedPhrase_) {
        /* Indices in markedPhrase_ are for sentence, not segment. */
        const color =
            this.severity_ ? this.severity_.color : this.markedPhrase_.color;
        for (let x = this.markedPhrase_.start; x <= this.markedPhrase_.end;
             x++) {
          tokenSpans[x].style.backgroundColor = color;
        }
      }
      this.noteTiming('visited-or-redrawn');
    }

    const hasBeenRead = this.cursor.hasBeenRead(seg, side, sent);

    if (!isCurr && hasBeenRead && !this.startedMarking_) {
      // anthea-segment-nav class makes the mouse a pointer on hover.
      sentence.sentSpan.classList.add('anthea-sentence-nav');
      sentence.clickListener = () => {
        this.revisitSentence(seg, side, sent);
      };
      sentence.sentSpan.addEventListener('click', sentence.clickListener);
    }

    const afterColor = (side == 0 && this.cursor.tgtFirst &&
                        !this.cursor.srcVisible(seg)) ?
                       'transparent' : this.afterColor_;
    sentence.sentSpan.style.color = isCurr ? this.currColor_ :
        (hasBeenRead ? this.beforeColor_ : afterColor);
    sentence.sentSpan.style.fontWeight = isCurr ? 'bold' : 'normal';
  }

  /**
   * Redraws all segments and calls setEvalButtonsAvailability().
   */
  redrawAllSegments() {
    for (let n = 0; n < this.segments_.length; n++) {
      const segment = this.segments_[n];
      for (let s = 0; s < segment.srcSents.length; s++) {
        this.redrawSentence(n, 0, s);
      }
      for (let t = 0; t < segment.tgtSents.length; t++) {
        this.redrawSentence(n, 1, t);
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
   * Displays the previously marked error in errors[index], alongside the
   *     current segment. The displayed error also includes an "x" button
   *     for deleting it.
   * @param {!Array<!Object>} errors
   * @param {number} index
   */
  displayError(errors, index) {
    const error = errors[index];
    const tr = document.createElement('tr');
    this.evalPanelErrors_.appendChild(tr);

    const delButton = googdom.createDom(
        'button', {class: 'anthea-stretchy-button anthea-eval-panel-short'},
        '×');
    tr.appendChild(googdom.createDom(
        'td', 'anthea-eval-panel-nav',
        googdom.createDom('div', 'anthea-eval-panel-nav', delButton)));

    const severity = this.config.severities[error.severity];
    let desc = error.display || severity.display;
    if (error.subtype) {
      const errorInfo = this.config.errors[error.type] || {};
      if (errorInfo.subtypes && errorInfo.subtypes[error.subtype]) {
        desc = desc + ': ' + errorInfo.subtypes[error.subtype].display;
      }
    }
    if (error.metadata && error.metadata.note) {
      desc = desc + ' [' + error.metadata.note + ']';
    }
    desc += ': ';

    /**
     * Use 0-width spaces to ensure leading/trailing spaces get shown.
     */
    tr.appendChild(googdom.createDom(
        'td', 'anthea-eval-panel-text', desc,
        googdom.createDom(
            'span', {
              dir: 'auto',
              style: 'background-color:' + severity.color,
            },
            '\u200b' + error.selected + '\u200b')));

    delButton.title = 'Delete this error';
    if (this.READ_ONLY) {
      delButton.disabled = true;
    } else {
      delButton.addEventListener('click', (e) => {
        e.preventDefault();
        errors.splice(index, 1);
        this.refreshCurrSentence();
      });
    }
  }

  /**
   * Called from the AntheaPhraseMarker object, this is set when a phrase-start
   *     has been marked.
   */
  setStartedMarking() {
    this.startedMarking_ = true;
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
    this.guidance_.style.display = '';
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
    const disableMarking = this.READ_ONLY ||
        (evalResult.errors && evalResult.errors.length > 0 &&
         (evalResult.errors[0].override_all_errors ||
          (this.config.MAX_ERRORS > 0 &&
           evalResult.errors.length >= this.config.MAX_ERRORS)));
    const disableSevErr = disableMarking || !this.markedPhrase_;
    const disableSeverity = disableSevErr || this.severity_;
    for (let s in this.config.severities) {
      this.buttons_[s].disabled = disableSeverity;
    }

    const disableErrors = disableSevErr || this.errorType_;
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
        if (errorInfo.override_all_errors &&
            this.severityId_ && this.severityId_ != 'major') {
          errorButton.disabled = true;
        }
        if (errorInfo.forced_severity && this.severityId_ &&
            this.severityId_ != errorInfo.forced_severity) {
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
    this.prevDocButton_.style.display = (this.cursor.doc == 0) ? 'none' : '';
    this.prevDocButton_.disabled = false;
    if (this.cursor.doc == this.docs_.length - 1) {
      this.nextDocButton_.style.display = 'none';
    } else {
      this.nextDocButton_.style.display = '';
      this.nextDocButton_.disabled = !this.READ_ONLY &&
                                     !this.cursor.seenDocEnd();
    }

    this.guidancePanel_.style.backgroundColor =
        (this.severity_ ? this.severity_.color : 'whitesmoke');
    this.guidance_.style.display = 'none';
    this.evalPanelBody_.style.display = 'none';
    this.cancel_.style.display = 'none';
    this.evalPanelErrors_.style.display = '';
    if (this.READ_ONLY) {
      return;
    }
    if (!this.startedMarking_) {
      if (!disableMarking) {
        this.phraseMarker_.getMarkedPhrase();
      }
      return;
    }

    // Marking has already been initiated.
    this.evalPanelBody_.style.display = '';
    this.cancel_.style.display = '';
    this.evalPanelErrors_.style.display = 'none';

    if (this.severity_) {
      this.showGuidance('Choose error type / sybtype');
    } else if (this.errorType_) {
      this.showGuidance('Choose error severity');
    } else {
      this.showGuidance('Choose error severity, type / subtype');
    }
    this.openSubtypes(null);
    this.prevButton_.disabled = true;
    this.nextButton_.disabled = true;
    this.prevDocButton_.disabled = true;
    this.nextDocButton_.disabled = true;
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
    this.evalResults_[seg].visited = true;
    this.saveResults();
    this.numWordsEvaluated_ += this.segments_[seg].numTgtWords;
    if (this.displayedProgress_) {
      this.displayedProgress_.innerHTML = this.getPercentEvaluated();
    }
  }

  /**
   * Called after a sentence should be done with. Returns false in
   *     the (rare) case that the sentence was a HOTW sentence with
   *     injected errors shown, which leads to the end of the HOTW check
   *     but makes the rater continue to rate the sentence.
   * @return {boolean}
   */
  finishCurrSentence() {
    const sentence = this.getCurrSentence();
    if (!this.READ_ONLY && sentence.hotw && !sentence.hotw.done) {
      const evalResult = this.currSegmentEval();
      this.noteTiming('missed-hands-on-the-wheel-error');
      sentence.hotw.done = true;
      sentence.hotw.timestamp = evalResult.timestamp;
      sentence.hotw.timing = evalResult.timing;
      evalResult.timing = {};
      this.redrawAllSegments();
      return false;
    }
    return true;
  }

  /**
   * Moves the current sentence into view, if off-screen.
   */
  ensureCurrSentVisible() {
    const sentSpan = this.getCurrSentence().sentSpan;
    const sentRect = sentSpan.getBoundingClientRect();
    if (sentRect.top >= 0 && sentRect.bottom < this.viewportHeight_) {
      return;
    }
    sentSpan.scrollIntoView({block: "center"});
  }

  /**
   * Navigates to the other side.
   */
  handleSwitch() {
    if (this.config.TARGET_SIDE_ONLY) {
      return;
    }
    this.cursor.switchSides();
    this.redrawAllSegments();
    this.ensureCurrSentVisible();
  }

  /**
   * Navigates to the previous sentence.
   */
  handlePrev() {
    this.cursor.prev();
    this.redrawAllSegments();
    this.ensureCurrSentVisible();
  }

  /**
   * Navigates to the next sentence.
   */
  handleNext() {
    if (!this.finishCurrSentence()) {
      return;
    }
    this.cursor.next();
    this.redrawAllSegments();
    this.ensureCurrSentVisible();
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
   * If the passed error requires additional user input, this function
   *     augments it with the needed info. Returns false if the user cancels.
   * @param {!Object} markedError Object with error details.
   * @return {boolean} Whether to continue with marking the error.
   */
  maybeAugmentError(markedError) {
    if (markedError.override_all_errors) {
      if (!confirm(
              'This error category will remove all other marked errors ' +
              'from this segment and will set the error span to ' +
              'be the whole segment. Please confirm!')) {
        this.noteTiming('cancelled-override-all-errors');
        return false;
      }
      this.noteTiming('confirmed-override-all-errors');
      markedError.location = 'translation';
      markedError.prefix = '';
      const spanArray = this.getCurrTokenSpans();
      markedError.selected = '';
      for (let x = 0; x < spanArray.length; x++) {
        markedError.selected += spanArray[x].innerText;
      }
      const sentence = this.getCurrSentence();
      markedError.start = sentence.firstToken;
      markedError.end = sentence.lastToken;
    }

    if (!markedError.metadata) {
      markedError.metadata = {};
    }
    markedError.metadata.sentence_index = this.cursor.sent;

    if (markedError.needs_note) {
      markedError.metadata.note = prompt(
          "Please enter a short error description", "");
      if (!markedError.metadata.note) {
        this.noteTiming('cancelled-error-note');
        return false;
      }
      this.noteTiming('added-error-note');
    }
    return true;
  }

  /**
   * Calling this marks the end of an error-marking in the current segment.
   * @param {?Object} markedError Object with error details and location,
   *     or null, if no error is to be marked.
   */
  concludeMarkingPhrase(markedError) {
    if (markedError) {
      const evalResult = this.currSegmentEval();
      if (markedError.override_all_errors) {
        evalResult.errors = [];
      }
      if (!markedError.metadata) {
        markedError.metadata = {};
      }
      markedError.metadata.timestamp = evalResult.timestamp;
      markedError.metadata.timing = evalResult.timing;
      if (!this.cursor.srcVisible(this.cursor.seg)) {
        markedError.metadata.source_not_seen = true;
      }
      evalResult.timing = {};
      evalResult.errors.push(markedError);
      this.displayError(evalResult.errors, evalResult.errors.length - 1);
    }
    this.resetMQMRating();

    this.saveResults();
    this.redrawAllSegments();
  }

  /**
   * Marks a "full-span" error that overrides all other errors and bypasses
   *     phrase selection.
   */
  markFullSpanError(severityId) {
    const severity = this.config.severities[severityId];
    const error = {
      severity: severityId,
      override_all_errors: true,
      needs_note: severity.full_span_error &&
                  severity.full_span_error.needs_note ? true : false,
      metadata: {},
    };
    if (!this.maybeAugmentError(error)) {
      this.concludeMarkingPhrase(null);
      return;
    }

    const sentence = this.getCurrSentence();
    if (sentence.hotw && !sentence.hotw.done) {
      const evalResult = this.currSegmentEval();
      this.noteTiming('found-hands-on-the-wheel-error');
      sentence.hotw.done = true;
      sentence.hotw.found = true;
      sentence.hotw.timestamp = evalResult.timestamp;
      sentence.hotw.timing = evalResult.timing;
      evalResult.timing = {};
      this.concludeMarkingPhrase(null);
      return;
    }

    this.concludeMarkingPhrase(error);
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
    this.guidance_.style.display = 'none';
    this.guidancePanel_.appendChild(this.guidance_);

    this.cancel_ = googdom.createDom(
        'button', 'anthea-stretchy-button anthea-eval-cancel', 'Cancel (Esc)');
    this.cancel_.style.display = 'none';
    const cancelListener = (e) => {
      if (e.type == 'click' ||
          (!this.cancel_.disabled && e.key && e.key === 'Escape')) {
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
            this.setMQMType(type, subtype);
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
          subtypesDiv.className =
              subtypesDiv.className + ' anthea-eval-panel-unflattened';
          subtypes.style.display = 'none';
          errorButton.addEventListener('click', (e) => {
            e.preventDefault();
            this.openSubtypes(subtypes);
          });
        }
      } else {
        errorButton.addEventListener('click', (e) => {
          e.preventDefault();
          this.setMQMType(type);
        });
      }
      errorButton.disabled = true;
      this.buttons_[this.errorButtonKey(type)] = errorButton;
    }
  }

  /**
   * Resets the current MQM rating.
   */
  resetMQMRating() {
    this.severityId_ = '';
    this.severity_ = null;
    this.errorType_ = '';
    this.errorSubtype_ = '';
    this.markedPhrase_ = null;
    this.startedMarking_ = false;
    this.evalPanelBody_.style.display = 'none';
  }

  /**
   * Called after any of the three parts of an MQM rating (error span,
   *     severity, error type) is added, this finishes the rating when
   *     all three parts have been received.
   */
  maybeSetMQMRating() {
    this.startedMarking_ = true;
    this.redrawAllSegments();
    if (!this.severity_ ||
        (!this.errorType_ && !this.severity_.forced_error) ||
        !this.markedPhrase_) {
      return;
    }
    const sentence = this.getCurrSentence();
    const errorInfo = this.config.errors[this.errorType_] ||
        this.severity_.forced_error || {};
    const display =
        errorInfo.display || this.severity_.display || 'Unspecified';
    const error = {
      location: this.cursor.side == 0 ? 'source' : 'translation',
      prefix: this.markedPhrase_.prefix,
      selected: this.markedPhrase_.selected,
      type: this.errorType_,
      subtype: this.errorSubtype_,
      display: display,
      start: this.markedPhrase_.start + sentence.firstToken,
      end: this.markedPhrase_.end + sentence.firstToken,
      severity: this.severityId_,
      override_all_errors: errorInfo.override_all_errors ? true : false,
      needs_note: errorInfo.needs_note ? true : false,
      metadata: {},
    };
    if (!this.maybeAugmentError(error)) {
      this.concludeMarkingPhrase(null);
      return;
    }
    this.concludeMarkingPhrase(error);
  }

  /**
   * Sets the severity level for the current MQM rating.
   * @param {string} severityId The severity of the error.
   */
  setMQMSeverity(severityId) {
    this.severityId_ = severityId;
    this.severity_ = this.config.severities[severityId];
    this.noteTiming('chose-severity-' + severityId);
    if (this.markedPhrase_ && this.severity_.forced_error) {
      this.setMQMType('');
      return;
    }
    this.maybeSetMQMRating();
  }

  /**
   * Sets the MQM error type and subtype for the current MQM rating.
   * @param {string} type
   * @param {string=} subtype
   */
  setMQMType(type, subtype = '') {
    this.errorType_ = type;
    this.errorSubtype_ = subtype;
    this.noteTiming('chose-error-' + type + (subtype ? '-' + subtype : ''));
    const errorInfo = this.config.errors[type];
    if (errorInfo.override_all_errors) {
      this.setMQMSeverity('major');
      return;
    }
    if (errorInfo.forced_severity) {
      this.setMQMSeverity(errorInfo.forced_severity);
      return;
    }
    this.maybeSetMQMRating();
  }

  /**
   * Sets the MQM error span for the current MQM rating.
   * @param {!Object} markedPhrase
   */
  setMQMPhrase(markedPhrase) {
    this.markedPhrase_ = markedPhrase;
    const sentence = this.getCurrSentence();
    if (sentence.hotw && !sentence.hotw.done) {
      const evalResult = this.currSegmentEval();
      this.noteTiming('found-hands-on-the-wheel-error');
      sentence.hotw.done = true;
      sentence.hotw.found = true;
      sentence.hotw.timestamp = evalResult.timestamp;
      sentence.hotw.timing = evalResult.timing;
      evalResult.timing = {};
      this.concludeMarkingPhrase(null);
      return;
    }
    this.noteTiming('ended-marking-phrase');
    if (this.severity_ && this.severity_.forced_error) {
      this.setMQMType('');
      return;
    }
    this.showGuidance('Choose error type / subtype');
    this.maybeSetMQMRating();
  }

  /**
   * Handles cancellation of the current error-marking.
   */
  handleCancel() {
    this.noteTiming('cancelled-marking-error');
    this.concludeMarkingPhrase(null);
  }

  /**
   * Handles a click on a "<severity>" button.
   * @param {string} severityId
   */
  handleSeverityClick(severityId) {
    if (this.buttons_[severityId].disabled) {
      return;
    }
    const severity = this.config.severities[severityId];
    this.noteTiming('chose-severity-' + severityId);
    if (severity.full_span_error) {
      // Just mark the full-span error directly.
      this.markFullSpanError(severityId);
    } else {
      this.setMQMSeverity(severityId);
    }
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
  revisitSentence(n, side, s) {
    if (!this.inCurrDoc(n) ||
        !this.cursor.hasBeenRead(n, side, s) ||
        this.cursor.equals(n, side, s) ||
        this.startedMarking_) {
      return;
    }
    const currSent = this.getCurrSentence();
    this.noteTiming('revisited');
    this.cursor.goto(n, side, s);
    this.redrawAllSegments();
    this.fadeTextSpan(currSent.sentSpan);
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
   * sub-phrase). See return format in injectHotw() documentation.
   * @param {string} text
   * @return {!Object}
   */
  static injectHotwWordsFlipped(text) {
    const ret = {
      corrupted: '',
      display: '',
    };
    const tokenization = AntheaEval.tokenize_with_zwsp(text);
    /**
     * Error injection is done by reversing a segment from tokens that starts
     * and ends on separators.
     */
    const tokens = tokenization.tokens;
    const seps = tokenization.separator_indices;
    if (seps.length <= 6) {
      // Too short.
      return ret;
    }
    // Start within the first half.
    const start = this.getRandomInt(seps.length / 2);
    const starti = seps[start];
    const end = Math.min(seps.length - 1, start + 4 + this.getRandomInt(4));
    const endi = seps[end];
    // Reverse
    ret.corrupted = tokens.slice(0, starti + 1).join('') +
      tokens.slice(starti + 1, endi).reverse().join('') +
      tokens.slice(endi).join('');
    ret.display = '<span class="anthea-hotw-revealed">' +
      tokens.slice(starti + 1, endi).reverse().join('') + '</span>';
    ret.hotwType = 'words-flipped';
    return ret;
  }

  /**
   * If the text has a sufficiently long word, then this function injects a
   * deliberate translation error in some part of the text by reversing a
   * long-enough sub-string in a word. See return format in injectHotw()
   * documentation.
   * @param {string} text
   * @return {!Object}
   */
  static injectHotwLettersFlipped(text) {
    const ret = {
      corrupted: '',
      display: '',
    };
    const tokenization = AntheaEval.tokenize_with_zwsp(text);
    /**
     * Error injection is done by reversing a long word.
     */
    const tokens = tokenization.tokens;
    const longTokenIndices = [];
    const MIN_LETTERS_FLIPPED = 4;
    const MIN_LETTERS = 5;
    for (let t = 0; t < tokens.length; t++) {
      if (tokens[t].length >= MIN_LETTERS) longTokenIndices.push(t);
    }

    if (longTokenIndices.length == 0) {
      return ret;
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
      return ret;
    }
    tokens[index] = tokenLetters.slice(0, startOffset).join('') + rev +
                    tokenLetters.slice(startOffset + sliceLength).join('');

    // Reverse
    ret.corrupted = tokens.join('');
    ret.display = '<span class="anthea-hotw-revealed">' + tokens[index] +
                  '</span>';
    ret.hotwType = 'letters-flipped';
    return ret;
  }

  /**
   * Pretend to inject an HOTW error, but don't actually do it. Only used
   * for training demos. See return format in injectHotw() documentation.
   * @param {string} text
   * @return {!Object}
   */
  static injectHotwPretend(text) {
    return {
      corrupted: text,
      display: '[Not a real injected error: only a training demo]',
      hotwType: 'pretend-hotw',
    };
  }

  /**
   * If possible, inject an HOTW error in the text. The returned object includes
   * a "corrupted" field that has the corrupted text (empty if no corruption
   * could be done), a "display" field suitable for revealing after the
   * corruption is undone, and an "hotwType" field.
   * @param {string} text
   * @param {boolean} hotwPretend Only pretend to insert error for training.
   * @return {!Object}
   */
  static injectHotw(text, hotwPretend) {
    if (hotwPretend) {
      return AntheaEval.injectHotwPretend(text);
    }
    /* 60% chance for words-flipped, 40% for letter-flipped */
    const tryWordsFlipped = this.getRandomInt(100) < 60;
    if (tryWordsFlipped) {
      const ret = AntheaEval.injectHotwWordsFlipped(text);
      if (ret.corrupted) {
        return ret;
      }
    }
    return AntheaEval.injectHotwLettersFlipped(text);
  }

  /**
   * Tokenizes text, splitting on space and on zero-width space. The zero-width
   * space character is not included in the returned tokens, but space is. Empty
   * strings are not emitted as tokens.
   * @param {string} text
   * @return {!Array<string>}
   */
  static tokenize(text) {
    let tokens = [];
    const textParts = text.split(' ');
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
   * Tokenizes text, splitting on space and on zero-width space. Spaces, ZWSPs,
   * and empty strings can occur in the output array. Returns an object that
   * has an array property named tokens and an array property named
   * separator_indices (that stores token indices for spaces and ZWSPs).
   * @param {string} text
   * @return {!Object}
   */
  static tokenize_with_zwsp(text) {
    const ret = {
      tokens: [],
      separator_indices: [],
    };
    let piece = '';
    for (let c of text) {
      if (c == ' ' || c == '\u200b') {
        if (piece) ret.tokens.push(piece);
        ret.separator_indices.push(ret.tokens.length);
        ret.tokens.push(c);
        piece = '';
      } else {
        piece += c;
      }
    }
    if (piece) ret.tokens.push(piece);
    return ret;
  }

  /**
   * Wraps each non-space token in text in a SPAN of class "anthea-word" and
   * each space in a SPAN of class "anthea-space".
   * @param {string} text
   * @param {boolean} appendSpace
   * @return {{numWords: number, numSpaces: number, spannified: string}}
   */
  static spannifyWords(text, appendSpace) {
    const ret = {
      numWords: 0,
      numSpaces: 0,
      spannified: '',
    };
    const tokens = AntheaEval.tokenize(text);
    const SEP = '<span class="anthea-space"> </span>';
    for (let token of tokens) {
      if (token == ' ') {
        ret.spannified += SEP;
        ret.numSpaces++;
      } else {
        ret.spannified += '<span class="anthea-word">' + token + '</span>';
        ret.numWords++;
      }
    }
    if (appendSpace) {
      ret.spannified += SEP;
      ret.numSpaces++;
    }
    return ret;
  }

  /**
   * Splits text into sentences (marked by two zero-width spaces) and tokens
   * (marked by spaces and zer-width spaces) and creates display-ready HTML
   * (including possibly adding HOTW errors). Returns an array of sentence-wise
   * objects. Each object includes the following fields:
   *   text: The raw text of the sentence.
   *   spanHTML: The HTML version of the sentence, with tokens wrapped in spans.
   *   hotwSpanHTML: Empty, or spanHTML variant with HOTW error-injected.
   *   injectedError: Empty, or the HOTW error.
   *   hotwType: Empty, or a description of the injected HOTW error.
   *   numWords: The number of word tokens in the sentence.
   *   numSPaces: The number of space tokens in the sentence.
   *   firstToken: The index of the first token (word or space).
   *   lastToken: The index of the last token (word or space).
   * In the HTML, each inter-word space is wrapped in a SPAN of class
   * "anthea-space" and each word is wrapped in a SPAN of class "anthea-word".
   * Adds a trailing space token to the last sentence.
   * 
   * @param {string} text
   * @param {number} hotwPercent
   * @param {boolean=} hotwPretend
   * @return {!Array<!Object>}
   */
  static splitAndSpannify(text, hotwPercent, hotwPretend=false) {
    const sentInfos = [];
    const sentences = text.split('\u200b\u200b');
    let tokenIndex = 0;
    for (let s = 0; s < sentences.length; s++) {
      const appendSpace = (s == sentences.length - 1);
      const sentence = sentences[s];
      const spannifyRet = AntheaEval.spannifyWords(sentence, appendSpace);
      const spanHTML = spannifyRet.spannified;
      let hotwSpanHTML = '';
      let injectedError = '';
      let hotwType = '';
      if (hotwPercent > 0 && (100 * Math.random()) < hotwPercent) {
        const ret = AntheaEval.injectHotw(sentence, hotwPretend);
        if (ret.corrupted) {
          const hotwSpannifyRet = AntheaEval.spannifyWords(
              ret.corrupted, appendSpace);
          /* Guard against any weird/unlikely change in number of tokens... */
          if (hotwSpannifyRet.numWords == spannifyRet.numWords &&
              hotwSpannifyRet.numSpaces == spannifyRet.numSpaces) {
            /* .. and commit to using this HOTW injected error. */
            hotwSpanHTML = hotwSpannifyRet.spannified;
            injectedError = ret.display;
            hotwType = ret.hotwType;
          }
        }
      }
      const sentStartToken = tokenIndex;
      tokenIndex += spannifyRet.numWords + spannifyRet.numSpaces;
      const sentInfo = {
        text: sentence,
        spanHTML: spanHTML,
        hotwSpanHTML: hotwSpanHTML,
        injectedError: injectedError,
        hotwType: hotwType,
        numWords: spannifyRet.numWords,
        numSpaces: spannifyRet.numSpaces,
        firstToken: sentStartToken,
        lastToken: tokenIndex - 1,
      };
      sentInfos.push(sentInfo);
    }
    return sentInfos;
  }

  /**
   * Computes the height of the viewport, useful for making sentences visible.
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
   * This function recomputes the tops of sentences in the current doc.
   */
  recomputeTops() {
    this.setViewportHeight();
    const start = this.docs_[this.cursor.doc].startSG;
    const num = this.docs_[this.cursor.doc].numSG;
    const docRowRect = this.docs_[this.cursor.doc].row.getBoundingClientRect();
    let maxTopPos = 0;
    for (let s = start; s < start + num; s++) {
      const segment = this.segments_[s];
      const allSents = segment.srcSents.concat(segment.tgtSents);
      for (let sent of allSents) {
        const sentRect = sent.sentSpan.getBoundingClientRect();
        sent.topPos = sentRect.top - docRowRect.top;
        sent.top = '' + sent.topPos + 'px';
        maxTopPos = Math.max(sent.topPos, maxTopPos);
      }
    }
    if (this.evalPanel_) {
      this.evalPanel_.style.top = this.getCurrSentence().top;
    }
    // Make sure the table height is sufficient.
    const docEvalCell = this.docs_[this.cursor.doc].eval;
    docEvalCell.style.height = '' + (maxTopPos + 600) + 'px';
  }

  /**
   * Returns to the previous document.
   */
  prevDocument() {
    if (!this.READ_ONLY && (this.startedMarking_ || this.cursor.doc == 0)) {
      return;
    }
    if (!this.finishCurrSentence()) {
      return;
    }
    this.docs_[this.cursor.doc].row.style.display = 'none';
    this.cursor.gotoDoc(this.cursor.doc - 1);
    this.displayedDocNum_.innerHTML = '' + (this.cursor.doc + 1);
    this.docs_[this.cursor.doc].row.style.display = '';
    const docEvalCell = this.docs_[this.cursor.doc].eval;
    docEvalCell.appendChild(this.evalPanel_);
    this.showPageContextIfPresent();
    this.redrawAllSegments();
    this.recomputeTops();
    this.refreshCurrSentence();
  }

  /**
   * Proceeds to the next document.
   */
  nextDocument() {
    if (!this.READ_ONLY &&
        (this.startedMarking_ || this.cursor.doc == this.docs_.length - 1 ||
         !this.cursor.seenDocEnd())) {
      return;
    }
    if (!this.finishCurrSentence()) {
      return;
    }
    this.docs_[this.cursor.doc].row.style.display = 'none';
    this.cursor.gotoDoc(this.cursor.doc + 1);
    this.displayedDocNum_.innerHTML = '' + (this.cursor.doc + 1);
    this.docs_[this.cursor.doc].row.style.display = '';
    const docEvalCell = this.docs_[this.cursor.doc].eval;
    docEvalCell.appendChild(this.evalPanel_);
    this.showPageContextIfPresent();
    this.redrawAllSegments();
    this.recomputeTops();
    this.refreshCurrSentence();
  }

  /**
   * Populates an instructions panel with instructions and lists of severities
   * and error types for MQM.
   * @param {!Element} panel The DIV element to populate.
   */
  populateMQMInstructions(panel) {
    panel.innerHTML = (this.config.instructions || '') +
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
          title: 'Go back to the previous sentence ' +
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
      const action = severity.action || severity.display;
      const buttonText =
          action + (severity.shortcut ? ' [' + severity.shortcut + ']' : '');
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

    this.nextButton_ = googdom.createDom(
        'button', {
          id: 'anthea-next-button',
          class: 'anthea-stretchy-button anthea-eval-panel-tall',
          title: 'Go to the next sentence ' +
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
      'span', null, '' + (this.cursor.doc + 1));
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
    this.prevDocButton_ = googdom.createDom(
      'button', {
        id: 'anthea-prev-doc-button', class: 'anthea-docnav-eval-button',
        title: 'Revisit the previous document' },
      'Prev Document');
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
      'Next Document');
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
    this.phraseMarker_ = new AntheaPhraseMarker(this);

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
   */
  adjustHeight(srcTD, tgtTD) {
    if (!srcTD || !tgtTD) {
      return;
    }
    srcTD.style.lineHeight = 1.3;
    tgtTD.style.lineHeight = 1.3;
    const srcLines = this.getApproxNumLines(srcTD, 'anthea-source-para');
    const tgtLines = this.getApproxNumLines(tgtTD, 'anthea-target-para');
    let perc = 100;
    if (srcLines < tgtLines) {
      perc = Math.floor(100 * tgtLines / srcLines);
    } else {
      perc = Math.floor(100 * srcLines / tgtLines);
    }
    if (perc >= 101) {
      const smaller = (srcLines < tgtLines) ? srcTD : tgtTD;
      smaller.style.lineHeight = 1.3 * perc / 100;
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
   *     sentences. The array also may have a "parameters" property.
   * @param {?Array<!Object>} projectResults Previously saved partial results.
   * @param {number=} hotwPercent Percent rate for HOTW testing.
   */
  setUpEval(evalDiv, templateName, projectData, projectResults, hotwPercent=0) {
    if (antheaTemplates[templateName]) {
      this.setUpEvalWithConfig(
          evalDiv, templateName, antheaTemplates[templateName], projectData,
          projectResults, hotwPercent);
      return;
    }
    /**
     * We set the template to an empty object so that we do not keep retrying
     * to load the template file.
     */
    antheaTemplates[templateName] = {};
    googdom.setInnerHtml(evalDiv, 'Loading template ' + templateName + '...');
    const scriptTag = document.createElement('script');
    scriptTag.src = this.scriptUrlPrefix_ + 'template-' +
                    templateName.toLowerCase() + '.js';
    scriptTag.onload = this.setUpEval.bind(
        this, evalDiv, templateName, projectData, projectResults, hotwPercent);
    document.head.append(scriptTag);
  }

  /**
   * Sets up the eval. This is the starting point called once the template has
   *     been loaded.
   *
   * @param {!Element} evalDiv The DIV in which to create the eval.
   * @param {string} templateName The template name.
   * @param {!Object} config The template configuration object.
   * @param {!Array<!Object>} projectData Project data, including src/tgt
   *     sentences. The array also may have a "parameters" property.
   * @param {?Array<!Object>} projectResults Previously saved partial results.
   * @param {number} hotwPercent Percent rate for HOTW testing.
   */
  setUpEvalWithConfig(
      evalDiv, templateName, config, projectData, projectResults, hotwPercent) {
    this.config = config;
    evalDiv.innerHTML = '';

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
    const srcLang = parameters.source_language || '';
    const tgtLang = parameters.target_language || '';
    if (parameters.hasOwnProperty('hotw_percent')) {
      /* Override the passed value */
      hotwPercent = parameters.hotw_percent;
    }
    /** Are we only pretending to add hotw errors, for training? */
    const hotwPretend = parameters.hotw_pretend || false;

    const srcHeading = srcLang ? ('Source (' + srcLang + ')') : 'Source';
    const srcHeadingTD = googdom.createDom(
        'td', 'anthea-text-heading',
        googdom.createDom('div', null, srcHeading));
    const targetLabel = config.TARGET_SIDE_ONLY ? 'Text' : 'Translation';
    const tgtHeading = tgtLang ?
        (targetLabel + ' (' + tgtLang + ')') : targetLabel;
    const tgtHeadingTD = googdom.createDom(
        'td', 'anthea-text-heading',
        googdom.createDom('div', null, tgtHeading));
    const evalHeading = this.READ_ONLY ?
        'Evaluations (view-only)' : 'Evaluations';
    const evalHeadingTD = googdom.createDom(
        'td', 'anthea-text-heading',
        googdom.createDom('div', null, evalHeading));
    const docTextTable = googdom.createDom(
        'table', 'anthea-document-text-table',
        googdom.createDom(
            'tr', null, srcHeadingTD, tgtHeadingTD, evalHeadingTD),
        this.contextRow_);
    if (config.TARGET_SIDE_ONLY) {
      srcHeadingTD.style.display = 'none';
    }
    evalDiv.appendChild(docTextTable);

    for (let docsys of projectData) {
      const doc = {
        'docsys': docsys,
      };
      this.docs_.push(doc);
      doc.eval = googdom.createDom('div', 'anthea-document-eval-div');

      const docTextSrcRow = googdom.createDom('td',
                                              'anthea-document-text-cell');
      const docTextTgtRow = googdom.createDom('td',
                                              'anthea-document-text-cell');
      doc.row = googdom.createDom(
          'tr', null, docTextSrcRow, docTextTgtRow,
          googdom.createDom('td', 'anthea-document-eval-cell', doc.eval));
      if (this.docs_.length > 1) {
        doc.row.style.display = 'none';
      }
      if (config.TARGET_SIDE_ONLY) {
        docTextSrcRow.style.display = 'none';
      }
      doc.startSG = this.evalResults_.length;
      doc.numSG = 0;

      docTextTable.appendChild(doc.row);

      const srcSegments = docsys.srcSegments;
      const tgtSegments = docsys.tgtSegments;
      let srcSpannified = '<p class="anthea-source-para" dir="auto">';
      let tgtSpannified = '<p class="anthea-target-para" dir="auto">';
      for (let i = 0; i < srcSegments.length; i++) {
        if (srcSegments[i].length == 0) {
          /* New paragraph. */
          srcSpannified += '</p><p class="anthea-source-para" dir="auto">';
          tgtSpannified += '</p><p class="anthea-target-para" dir="auto">';
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
        this.evalResults_.push(evalResult);

        const segment = {
          doc: this.docs_.length - 1,
          srcText: srcSegments[i],
          tgtText: tgtSegments[i],
          numTgtWords: 0,
          srcSents: AntheaEval.splitAndSpannify(srcSegments[i], 0),
          tgtSents: AntheaEval.splitAndSpannify(
              tgtSegments[i], this.READ_ONLY ? 0 : hotwPercent, hotwPretend),
        };
        this.segments_.push(segment);

        srcSpannified += '<span class="anthea-source-segment">';
        for (let srcSent of segment.srcSents) {
          srcSpannified += '<span class="anthea-source-sentence">' +
                           srcSent.spanHTML + '</span>';
        }
        srcSpannified += '</span>';

        tgtSpannified += '<span class="anthea-target-segment">';
        for (let t = 0; t < segment.tgtSents.length; t++) {
          const tgtSent = segment.tgtSents[t];
          tgtSpannified += '<span class="anthea-target-sentence">' +
                           (tgtSent.hotwSpanHTML || tgtSent.spanHTML) +
                           '</span>';
          segment.numTgtWords += tgtSent.numWords;
          if (tgtSent.injectedError) {
            tgtSent.hotw = {
              'timestamp': this.lastTimestampMS_,
              'injected_error': tgtSent.injectedError,
              'hotw_type': tgtSent.hotwType,
              'sentence_index': t,
              'done': false,
              'found': false,
            };
            evalResult['hotw_list'].push(tgtSent.hotw);
          }
        }
        this.numTgtWordsTotal_ += segment.numTgtWords;
        tgtSpannified += '</span>';

        doc.numSG++;
      }
      googdom.setInnerHtml(docTextSrcRow, srcSpannified + '</p>');
      googdom.setInnerHtml(docTextTgtRow, tgtSpannified + '</p>');
      this.adjustHeight(docTextSrcRow, docTextTgtRow);
    }

    /* Grab segment and sentence span elements */
    const srcSegmentSpans = document.getElementsByClassName(
      'anthea-source-segment');
    const tgtSegmentSpans = document.getElementsByClassName(
      'anthea-target-segment');
    console.assert(srcSegmentSpans.length == this.segments_.length);
    console.assert(tgtSegmentSpans.length == this.segments_.length);
    for (let i = 0; i < this.segments_.length; i++) {
      const segment = this.segments_[i];
      segment.srcSegmentSpan = srcSegmentSpans[i];
      segment.tgtSegmentSpan = tgtSegmentSpans[i];
      const srcSentSpans = segment.srcSegmentSpan.getElementsByClassName(
        'anthea-source-sentence');
      const tgtSentSpans = segment.tgtSegmentSpan.getElementsByClassName(
        'anthea-target-sentence');
      console.assert(srcSentSpans.length == segment.srcSents.length);
      console.assert(tgtSentSpans.length == segment.tgtSents.length);
      for (let s = 0; s < segment.srcSents.length; s++) {
        segment.srcSents[s].sentSpan = srcSentSpans[s];
      } 
      for (let t = 0; t < segment.tgtSents.length; t++) {
        segment.tgtSents[t].sentSpan = tgtSentSpans[t];
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

    this.cursor = new AntheaCursor(this.segments_,
                                   config.TARGET_SIDE_ONLY || false,
                                   config.TARGET_SIDE_FIRST || false,
                                   this.updateProgressForSegment.bind(this));

    this.createUI(instructionsPanel, controlPanel);

    this.restoreResults(projectResults);
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
    this.showPageContextIfPresent();

    this.redrawAllSegments();
    this.recomputeTops();
    this.resizeListener_ = () => { this.recomputeTops(); };
    window.addEventListener('resize', this.resizeListener_);
  }
}

/**
 * The AntheaPhraseMarker class is used to collect highlighted phrases for the
 *     current sentence.
 * @final
 */
class AntheaPhraseMarker {
  /**
   * @param {!AntheaEval} contextedEval
   */
  constructor(contextedEval) {
    /** @private @const {!AntheaEval} */
    this.contextedEval_ = contextedEval;

    /** @private @const {string} */
    this.DEFAULT_COLOR = 'gainsboro';
    /** @private {string} */
    this.color_ = this.DEFAULT_COLOR;

    /** @private {number} */
    this.startSpanIndex_ = -1;
    /** @private {number} */
    this.endSpanIndex_ = -1;

    /** @private {!Array<!Element>} Token span elements */
    this.tokenSpans_ = [];
    /** @private {!Array<string>} Saved colors of token spans during marking */
    this.tokenSpanColors_ = [];
  }

  /**
   * Resets the word spans in the current sentence, getting rid of any
   *     event listeners from spannification done in the previous state. Sets
   *     element class to 'anthea-word-active' or
   *     'anthea-space-active'.
   */
  resetWordSpans() {
    const ce = this.contextedEval_;
    ce.redrawCurrSentence();

    this.tokenSpans_ = ce.getCurrTokenSpans();

    const allowSpaceStart = ce.config.ALLOW_SPANS_STARTING_ON_SPACE || false;

    this.tokenSpanColors_ = [];
    const spanClassSuffix = (this.startSpanIndex_ < 0) ? '-begin' : '';
    const suffix = '-active' + spanClassSuffix;
    const spaceClass = 'anthea-space' + suffix;
    const wordClass = 'anthea-word' + suffix;
    for (let x = 0; x < this.tokenSpans_.length; x++) {
      this.tokenSpans_[x].className =
          this.tokenSpans_[x].className + suffix;
      if (allowSpaceStart && this.tokenSpans_[x].className == spaceClass) {
        this.tokenSpans_[x].className = wordClass;
      }
      this.tokenSpanColors_.push(this.tokenSpans_[x].style.backgroundColor);
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

    const markedPhrase = {
      start: this.startSpanIndex_,
      end: this.endSpanIndex_,
      prefix: prefix,
      selected: selected,
      color: this.color_,
    };
    this.contextedEval_.setMQMPhrase(markedPhrase);
  }

  /**
   * Notes that startSpanIndex for the highlighted phrase and sets up the UI for
   *     picking the end of the phrase.
   * @param {number} spanIndex
   */
  prepareToPickEnd(spanIndex) {
    const ce = this.contextedEval_;
    ce.setStartedMarking();
    ce.showGuidance('Click on the end of the error span');

    this.startSpanIndex_ = spanIndex;

    /* Remove anthea-word listeners, add new ones. */
    this.resetWordSpans();

    const span = this.tokenSpans_[spanIndex];
    span.style.backgroundColor = this.color_;

    for (let x = 0; x < this.tokenSpans_.length; x++) {
      this.tokenSpans_[x].addEventListener(
        'mouseover', () => { this.highlightTo(x); });
      this.tokenSpans_[x].addEventListener('click', () => { this.pickEnd(x); });
    }
    ce.noteTiming('began-marking-phrase');
  }

  /**
   * Sets state and UI to wait for the start of the highlighted phrase to get
   *     picked.
   */
  prepareToPickStart() {
    const ce = this.contextedEval_;
    this.startSpanIndex_ = -1;
    this.endSpanIndex_ = -1;
    this.resetWordSpans();

    ce.showGuidance('Click on the start of an error span not yet marked');
    const cls = 'anthea-word-active-begin';
    for (let x = 0; x < this.tokenSpans_.length; x++) {
      if (this.tokenSpans_[x].className == cls) {
        this.tokenSpans_[x].addEventListener(
          'click',
          () => { this.prepareToPickEnd(x); });
      }
    }
  }

  /**
   * The public entrypoint in the AntheaPhraseMarker object. Sets up the UI to
   *     collect a highlighted phrase from the current sentence.
   *     When phrase-marking is done, the contextedEval_ object's
   *     setMQMPhrase() function will get called.
   * @param {string} color The color to use for highlighting.
   */
  getMarkedPhrase(color) {
    this.color_ = color || this.DEFAULT_COLOR;
    this.prepareToPickStart();
  }
}
