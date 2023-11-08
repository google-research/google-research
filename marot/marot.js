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
 * This file contains the JavaScript code for Marot, an interactive tool for
 * visualizing and analyzing translation quality metrics.
 *
 * It creates a global object of type Marot, named "marot." From the
 * application, marot.init() should be called to finish setting up the tool.
 */

/**
 * MQM scoring weights. Each weight has a name and a regular expression pattern
 * for matching <severity>:<category>[/<subcategory>] (case-insensitively).
 * The weights are tried out in the sequence shown and for a given annotation,
 * the first matching weight is used. While you can interactively change these
 * for experimentation, you should set this default array to values suitable
 * for your application. The best place to do this is in your own version of
 * marot.html.
 *
 * The "name" fields should be unique, short (<= 10 characters), and composed
 * only of [a-zA-Z-] (no periods please).
 */
let mqmDefaultWeights = [
  {
    'name': 'Trivial',
    'weight': 0.1,
    'pattern': 'minor:.*punctuation|trivial:',
  },
  {
    'name': 'Creative',
    'weight': 0,
    'pattern': ':.*reinterpretation',
  },
  {
    'name': 'Source',
    'weight': 0,
    'pattern': ':source',
  },
  {
    'name': 'Non-trans',
    'weight': 25,
    'pattern': 'non.translation',
  },
  {
    'name': 'Minor',
    'weight': 1,
    'pattern': 'minor:',
  },
  {
    'name': 'Major',
    'weight': 5,
    'pattern': 'major:',
  },
  {
    'name': 'Critical',
    'weight': 5,
    'pattern': 'critical:',
  },
];

/**
 * MQM Scores can be sliced along a second dimension, which is typically
 * Accuracy/Fluency, but can be customized in any desired manner. The
 * slicing is done by matching the pattern regular expressions
 * case-insensitively in order to find the first matching slice. Similarly
 * as with mqmDefaultWeights, you may want to override these defaults in
 * your own application.
 *
 * See comment above mqmDefaultWeights for requirements on the "name" field.
 */
let mqmDefaultSlices = [
  {
    'name': 'Accuracy',
    'pattern': 'accuracy|terminology|non.translation',
  },
  {
    'name': 'Fluency',
    'pattern': 'fluency|style|locale',
  },
  {
    'name': 'Other',
    'pattern': '.*',
  },
];

/**
 * An object that captures all the data needed for running pairwise significance
 * tests on one particular metric over systems/raters.
 */
class MarotSigtestsMetricData {
  constructor() {
    /** {boolean} */
    this.lowerBetter = false;
    /**
     * {!Array<string>} Sorted array ordered by degrading scores.
     */
    this.comparables = [];

     /**
      * {!Object} Scores by system/rater. Each score itself is an object
      *     containing score and numSegments.
      */
    this.scores = {};
    /**
     * {!Object} Segment scores by system/rater. Each value is an array of
     *     scores that are aligned such that elements at the n-th position of
     *     all arrays correspond to the same segment. Note that some scores
     *     might be null since some systems/raters might be missing scores for
     *     some segments.
     */
    this.segScores = {};
    /**
     * {!Object} Common segments shared by a pair of systems/raters. This stores
     *     positions in segScores.
     */
    this.commonPosByItemPair = {};
    /**
     * {!Array<!Array<number>>} Computed matrix of p-values.
     */
    this.pValues = [];
  }
}

/**
 * The main class, encapsulating all the data and interactive UI.
 */
class Marot {
  constructor() {
    /**
     * Raw data read from TSV data files/URLs. Each entry is an array with 10
     * entries, in this order (slightly different from the original order in the
     * TSV data, as we keep the fields in their more natural presentation order,
     * as used in the HTML table we display):
     *
     *   0: system, 1: doc, 2: docSegId, 3: globalSegId, 4: source, 5: target,
     *   6: rater, 7: category, 8: severity, 9: metadata
     *
     * The docSegId field is the 1-based index of the segment within the doc.
     *
     * The globalSegId field is an arbitrary, application-specific segment
     * identifier. If such an identifier is not needed or available, then set
     * this field to some constant value, such as 0. It is ignored by Marot, but
     * is available for use in filter expressions.
     *
     * The last field, "metadata", is an object that includes the timestamp of
     * the rating, any note the rater may have left, and other metadata.
     *
     * There is a special severity, "HOTW-test", reserved for hands-on-the-wheel
     * test results. These are test sentences in which a deliberate error is
     * injected, just to help the rater stay on task. The test result is
     * captured by category = "Found" or "Missed" and there is a note in the
     * metadata that captures the injected error.
     */
    this.data = [];

    /**
     * A data structure that provides a convenient way to iterate over this.data
     * in nested loops on doc, docSegId, system.
     */
    this.dataIter = {
      docs: [],
      docSegs: {},
      docSys: {},
      docSegSys: {},
      systems: null,  /** Set of all systems in the data. */
      /**
       * Set of all raters in the data, including prior-raters (recorded here
       * with a "prior_rater:" prefix), and faux, "all-raters-other-than"
       * raters (recorded with an "not:" prefix).
       */
      raters: null,
    };

    /**
     * Each call to show() figures out the rowIds that pass current filters and
     * populates selectedRows with that set.
     */
    this.selectedRows = new Set();

    /**
     * If there are "viewing constraints" currently in place (because of a
     * drill-down click on a histogram bar), then the selected histogram rect
     * is recorded here (so that its styling can be cleared when the constraint)
     * is removed.
     */
    this.histRectSelected = null;

    /** Array indices in each data[] entry */
    this.DATA_COL_SYSTEM = 0;
    this.DATA_COL_DOC = 1;
    this.DATA_COL_DOC_SEG_ID = 2;
    this.DATA_COL_GLOBAL_SEG_ID = 3;
    this.DATA_COL_SOURCE = 4;
    this.DATA_COL_TARGET = 5;
    this.DATA_COL_RATER = 6;
    this.DATA_COL_CATEGORY = 7;
    this.DATA_COL_SEVERITY = 8;
    this.DATA_COL_METADATA = 9;
    this.DATA_COL_NUM_PARTS = 10;

    /** Column filter id mappings */
    this.filterColumns = {
      'marot-filter-doc': this.DATA_COL_DOC,
      'marot-filter-doc-seg': this.DATA_COL_DOC_SEG_ID,
      'marot-filter-system': this.DATA_COL_SYSTEM,
      'marot-filter-source': this.DATA_COL_SOURCE,
      'marot-filter-target': this.DATA_COL_TARGET,
      'marot-filter-rater': this.DATA_COL_RATER,
      'marot-filter-category': this.DATA_COL_CATEGORY,
      'marot-filter-severity': this.DATA_COL_SEVERITY,
    };

    /**
     * If TSV data was supplied (instead of being chosen from a file), then it
     * is saved here (for possible downloading).
     */
    this.tsvData = '';

    /**
     * The following are all arrays of per-rater stats (scores, score slices,
     * error counts) for one segment.
     *
     *   statsBySystem[system][doc][docSegId]
     *   statsByRaterSystem[rater][system][doc][docSegId]
     *   statsByRater[rater][doc][docSegId]
     *      stats has a special system value ('MAROT_TOTAL') that aggregates
     *      ratings for all systems, by using faux "doc" keys that look like
     *      "[doc]:[system]". The "doc" keys in statsByRater also look like
     *      "[doc]:[system]".
     *
     * sevcatStats[severity][category][system] is the total count of
     * annotations of a specific severity+category in a specific system.
     *
     * Each of these stats objects is recomputed for any filtering applied to
     * the data.
     */
    this.statsBySystem = {};
    this.statsByRaterSystem = {};
    this.statsByRater = {};
    this.sevcatStats = {};

    /** {!Element} HTML table body elements for various tables */
    this.segmentsTable = null;
    this.statsTable = null;
    this.sevcatStatsTable = null;
    this.eventsTable = null;

    /** Events timing info for current filtered data. **/
    this.events = {};

    /**
     * Max number of annotations to show in the sample of ratings shown. Note
     * that this is not a hard limit, as we include all systems + raters for any
     * document segment that pass the current filter (if any).
     */
    this.rowLimit = 200;

    /** Clause built by helper menus, for appending to the filter expression **/
    this.filterClause = '';

    /** UI elements for clause-builder. */
    this.filterClauseKey = null;
    this.filterClauseInclExcl = null;
    this.filterClauseSev = null;
    this.filterClauseCat = null;
    this.filterClauseAddAnd = null;
    this.filterClauseAddOr = null;

    /** Selected system and rater names for comparisons. */
    this.comparisons = {
      sys1: null,
      sys2: null,
      rater1: null,
      rater2: null,
    };

    /** A distinctive name used as the key for aggregate stats. */
    this.TOTAL = '_MAROT_TOTAL_';

    this.PVALUE_THRESHOLD = 0.05;
    this.SIGTEST_TRIALS = 10000;

    /**
     * An object with data for computing significance tests. This data is sent
     * to a background Worker thread. See computation details in
     * marot-sigtests.js. The data object potentially has keys 'sys' and 'rater'
     * and the values are objects keyed by metric id, with those values being
     * objects of type MarotSigtestsMetricData.
     */
    this.sigtestsData = {
      data: {},
      /** {number} Number of trials. */
      numTrials: this.SIGTEST_TRIALS,
    };

    /** {!Worker} A background Worker thread that computes sigtests */
    this.sigtestsWorker = null;

    /**
     * The Sigtests Worker loads its code from 'marot-sigtests.js'. If that file
     * is not servable for some reason, then set marot.sigtestsWorkerJS
     * variable to its contents before calling marot.init().
     */
    this.sigtestsWorkerJS = '';
    /**
     * {!Element} HTML spans that shows sigtest computation status messages.
     */
    this.sigtestsSysMsg = null;
    this.sigtestsRaterMsg = null;

    /**
     * mqmWeights and mqmSlices are set from current settings in
     * parseScoreSettings() and resetSettings().
     */
    this.mqmWeights = [];
    this.mqmSlices = [];

    /**
     * Score aggregates include 'mqm-weighted-" and "mqm-slice-" prefixed
     * scores. The names beyond the prefixes are taken from the "name" field in
     * this.mqmWeights and this.mqmSlices.
     */
    this.MQM_WEIGHTED_PREFIX = 'mqm-weighted-';
    this.MQM_SLICE_PREFIX = 'mqm-slice-';

    /**
     * Arrays of names of currently being displayed MQM score components, sorted
     * in decreasing score order.
     */
    this.mqmWeightedFields = [];
    this.mqmSliceFields = [];

    /**
     * The field to sort the score table rows by. By default, sort by
     * overall MQM score. `sortReverse` indicates whether it is sorted in
     * ascending order (false, default) or descending order (true).
     *
     * The value of this is something like 'metric-<k>' (where k is an index
     * into this.metrics[]), or a name from this.mqmWeightedFields[] or
     * this.mqmSliceFields[].
     */
    this.sortByField = 'metric-0';
    this.sortReverse = false;

    /**
     * All metrics possibly available in the current data. The entries will be
     * like 'MQM', 'BLEURT-X', etc. 'MQM' is the always the first entry in this
     * array.
     * {!Array<string>}
     */
    this.metrics = ['MQM'];
    /**
     * Info about metrics.
     */
    this.metricsInfo = {
      'MQM': {
        index: 0,  /** index into this.metrics[] */
        lowerBetter: true,  /** default is false */
      },
    };
    /**
     * The metrics that are available for the data with the current filtering.
     * {!Array<number>} Indices into this.metrics.
     */
    this.metricsVisible = [];

    /**
     * Max number of rows to show in a rater's event timeline.
     */
    this.RATER_TIMELINE_LIMIT = 200;

    /**
     * Prefix attached to a rater id to identify the complement set of raters.
     */
    this.NOT_PREFIX = 'not:';

    /**
     * Prefix attached to a rater id to mark it as coming from a prior rater.
     */
    this.PRIOR_RATER_PREFIX = 'prior:';

    /**
     * Maximum number of lines of data that we'll consume. Human eval data is
     * generally of modest size, but automated metrics data can be arbitrary
     * large. Users should limit and curate such data.
     * 1000 docs * 100 lines * 10 systems * 10 raters = 10,000,000
     */
    this.MAX_DATA_LINES = 10000000;
  }

  /**
   * Listener for changes to the input field that specifies the limit on
   * the number of rows shown.
   */
  setShownRowsLimit() {
    const limitElt = document.getElementById('marot-limit');
    const limit = limitElt.value.trim();
    if (limit > 0) {
      this.rowLimit = limit;
      this.show();
    } else {
      limitElt.value = this.rowLimit;
    }
  }

  /**
   * This function returns a "comparable" version of docSegId by padding it
   * with leading zeros. When docSegId is a non-negative integer (reasonably
   * bounded), then this ensures numeric ordering.
   * @param {string} s
   * @return {string}
   */
  cmpDocSegId(s) {
    return ('' + s).padStart(10, '0');
  }

  /**
   * This sorts 10-column Marot data by fields in the order doc, docSegId,
   *   system, rater, severity, category.
   * @param {!Array<!Array>} data The Marot-10-column data to be sorted.
   */
  sortData(data) {
    data.sort((e1, e2) => {
      let diff = 0;
      const docSegId1 = this.cmpDocSegId(e1[this.DATA_COL_DOC_SEG_ID]);
      const docSegId2 = this.cmpDocSegId(e2[this.DATA_COL_DOC_SEG_ID]);
      if (e1[this.DATA_COL_DOC] < e2[this.DATA_COL_DOC]) {
        diff = -1;
      } else if (e1[this.DATA_COL_DOC] > e2[this.DATA_COL_DOC]) {
        diff = 1;
      } else if (docSegId1 < docSegId2) {
        diff = -1;
      } else if (docSegId1 > docSegId2) {
        diff = 1;
      } else if (e1[this.DATA_COL_SYSTEM] < e2[this.DATA_COL_SYSTEM]) {
        diff = -1;
      } else if (e1[this.DATA_COL_SYSTEM] > e2[this.DATA_COL_SYSTEM]) {
        diff = 1;
      } else if (e1[this.DATA_COL_RATER] < e2[this.DATA_COL_RATER]) {
        diff = -1;
      } else if (e1[this.DATA_COL_RATER] > e2[this.DATA_COL_RATER]) {
        diff = 1;
      } else if (e1[this.DATA_COL_SEVERITY] < e2[this.DATA_COL_SEVERITY]) {
        diff = -1;
      } else if (e1[this.DATA_COL_SEVERITY] > e2[this.DATA_COL_SEVERITY]) {
        diff = 1;
      } else if (e1[this.DATA_COL_CATEGORY] < e2[this.DATA_COL_CATEGORY]) {
        diff = -1;
      } else if (e1[this.DATA_COL_CATEGORY] > e2[this.DATA_COL_CATEGORY]) {
        diff = 1;
      }
      return diff;
    });
  }

  /**
   * Sets this.dataIter to a data structure that can be used to iterate over
   * this.data[] rows by looping over documents, segments, and systems.
   */
  createDataIter() {
    this.dataIter = {
      docs: [],
      docSegs: {},
      docSys: {},
      docSegSys: {},
      systems: new Set,
      raters: new Set,
      evaluation: {},
    };
    let lastRow = null;
    for (let rowId = 0; rowId < this.data.length; rowId++) {
      const parts = this.data[rowId];
      const doc = parts[this.DATA_COL_DOC];
      const docSegId = parts[this.DATA_COL_DOC_SEG_ID];
      const system = parts[this.DATA_COL_SYSTEM];
      this.dataIter.systems.add(system);
      const rater = parts[this.DATA_COL_RATER];
      if (rater) {
        this.dataIter.raters.add(rater);
        this.dataIter.raters.add(this.NOT_PREFIX + rater);
      }
      const metadata = parts[this.DATA_COL_METADATA];
      const priorRater = metadata.prior_rater ?? '';
      if (priorRater) {
        this.dataIter.raters.add(this.PRIOR_RATER_PREFIX + priorRater);
      }

      const sameDoc = lastRow && (doc == lastRow[this.DATA_COL_DOC]);
      const sameDocSeg = sameDoc &&
                         (docSegId == lastRow[this.DATA_COL_DOC_SEG_ID]);
      const sameDocSys = sameDoc && (system == lastRow[this.DATA_COL_SYSTEM]);
      if (!sameDoc) {
        this.dataIter.docs.push(doc);
        this.dataIter.docSegs[doc] = [];
        this.dataIter.docSys[doc] = [];
      }
      if (!sameDocSeg) {
        console.assert(!this.dataIter.docSegs[doc].includes(docSegId),
                       doc, docSegId);
        this.dataIter.docSegs[doc].push(docSegId);
      }
      if (!sameDocSys && !this.dataIter.docSys[doc].includes(system)) {
        this.dataIter.docSys[doc].push(system);
      }
      lastRow = parts;
    }
    /**
     * Ensure that there are entries in docSegSys for each
     * docSegId x system.
     */
    for (const doc of this.dataIter.docs) {
      this.dataIter.docSegSys[doc] = {};
      for (const docSegId of this.dataIter.docSegs[doc]) {
        this.dataIter.docSegSys[doc][docSegId] = {};
        for (const system of this.dataIter.systems) {
          this.dataIter.docSegSys[doc][docSegId][system] = {
            rows: [-1, -1],
            segment: {},
          };
        }
      }
    }
    lastRow = null;
    let segment = null;
    for (let rowId = 0; rowId < this.data.length; rowId++) {
      const parts = this.data[rowId];
      const doc = parts[this.DATA_COL_DOC];
      const docSegId = parts[this.DATA_COL_DOC_SEG_ID];
      const system = parts[this.DATA_COL_SYSTEM];
      const metadata = parts[this.DATA_COL_METADATA];
      if (metadata.evaluation) {
        this.dataIter.evaluation = {
          ...this.dataIter.evaluation,
          ...metadata.evaluation
        };
      }
      const sameDoc = lastRow && (doc == lastRow[this.DATA_COL_DOC]);
      const sameDocSeg = sameDoc &&
                         (docSegId == lastRow[this.DATA_COL_DOC_SEG_ID]);
      const sameDocSegSys = sameDocSeg &&
                            (system == lastRow[this.DATA_COL_SYSTEM]);

      if (!sameDocSegSys) {
        this.dataIter.docSegSys[doc][docSegId][system].rows =
            [rowId, rowId + 1];
        segment = metadata.segment || {};
      } else {
        this.dataIter.docSegSys[doc][docSegId][system].rows[1] = rowId + 1;
      }
      this.dataIter.docSegSys[doc][docSegId][system].segment = segment;
      lastRow = parts;
    }
  }

  /**
   * If obj does not have an array property named key, creates an empty array.
   * Pushes val into the obj[key] array.
   * @param {!Object} obj
   * @param {string} key
   * @param {string} val
   */
  addToArray(obj, key, val) {
    if (!obj.hasOwnProperty(key)) obj[key] = [];
    obj[key].push(val);
  }


  /**
   * Returns the location of elt in sorted array arr using binary search. if
   * elt is not present in arr, then returns the slot where it belongs in sorted
   * order.
   * @param {!Array<number>} arr Sorted array of numbers.
   * @param {number} elt
   * @return {number}
   */
  binSearch(arr, elt) {
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
    return l;
  }

  /**
   * Given an array of all instances of annotated text for a segment (where
   * annotations have been marked using <v>..</v> spans), generates a
   * tokenization that starts with space-based splitting, but refines it to
   * ensure that each <v> and </v> is at a token boundary. Returns the
   * tokenization as well as an array containing the marked spans encoded as
   * [start, end] token indices (both inclusive).
   *
   * The structure of the returned object is: {
   *   tokens: !Array<string>,
   *   spans: !Array<Pair<number, number>>
   * }
   * @param {!Array<string>} annotations
   * @return {!Object}
   */
  tokenizeLegacyText(annotations) {
    let cleanText = '';
    for (const text of annotations) {
      const noMarkers = text.replace(/<\/?v>/g, '');
      if (noMarkers.length > cleanText.length) {
        cleanText = noMarkers;
      }
    }
    const spacedTokens = cleanText.split(' ');
    const tokens = [];
    for (let i = 0; i < spacedTokens.length; i++) {
      tokens.push(spacedTokens[i]);
      tokens.push(' ');
    }
    const tokenOffsets = [];
    let tokenOffset = 0;
    for (const token of tokens) {
      tokenOffsets.push(tokenOffset);
      tokenOffset += token.length;
    }

    const MARKERS = ['<v>', '</v>'];
    const markerOffsets = [];
    for (const text of annotations) {
      const offsets = [];
      let markerIdx = 0;
      let modText = text;
      let x;
      while ((x = modText.indexOf(MARKERS[markerIdx])) >= 0) {
        const marker = MARKERS[markerIdx];
        offsets.push(x);
        modText = modText.substr(0, x) + modText.substr(x + marker.length);
        markerIdx = 1 - markerIdx;

        const loc = this.binSearch(tokenOffsets, x);
        if (tokenOffsets.length > loc && tokenOffsets[loc] == x) {
          continue;
        }
        /**
         * The current marker (<v> or </v>) lies inside a token. Split that
         * token.
         */
        const toSplit = loc - 1;
        if (toSplit < 0) {
          console.log('Weird splitting situation for offset: ' + x +
                      ' in [' + modText + ']');
          continue;
        }
        console.assert(toSplit < tokenOffsets.length);
        console.assert(tokenOffsets[toSplit] < x);
        const oldToken = tokens[toSplit];
        console.assert(tokenOffsets[toSplit] + oldToken.length > x);
        const newLen = x - tokenOffsets[toSplit];
        tokens[toSplit] = oldToken.substr(0, newLen);
        tokens.splice(loc, 0, oldToken.substr(newLen));
        tokenOffsets.splice(loc, 0, x);
      }
      markerOffsets.push(offsets);
    }
    const spansList = [];
    for (const offsets of markerOffsets) {
      const spans = [];
      for (let i = 0; i < offsets.length; i+= 2) {
        if (i + 1 >= offsets.length) break;
        spans.push([this.binSearch(tokenOffsets, offsets[i]),
                    this.binSearch(tokenOffsets, offsets[i + 1]) - 1]);
      }
      spansList.push(spans);
    }
    return {
      tokens: tokens,
      spans: spansList,
    };
  }

  /**
   * Given the full range of rows for the same doc+docSegId+system, tokenizes
   * the source and target side using spaces, but refining the tokenization to
   * make each <v> and </v> fall on a token boundary. Sets
   * segment.{source,target}_tokens as well as
   *     this.data[row][this.DATA_COL_METADATA].{source,target}_spans.
   *
   * If segment.source/target_tokens is already present in the data (as
   * will be the case with newer data), this function is a no-op.
   * If rowRange does not cover any rows, then this function is a no-op.
   * @param {!Array<number>} rowRange The start (inclusive) and limit
   * (exclusive)
   *     rowId for the segment, in this.data[].
   * @param {!Object} segment The segment-level aggregate data.
   */
  tokenizeLegacySegment(rowRange, segment) {
    if (rowRange[0] >= rowRange[1]) {
      return;
    }
    const sources = [];
    const targets = [];
    for (let row = rowRange[0]; row < rowRange[1]; row++) {
      const parts = this.data[row];
      sources.push(parts[this.DATA_COL_SOURCE]);
      targets.push(parts[this.DATA_COL_TARGET]);
    }
    const sourceTokenization = this.tokenizeLegacyText(sources);
    segment.source_tokens = sourceTokenization.tokens;
    const targetTokenization = this.tokenizeLegacyText(targets);
    segment.target_tokens = targetTokenization.tokens;
    for (let row = rowRange[0]; row < rowRange[1]; row++) {
      const parts = this.data[row];
      const idx = row - rowRange[0];
      parts[this.DATA_COL_METADATA].source_spans =
          sourceTokenization.spans[idx];
      parts[this.DATA_COL_METADATA].target_spans =
          targetTokenization.spans[idx];
    }
  }

  /**
   * When the input data has the older "segment.references" data (rather than
   * the newer "segment.reference_tokens" and
   * "segment.reference_sentence_splits" data), convert it to the newer format.
   * @param {!Object} segment
   */
  tokenizeLegacyReferences(segment) {
    if (!segment.references || segment.reference_tokens) {
      return;
    }
    segment.reference_tokens = {};
    segment.reference_sentence_splits = {};
    for (const refKey in segment.references) {
      const text = segment.references[refKey];
      const tokens = [];
      const splits = [];
      const paragraphs = text.split('\n\n');
      for (let p = 0; p < paragraphs.length; p++) {
        const para = paragraphs[p];
        const lines = para.split('\n');
        for (let l = 0; l < lines.length; l++) {
          const line = lines[l];
          const spacedTokens = line.split(' ');
          let lineTokens = 0;
          for (const t of spacedTokens) {
            tokens.push(t);
            tokens.push(' ');
            lineTokens += 2;
          }
          const splitInfo = {
            num_tokens: lineTokens
          };
          if (l < lines.length - 1) {
            splitInfo.ends_with_line_break = true;
          } else if (p < paragraphs.length - 1) {
            splitInfo.ends_with_para_break = true;
          }
          splits.push(splitInfo);
        }
      }
      segment.reference_tokens[refKey] = tokens;
      segment.reference_sentence_splits[refKey] = splits;
    }
  }

  /**
   * Aggregates this.data, collecting all data for a particular segment
   * translation (i.e., for a given (doc, docSegId) pair) into the aggrDocSeg
   * object in the metadata.segment field, adding to it the following
   * properties:
   *         {cats,sevs,sevcats}By{Rater,System}.
   *     Each of these properties is an object keyed by system or rater, with
   *     the values being arrays of strings that are categories, severities,
   *     and <sev>[/<cat>], * respectively.
   *
   *     Also added are aggrDocSeg.metrics[metric][system] values.
   * Makes sure that the metadata.segment object is common for each row from
   * the same doc+seg+sys.
   */
  addSegmentAggregations() {
    for (const doc of this.dataIter.docs) {
      const aggrDoc = {
        doc: doc,
        thumbsUpCount: 0,
        thumbsDownCount: 0,
      };
      for (const docSegId of this.dataIter.docSegs[doc]) {
        const aggrDocSeg = {
          catsBySystem: {},
          catsByRater: {},
          sevsBySystem: {},
          sevsByRater: {},
          sevcatsBySystem: {},
          sevcatsByRater: {},
          metrics: {},
          aggrDoc: aggrDoc,
        };
        for (const system of this.dataIter.docSys[doc]) {
          const range = this.dataIter.docSegSys[doc][docSegId][system].rows;
          let aggrDocSegSys = {
            aggrDocSeg: aggrDocSeg,
            metrics: {},
          };
          for (let rowId = range[0]; rowId < range[1]; rowId++) {
            const parts = this.data[rowId];
            const segment = parts[this.DATA_COL_METADATA].segment || {};
            if (segment.hasOwnProperty('metrics')) {
              aggrDocSegSys.metrics = {
                ...segment.metrics,
                ...aggrDocSegSys.metrics,
              };
            }
            aggrDocSegSys = {...segment, ...aggrDocSegSys};
            if (!aggrDocSegSys.hasOwnProperty('num_source_chars')) {
              aggrDocSegSys.num_source_chars = parts.num_source_chars;
            }
            if (!aggrDocSegSys.hasOwnProperty('num_target_chars')) {
              aggrDocSegSys.num_target_chars = parts.num_target_chars;
            }
          }
          for (let metric in aggrDocSegSys.metrics) {
            if (!aggrDocSeg.metrics.hasOwnProperty(metric)) {
              aggrDocSeg.metrics[metric] = {};
            }
            aggrDocSeg.metrics[metric][system] = aggrDocSegSys.metrics[metric];
          }
          if (!aggrDocSegSys.source_tokens ||
              aggrDocSegSys.source_tokens.length == 0) {
            this.tokenizeLegacySegment(range, aggrDocSegSys);
          }
          if (!aggrDocSeg.hasOwnProperty('source_tokens') &&
              aggrDocSegSys.hasOwnProperty('source_tokens')) {
            aggrDocSeg.source_tokens = aggrDocSegSys.source_tokens;
          }
          if (!aggrDocSeg.hasOwnProperty('num_source_chars') &&
              aggrDocSegSys.hasOwnProperty('num_source_chars')) {
            aggrDocSeg.num_source_chars = aggrDocSegSys.num_source_chars;
          }
          if (!aggrDocSeg.hasOwnProperty('source_sentence_splits') &&
              aggrDocSegSys.hasOwnProperty('source_sentence_splits')) {
            aggrDocSeg.source_sentence_splits =
                aggrDocSegSys.source_sentence_splits;
          }
          if (!aggrDocSeg.hasOwnProperty('starts_paragraph') &&
              aggrDocSegSys.hasOwnProperty('starts_paragraph')) {
            aggrDocSeg.starts_paragraph = aggrDocSegSys.starts_paragraph;
          }
          if (!aggrDocSegSys.reference_tokens && aggrDocSegSys.references) {
            this.tokenizeLegacyReferences(aggrDocSegSys);
          }
          if (aggrDocSegSys.hasOwnProperty('reference_tokens')) {
            if (!aggrDocSeg.hasOwnProperty('reference_tokens')) {
              aggrDocSeg.reference_tokens = {};
            }
            aggrDocSeg.reference_tokens = {
              ...aggrDocSeg.reference_tokens,
              ...aggrDocSegSys.reference_tokens
            };
          }
          if (aggrDocSegSys.hasOwnProperty('reference_sentence_splits')) {
            if (!aggrDocSeg.hasOwnProperty('reference_sentence_splits')) {
              aggrDocSeg.reference_sentence_splits = {};
            }
            aggrDocSeg.reference_sentence_splits = {
              ...aggrDocSeg.reference_sentence_splits,
              ...aggrDocSegSys.reference_sentence_splits
            };
          }
          if (!aggrDocSeg.hasOwnProperty('primary_reference') &&
              aggrDocSegSys.hasOwnProperty('primary_reference')) {
            aggrDocSeg.primary_reference = aggrDocSegSys.primary_reference;
          }
          for (let rowId = range[0]; rowId < range[1]; rowId++) {
            const parts = this.data[rowId];
            const metadata = parts[this.DATA_COL_METADATA];
            metadata.segment = aggrDocSegSys;
            this.setMarkedText(rowId, metadata);

            const rater = parts[this.DATA_COL_RATER];
            if (!rater) {
              /**
               * This row is purely for metadata, such as references and/or
               * automated metrics
               */
              continue;
            }
            const category = parts[this.DATA_COL_CATEGORY];
            const severity = parts[this.DATA_COL_SEVERITY];

            this.addToArray(aggrDocSeg.catsBySystem, system, category);
            this.addToArray(aggrDocSeg.catsByRater, rater, category);
            this.addToArray(aggrDocSeg.sevsBySystem, system, severity);
            this.addToArray(aggrDocSeg.sevsByRater, rater, severity);
            const sevcat = severity + (category ? '/' + category : '');
            this.addToArray(aggrDocSeg.sevcatsBySystem, system, sevcat);
            this.addToArray(aggrDocSeg.sevcatsByRater, rater, sevcat);
            if (metadata.feedback && metadata.feedback.thumbs) {
              if (metadata.feedback.thumbs == 'up') {
                aggrDoc.thumbsUpCount++;
              } else if (metadata.feedback.thumbs == 'down') {
                aggrDoc.thumbsDownCount++;
              }
            }
            if (metadata.feedback && metadata.feedback.notes) {
              aggrDoc.feedbackNotes = (aggrDoc.feedbackNotes || '') +
                                      metadata.feedback.notes;
            }
          }
        }
      }
    }
  }

  /**
   * Returns an object consisting of filterREs (a dictionary of column
   * filter REs keyed by the id of the filter), filterExpr (a JavaScript
   * expression for filtering, possibly entered by the user), and onlyAllSysSegs
   * (a boolean that captures the value of the corresponding checkbox).
   *
   * Also sets the value of the select menus for column filters (if they exist).
   * @return {!Object}
   */
  getAllFilters() {
    const res = {};
    const filters = document.getElementsByClassName('marot-filter-re');
    for (let i = 0; i < filters.length; i++) {
      const filter = filters[i].value.trim();
      const id = filters[i].id;
      const selectId = id.replace(/filter/, 'select');
      const sel = document.getElementById(selectId);
      if (sel) sel.value = filter;
      if (!filter) {
        res[id] = null;
        continue;
      }
      res[id] = new RegExp(filter);
    }
    const filterExpr =
        document.getElementById('marot-filter-expr').value.trim();
    const onlyAllSysSegs = document.getElementById(
        'marot-only-all-systems-segments').checked;
    return {
      filterREs: res,
      filterExpr: filterExpr,
      onlyAllSysSegs: onlyAllSysSegs,
    };
  }

  /**
   * Short-cut function (convenient in filter expressions) for:
   *     "obj has array property prop that includes val"
   * @param {!Object} obj
   * @param {string} prop
   * @param {string} val
   * @return {boolean}
   */
  incl(obj, prop, val) {
    return obj.hasOwnProperty(prop) &&
      obj[prop].includes(val);
  }

  /**
   * Short-cut function (convenient in filter expressions) for:
   *     "obj has array property prop that excludes val"
   * @param {!Object} obj
   * @param {string} prop
   * @param {string} val
   * @return {boolean}
   */
  excl(obj, prop, val) {
    return obj.hasOwnProperty(prop) &&
      !obj[prop].includes(val);
  }

  /**
   * Clears this.filterClause and disables associated buttons.
   */
  clearClause() {
    this.filterClause = '';
    this.filterClauseKey.value = '';
    this.filterClauseInclExcl.value = 'includes';
    this.filterClauseSev.value = '';
    this.filterClauseCat.value = '';
    this.filterClauseAddAnd.disabled = true;
    this.filterClauseAddOr.disabled = true;
  }

  /**
   * Checks if filter expression clause is fully specified, enables "add"
   *     buttons if so.
   */
  checkClause() {
    this.filterClause = '';
    this.filterClauseAddAnd.disabled = true;
    this.filterClauseAddOr.disabled = true;
    if (!this.filterClauseKey.value) return;
    if (!this.filterClauseSev.value && !this.filterClauseCat.value) return;

    let sevcats = 'aggrDocSeg.sevcats';
    let key = '';
    let err = this.filterClauseSev.value + '/' + this.filterClauseCat.value;
    if (!this.filterClauseSev.value) {
      sevcats = 'aggrDocSeg.cats';
      err = this.filterClauseCat.value;
    }
    if (!this.filterClauseCat.value) {
      sevcats = 'aggrDocSeg.sevs';
      err = this.filterClauseSev.value;
    }
    if (this.filterClauseKey.value.startsWith('System: ')) {
      sevcats += 'BySystem';
      key = this.filterClauseKey.value.substr(8);
    } else {
      console.assert(this.filterClauseKey.value.startsWith('Rater: '),
                     this.filterClauseKey.value);
      sevcats += 'ByRater';
      key = this.filterClauseKey.value.substr(7);
    }
    const inclexcl = (this.filterClauseInclExcl.value == 'excludes') ?
        'marot.excl' : 'marot.incl';
    this.filterClause = `${inclexcl}(${sevcats}, "${key}", "${err}")`;
    this.filterClauseAddAnd.disabled = false;
    this.filterClauseAddOr.disabled = false;
  }

  /**
   * Adds this.filterClause with and/or to the filter expression.
   * @param {string} andor
   */
  addClause(andor) {
    if (!this.filterClause) return;
    const elt = document.getElementById('marot-filter-expr');
    let expr = elt.value.trim();
    if (expr) expr += ' ' + andor + ' ';
    expr += this.filterClause;
    elt.value = expr;
    this.clearClause();
    marot.show();
  }

  /**
   * Evaluates the JavaScript filterExpr on an this.data[] row and returns true
   *     only if the filter passes.
   * @param {string} filterExpr
   * @param {!Array<string>} parts
   * @return {boolean}
   */
  filterExprPasses(filterExpr, parts) {
    if (!filterExpr.trim()) return true;
    try {
      return Function(
          '"use strict";' +
          `
      const system = arguments[marot.DATA_COL_SYSTEM];
      const doc = arguments[marot.DATA_COL_DOC];
      const docSegId = arguments[marot.DATA_COL_DOC_SEG_ID];
      const globalSegId = arguments[marot.DATA_COL_GLOBAL_SEG_ID];
      const source = arguments[marot.DATA_COL_SOURCE];
      const target = arguments[marot.DATA_COL_TARGET];
      const rater = arguments[marot.DATA_COL_RATER];
      const category = arguments[marot.DATA_COL_CATEGORY];
      const severity = arguments[marot.DATA_COL_SEVERITY];
      const metadata = arguments[marot.DATA_COL_METADATA];
      const segment = metadata.segment;
      const reference = (
          segment.reference_tokens[segment.primary_reference] ?? []).join('');
      const aggrDocSegSys = segment;
      const aggrDocSeg = aggrDocSegSys.aggrDocSeg;
      const aggrDoc = aggrDocSeg.aggrDoc;` +
          'return (' + filterExpr + ')')(
          parts[this.DATA_COL_SYSTEM], parts[this.DATA_COL_DOC],
          parts[this.DATA_COL_DOC_SEG_ID], parts[this.DATA_COL_GLOBAL_SEG_ID],
          parts[this.DATA_COL_SOURCE], parts[this.DATA_COL_TARGET],
          parts[this.DATA_COL_RATER], parts[this.DATA_COL_CATEGORY],
          parts[this.DATA_COL_SEVERITY], parts[this.DATA_COL_METADATA]);
    } catch (err) {
      document.getElementById('marot-filter-expr-error').innerHTML =
          'Some rows were filtered out because of invalid fields: ' + err;
      return false;
    }
  }

  /**
   * This function will return false for segments that have some metric (MQM or
   * other) only available for some of the systems, not all/none.
   * @param {!Object} metadata
   * @return {boolean}
   */
  allSystemsFilterPasses(metadata) {
    const segment = metadata.segment;
    const aggrDocSeg = segment.aggrDocSeg;
    for (let metric in aggrDocSeg.metrics) {
      const numSystemsWithMetric =
          Object.keys(aggrDocSeg.metrics[metric]).length;
      if (numSystemsWithMetric > 0 &&
          numSystemsWithMetric != this.dataIter.systems.size) {
        return false;
      }
    }
    return true;
  }

  /**
   * Logs the metadata from one particular row to the JavaScript console. The
   * row number is provided by the user in an <input> element. This is
   * useful when formulating filter functions, to see what metadata fields are
   * available.
   */
  logRowMetadata() {
    const rowInput = document.getElementById('marot-metadata-row');
    const rowInputVal = rowInput.value.trim();
    if (!rowInputVal) return;
    const row = parseInt(rowInputVal);
    if (row < 0 || row >= this.data.length) {
      console.log(`Row must be in the range 0-${this.data.length - 1}`);
      rowInput.value = '';
      return;
    }
    const doc = this.data[row][this.DATA_COL_DOC];
    const docSegId = this.data[row][this.DATA_COL_DOC_SEG_ID];
    const system = this.data[row][this.DATA_COL_SYSTEM];
    const rater = this.data[row][this.DATA_COL_RATER];
    console.log('Metadata for row ' + row +
                ' - doc [' + doc + '], docSegId [' + docSegId +
                '], system [' + system + '], rater [' + rater + ']:');
    console.log(this.data[row][this.DATA_COL_METADATA]);
    console.log('Note that aggrDocSegSys is an alias for metadata.segment, ' +
                'aggrDocSeg for aggrDocSegSys.aggrDocSeg, ' +
                'and aggrDoc for aggrDocSeg.aggrDoc');
  }

  /**
   * In the weights/slices settings table with the given element id, add a row.
   * @param {string} id
   * @param {number} cols The number of columns to use.
   */
  settingsAddRow(id, cols) {
    let html = '<tr>';
    for (let i = 0; i < cols; i++) {
      html += `
          <td><span contenteditable="true"
                   class="marot-settings-editable"></span></td>`;
    }
    html += '</tr>';
    const elt = document.getElementById(id);
    const rowNum = document.getElementById(id + '-add-row').value ?? 1;
    /** The new row needs to be the 1-based position "rowNum" */
    const rows = elt.getElementsByTagName('tr');
    if (rows.length == 0 || rowNum <= 1) {
      elt.insertAdjacentHTML('afterbegin', html);
    } else {
      if (rowNum > rows.length) {
        rows[rows.length - 1].insertAdjacentHTML('afterend', html);
      } else {
        rows[rowNum - 1].insertAdjacentHTML('beforebegin', html);
      }
    }
  }

  /**
   * Displays settings tables for score weights and slices.
   */
  setUpScoreSettings() {
    const weightSettings = document.getElementById('marot-settings-weights');
    weightSettings.innerHTML = '';
    for (const sc of this.mqmWeights) {
      sc.regex = new RegExp(sc.pattern, 'i');
      weightSettings.insertAdjacentHTML('beforeend', `
          <tr>
            <td><span contenteditable="true"
                     class="marot-settings-editable">${sc.name}</span></td>
            <td><span contenteditable="true"
                     class="marot-settings-editable">${sc.pattern}</span></td>
            <td><span contenteditable="true"
                     class="marot-settings-editable">${sc.weight}</span></td>
          </tr>`);
    }
    const sliceSettings = document.getElementById('marot-settings-slices');
    sliceSettings.innerHTML = '';
    for (const sc of this.mqmSlices) {
      sc.regex = new RegExp(sc.pattern, 'i');
      sliceSettings.insertAdjacentHTML('beforeend', `
          <tr>
            <td><span contenteditable="true"
                     class="marot-settings-editable">${sc.name}</span></td>
            <td><span contenteditable="true"
                     class="marot-settings-editable">${sc.pattern}</span></td>
          </tr>`);
    }
  }

  /**
   * Parses score weights/slices from the user-edited table identified by id.
   * If there are errors in parsing then they are displayed to the user in the
   * marot-errors element and null is returned.
   * @param {string} id
   * @param {boolean} hasWeight True if this is the weights table.
   * @return {?Array<!Object>} Array of parsed weights/slices, or null if
   *     errors.
   */
  parseScoreSettingsInner(id, hasWeight) {
    const nameChecker = new RegExp(/^[a-z0-9\.-]+$/i);
    const errorsFound = [];
    const rows = document.getElementById(id).getElementsByTagName('tr');
    const parsed = [];
    const names = {};
    for (let i = 0; i < rows.length; i++) {
      if (!rows[i].textContent.trim()) {
        /* Allow skipping blank lines */
        continue;
      }
      const parsedRow = {};
      const spans = rows[i].getElementsByTagName('span');
      console.assert(spans.length == 2 + (hasWeight ? 1 : 0), spans);
      parsedRow.name = spans[0].textContent.trim();
      if (parsedRow.name.length > 10) {
        errorsFound.push(
            'The name [' + parsedRow.name + '] is longer than 10 chars');
        continue;
      }
      if (!nameChecker.test(parsedRow.name)) {
        errorsFound.push(
            'The name [' + parsedRow.name +
            '] cannot be empty, must use characters [a-zA-Z0-9.-]');
        continue;
      }
      if (names[parsedRow.name]) {
        errorsFound.push('The name [' + parsedRow.name + '] is a duplicate');
        continue;
      }
      parsedRow.pattern = spans[1].textContent.trim();
      try {
        parsedRow.regex = new RegExp(parsedRow.pattern, 'i');
      } catch (err) {
        console.log(err);
        parsedRow.pattern = '';
      }
      if (!parsedRow.pattern) {
        errorsFound.push(
            'The pattern in row [' + parsedRow.name + '] is empty/invalid');
        continue;
      }
      if (hasWeight) {
        parsedRow.weight = parseFloat(spans[2].textContent.trim());
        if (isNaN(parsedRow.weight) || parsedRow.weight < 0) {
          errorsFound.push(
              'The weight in row [' + parsedRow.name + '] is invalid');
          continue;
        }
      }
      /* All good! */
      names[parsedRow.name] = true;
      parsed.push(parsedRow);
    }
    const errorsElt = document.getElementById('marot-errors');
    for (const error of errorsFound) {
      errorsElt.insertAdjacentHTML('beforeend', `<div>${error}</div>\n`);
    }
    return (errorsFound.length == 0) ? parsed : null;
  }

  /**
   * Parses score weights and slices from the user-edited settings tables. Sets
   * this.mqmWeights and this.mqmSlices if successful.
   * @return {boolean} True if the parsing was successful.
   */
  parseScoreSettings() {
    const errors = document.getElementById('marot-errors');
    errors.innerHTML = '';
    const newWeights = this.parseScoreSettingsInner(
        'marot-settings-weights', true);
    const newSlices = this.parseScoreSettingsInner(
        'marot-settings-slices', false);
    if (!newWeights || !newSlices) {
      return false;
    }
    this.mqmWeights = newWeights;
    this.mqmSlices = newSlices;
    return true;
  }

  /**
   * This checks if the annotation matches the pattern in the MQM weight/slice
   * component.
   * @param {!Object} sc Score component, with a regex property.
   * @param {string} sev Severity of the annotation.
   * @param {string} cat Category (and optional "/"+subcat.) of the annotation.
   * @return {boolean}
   */
  matchesMQMSplit(sc, sev, cat) {
    return sc.regex.test(sev + ':' + cat);
  }

  /**
   * Returns a string that shows the value of the metric to three decimal
   * places. If denominator is <= 0, then returns "-".
   * @param {number} metric
   * @param {number} denominator
   * @return {string}
   */
  metricDisplay(metric, denominator) {
    return (denominator > 0) ? metric.toFixed(3) : '-';
  }

  /**
   * Returns an HTML string for a '<td>' tag containing the score, including
   * hover text for showing the number of segments and source characters.
   * @param {!Object} s Score object containing the fields numSegments,
   *     num_source_chars, score.
   * @param {string=} cls Optional span class to place the score text within.
   * @return {string}
   */
  tdForScore(s, cls='') {
    const title =
        `#Segments: ${s.numSegments}, #SrcChars: ${s.num_source_chars}`;
    let td = `<td title="${title}">`;
    if (cls) {
      td += `<span class="${cls}">`;
    }
    td += this.metricDisplay(s.score, s.numSegments);
    if (cls) {
      td += '</span>';
    }
    td += '</td>';
    return td;
  }

  /**
   * Initializes and returns a rater stats object, which captures the score,
   * score slices, total error counts, and some more stats from one rater on
   * one specific segment (doc+sys+seg).
   * @param {string} rater
   * @return {!Object}
   */
  initRaterStats(rater) {
    return {
      rater: rater,
      score: 0,

      errorSpans: 0,
      numWithErrors: 0,

      hotwFound: 0,
      hotwMissed: 0,
      timeSpentMS: 0,
    };
  }

  /**
   * Creates the key for an MQM weighted component or slice.
   * @param {string} name
   * @param {boolean=} isSlice
   * @return {string}
   */
  mqmKey(name, isSlice = false) {
    return (isSlice ? this.MQM_SLICE_PREFIX : this.MQM_WEIGHTED_PREFIX) + name;
  }

  /**
   * Strips the prefix from a key for an MQM component (previously assembled by
   * mqmKey()).
   * @param {string} key
   * @return {string}
   */
  mqmKeyToName(key) {
    if (key.startsWith(this.MQM_WEIGHTED_PREFIX)) {
      return key.substr(this.MQM_WEIGHTED_PREFIX.length);
    } else if (key.startsWith(this.MQM_SLICE_PREFIX)) {
      return key.substr(this.MQM_SLICE_PREFIX.length);
    }
    return key;
  }

  /**
   * Appends stats from delta into raterStats.
   * @param {!Object} raterStats
   * @param {!Object} delta
   */
  addRaterStats(raterStats, delta) {
    raterStats.score += delta.score;
    for (const sc of this.mqmWeights) {
      const key = this.mqmKey(sc.name);
      if (delta[key]) {
        raterStats[key] = (raterStats[key] ?? 0) + delta[key];
      }
    }
    for (const sc of this.mqmSlices) {
      const key = this.mqmKey(sc.name, true);
      if (delta[key]) {
        raterStats[key] = (raterStats[key] ?? 0) + delta[key];
      }
    }
    raterStats.errorSpans += delta.errorSpans;
    raterStats.numWithErrors += delta.numWithErrors;
    raterStats.hotwFound += delta.hotwFound;
    raterStats.hotwMissed += delta.hotwMissed;
    raterStats.timeSpentMS += delta.timeSpentMS;
  }

  /**
   * Divides all metrics in raterStats by num.
   * @param {!Object} raterStats
   * @param {number} num
   */
  avgRaterStats(raterStats, num) {
    if (!num) return;
    raterStats.score /= num;
    raterStats.timeSpentMS /= num;
    for (const sc of this.mqmWeights) {
      const key = this.mqmKey(sc.name);
      if (raterStats[key]) {
        raterStats[key] /= num;
      }
    }
    for (const sc of this.mqmSlices) {
      const key = this.mqmKey(sc.name, true);
      if (raterStats[key]) {
        raterStats[key] /= num;
      }
    }
  }

  /**
   * Aggregates segment stats. This returns an object that has aggregate MQM
   * score in the "score" field and these additional properties:
   *       numSegments
   *       num_source_chars
   *       metrics['<metric-name>']:
   *          score
   *          numSegments
   *          num_source_chars
   *       metric-[index in this.metrics]
   *           (repeated from metrics[...].score, as a convenient sorting key)
   *       mqmStats:
   *         numSegments # of segments with MQM ratings
   *         num_source_chars # of src-chars with MQM ratings
   *         numRatings: # segment X rater combinations for MQM
   *         timeSpentMS
   * @param {!Array} segs
   * @return {!Object}
   */
  aggregateSegStats(segs) {
    const aggregates = {
      numSegments: 0,
      num_source_chars: 0,
      metrics: {},
    };
    aggregates.mqmStats = {
      numSegments: 0,
      num_source_chars: 0,
      numRatings: 0,
      ...(this.initRaterStats('')),
    };
    if (!segs || !segs.length) {
      return aggregates;
    }
    for (const segStats of segs) {
      aggregates.numSegments++;
      aggregates.num_source_chars += segStats.num_source_chars;
      const allRaterStats = this.initRaterStats('');
      for (const r of segStats) {
        this.addRaterStats(allRaterStats, r);
      }
      if (segStats.length > 0) {
        this.avgRaterStats(allRaterStats, segStats.length);
        aggregates.mqmStats.numSegments++;
        aggregates.mqmStats.num_source_chars += segStats.num_source_chars;
        aggregates.mqmStats.numRatings += segStats.length;
        this.addRaterStats(aggregates.mqmStats, allRaterStats);
      }
      if (segStats.hasOwnProperty('metrics')) {
        for (let metric in segStats.metrics) {
          if (metric == 'MQM') {
            /**
             * Ignore any MQM values that may be present in the segment metadata
             * as we compute them from the annotations.
             */
            continue;
          }
          if (!aggregates.metrics.hasOwnProperty(metric)) {
            aggregates.metrics[metric] = {
              score: 0,
              numSegments: 0,
              num_source_chars: 0,
            };
          }
        }
      }
    }

    this.avgRaterStats(aggregates.mqmStats, aggregates.mqmStats.numSegments);

    for (let metric in aggregates.metrics) {
      const metricStats = aggregates.metrics[metric];
      for (const segStats of segs) {
        if (!segStats.hasOwnProperty('metrics') ||
            !segStats.metrics.hasOwnProperty(metric)) {
          continue;
        }
        metricStats.numSegments++;
        metricStats.num_source_chars += segStats.num_source_chars;
        metricStats.score += segStats.metrics[metric];
      }
      if (metricStats.numSegments > 0) {
        metricStats.score /= metricStats.numSegments;
      }
    }
    /** Copy MQM score into aggregate.metrics['MQM'] */
    if (aggregates.mqmStats.numRatings > 0) {
      aggregates.metrics['MQM'] = {
        score: aggregates.mqmStats.score,
        numSegments: aggregates.mqmStats.numSegments,
        num_source_chars: aggregates.mqmStats.num_source_chars,
        numRatings: aggregates.mqmStats.numRatings,
      };
    }
    for (let metric in aggregates.metrics) {
      const metricStats = aggregates.metrics[metric];
      const metricIndex = this.metricsInfo[metric].index;
      aggregates['metric-' + metricIndex] = metricStats.score;
    }
    return aggregates;
  }

  /**
   * Creates a background Worker for running significance tests. If sigtests
   * are not applicable, then this returns without creating a Worker.
   */
  startSigtests() {
    let noop = true;
    if (this.sigtestsData.data.hasOwnProperty('sys')) {
      noop = false;
      this.sigtestsSysMsg.innerHTML = 'Computing p-values...';
    }
    if (this.sigtestsData.data.hasOwnProperty('rater')) {
      noop = false;
      this.sigtestsRaterMsg.innerHTML = 'Computing p-values...';
    }
    if (noop) {
      return;
    }
    const elt = document.getElementById('marot-sigtests-num-trials');
    this.sigtestsData.numTrials = parseInt(elt.value);
    console.assert(this.sigtestsWorkerJS,
                   'Missing code from marot-sigtests.js');
    const blob = new Blob([this.sigtestsWorkerJS],
                          {type: "text/javascript" });
    this.sigtestsWorker = new Worker(window.URL.createObjectURL(blob));
    this.sigtestsWorker.postMessage(this.sigtestsData);
    this.sigtestsWorker.onmessage = this.sigtestsUpdate.bind(this);
  }

  /**
   * This resets the significance tests data and terminates the active sigtests
   * computation Worker if it exists.
   */
  resetSigtests() {
    this.sigtestsSysMsg.innerHTML = '';
    this.sigtestsRaterMsg.innerHTML = '';
    this.sigtestsData.data = {};
    if (this.sigtestsWorker) {
      this.sigtestsWorker.terminate();
    }
    this.sigtestsWorker = null;
  }

  /**
   * Runs Pearson correlation between scores from baseline and item, returning
   * the rho value.
   *
   * @param {!MarotSigtestsMetricData} data
   * @param {string} baseline
   * @param {string} item
   * @return {number}
   */
  pearson(data, baseline, item) {
    const baselineScores = data.segScores[baseline];
    const itemScores = data.segScores[item];
    const commonPos = data.commonPosByItemPair[baseline][item];

    const n = commonPos.length ?? 0;
    if (n <= 1) {
      return NaN;
    }
    let sumX = 0.0, sumY = 0.0;
    let sumX2 = 0.0, sumY2 = 0.0;
    let sumXY = 0.0;
    for (const pos of commonPos) {
      const x = baselineScores[pos];
      const y = itemScores[pos];
      sumX += x;
      sumY += y;
      sumX2 += (x * x);
      sumY2 += (y * y);
      sumXY += (x * y);
    }
    return ((n * sumXY) - (sumX * sumY)) / Math.sqrt(
      ((n * sumX2) - (sumX * sumX)) * ((n * sumY2) - (sumY * sumY)));
  }

  /**
   * This prepares significance tests data for systems (sysOrRater = 'sys') or
   * raters (sysOrRater = 'rater').
   *
   * @param {string} sysOrRater
   * @param {!Object} stats Stats object keyed by system/rater.
   * @param {!Object} statsAggregates Aggregation (also keyed by system/rater)
   *     of stats over segments.
   * @param {!Array<number>} metricIds
   * @return {!Object} Returns an object keyed by metric name, where each
   *     value is a MarotSigtestsMetricData object.
   */
  prepareSigtests(sysOrRater, stats, statsAggregates, metricIds) {
    /**
     * Each segment is uniquely determined by the (doc, docSegId) pair. We use
     * `pairToPos` to track which pair goes to which position in the aligned
     * segScores[] array.
     */
    const pairToPos = {};
    let maxPos = 0;
    for (const key in stats) {
      for (const doc in stats[key]) {
        if (!pairToPos.hasOwnProperty(doc)) {
          pairToPos[doc] = {};
        }
        const docsegs = stats[key][doc];
        for (const docSegId in docsegs) {
          if (pairToPos[doc].hasOwnProperty(docSegId)) {
            continue;
          }
          pairToPos[doc][docSegId] = maxPos;
          maxPos += 1;
        }
      }
    }
    const sigtestsData = {
    };

    const comparables = this.excludeRaterComplements(
        Object.keys(statsAggregates));
    const indexOfTotal = comparables.indexOf(this.TOTAL);
    if (indexOfTotal >= 0) {
      comparables.splice(indexOfTotal, 1);
    }

    for (const m of metricIds) {
      const metricKey = 'metric-' + m;
      const metric = this.metrics[m];
      const metricInfo = this.metricsInfo[metric];
      const data = new MarotSigtestsMetricData();
      sigtestsData[metric] = data;
      data.comparables = comparables.slice();
      data.lowerBetter = metricInfo.lowerBetter || false;
      const signReverser = metricInfo.lowerBetter ? 1.0 : -1.0;
      data.comparables.sort(
          (s1, s2) => signReverser * (
                          (statsAggregates[s1][metricKey] ?? 0) -
                          (statsAggregates[s2][metricKey] ?? 0)));
      for (const item of data.comparables) {
        data.scores[item] =
            statsAggregates[item].metrics[metric] ??
            {score: 0, numSegments: 0};
      }
      const augmentedItems = data.comparables.slice();
      if (sysOrRater == 'rater') {
        for (const rater of data.comparables) {
          const notRater = this.NOT_PREFIX + rater;
          augmentedItems.push(notRater);
          data.scores[notRater] =
              statsAggregates[notRater].metrics[metric] ??
              {score: 0, numSegments: 0};
        }
      }
      const segScores = data.segScores;
      for (const item of augmentedItems) {
        /**
         * For each item, we first compute the mapping from position to score.
         * Any missing key correponds to one missing segment for this item.
         */
        const posToScore = {};
        for (const doc of Object.keys(stats[item])) {
          for (const docSegId of Object.keys(stats[item][doc])) {
            const pos = pairToPos[doc][docSegId];
            const segs = stats[item][doc][docSegId];
            /** Note the extra "[]". */
            const aggregate = this.aggregateSegStats([segs]);
            const metricStats = aggregate.metrics[metric] ?? null;
            if (metricStats && metricStats.numSegments > 0) {
              posToScore[pos] = metricStats.score;
            }
          }
        }
        /** Now we can compute "segScores". */
        segScores[item] = [];
        for (let pos = 0; pos < maxPos; pos++) {
          if (posToScore.hasOwnProperty(pos)) {
            segScores[item].push(posToScore[pos]);
          } else {
            /** This item is missing this specific segment. */
            segScores[item].push(null);
          }
        }
      }

      /** Compute common positions for each item pair in `commonPos`. */
      const commonPos = data.commonPosByItemPair;
      for (const [idx, baseline] of data.comparables.entries()) {
        if (!commonPos.hasOwnProperty(baseline)) {
          commonPos[baseline] = {};
        }
        /**
         * We only need the upper triangle in the comparison table, plus the
         * complement in case of raters.
         */
        const columns = (sysOrRater == 'rater' ?
                         [this.NOT_PREFIX + baseline] : []).concat(
                             data.comparables.slice(idx + 1));
        for (const item of columns) {
          if (!commonPos[baseline].hasOwnProperty(item)) {
            commonPos[baseline][item] = [];
          }
          for (let pos = 0; pos < maxPos; pos++) {
            if ((segScores[item][pos] != null) &&
                (segScores[baseline][pos] != null)) {
              commonPos[baseline][item].push(pos);
            }
          }
        }
      }

      /**
       * Create p-values matrix, to be populated with updates from the Worker.
       */
      const numComparables = data.comparables.length;
      data.pValues = Array(numComparables);
      for (let row = 0; row < numComparables; row++) {
        data.pValues[row] = Array(numComparables);
        for (let col = 0; col < numComparables; col++) {
          data.pValues[row][col] = NaN;
        }
      }
    }
    return sigtestsData;
  }

  /**
   * In the significance tests table, draw a solid line under every prefix of
   * items that is significantly better than all subsequent items. Draw a
   * dotted line to separate clusters within which no item is significantly
   * better than any other.
   * @param {string} sysOrRater
   * @param {string} metric
   */
  clusterSigtests(sysOrRater, metric) {
    const m = this.metricsInfo[metric].index;
    const data = this.sigtestsData.data[sysOrRater][metric];
    const numComparables = data.comparables.length;
    const itemBetterThanAllAfter = Array(numComparables);
    for (let row = 0; row < numComparables; row++) {
      itemBetterThanAllAfter[row] = numComparables - 1;
      for (let col = numComparables - 1; col > row; col--) {
        const pValue = data.pValues[row][col];
        if (isNaN(pValue) || pValue >= this.PVALUE_THRESHOLD) {
          break;
        }
        itemBetterThanAllAfter[row] = col - 1;
      }
    }
    let maxBetterThanAllAfter = 0;  /** Max over rows 0..row */
    let dottedClusterStart = 0;
    for (let row = 0; row < numComparables - 1; row++) {
      const tr = document.getElementById(
          `marot-${sysOrRater}-sigtests-${m}-row-${row}`);
      maxBetterThanAllAfter = Math.max(maxBetterThanAllAfter,
                                       itemBetterThanAllAfter[row]);
      if (maxBetterThanAllAfter == row) {
        tr.className = 'marot-bottomed-tr';
        dottedClusterStart = row + 1;
        continue;
      }
      /** Is no item in dottedClusterStart..row signif. better than row+1? */
      let noneSigBetter = true;
      for (let dottedClusterRow = dottedClusterStart;
           dottedClusterRow <= row; dottedClusterRow++) {
        const pValue = data.pValues[dottedClusterRow][row + 1];
        if (!isNaN(pValue) && pValue < this.PVALUE_THRESHOLD) {
          noneSigBetter = false;
          break;
        }
      }
      if (!noneSigBetter) {
        tr.className = 'marot-dotted-bottomed-tr';
        dottedClusterStart = row + 1;
      }
    }
  }

  /**
   * This receives a computation update from the Sigtests Worker thread. The
   * update consists of one p-value for a metric, row, col, or marks the
   * computation for that metric as done, or marks all computations as finished.
   * @param {!Event} e
   */
  sigtestsUpdate(e) {
    const update = e.data;
    if (update.finished) {
      this.resetSigtests();
      return;
    }
    if (update.sysOrRaterDone) {
      if (update.sysOrRater == 'sys') {
        this.sigtestsSysMsg.innerHTML = '';
      } else {
        this.sigtestsRaterMsg.innerHTML = '';
      }
      return;
    }
    const metric = update.metric;
    if (update.metricDone) {
      this.clusterSigtests(update.sysOrRater, metric);
      return;
    }
    const pValues = this.sigtestsData.data[update.sysOrRater][metric].pValues;
    const row = update.row;
    const col = update.col;
    pValues[row][col] = update.pValue;
    const m = this.metricsInfo[metric].index;
    const spanId = `marot-${update.sysOrRater}-sigtest-${m}-${row}-${col}`;
    const span = document.getElementById(spanId).firstElementChild;
    span.innerHTML = isNaN(update.pValue) ? '-' : update.pValue.toFixed(3);
    if (update.pValue < this.PVALUE_THRESHOLD) {
      span.classList.add('marot-pvalue-significant');
    }
  }

  /**
   * Toggle system/rater comparison tables between p-value display and Pearson
   * rho display.
   */
  switchCmpTable(sysOrRater) {
    const section = document.getElementById(`marot-${sysOrRater}-comparison`);
    const key = document.getElementById(
        `marot-${sysOrRater}-v-${sysOrRater}-tables-type`).value;
    if (key == 'pValue') {
      section.className = 'marot-comparison-tables-pvalue';
    } else {
      section.className = 'marot-comparison-tables-rho';
    }
  }

  /**
   * Shows the table for significance tests for system/rater differences.
   * @param {string} sysOrRater
   * @param {!Object} stats
   * @param {!Object} statsAggregates
   * @param {!Array<number>} metricIds
   */
  showSigtests(sysOrRater, stats, statsAggregates, metricIds) {
    const div = document.getElementById(
        `marot-${sysOrRater}-v-${sysOrRater}-tables`);
    div.innerHTML = '';
    const sigtestsData = this.prepareSigtests(
        sysOrRater, stats, statsAggregates, metricIds);
    let firstTable = true;
    for (const m of metricIds) {
      const metric = this.metrics[m];
      const data = sigtestsData[metric];
      const comparables = data.comparables;
      if (comparables.length == 0) {
        continue;
      }
      const scores = data.scores;

      /** Header. */
      let html = `
      ${firstTable ? '' : '<br>'}
      <table id="marot-${sysOrRater}-sigtests-${m}"
          class="marot-table marot-numbers-table">
        <thead>
          <tr>
            <th>${sysOrRater == 'sys' ? 'System' : 'Rater'}</th>
            <th>${this.metrics[m]}</th>`;
      if (sysOrRater == 'rater') {
        html += '<th colspan="2">All other raters</th>';
      }
      for (const item of comparables) {
        html += `<th>${item}</th>`;
      }
      html += `</tr></thead>\n<tbody>\n`;

      /** Show Pearson correlations & significance test p-value placeholders. */
      for (const [rowIdx, baseline] of comparables.entries()) {
        /** Show metric score in the second column. */
        const s = scores[baseline];
        html += `
          <tr id="marot-${sysOrRater}-sigtests-${m}-row-${rowIdx}">
            <td>${baseline}</td>
            ${this.tdForScore(s)}`;
        const othersColumn = [];
        if (sysOrRater == 'rater') {
          const notRater = this.NOT_PREFIX + baseline;
          const otherRatersScore = scores[notRater];
          html += this.tdForScore(otherRatersScore, 'marot-gray');
          othersColumn.push(notRater);
        }
        const columns = othersColumn.concat(comparables);
        for (const [itemIdx, item] of columns.entries()) {
          const colIdx = itemIdx - othersColumn.length;
          /**
           * Note that p-value against "all other raters" is shown in colIdx = -1
           */
          if (rowIdx >= colIdx && colIdx != -1) {
            html += '<td></td>';
            continue;
          }
          const commonPos = data.commonPosByItemPair[baseline][item] ?? [];
          const title = 'Based on ' + commonPos.length + ' common segments';
          html += '<td title="' + title + '">';
          const s2 = scores[item];
          if (s2.numSegments == 0) {
            html += '</td>';
            continue;
          }
          const spanId = `marot-${sysOrRater}-sigtest-${m}-${rowIdx}-${colIdx}`;
          const rho = this.pearson(data, baseline, item);
          const rhoDisp = isNaN(rho) ? '-' : rho.toFixed(3);
          html += `<span id="${spanId}">
              <span class="marot-pvalue">-.---</span>
              <span class="marot-rho">${rhoDisp}</span>
          <span></td>`;
        }
        html += `</tr>`;
      }
      html += `</tbody></table>`;
      div.insertAdjacentHTML('beforeend', html);
      firstTable = false;
    }
    if (!firstTable) {
      this.sigtestsData.data[sysOrRater] = sigtestsData;
    }
  }

  /**
   * Listener for changes to the input field that specifies the number of trials
   * for paired one-sided approximate randomization.
   */
  setSigtestsNumTrials() {
    const elt = document.getElementById('marot-sigtests-num-trials');
    const numTrials = parseInt(elt.value);
    if (numTrials <= 0 || numTrials == this.sigtestsData.numTrials) {
      elt.value = this.sigtestsData.numTrials;
      return;
    }
    this.show();
  }

  /**
   * Shows the table header for the marot scores table. The score weighted
   * components and slices to display should be available in
   * mqmWeightedFields and mqmSliceFields.
   * @param {boolean} hasRatings set to true if there are some MQM annotations.
   */
  showScoresHeader(hasRatings) {
    const header = document.getElementById('marot-stats-thead');
    let html = `
        <tr>
          <th>Scores are per segment</th>`;
    const metricFields = [];
    for (const m of this.metricsVisible) {
      const metric = this.metrics[m];
      html +=  `<th id="marot-metric-${m}-th">${metric}</th>`;
      metricFields.push('metric-' + m);
    }
    html += `
            <th title="Number of segments"><b>#Segments</b></th>
            <th title="Number of source characters"><b>#Source-chars</b></th>`;
    if (hasRatings) {
      html += `
            <th title="Number of segment ratings"><b>#Segment ratings</b></th>`;
    }

    const mqmPartFields =
        this.mqmWeightedFields.map(x => this.MQM_WEIGHTED_PREFIX + x)
            .concat(this.mqmSliceFields.map(x => this.MQM_SLICE_PREFIX + x));
    for (let i = 0; i < mqmPartFields.length; i++) {
      const scoreKey = mqmPartFields[i];
      const scoreName = this.mqmKeyToName(scoreKey);
      const partType = (i < this.mqmWeightedFields.length) ? 'weighted' :
          'slice';
      const cls = 'marot-stats-' + partType;
      const tooltip = 'Score part: ' + scoreName + '-' + partType;
      html += `
          <th id="marot-${scoreKey}-th" class="marot-score-th ${cls}"
              title="${tooltip}">
            <b>${scoreName}</b>
          </th>`;
    }
    if (hasRatings) {
      html += `
            <th title="Average time (seconds) per rater per segment">
                <b>Time (s)</b></th>
            <th title="Average length of error span"><b>Err span</b></th>
            <th title="Hands-on-the-wheel test"><b>HOTW Test</b></th>
          </tr>`;
    }
    header.innerHTML = html;

    /** Make columns clickable for sorting purposes. */

    const upArrow = '<span class="marot-arrow marot-arrow-up">&#129041;</span>';
    const downArrow =
        '<span class="marot-arrow marot-arrow-down">&#129043;</span>';
    for (const field of metricFields.concat(mqmPartFields)) {
      const headerId = `marot-${field}-th`;
      const th = document.getElementById(headerId);
      th.insertAdjacentHTML('beforeend', ` ${upArrow}${downArrow}`);
      th.addEventListener('click', (e) => {
        // Click again for reversing order. Otherwise sort in ascending order.
        if (field == this.sortByField) {
          this.sortReverse = !this.sortReverse;
        } else {
          this.sortReverse = false;
        }
        this.sortByField = field;
        this.show();
      });
    }
    this.showSortArrow();
  }

  /**
   * In the stats table, display a separator line, optionally followed by a row
   * that displays a title (if non-empty).
   * @param {boolean} hasRatings set to true if there are some marot annotations.
   * @param {string=} title
   */
  showScoresSeparator(hasRatings, title='') {
    const NUM_COLS = (hasRatings ? 7 : 3) + this.metricsVisible.length +
                     this.mqmWeightedFields.length +
                     this.mqmSliceFields.length;
    this.statsTable.insertAdjacentHTML(
        'beforeend',
        `<tr><td colspan="${NUM_COLS}"><hr></td></tr>` +
        (title ?
        `<tr><td colspan="${NUM_COLS}"><b>${title}</b></td></tr>\n` :
        ''));
  }

  /**
   * Appends a row with score details for "label" (shown in the first column)
   * from the stats object to this.statsTable.
   * @param {string} label
   * @param {boolean} hasRatings set to true if there are some MQM annotations.
   * @param {!Object} stats
   * @param {!Object} aggregates
   */
  showScores(label, hasRatings, stats, aggregates) {
    const scoreFields =
        this.mqmWeightedFields.map(x => this.MQM_WEIGHTED_PREFIX + x).concat(
            this.mqmSliceFields.map(x => this.MQM_SLICE_PREFIX + x));
    let rowHTML = `<tr><td>${label}</td>`;
    for (const m of this.metricsVisible) {
      const metric = this.metrics[m];
      if (!aggregates.metrics.hasOwnProperty(metric)) {
        rowHTML += '<td>-</td>';
        continue;
      }
      const s = aggregates.metrics[metric];
      rowHTML += this.tdForScore(s);
    }
    rowHTML +=
        `<td>${aggregates.numSegments}</td>` +
        `<td>${aggregates.num_source_chars}</td>`;
    if (hasRatings) {
      const mqmStats = aggregates.mqmStats;
      console.assert(mqmStats.numRatings > 0, mqmStats);
      rowHTML +=
        `<td>${mqmStats.numRatings}</td>`;
      if (mqmStats.numSegments <= 0) {
        for (let i = 0; i < scoreFields.length + 3; i++) {
          rowHTML += '<td>-</td>';
        }
      } else {
        for (const s of scoreFields) {
          let content =
              mqmStats.hasOwnProperty(s) ? mqmStats[s].toFixed(3) : '-';
          const nameParts = s.split('-', 2);
          const cls = (nameParts.length == 2) ?
            ' class="marot-stats-' + nameParts[0] + '"' :
            '';
          rowHTML += `<td${cls}>${content}</td>`;
        }
        let errorSpan = 0;
        if (mqmStats.numWithErrors > 0) {
          errorSpan = mqmStats.errorSpans / mqmStats.numWithErrors;
        }
        rowHTML += `<td>${(mqmStats.timeSpentMS/1000.0).toFixed(1)}</td>`;
        rowHTML += `<td>${(errorSpan).toFixed(1)}</td>`;
        const hotw = mqmStats.hotwFound + mqmStats.hotwMissed;
        if (hotw > 0) {
          const perc = ((mqmStats.hotwFound * 100.0) / hotw).toFixed(1);
          rowHTML += `<td>${mqmStats.hotwFound}/${hotw} (${perc}%)</td>`;
        } else {
          rowHTML += '<td>-</td>';
        }
      }
    }
    rowHTML += '</tr>\n';
    this.statsTable.insertAdjacentHTML('beforeend', rowHTML);
  }

  /**
   * Shows the system x rater matrix of scores. The rows and columns are
   * ordered by total marot score.
   */
  showSystemRaterStats() {
    const table = document.getElementById('marot-system-x-rater');

    const systems = Object.keys(this.statsBySystem);
    const systemAggregates = {};
    for (const sys of systems) {
      const segs = this.getSegStatsAsArray(this.statsBySystem[sys]);
      systemAggregates[sys] = this.aggregateSegStats(segs);
    }

    const SORT_FIELD = 'metric-0';
    systems.sort(
        (sys1, sys2) =>
            systemAggregates[sys1][SORT_FIELD] -
            systemAggregates[sys2][SORT_FIELD]);

    const raters = this.excludeRaterComplements(Object.keys(this.statsByRater));
    const raterAggregates = {};
    for (const rater of raters) {
      const segs = this.getSegStatsAsArray(this.statsByRater[rater]);
      raterAggregates[rater] = this.aggregateSegStats(segs);
    }
    raters.sort(
        (rater1, rater2) =>
            raterAggregates[rater1][SORT_FIELD] -
            raterAggregates[rater2][SORT_FIELD]);

    let html = `
      <thead>
        <tr>
          <th>System</th>
          <th>All raters</th>`;
    for (const rater of raters) {
      html += `
          <th>${rater}</th>`;
    }
    html += `
        </tr>
      </thead>
      <tbody>`;
    /**
     * State for detecting "out-of-order" raters. We say a rater is out-of-order
     * if their rating for a system is oppositely related to the previous
     * system's rating, when compared with the aggregate over all raters.
     */
    let lastAllRaters = 0;
    const lastForRater = {};
    for (const rater of raters) {
      lastForRater[rater] = 0;
    }
    for (const sys of systems) {
      if (sys == this.TOTAL) {
        continue;
      }
      const allRatersScore = systemAggregates[sys].mqmStats;
      html += `
        <tr><td>${sys}</td>${this.tdForScore(allRatersScore)}`;
      for (const rater of raters) {
        const segs = this.getSegStatsAsArray(
            (this.statsByRaterSystem[rater] ?? {})[sys] ?? {});
        if (segs && segs.length > 0) {
          const aggregate = this.aggregateSegStats(segs);
          const cls = ((aggregate.mqmStats.score < lastForRater[rater] &&
                        allRatersScore.score > lastAllRaters) ||
                       (aggregate.mqmStats.score > lastForRater[rater] &&
                        allRatersScore.score < lastAllRaters)) ?
              'marot-out-of-order' : '';
          html += this.tdForScore(aggregate.mqmStats, cls);
          lastForRater[rater] = aggregate.mqmStats.score;
        } else {
          html += '<td>-</td>';
          /**
           * Ensure that the next score for this rater is marked as out-of-order
           * or not in some reasonable manner:
           */
          lastForRater[rater] = allRatersScore.score;
        }
      }
      lastAllRaters = allRatersScore.score;
      html += `
        </tr>`;
    }
    html += `
      </tbody>`;
    table.innerHTML = html;
  }

  /**
   * Sorter function where a & b are both a [doc, seg] pair. The seg part can
   * be a number (0,1,2,3,etc.) or a string, so we have to be careful to ensure
   * transitivity. We do that by assuming that the numbers are not too big, so
   * padding them with up to 10 zeros should be good enough.
   * @param {!Array<string>} a
   * @param {!Array<string>} b
   * @return {number} Comparison for sorting a & b.
   */
  docsegsSorter(a, b) {
    if (a[0] < b[0]) return -1;
    if (a[0] > b[0]) return 1;
    const seg1 = this.cmpDocSegId(a[1]);
    const seg2 = this.cmpDocSegId(b[1]);
    if (seg1 < seg2) return -1;
    if (seg1 > seg2) return 1;
    return 0;
  }

  /**
   * From a stats object that's keyed on doc and then on segs, extracts all
   * [doc, seg] pairs into an array, sorts the array, and returns it.
   * @param {!Object} stats
   * @return {!Array}
   */
  getDocSegs(stats) {
    const segs = [];
    for (let doc in stats) {
      const docstats = stats[doc];
      for (let docSeg in docstats) {
        segs.push([doc, docSeg]);
      }
    }
    return segs.sort(this.docsegsSorter.bind(this));
  }

  /**
   * Makes a convenient key that captures a doc name and a docSegId.
   * @param {string} doc
   * @param {string|number} seg
   * @return {string}
   */
  docsegKey(doc, seg) {
    return doc + ':' + seg;
  }

  /**
   * Returns a pruned copy of the input array, removing the faux 'raters' named
   * with a 'not:' prefix.
   * @param {!Array<string>} comparables
   * @return {!Array<string>}
   */
  excludeRaterComplements(comparables) {
    const pruned = [];
    for (const item of comparables) {
      if (!item.startsWith(this.NOT_PREFIX)) {
        pruned.push(item);
      }
    }
    return pruned;
  }

  /**
   * Creates the "system vs system" or "rater vs rater" plots comparing the
   * pairs for all available metrics. This sets up the menus for selecting the
   * systems/raters, creates skeletal tables, and then calls this.showCmp() to
   * populate the tables.
   *
   * @param {string} sysOrRater 'sys' or 'rater'
   */
  createCmpPlots(sysOrRater) {
    const versus = `${sysOrRater}-v-${sysOrRater}`;
    const plots = document.getElementById(`marot-${versus}-plots`);
    const metricIds = (sysOrRater == 'sys') ? this.metricsVisible : [0];
    plots.innerHTML = '';
    for (const m of metricIds) {
      const metric = this.metrics[m];
      const html = `
      <p id="marot-${versus}-${m}">
        <b>${metric}</b><br>
        <table>
          <tr>
          <td colspan="2">
            <svg class="marot-${versus}-plot" zoomAndPan="disable"
                id="marot-${versus}-plot-${m}">
            </svg>
          </td>
          </tr>
          <tr style="vertical-align:bottom">
          <td>
            <svg class="marot-${versus}-plot" zoomAndPan="disable"
                id="marot-${sysOrRater}1-plot-${m}">
            </svg>
          </td>
          <td>
            <svg class="marot-${versus}-plot" zoomAndPan="disable"
                id="marot-${sysOrRater}2-plot-${m}">
            </svg>
          </td>
          </tr>
        </table>
      </p>`;
      plots.insertAdjacentHTML('beforeend', html);
    }

    /** Populate menu choices. */
    const select1 = document.getElementById(`marot-${versus}-1`);
    select1.innerHTML = '';
    const select2 = document.getElementById(`marot-${versus}-2`);
    select2.innerHTML = '';
    const stats = sysOrRater == 'sys' ? this.statsBySystem : this.statsByRater;

    const comparables = this.excludeRaterComplements(Object.keys(stats));
    const indexOfTotal = comparables.indexOf(this.TOTAL);
    if (indexOfTotal >= 0) {
      comparables.splice(indexOfTotal, 1);
    }
    const key1 = sysOrRater + '1';
    const key2 = sysOrRater + '2';
    /**
     * If possible, use the previously set values.
     */
    if (this.comparisons[key1] &&
        !stats.hasOwnProperty(this.comparisons[key1])) {
      this.comparisons[key1] = '';
    }
    if (this.comparisons[key2] &&
        !stats.hasOwnProperty(this.comparisons[key2]) &&
        (sysOrRater == 'sys' || this.comparisons[key2] != this.NOT_PREFIX)) {
      this.comparisons[key2] = '';
    }
    for (const [index, sr] of comparables.entries()) {
      if (!this.comparisons[key1]) {
        this.comparisons[key1] = sr;
      }
      if (!this.comparisons[key2] &&
          (sr != this.comparisons[key1] || (index == comparables.length - 1))) {
        this.comparisons[key2] = sr;
      }
      const option1 = document.createElement('option');
      option1.value = sr;
      option1.innerHTML = sr;
      if (sr == this.comparisons[key1]) {
        option1.selected = true;
      }
      select1.insertAdjacentElement('beforeend', option1);
      const option2 = document.createElement('option');
      option2.value = sr;
      option2.innerHTML = sr;
      if (sr == this.comparisons[key2]) {
        option2.selected = true;
      }
      select2.insertAdjacentElement('beforeend', option2);
    }
    if (sysOrRater == 'rater') {
      const option2 = document.createElement('option');
      option2.value = this.NOT_PREFIX;
      option2.innerHTML = 'All other raters';
      if (this.NOT_PREFIX == this.comparisons[key2]) {
        option2.selected = true;
      }
      select2.insertAdjacentElement('beforeend', option2);
    }
    this.showCmp(sysOrRater);
  }

  /**
   * Shows the system v system histograms of segment score differences.
   * @param {string} sysOrRater 'sys' or 'rater'
   */
  showCmp(sysOrRater) {
    const versus = `${sysOrRater}-v-${sysOrRater}`;
    const select1 = document.getElementById(`marot-${versus}-1`);
    const select2 = document.getElementById(`marot-${versus}-2`);
    const sr1 = select1.value;
    let sr2 = select2.value;
    if (sysOrRater == 'rater' && sr2 == this.NOT_PREFIX) {
      sr2 = this.NOT_PREFIX + sr1;
    }
    const stats = sysOrRater == 'sys' ? this.statsBySystem : this.statsByRater;
    const docsegs1 = this.getDocSegs(stats[sr1] || {});
    const docsegs2 = this.getDocSegs(stats[sr2] || {});
    /**
     * Find common segments.
     */
    let i1 = 0;
    let i2 = 0;
    const docsegs12 = [];
    while (i1 < docsegs1.length && i2 < docsegs2.length) {
      const ds1 = docsegs1[i1];
      const ds2 = docsegs2[i2];
      const sort = this.docsegsSorter(ds1, ds2);
      if (sort < 0) {
        i1++;
      } else if (sort > 0) {
        i2++;
      } else {
        docsegs12.push(ds1);
        i1++;
        i2++;
      }
    }
    document.getElementById(`marot-${versus}-xsegs`).innerHTML =
        docsegs12.length;
    document.getElementById(`marot-${versus}-1-segs`).innerHTML =
        docsegs1.length;
    document.getElementById(`marot-${versus}-2-segs`).innerHTML =
        docsegs2.length;

    const sameSR = sr1 == sr2;

    const metricIds = (sysOrRater == 'sys') ? this.metricsVisible : [0];

    for (const m of metricIds) {
      const metricKey = 'metric-' + m;
      /**
       * We draw up to 3 plots for a metric: sr-1, sr-2, and their diff.
       */
      const hists = [
        {
          docsegs: docsegs1,
          hide: !sr1,
          sr: sr1,
          color: 'lightgreen',
          srCmp: '',
          colorCmp: '',
          id: `marot-${sysOrRater}1-plot-` + m,
        },
        {
          docsegs: docsegs2,
          hide: sameSR,
          sr: sr2,
          color: 'lightblue',
          srCmp: '',
          colorCmp: '',
          id: `marot-${sysOrRater}2-plot-` + m,
        },
        {
          docsegs: docsegs12,
          hide: sameSR,
          sr: sr1,
          color: 'lightgreen',
          srCmp: sr2,
          colorCmp: 'lightblue',
          id: `marot-${versus}-plot-` + m,
        },
      ];
      for (const hist of hists) {
        const histElt = document.getElementById(hist.id);
        histElt.style.display = hist.hide ? 'none' : '';
        if (hist.hide) {
          continue;
        }
        const histBuilder = new MarotHistogram(sysOrRater,
                                               m, hist.sr, hist.color,
                                               hist.srCmp, hist.colorCmp);
        for (let i = 0; i < hist.docsegs.length; i++) {
          const doc = hist.docsegs[i][0];
          const docSegId = hist.docsegs[i][1];
          const aggregate1 = this.aggregateSegStats(
              [stats[hist.sr][doc][docSegId]]);
          if (!aggregate1.hasOwnProperty(metricKey)) {
            continue;
          }
          let score = aggregate1[metricKey];
          if (hist.srCmp) {
            const aggregate2 = this.aggregateSegStats(
                [stats[hist.srCmp][doc][docSegId]]);
            if (!aggregate2.hasOwnProperty(metricKey)) {
              continue;
            }
            score -= aggregate2[metricKey];
          }
          histBuilder.addSegment(doc, docSegId, score);
        }
        histBuilder.display(histElt);
      }
    }
  }

  /**
   * Shows details of severity- and category-wise scores (from the
   * this.sevcatStats object) in the categories table.
   */
  showSevCatStats() {
    const stats = this.sevcatStats;
    const systems = {};
    for (let severity in stats) {
      for (let category in stats[severity]) {
        for (let system in stats[severity][category]) {
          if (!systems[system]) systems[system] = 0;
          systems[system] += stats[severity][category][system];
        }
      }
    }
    const systemsList = Object.keys(systems);
    const colspan = systemsList.length || 1;
    const th = document.getElementById('marot-sevcat-stats-th');
    th.colSpan = colspan;

    systemsList.sort((sys1, sys2) => systems[sys2] - systems[sys1]);

    let rowHTML = '<tr><td></td><td></td><td></td>';
    for (const system of systemsList) {
      rowHTML += `<td><b>${system == this.TOTAL ? 'Total' : system}</b></td>`;
    }
    rowHTML += '</tr>\n';
    this.sevcatStatsTable.insertAdjacentHTML('beforeend', rowHTML);

    const sevKeys = Object.keys(stats);
    sevKeys.sort();
    for (const severity of sevKeys) {
      this.sevcatStatsTable.insertAdjacentHTML(
          'beforeend', `<tr><td colspan="${3 + colspan}"><hr></td></tr>`);
      const sevStats = stats[severity];
      const catKeys = Object.keys(sevStats);
      catKeys.sort(
          (k1, k2) => sevStats[k2][this.TOTAL] - sevStats[k1][this.TOTAL]);
      for (const category of catKeys) {
        const row = sevStats[category];
        let rowHTML = `<tr><td>${severity}</td><td>${category}</td><td></td>`;
        for (const system of systemsList) {
          const val = row.hasOwnProperty(system) ? row[system] : '';
          rowHTML += `<td>${val ? val : ''}</td>`;
        }
        rowHTML += '</tr>\n';
        this.sevcatStatsTable.insertAdjacentHTML('beforeend', rowHTML);
      }
    }
  }

  /**
   * Shows UI event counts and timespans.
   */
  showEventTimespans() {
    const sortedEvents = [];
    for (const e of Object.keys(this.events.aggregates)) {
      const event = {
        'name': e,
      };
      const eventInfo = this.events.aggregates[e];
      event.count = eventInfo.count;
      if (e.indexOf('visited-or-redrawn') >= 0) {
        /** Deprecated event for which timespan did not make sense. */
        event.avgTimeMS = '';
      } else {
        event.avgTimeMS = eventInfo.timeMS / eventInfo.count;
      }
      sortedEvents.push(event);
    }
    sortedEvents.sort((e1, e2) => {
      const t1 = e1.avgTimeMS || 0;
      const t2 = e2.avgTimeMS || 0;
      return t2 - t1;
    });
    for (const event of sortedEvents) {
      let rowHTML = '<tr>';
      rowHTML += '<td>' + event.name + '</td>';
      rowHTML += '<td>' + event.count + '</td>';
      let t = event.avgTimeMS;
      if (t) {
        t = Math.round(t);
      }
      rowHTML += '<td>' + t + '</td>';
      rowHTML += '</tr>\n';
      this.eventsTable.insertAdjacentHTML('beforeend', rowHTML);
    }
  }

  /**
   * Make the timeline for the currently selected rater visible, hiding others.
   */
  raterTimelineSelect() {
    const raterIndex = document.getElementById(
        'marot-rater-timelines-rater').value;
    const tbodyId = `marot-rater-timeline-${raterIndex}`;
    const table = document.getElementById('marot-rater-timelines');
    const tbodies = table.getElementsByTagName('tbody');
    for (let i = 0; i < tbodies.length; i++) {
      tbodies[i].style.display = (tbodies[i].id == tbodyId) ? '' : 'none';
    }
  }

  /**
   * Shows rater-wise UI event timelines.
   */
  showRaterTimelines() {
    const raters = Object.keys(this.events.raters);
    const raterSelect = document.getElementById('marot-rater-timelines-rater');
    raterSelect.innerHTML = '';
    const table = document.getElementById('marot-rater-timelines');
    const tbodies = table.getElementsByTagName('tbody');
    for (let i = 0; i < tbodies.length; i++) {
      tbodies[i].remove();
    }
    for (let i = 0; i < raters.length; i++) {
      const rater = raters[i];
      raterSelect.insertAdjacentHTML('beforeend', `
                                     <option value="${i}">${rater}</option>`);
      const tbody = document.createElement('tbody');
      tbody.setAttribute('id', `marot-rater-timeline-${i}`);
      table.appendChild(tbody);
      const log = this.events.raters[rater];
      log.sort((e1, e2) => e1.ts - e2.ts);
      let num = 0;
      for (const e of log) {
        let rowHTML = '<tr>';
        rowHTML += '<td>' + (new Date(e.ts)).toLocaleString() + '</td>';
        rowHTML += '<td>' + e.action + '</td>';
        rowHTML += '<td>' + e.doc + '</td>';
        rowHTML += '<td>' + e.system + '</td>';
        rowHTML += '<td>' + e.docSegId + '</td>';
        rowHTML += '<td>' + (e.side == 0 ? 'Source' : 'Translation') + '</td>';
        rowHTML += '<td>' +
                   (e.source_not_seen ? 'Translation' : 'Source, Translation') +
                   '</td>';
        rowHTML += '</tr>\n';
        tbody.insertAdjacentHTML('beforeend', rowHTML);
        num++;
        if (num >= this.RATER_TIMELINE_LIMIT) {
          break;
        }
      }
      tbody.style.display = 'none';
    }
    this.raterTimelineSelect();
  }

  /**
   * Shows UI event counts, timespans, and rater timelines
   */
  showEvents() {
    this.showEventTimespans();
    this.showRaterTimelines();
  }

  /**
   * This is called from show() after looping through the data. It creates
   * "complementary raters" stats. I.e., for each rater R, for each segment, it
   * creates stats for the "not:R" rater, that are the ratings from all other
   * raters.
   */
  buildRaterComplementStats() {
    const raters = Object.keys(this.statsByRater);
    for (const rater of raters) {
      const notRater = this.NOT_PREFIX + rater;
      if (!this.statsByRater.hasOwnProperty(notRater)) {
        this.statsByRater[notRater] = {};
      }
    }
    for (const rater of raters) {
      const raterStats = this.statsByRater[rater];
      for (const docsys in raterStats) {
        /**
         * Note that this.statsByRater[rater] (raterStats) is keyed by docsys
         * (which is like "doc:sys"). Each this.statsByRater[rater][docsys][seg]
         * is an array consisting of exactly one element, which contains the
         * aggregate score given by this rater to this particular segment in
         * this doc:sys.
         * 
         * When we build the complement ratings ('not:<r>') for a rater '<r>',
         * this.statsByRater['not:<r>'][docsys][seg] will be an array consisting
         * of all existing this.statsByRater[rater][docsys][seg][0] such that
         * rater != '<r>'.
         */
        const raterDocsysStats = raterStats[docsys];
        for (const seg in raterDocsysStats) {
          const raterDocsysSegStats = raterDocsysStats[seg];
          console.assert(raterDocsysSegStats.length == 1, raterDocsysSegStats);
          for (const otherRater of raters) {
            if (rater == otherRater) {
              continue;
            }
            const notOtherRater = this.NOT_PREFIX + otherRater;
            if (!this.statsByRater[notOtherRater].hasOwnProperty(docsys)) {
              this.statsByRater[notOtherRater][docsys] = {};
            }
            if (!this.statsByRater[notOtherRater][docsys].hasOwnProperty(seg)) {
              this.statsByRater[notOtherRater][docsys][seg] = [];
              this.statsByRater[notOtherRater][docsys][seg].num_source_chars =
                  raterDocsysSegStats.num_source_chars;
            }
            this.statsByRater[notOtherRater][docsys][seg].push(
                raterDocsysSegStats[0]);
          }
        }
      }
    }

  }

  /**
   * Shows all the stats.
   */
  showStats() {
    /**
     * Get aggregates for the stats by system, including the special
     * '_MAROT_TOTAL_' system (which lets us decide which score splits have
     * non-zero values and we show score columns for only those splits).
     */
    const systems = Object.keys(this.statsBySystem);
    const statsBySysAggregates = {};
    for (const system of systems) {
      const segs = this.getSegStatsAsArray(this.statsBySystem[system]);
      statsBySysAggregates[system] = this.aggregateSegStats(segs);
    }
    const overallMQMStats = statsBySysAggregates[this.TOTAL].mqmStats ?? {};
    this.mqmWeightedFields = [];
    this.mqmSliceFields = [];
    for (let key in overallMQMStats) {
      if (!overallMQMStats[key]) continue;
      if (key.startsWith(this.MQM_WEIGHTED_PREFIX)) {
        this.mqmWeightedFields.push(this.mqmKeyToName(key));
      } else if (key.startsWith(this.MQM_SLICE_PREFIX)) {
        this.mqmSliceFields.push(this.mqmKeyToName(key));
      }
    }
    this.mqmWeightedFields.sort(
        (k1, k2) => (overallMQMStats[this.MQM_WEIGHTED_PREFIX + k2] ?? 0) -
            (overallMQMStats[this.MQM_WEIGHTED_PREFIX + k1] ?? 0));
    this.mqmSliceFields.sort(
        (k1, k2) => (overallMQMStats[this.MQM_SLICE_PREFIX + k2] ?? 0) -
            (overallMQMStats[this.MQM_SLICE_PREFIX + k1] ?? 0));

    const statsByRaterAggregates = {};
    const ratersOrGroups = Object.keys(this.statsByRater);
    for (const rater of ratersOrGroups) {
      /** Build aggregates for all, including the 'not:' rater groups. */
      const segs = this.getSegStatsAsArray(this.statsByRater[rater]);
      statsByRaterAggregates[rater] = this.aggregateSegStats(segs);
    }

    const raters = this.excludeRaterComplements(ratersOrGroups);

    const indexOfTotal = systems.indexOf(this.TOTAL);
    systems.splice(indexOfTotal, 1);

    systems.sort(
        (k1, k2) => (statsBySysAggregates[k1][this.sortByField] ?? 0) -
                    (statsBySysAggregates[k2][this.sortByField] ?? 0));
    raters.sort(
        (k1, k2) => (statsByRaterAggregates[k1][this.sortByField] ?? 0) -
                    (statsByRaterAggregates[k2][this.sortByField] ?? 0));
    if (this.sortReverse) {
      systems.reverse();
      raters.reverse();
    }

    /**
     * First show the scores table header with the sorted columns from
     * this.mqmWeightedFields and this.mqmSliceFields. Then add scores rows to
     * the table: by system, and then by rater.
     */
    const haveRaters = raters.length > 0;
    this.showScoresHeader(haveRaters);
    if (systems.length > 0) {
      this.showScoresSeparator(haveRaters, 'By system');
      for (const system of systems) {
        this.showScores(system, haveRaters, this.statsBySystem[system],
                        statsBySysAggregates[system]);
      }
    }
    if (haveRaters) {
      this.showScoresSeparator(haveRaters, 'By rater');
      for (const rater of raters) {
        this.showScores(rater, haveRaters, this.statsByRater[rater],
                        statsByRaterAggregates[rater]);
      }
      this.raterRelatedSections.style.display = '';
    } else {
      this.raterRelatedSections.style.display = 'none';
    }
    this.showScoresSeparator(haveRaters);

    this.showSystemRaterStats();
    this.createCmpPlots('sys');
    this.createCmpPlots('rater');
    this.showSevCatStats();
    this.showEvents();
    this.showSigtests('sys', this.statsBySystem, statsBySysAggregates,
                      this.metricsVisible);
    this.showSigtests('rater', this.statsByRater, statsByRaterAggregates, [0]);
    this.startSigtests();
  }

  /**
   * Increments the counts statsArray[severity][category][system] and
   *   statsArray[severity][category][this.TOTAL].
   * @param {!Object} statsArray
   * @param {string} system
   * @param {string} category
   * @param {string} severity
   */
  addSevCatStats(statsArray, system, category, severity) {
    if (!statsArray.hasOwnProperty(severity)) {
      statsArray[severity] = {};
    }
    if (!statsArray[severity].hasOwnProperty(category)) {
      statsArray[severity][category] = {};
      statsArray[severity][category][this.TOTAL] = 0;
    }
    if (!statsArray[severity][category].hasOwnProperty(system)) {
      statsArray[severity][category][system] = 0;
    }
    statsArray[severity][category][this.TOTAL]++;
    statsArray[severity][category][system]++;
  }

  /**
   * Returns total time spent, across various timing events in metadata.timing.
   * @param {!Object} metadata
   * @return {number}
   */
  timeSpent(metadata) {
    let timeSpentMS = 0;
    if (metadata.timing) {
      for (let e in metadata.timing) {
        timeSpentMS += metadata.timing[e].timeMS;
      }
    }
    if (metadata.deleted_errors) {
      for (const deletedError of metadata.deleted_errors) {
        timeSpentMS += this.timeSpent(deletedError.metadata);
      }
    }
    return timeSpentMS;
  }

  /**
   * Adds UI events and timings from metadata into events.
   * @param {!Object} events
   * @param {!Object} metadata
   * @param {string} doc
   * @param {string} docSegId
   * @param {string} system
   * @param {string} rater
   */
  addEvents(events, metadata, doc, docSegId, system, rater) {
    if (metadata.deleted_errors) {
      for (const deletedError of metadata.deleted_errors) {
        this.addEvents(
            events, deletedError.metadata, doc, docSegId, system, rater);
      }
    }
    if (!metadata.timing) {
      return;
    }
    for (const e of Object.keys(metadata.timing)) {
      if (!events.aggregates.hasOwnProperty(e)) {
        events.aggregates[e] = {
          count: 0,
          timeMS: 0,
        };
      }
      events.aggregates[e].count += metadata.timing[e].count;
      events.aggregates[e].timeMS += metadata.timing[e].timeMS;
      if (!events.raters.hasOwnProperty(rater)) {
        events.raters[rater] = [];
      }
      const log = metadata.timing[e].log ?? [];
      for (const detail of log) {
        events.raters[rater].push({
          ...detail,
          action: e,
          doc: doc,
          system: system,
          docSegId: docSegId,
        });
      }
    }
  }

  /**
   * Updates stats with an error of (category, severity). The weighted score
   * component to use is the first matching one in this.mqmWeights[]. Similarly,
   * the slice to attribute the score to is the first matching one in
   * this.mqmSlices[].
   *
   * @param {!Object} stats
   * @param {number} timeSpentMS
   * @param {string} category
   * @param {string} severity
   * @param {number} span
   */
  addErrorStats(stats, timeSpentMS, category, severity, span) {
    stats.timeSpentMS += timeSpentMS;

    const lcat = category.toLowerCase().trim();
    if (lcat == 'no-error' || lcat == 'no_error') {
      return;
    }

    const lsev = severity.toLowerCase().trim();
    if (lsev == 'hotw-test' || lsev == 'hotw_test') {
      if (lcat == 'found') {
        stats.hotwFound++;
      } else if (lcat == 'missed') {
        stats.hotwMissed++;
      }
      return;
    }
    if (lsev == 'unrateable') {
      stats.unrateable++;
      return;
    }
    if (lsev == 'neutral') {
      return;
    }

    if (span > 0) {
      /* There is a scoreable error span.  */
      stats.numWithErrors++;
      stats.errorSpans += span;
    }

    let score = 0;
    for (const sc of this.mqmWeights) {
      if (this.matchesMQMSplit(sc, lsev, lcat)) {
        score = sc.weight;
        stats.score += score;
        const key = this.mqmKey(sc.name);
        stats[key] = (stats[key] ?? 0) + score;
        break;
      }
    }
    if (score > 0) {
      for (const sc of this.mqmSlices) {
        if (this.matchesMQMSplit(sc, lsev, lcat)) {
          const key = this.mqmKey(sc.name, true);
          stats[key] = (stats[key] ?? 0) + score;
          break;
        }
      }
    }
  }

  /**
   * Returns the last element of array a.
   * @param {!Array<T>} a
   * @return {T}
   * @template T
   */
  arrayLast(a) {
    return a[a.length - 1];
  }

  /**
   * Returns the segment stats keyed by doc and docSegId. This will
   * create an empty array if the associated segment stats array doesn't exist.
   * @param {!Object} statsByDocAndDocSegId
   * @param {string} doc
   * @param {string} docSegId
   * @return {!Array}
   */
  getSegStats(statsByDocAndDocSegId, doc, docSegId) {
    if (!statsByDocAndDocSegId.hasOwnProperty(doc)) {
      statsByDocAndDocSegId[doc] = {};
    }
    if (!statsByDocAndDocSegId[doc].hasOwnProperty(docSegId)) {
      statsByDocAndDocSegId[doc][docSegId] = [];
    }
    return statsByDocAndDocSegId[doc][docSegId];
  }

  /**
   * Flattens the nested stats object into an array of segment stats.
   * @param {!Object} statsByDocAndDocSegId
   * @return {!Array}
   */
  getSegStatsAsArray(statsByDocAndDocSegId) {
    let arr = [];
    for (const doc of Object.keys(statsByDocAndDocSegId)) {
      let statsByDocSegId = statsByDocAndDocSegId[doc];
      for (const docSegId of Object.keys(statsByDocSegId)) {
        arr.push(statsByDocSegId[docSegId]);
      }
    }
    return arr;
  }

  /**
   * Shows up or down arrow to show which field is used for sorting.
   */
  showSortArrow() {
    // Remove existing active arrows first.
    const active = document.querySelector('.marot-arrow-active');
    if (active) {
      active.classList.remove('marot-arrow-active');
    }
    // Highlight the appropriate arrow for the sorting field.
    const className = this.sortReverse ? 'marot-arrow-down' : 'marot-arrow-up';
    const arrow = document.querySelector(
      `#marot-${this.sortByField}-th .${className}`);
    if (arrow) {
      arrow.classList.add('marot-arrow-active');
    }
  }

  /**
   * Scoops out the text in the tokens identified by the ranges in spanBounds.
   * Each range (usually there is just one) is a pair of inclusive indices,
   * [start, end].
   * @param {!Array<string>} tokens
   * @param {!Array<?Array<number>>} spanBounds
   * @return {string}
   */
  getSpan(tokens, spanBounds) {
    const parts = [];
    for (const bound of spanBounds) {
      const part = tokens.slice(bound[0], bound[1] + 1).join('');
      if (part) parts.push(part);
    }
    return parts.join('...');
  }

  /**
   * From a segment with spans marked using <v>..</v>, scoops out and returns
   * just the marked spans. This is the fallback for finding the span to
   * display, for legacy data where detailed tokenization info may not be
   * available.
   * @param {string} text
   * @return {string}
   */
  getLegacySpan(text) {
    const tokens = text.split(/<v>|<\/v>/);
    const oddOnes = [];
    for (let i = 1; i < tokens.length; i += 2) {
      oddOnes.push(tokens[i]);
    }
    return oddOnes.join('...');
  }

  /**
   * Returns a CSS class name suitable for displaying an error with the given
   * severity level.
   * @param {string} severity
   * @return {string}
   */
  mqmSeverityClass(severity) {
    let cls = 'mqm-neutral';
    severity = severity.toLowerCase();
    if (severity == 'major' ||
        severity.startsWith('non-translation') ||
        severity.startsWith('non_translation')) {
      cls = 'mqm-major';
    } else if (severity == 'minor') {
      cls = 'mqm-minor';
    } else if (severity == 'trivial') {
      cls = 'mqm-trivial';
    } else if (severity == 'critical') {
      cls = 'mqm-critical';
    }
    return cls;
  }

  /**
   * For the annotation defined in metadata, (for row rowId in this.data), sets
   * metadata.marked_text as the text that has been marked by the rater (or
   * sets it to the empty string). The rowId is only used for legacy formats
   * where tokenization is not available in metadata.
   * @param {number} rowId
   * @param {!Object} metadata
   */
  setMarkedText(rowId, metadata) {
    let sourceSpan = this.getSpan(metadata.segment.source_tokens,
                                  metadata.source_spans || []);
    if (!sourceSpan) {
      const source = this.data[rowId][this.DATA_COL_SOURCE];
      sourceSpan = this.getLegacySpan(source);
    }
    let targetSpan = this.getSpan(metadata.segment.target_tokens,
                                  metadata.target_spans || []);
    if (!targetSpan) {
      const target = this.data[rowId][this.DATA_COL_TARGET];
      targetSpan = this.getLegacySpan(target);
    }
    metadata.marked_text = sourceSpan + targetSpan;
  }

  /**
   * Wrap a marked span in an HTML span with the given class.
   * @param {string} spanText
   * @param {string} cls
   * @return {string}
   */
  spanHTML(spanText, cls) {
    const s = spanText.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return '<span class="' + cls + '">[' + s + ']</span>';
  }

  /**
   * For the given severity level, return an HTML string suitable for displaying
   * it, including an identifier that includes rowId (for creating a filter upon
   * clicking).
   * @param {number} rowId
   * @param {string} severity
   * @return {string}
   */
  severityHTML(rowId, severity) {
    let html = '';
    html += `<span class="marot-val"
               id="marot-val-${rowId}-${this.DATA_COL_SEVERITY}">` +
            severity + '</span>';
    return html;
  }

  /**
   * For the given annotation category, return an HTML string suitable for
   * displaying it, including an identifier that includes rowId (for creating a
   * filter upon clicking).
   * @param {number} rowId
   * @param {string} category
   * @return {string}
   */
  categoryHTML(rowId, category) {
    let html = '';
    html += `<span class="marot-val"
               id="marot-val-${rowId}-${this.DATA_COL_CATEGORY}">` +
            category + '</span>';
    return html;
  }

  /**
   * For the given rater name/id, return an HTML string suitable for displaying
   * it, including an identifier that includes rowId (for creating a filter upon
   * clicking).
   * @param {number} rowId
   * @param {string} rater
   * @return {string}
   */
  raterHTML(rowId, rater) {
    return `<span class="marot-val"
               id="marot-val-${rowId}-${this.DATA_COL_RATER}">` +
            rater + '</span>';
  }

  /**
   * Render in HTML a "Raw Error," which is the format used for storing
   * deleted errors and prior-rater's original errors in Marot metadata.
   * This format uses fields named "selected", "severity", "type", "subtype".
   * @param {!Object} e
   * @return {string}
   */
  rawErrorHTML(e) {
    let cat = e.type;
    let subcat = e.subtype;
    if (this.dataIter.evaluation && this.dataIter.evaluation.config) {
      const config = this.dataIter.evaluation.config;
      if (config.errors && config.errors.hasOwnProperty(cat)) {
        const configError = config.errors[cat];
        cat = configError.display ?? cat;
        if (subcat && configError.hasOwnProperty('subtypes') &&
            configError.subtypes.hasOwnProperty(subcat)) {
          subcat = configError.subtypes[subcat].display ?? subcat;
        }
      }
    }
    let html = this.spanHTML(e.selected, this.mqmSeverityClass(e.severity));
    html += '<br>' + this.casedSeverity(e.severity) +
            '&nbsp;' + cat + (subcat ? '/' + subcat : '');
    html += this.metadataHTML(e.metadata);
    return html;
  }

  /**
   * Render the metadata (timestamp, note, feedback, prior_rater, etc.)
   * associated with a rating.
   * @param {!Object} metadata
   * @return {string}
   */
  metadataHTML(metadata) {
    let html = '';
    if (metadata.hotw_error) {
      html += '<br>HOTW error: <span class="marot-note">' +
              metadata.hotw_error + '</span>';
      if (metadata.hotw_type) {
        html += '[' + metadata.hotw_type + ']';
      }
      html += '\n';
    }
    if (metadata.note) {
      /* There is a note from the rater */
      html += '<br>Note: <span class="marot-note">' + metadata.note +
              '</span>\n';
    }
    if (metadata.timestamp) {
      /* There is a timestamp, but it might have been stringified */
      const timestamp = parseInt(metadata.timestamp, 10);
      html += '<br><span class="marot-rater-details">' +
              (new Date(timestamp)).toLocaleString() + '</span>\n';
    }
    if (metadata.feedback) {
      /* There might be feedback */
      const feedback = metadata.feedback;
      const thumbs = feedback.thumbs || '';
      const notes = feedback.notes || '';
      let feedbackHTML = '';
      if (thumbs || notes) {
        feedbackHTML = '<br>Feedback:';
      }
      if (thumbs == 'up') {
        feedbackHTML += ' &#x1F44D;';
      } else if (thumbs == 'down') {
        feedbackHTML += ' &#x1F44E;';
      }
      if (notes) {
        feedbackHTML += '<br><span class="marot-note">' + notes + '</span>\n';
      }
      html += feedbackHTML;
    }
    if (metadata.prior_rater) {
      html += '<br><span">Prior rater: ' + metadata.prior_rater + '</span>\n';
    }
    if (metadata.prior_error) {
      html += '<br><details><summary>Annotation from prior rater:</summary>' +
              '<div class="marot-prior-error">' +
                this.rawErrorHTML(metadata.prior_error) +
              '</div></details>\n';
    }
    if (metadata.deleted_errors && metadata.deleted_errors.length > 0) {
      html += '<br><details class="marot-deleted"><summary>';
      html += metadata.deleted_errors.length + ' deleted error(s)</summary>';
      html += '<table class="marot-table">';
      for (let x = 0; x < metadata.deleted_errors.length; x++) {
        const de = metadata.deleted_errors[x];
        html += '<tr><td><span class="marot-deleted-index">' +
                (x + 1) + '.</span></td><td>';
        html += this.rawErrorHTML(de);
        html += '</td></tr>';
      }
      html += '</table></details>\n';
    }
    return html;
  }

  /**
   * Returns the "metrics line" to display for the current segment, which
   * includes MQM score as well as any available automated metrics.
   * @param {!Object} currSegStatsBySys
   * @return {string}
   */
  getSegScoresHTML(currSegStatsBySys) {
    const segScoresParts = [];
    const metrics = currSegStatsBySys.metrics;
    for (let metric in metrics) {
      segScoresParts.push([metric, metrics[metric]]);
      if (metric == 'MQM') {
        /** metrics.MQM is the unfiltered value; recompute with filtering */
        const aggregate = this.aggregateSegStats([currSegStatsBySys]);
        const score = aggregate.mqmStats.score;
        if (score != metrics[metric]) {
          segScoresParts.push(['MQM-filtered', score]);
        }
      }
    }
    if (segScoresParts.length == 0) {
      return '';
    }
    let scoresRows = '';
    for (const part of segScoresParts) {
      scoresRows += '<tr><td>' + part[0] + ':&nbsp;</td>' +
      `<td><span class="marot-seg-score">${this.metricDisplay(part[1], 1)}
           </span></td></tr>`;
    }
    return '<tr><td><table class="marot-scores-table">' +
           scoresRows +
           '</table></td></tr>\n';
  }

  /**
   * Returns HTML from tokens and sentence-splits. This involves wrapping
   * the joined text from the tokens within a <p>..</p> tag. In addition
   * at the end of each sentence, if the sentenceSplit object indicates
   * that there is a paragraph break, then a new paragraph is initiated, while
   * if it indicates that there is a line break, then a <br> is inserted.
   * @param {!Array<string>} tokens
   * @param {!Array<!Object>} sentenceSplits
   * @return {string}
   */
  htmlFromTokens(tokens, sentenceSplits) {
    let html = '<p>\n';
    let nextToken = 0;
    for (const split of sentenceSplits) {
      const end = nextToken + split.num_tokens;
      html += tokens.slice(nextToken, end).join('');
      nextToken = end;
      if (split.ends_with_para_break) {
        html += '</p>\n<p>\n';
      } else if (split.ends_with_line_break) {
        html += '<br>\n';
      }
    }
    html += tokens.slice(nextToken).join('');
    html += '</p>\n';
    return html;
  }

  /**
   * Updates the display to show the segment data and scores according to the
   * current filters.
   */
  show() {
    // Cancel existing Sigtest computation when a new `this.show` is called.
    this.resetSigtests();

    this.statsTable.innerHTML = '';
    this.sevcatStatsTable.innerHTML = '';
    this.eventsTable.innerHTML = '';

    this.statsBySystem = {};
    this.statsBySystem[this.TOTAL] = {};
    this.statsByRaterSystem = {};
    this.statsByRater = {};
    this.sevcatStats = {};

    this.selectedRows.clear();

    this.events = {
      aggregates: {},
      raters: {},
    };
    const visibleMetrics = {};
    this.metricsVisible = [];

    document.getElementById('marot-filter-expr-error').innerHTML = '';
    const allFilters = this.getAllFilters();

    let currSegStats = [];
    let currSegStatsBySys = [];
    let currSegStatsByRater = [];
    let currSegStatsByRaterSys = [];

    document.body.style.cursor = 'wait';
    for (const doc of this.dataIter.docs) {
      for (const docSegId of this.dataIter.docSegs[doc]) {
        for (const system of this.dataIter.docSys[doc]) {
          let firstRowId = -1;
          let numSrcChars = 0;
          let segmentMetadata = null;
          let lastRater = '';
          const range = this.dataIter.docSegSys[doc][docSegId][system].rows;
          const docColonSys = doc + ':' + system;
          const priorRaterStats = {};  /** prior_rater -> RaterStats */
          for (let rowId = range[0]; rowId < range[1]; rowId++) {
            const parts = this.data[rowId];
            numSrcChars = parts.num_source_chars;
            let match = true;
            for (let id in allFilters.filterREs) {
              const col = this.filterColumns[id];
              if (allFilters.filterREs[id] &&
                  !allFilters.filterREs[id].test(parts[col])) {
                match = false;
                break;
              }
            }
            if (!match) {
              continue;
            }
            if (!this.filterExprPasses(allFilters.filterExpr, parts)) {
              continue;
            }
            const metadata = parts[this.DATA_COL_METADATA];
            if (allFilters.onlyAllSysSegs &&
                !this.allSystemsFilterPasses(metadata)) {
              continue;
            }

            this.selectedRows.add(rowId);
            const rater = parts[this.DATA_COL_RATER];
            const category = parts[this.DATA_COL_CATEGORY];
            const severity = parts[this.DATA_COL_SEVERITY];

            if (firstRowId < 0) {
              firstRowId = rowId;

              segmentMetadata = metadata.segment;

              currSegStats = this.getSegStats(
                  this.statsBySystem[this.TOTAL], docColonSys, docSegId);
              if (!this.statsBySystem.hasOwnProperty(system)) {
                this.statsBySystem[system] = {};
              }
              currSegStatsBySys =
                  this.getSegStats(this.statsBySystem[system], doc, docSegId);
              currSegStats.num_source_chars = numSrcChars;
              currSegStatsBySys.num_source_chars = numSrcChars;
              if (segmentMetadata.hasOwnProperty('metrics')) {
                currSegStatsBySys.metrics = segmentMetadata.metrics;
                for (let metric in currSegStatsBySys.metrics) {
                  visibleMetrics[metric] = true;
                }
              }
            }

            if (rater && (rater != lastRater)) {
              lastRater = rater;
              visibleMetrics['MQM'] = true;  /** We do have some MQM scores. */

              currSegStats.push(this.initRaterStats(rater));
              currSegStatsBySys.push(this.initRaterStats(rater));
              if (!this.statsByRater.hasOwnProperty(rater)) {
                this.statsByRater[rater] = {};
              }
              currSegStatsByRater = this.getSegStats(
                  this.statsByRater[rater], docColonSys, docSegId);
              currSegStatsByRater.push(this.initRaterStats(rater));
              currSegStatsByRater.num_source_chars = numSrcChars;

              if (!this.statsByRaterSystem.hasOwnProperty(rater)) {
                this.statsByRaterSystem[rater] = {};
              }
              if (!this.statsByRaterSystem[rater].hasOwnProperty(system)) {
                this.statsByRaterSystem[rater][system] = {};
              }
              currSegStatsByRaterSys = this.getSegStats(
                  this.statsByRaterSystem[rater][system], doc, docSegId);
              currSegStatsByRaterSys.push(this.initRaterStats(rater));
              currSegStatsByRaterSys.num_source_chars = numSrcChars;
            }
            if (rater) {
              /** An actual rater-annotation row, not just a metadata row */
              const span = metadata.marked_text.length;
              const timeSpentMS = this.timeSpent(metadata);
              this.addErrorStats(this.arrayLast(currSegStats),
                                 timeSpentMS, category, severity, span);
              this.addErrorStats(this.arrayLast(currSegStatsBySys),
                                 timeSpentMS, category, severity, span);
              this.addErrorStats(this.arrayLast(currSegStatsByRater),
                                 timeSpentMS, category, severity, span);
              this.addErrorStats(this.arrayLast(currSegStatsByRaterSys),
                                 timeSpentMS, category, severity, span);
              this.addSevCatStats(this.sevcatStats, system, category, severity);
              this.addEvents(this.events, metadata,
                             doc, docSegId, system, rater);
              if (metadata.prior_rater) {
                const priorRater = this.PRIOR_RATER_PREFIX +
                                   metadata.prior_rater;
                if (!priorRaterStats.hasOwnProperty(priorRater)) {
                  priorRaterStats[priorRater] = this.initRaterStats(priorRater);
                }
                const priorErrors = [];
                if (metadata.prior_error) {
                  priorErrors.push(metadata.prior_error);
                }
                for (const e of (metadata.deleted_errors ?? [])) {
                  if (e.metadata &&
                      e.metadata.prior_rater == metadata.prior_rater) {
                    priorErrors.push(e);
                  }
                }
                for (const priorError of priorErrors) {
                  let cat = priorError.type;
                  if (priorError.subtype) {
                    cat += '/' + priorError.subtype;
                  }
                  const span = (priorError.selected ?? '').length;
                  this.addErrorStats(priorRaterStats[priorRater],
                                     this.timeSpent(priorError.metadata),
                                     cat, priorError.severity, span);
                }
              }
            }
          }
          for (const priorRater in priorRaterStats) {
            /**
             * Note that we accumulate prior-rater stats *only* in statsByRater,
             * to compare against other raters. We do not add it to
             * statsBySystem, as that would be double-counting.
             */
            if (!this.statsByRater.hasOwnProperty(priorRater)) {
              this.statsByRater[priorRater] = {};
            }
            currSegStatsByRater = this.getSegStats(
              this.statsByRater[priorRater], docColonSys, docSegId);
            currSegStatsByRater.num_source_chars = numSrcChars;
            currSegStatsByRater.push(priorRaterStats[priorRater]);
          }
        }
      }
    }
    this.buildRaterComplementStats();

    /**
     * Update #unfiltered rows display.
     */
    document.getElementById('marot-num-rows').innerText = this.data.length;
    document.getElementById('marot-num-unfiltered-rows').innerText =
        this.selectedRows.size;

    for (let m = 0; m < this.metrics.length; m++) {
      const metric = this.metrics[m];
      if (visibleMetrics[metric]) {
        this.metricsVisible.push(m);
      }
    }
    if (this.sortByField.startsWith('metric-')) {
      /**
       * If the currently chosen sort-by field is a metric that is not visible,
       * then change it to be the first metric that *is* visible (if any,
       * defaulting to metric-0, which is marot). Set the default direction
       * based upon whether lower numbers are better for the chosen metric.
       */
      let sortingMetric = parseInt(this.sortByField.substr(7));
      if (!this.metricsVisible.includes(sortingMetric)) {
        sortingMetric = 0;
        for (let m = 0; m < this.metrics.length; m++) {
          const metric = this.metrics[m];
          if (visibleMetrics[metric]) {
            sortingMetric = m;
            break;
          }
        }
        this.sortByField = 'metric-' + sortingMetric;
        this.sortReverse =
            this.metricsInfo[this.metrics[sortingMetric]].lowerBetter ?
            false : true;
      }
    }
    this.showStats();
    this.showSegments();
    if (this.data.length > 0) {
      this.showViewer();
    }
    document.body.style.cursor = 'auto';
  }

  /**
   * Shows the segment data according to the current filters.
   * @param {?Object=} viewingConstraints Optional dict of doc:seg or
   *     doc:sys:seg to view. When not null, only these segments are shown.
   *     When not null, this parameter object should have two additional
   *     properties:
   *       description: Shown to the user, describing the constrained view.
   *       color: A useful identifying color that highlights the description.
   */
  showSegments(viewingConstraints=null) {
    this.segmentsTable.innerHTML = '';
    const viewingConstraintsDesc = document.getElementById(
        'marot-viewing-constraints');
    if (this.histRectSelected) {
      this.histRectSelected.classList.remove('marot-histogram-selected');
      this.histRectSelected = null;
    }
    if (viewingConstraints) {
      viewingConstraintsDesc.innerHTML = 'View limited to ' +
          viewingConstraints.description +
          ' Click on this text to remove this constraint.';
      viewingConstraintsDesc.style.backgroundColor = viewingConstraints.color;
      viewingConstraintsDesc.style.display = '';
      this.histRectSelected = viewingConstraints.rect;
      this.histRectSelected.classList.add('marot-histogram-selected');
    } else {
      viewingConstraintsDesc.innerHTML = '';
      viewingConstraintsDesc.style.display = 'none';
    }

    let shownCount = 0;
    const shownRows = [];

    for (const doc of this.dataIter.docs) {
      for (const docSegId of this.dataIter.docSegs[doc]) {
        let shownForDocSeg = 0;
        let aggrDocSeg = null;
        for (const system of this.dataIter.docSys[doc]) {
          let shownForDocSegSys = 0;
          let aggrDocSegSys = null;
          let firstRowId = -1;
          let ratingRowsHTML = '';
          let segmentMetadata = null;
          let sourceTokens = null;
          let targetTokens = null;
          let lastRater = '';
          const range = this.dataIter.docSegSys[doc][docSegId][system].rows;
          const docColonSys = doc + ':' + system;
          for (let rowId = range[0]; rowId < range[1]; rowId++) {
            if (!this.selectedRows.has(rowId)) {
              continue;
            }
            const parts = this.data[rowId];
            const metadata = parts[this.DATA_COL_METADATA];
            const rater = parts[this.DATA_COL_RATER];
            const category = parts[this.DATA_COL_CATEGORY];
            const severity = parts[this.DATA_COL_SEVERITY];
            if (!aggrDocSeg && metadata.segment &&
                metadata.segment.aggrDocSeg) {
              aggrDocSeg = metadata.segment.aggrDocSeg;
            }
            if (!aggrDocSegSys) {
              aggrDocSegSys = metadata.segment;
            }

            if (firstRowId < 0) {
              firstRowId = rowId;

              segmentMetadata = metadata.segment;
              sourceTokens = (segmentMetadata.source_tokens || []).slice();
              targetTokens = (segmentMetadata.target_tokens || []).slice();
            }

            if (rater && (rater != lastRater)) {
              lastRater = rater;
            }
            let spanClass = '';
            if (rater) {
              /** An actual rater-annotation row, not just a metadata row */
              spanClass = this.mqmSeverityClass(severity) +
                          ` marot-anno-${shownRows.length}`;
              this.markSpans(
                  sourceTokens, metadata.source_spans || [], spanClass);
              this.markSpans(
                  targetTokens, metadata.target_spans || [], spanClass);
            }

            if (viewingConstraints &&
                !viewingConstraints[this.docsegKey(doc, docSegId)] &&
                !viewingConstraints[this.docsegKey(docColonSys, docSegId)]) {
              continue;
            }
            if (shownCount >= this.rowLimit) {
              continue;
            }

            shownRows.push(rowId);
            shownForDocSegSys++;

            if (!rater) {
              /**
               * This matching row only has segment metadata, there is no rater
               * annotation to show from this row.
               */
              continue;
            }
            ratingRowsHTML += '<tr><td><div>';
            if (metadata.marked_text) {
              ratingRowsHTML += this.spanHTML(metadata.marked_text, spanClass) +
                                '<br>';
            }
            ratingRowsHTML += this.severityHTML(rowId, severity) +
                              '&nbsp;';
            ratingRowsHTML += this.categoryHTML(rowId, category) +
                              '<br>';
            ratingRowsHTML += this.raterHTML(rowId, rater);
            ratingRowsHTML += this.metadataHTML(metadata);
            ratingRowsHTML += '</div></td></tr>\n';
          }
          if (shownForDocSegSys == 0) {
            continue;
          }
          console.assert(firstRowId >= 0, firstRowId);

          if (shownForDocSeg == 0 && aggrDocSeg &&
              aggrDocSeg.reference_tokens) {
            for (const ref of Object.keys(aggrDocSeg.reference_tokens)) {
              let refRowHTML = '<tr class="marot-row marot-ref-row">';
              refRowHTML += '<td><div>' + doc + '</div></td>';
              refRowHTML += '<td><div>' + docSegId + '</div></td>';
              refRowHTML += '<td><div><b>Ref</b>: ' + ref + '</div></td>';
              const sourceTokens = aggrDocSeg.source_tokens || [];
              refRowHTML += '<td><div>' +
                  this.htmlFromTokens(
                      sourceTokens,
                      segmentMetadata.source_sentence_splits ?? []) +
                  '</div></td>';
              const refTokens = aggrDocSeg.reference_tokens[ref] ?? [];
              const refSplits = aggrDocSeg.reference_sentence_splits[ref] ?? [];
              refRowHTML += '<td><div>' +
                            this.htmlFromTokens(refTokens, refSplits) +
                            '</div></td>';
              refRowHTML += '<td></td></tr>\n';
              this.segmentsTable.insertAdjacentHTML('beforeend', refRowHTML);
            }
          }
          let rowHTML = '';
          rowHTML += '<td><div class="marot-val" ';
          rowHTML += `id="marot-val-${firstRowId}-${this.DATA_COL_DOC}">` +
                     doc + '</div></td>';
          rowHTML += '<td><div class="marot-val" ';
          rowHTML +=
              `id="marot-val-${firstRowId}-${this.DATA_COL_DOC_SEG_ID}">` +
              docSegId + '</div></td>';
          rowHTML += '<td><div class="marot-val" ';
          rowHTML += `id="marot-val-${firstRowId}-${this.DATA_COL_SYSTEM}">` +
                     system + '</div></td>';

          const source = sourceTokens.length > 0 ?
              this.htmlFromTokens(
                  sourceTokens, segmentMetadata.source_sentence_splits ?? []) :
              this.data[firstRowId][this.DATA_COL_SOURCE].replace(
                  /<\/?v>/g, '');
          const target = targetTokens.length > 0 ?
              this.htmlFromTokens(
                  targetTokens, segmentMetadata.target_sentence_splits ?? []) :
              this.data[firstRowId][this.DATA_COL_TARGET].replace(
                  /<\/?v>/g, '');

          rowHTML += '<td><div>' + source + '</div></td>';
          rowHTML += '<td><div>' + target + '</div></td>';

          const currSegStatsBySys =
              this.getSegStats(this.statsBySystem[system], doc, docSegId);
          rowHTML += '<td><table class="marot-table-ratings">' +
                     ratingRowsHTML + this.getSegScoresHTML(currSegStatsBySys) +
                     '</table></td>';

          this.segmentsTable.insertAdjacentHTML(
              'beforeend', `<tr class="marot-row">${rowHTML}</tr>\n`);
          shownForDocSeg += shownForDocSegSys;
        }
        if (shownForDocSeg > 0) {
          shownCount += shownForDocSeg;
        }
      }
    }
    /**
     * Add cross-highlighting listeners.
     */
    const annoHighlighter = (a, shouldShow) => {
      const elts = document.getElementsByClassName('marot-anno-' + a);
      const fontWeight = shouldShow ? 'bold' : 'inherit';
      const border = shouldShow ? '1px solid blue' : 'none';
      for (let i = 0; i < elts.length; i++) {
        const style = elts[i].style;
        style.fontWeight = fontWeight;
        style.borderTop = border;
        style.borderBottom = border;
      }
    };
    for (let a = 0; a < shownRows.length; a++) {
      const elts = document.getElementsByClassName('marot-anno-' + a);
      if (elts.length == 0) continue;
      const onHover = (e) => {
        annoHighlighter(a, true);
      };
      const onNonHover = (e) => {
        annoHighlighter(a, false);
      };
      for (let i = 0; i < elts.length; i++) {
        elts[i].addEventListener('mouseover', onHover);
        elts[i].addEventListener('mouseout', onNonHover);
      }
    }

    /**
     * Add filter listeners.
     */
    const filters = document.getElementsByClassName('marot-filter-re');
    for (const rowId of shownRows) {
      const parts = this.data[rowId];
      for (let i = 0; i < filters.length; i++) {
        const filter = filters[i];
        const col = this.filterColumns[filter.id];
        const v = document.getElementById(`marot-val-${rowId}-${col}`);
        if (!v) continue;
        v.addEventListener('click', (e) => {
          filter.value = '^' + parts[col] + '$';
          this.show();
        });
      }
    }
  }

  /**
   * Recomputes MQM score for each segment (using current weight settings) and
   * sets it in segment.metrics['MQM'].
   */
  recomputeMQM() {
    const statsBySystem = {};
    let currSegStatsBySys = [];
    for (const doc of this.dataIter.docs) {
      for (const docSegId of this.dataIter.docSegs[doc]) {
        for (const system of this.dataIter.docSys[doc]) {
          let lastRater = '';
          const range = this.dataIter.docSegSys[doc][docSegId][system].rows;
          let aggrDocSegSys = null;
          for (let rowId = range[0]; rowId < range[1]; rowId++) {
            const parts = this.data[rowId];
            const metadata = parts[this.DATA_COL_METADATA];
            if (!aggrDocSegSys) {
              aggrDocSegSys = metadata.segment;
              if (!statsBySystem.hasOwnProperty(system)) {
                statsBySystem[system] = {};
              }
              currSegStatsBySys =
                  this.getSegStats(statsBySystem[system], doc, docSegId);
              currSegStatsBySys.num_source_chars = parts.num_source_chars;
            }
            const rater = parts[this.DATA_COL_RATER];
            if (!rater) {
              continue;
            }
            if (rater != lastRater) {
              lastRater = rater;
              currSegStatsBySys.push(this.initRaterStats(rater));
            }
            const category = parts[this.DATA_COL_CATEGORY];
            const severity = parts[this.DATA_COL_SEVERITY];
            /** We don't care about computing avg span/time here, pass as 0. */
            this.addErrorStats(this.arrayLast(currSegStatsBySys),
                               0, category, severity, 0);
          }
          if (aggrDocSegSys) {
            const aggrScores = this.aggregateSegStats([currSegStatsBySys]);
            if (aggrScores.mqmStats.numSegments > 0) {
              aggrDocSegSys.metrics['MQM'] = aggrScores.mqmStats.score;
            }
          }
        }
      }
    }
  }

  /**
   * Wraps tokens within ranges specified in each bounds entry, in HTML
   * spans with the specified class.
   * @param {!Array<string>} tokens
   * @param {!Array<!Array<number>>} bounds
   * @param {string} cls
   */
  markSpans(tokens, bounds, cls) {
    for (const bound of bounds) {
      for (let i = bound[0]; i <= bound[1]; i++) {
        if (i < 0 || i >= tokens.length) {
          continue;
        }
        tokens[i] = '<span class="' + cls + '">' + tokens[i] + '</span>';
      }
    }
  }

  /**
   * Clears all filters, except possibly 'marot-only-all-systems-segments'.
   * @param {boolean=} resetOnlyAllSys
   */
  clearFilters(resetOnlyAllSys=false) {
    const filters = document.getElementsByClassName('marot-filter-re');
    for (let i = 0; i < filters.length; i++) {
      filters[i].value = '';
    }
    document.getElementById('marot-filter-expr').value = '';
    document.getElementById('marot-filter-expr-error').innerHTML = '';
    if (resetOnlyAllSys) {
      document.getElementById('marot-only-all-systems-segments').checked =
          false;
    }
  }

  /**
   * Clears all filters and shows stats again.
   */
  clearFiltersAndShow() {
    this.clearFilters(true);
    this.show();
  }

  /**
   * For the column named by "what", sets the filter to the currently picked
   * value from its drop-down list.
   * @param {string} what
   */
  pickFilter(what) {
    const filter = document.getElementById('marot-filter-' + what);
    if (!filter) return;
    const sel = document.getElementById('marot-select-' + what);
    if (!sel) return;
    filter.value = sel.value;
    this.show();
  }

  /**
   * Populates the column drop-down lists and filter-expression builder with
   * unique values.
   */
  setSelectOptions() {
    const options = {};
    for (let id in this.filterColumns) {
      options[id] = {};
    }
    for (const parts of this.data) {
      for (let id in this.filterColumns) {
        const col = this.filterColumns[id];
        if (col == this.DATA_COL_SOURCE || col == this.DATA_COL_TARGET) {
          continue;
        }
        options[id][parts[col].trim()] = true;
      }
    }
    for (let id in this.filterColumns) {
      const selectId = id.replace(/filter/, 'select');
      const sel = document.getElementById(selectId);
      if (!sel) continue;
      const opt = options[id];
      let html = '<option value=""></option>\n';
      for (let o in opt) {
        if (!o) continue;
        html += `<option value="^${o}$">${o}</option>\n`;
      }
      sel.innerHTML = html;
    }

    /**
     * Populate filter clause builder's selects:
     */
    this.filterClauseKey = document.getElementById('marot-clause-key');
    let html = '<option value=""></option>\n';
    const SYSTEM_FILTER_ID = 'marot-filter-system';
    for (let sys in options[SYSTEM_FILTER_ID]) {
      html += `<option value="System: ${sys}">System: ${sys}</option>\n`;
    }
    const RATER_FILTER_ID = 'marot-filter-rater';
    for (let rater in options[RATER_FILTER_ID]) {
      html += `<option value="Rater: ${rater}">Rater: ${rater}</option>\n`;
    }
    this.filterClauseKey.innerHTML = html;

    this.filterClauseInclExcl =
        document.getElementById('marot-clause-inclexcl');

    this.filterClauseCat = document.getElementById('marot-clause-cat');
    html = '<option value=""></option>\n';
    const CATEGORY_FILTER_ID = 'marot-filter-category';
    for (let cat in options[CATEGORY_FILTER_ID]) {
      html += `<option value="${cat}">${cat}</option>\n`;
    }
    this.filterClauseCat.innerHTML = html;

    this.filterClauseSev = document.getElementById('marot-clause-sev');
    html = '<option value=""></option>\n';
    const SEVERITY_FILTER_ID = 'marot-filter-severity';
    for (let sev in options[SEVERITY_FILTER_ID]) {
      html += `<option value="${sev}">${sev}</option>\n`;
    }
    this.filterClauseSev.innerHTML = html;

    this.filterClauseAddAnd = document.getElementById('marot-clause-add-and');
    this.filterClauseAddOr = document.getElementById('marot-clause-add-or');
    this.clearClause();
  }

  /**
   * This resets information derived from or associated with the current data
   * (if any), preparing for new data.
   */
  resetData() {
    this.clearFilters();
    this.data = [];
    this.metrics = ['MQM'];
    for (let key in this.metricsInfo) {
      /** Only retain the entry for 'MQM'. */
      if (key == 'MQM') continue;
      delete this.metricsInfo[key];
    }
    this.metricsVisible = [];
    this.sortByField = 'metric-0';
    this.sortReverse = false;
    this.closeMenuEntries('');
  }

  /**
   * Make the first letter of a severity level uppercase.
   * @param {string} sev
   * @return {string}
   */
  casedSeverity(sev) {
    return sev.charAt(0).toUpperCase() + sev.substr(1);
  }

  /**
   * Sets this.tsvData from the passed TSV data string or array of strings, and
   * parses it into this.data. If the UI option marot-load-file-append is
   * checked, then the new data is appended to the existing data, else it
   * replaces it.
   * @param {string|!Array<string>} tsvData
   */
  setData(tsvData) {
    const errors = document.getElementById('marot-errors');
    errors.innerHTML = '';
    if (Array.isArray(tsvData)) {
      let allTsvData = '';
      for (const tsvDataItem of tsvData) {
        if (!tsvDataItem) continue;
        if (allTsvData && !allTsvData.endsWith('\n')) {
          allTsvData += '\n';
        }
        allTsvData += tsvDataItem;
      }
      tsvData = allTsvData;
    }
    if (!tsvData) {
      errors.innerHTML = 'Empty data passed to this.setData()';
      return;
    }
    if (document.getElementById('marot-load-file-append').checked) {
      if (this.tsvData && !this.tsvData.endsWith('\n')) {
        this.tsvData += '\n';
      }
    } else {
      this.tsvData = '';
    }
    this.tsvData += tsvData;

    this.resetData();
    const data = this.tsvData.split('\n');
    for (let line of data) {
      if (this.data.length >= this.MAX_DATA_LINES) {
        errors.insertAdjacentHTML('beforeend',
            'Skipping data lines beyond number ' + this.MAX_DATA_LINES);
        break;
      }
      line = line.trim();
      if (!line) {
        continue;
      }
      if (line.toLowerCase().indexOf('system\tdoc\t') >= 0) {
        /**
         * Skip header line. It may be present anywhere, as we may be looking
         * at data concatenated from multiple files.
         */
        continue;
      }
      const parts = line.split('\t');

      let metadata = {};
      if (parts.length < this.DATA_COL_METADATA) {
        errors.insertAdjacentHTML('beforeend',
            `Could not parse: ${line.substr(0, 80)}...<br>`);
        continue;
      } else if (parts.length == this.DATA_COL_METADATA) {
        /** TSV data is missing the last metadata column. Create it. */
        parts.push(metadata);
      } else {
        /**
         * The 10th column should be a JSON-encoded "metadata" object. Prior to
         * May 2022, the 10th column, when present, was just a string that was a
         * "note" from the rater, so convert that to a metadata object if
         * needed.
         */
        try {
          metadata = JSON.parse(parts[this.DATA_COL_METADATA]);
        } catch (err) {
          console.log(err);
          console.log(parts[this.DATA_COL_METADATA]);
          metadata = {};
          const note = parts[this.DATA_COL_METADATA].trim();
          if (note) {
            metadata['note'] = note;
          }
        }
        parts[this.DATA_COL_METADATA] = metadata;
      }
      /**
       * Make sure metadata has the keys for object members, so that they
       * can be used in filter expressions freely.
       */
      if (!metadata.segment) {
        metadata.segment = {};
      }
      if (!metadata.segment.references) {
        metadata.segment.references = {};
      }
      /**
       * Convert legacy format for sentence-splits info.
       */
      if (metadata.segment.hasOwnProperty('source_sentence_tokens') &&
          !metadata.segment.hasOwnProperty('source_sentence_splits')) {
        metadata.segment.source_sentence_splits = [];
        for (const num_tokens of metadata.segment.source_sentence_tokens) {
          metadata.segment.source_sentence_splits.push({
            num_tokens: num_tokens
          });
        }
      }
      if (metadata.segment.hasOwnProperty('target_sentence_tokens') &&
          !metadata.segment.hasOwnProperty('target_sentence_splits')) {
        metadata.segment.target_sentence_splits = [];
        for (const num_tokens of metadata.segment.target_sentence_tokens) {
          metadata.segment.target_sentence_splits.push({
            num_tokens: num_tokens
          });
        }
      }
      if (!metadata.segment.metrics) {
        metadata.segment.metrics = {};
      }
      if (!metadata.feedback) {
        metadata.feedback = {};
      }
      if (metadata.evaluation) {
        /* Show the evaluation metadata in the log. */
        console.log('Evaluation info found in row ' + this.data.length + ':');
        console.log(metadata.evaluation);
      }
      if (parts[this.DATA_COL_SEVERITY].toLowerCase().trim().startsWith(
              'hotw')) {
        /** HOTW error used to be copied into the "note" field. */
        if (!metadata.hotw_error && metadata.note) {
          metadata.hotw_error = metadata.note;
          metadata.note = '';
        }
      }
      /** Note any metrics that might be in the data. */
      const metrics = metadata.segment.metrics;
      for (let metric in metrics) {
        if (this.metricsInfo.hasOwnProperty(metric)) continue;
        this.metricsInfo[metric] = {
          index: this.metrics.length,
        };
        this.metrics.push(metric);
      }
      /** Move "Rater" down from its position in the TSV data. */
      const temp = parts[4];
      parts[this.DATA_COL_SOURCE] = parts[5];
      parts[this.DATA_COL_TARGET] = parts[6];
      parts[this.DATA_COL_RATER] = temp;
      /** Make severity start with an uppercase letter. */
      parts[this.DATA_COL_SEVERITY] =
          this.casedSeverity(parts[this.DATA_COL_SEVERITY]);
      /**
       * Count all characters, including spaces, in src/tgt length, excluding
       * the span-marking <v> and </v> tags.
       */
      parts.num_source_chars =
          parts[this.DATA_COL_SOURCE].replace(/<\/?v>/g, '').length;
      parts.num_target_chars =
          parts[this.DATA_COL_TARGET].replace(/<\/?v>/g, '').length;
      this.data.push(parts);
    }
    this.sortData(this.data);
    this.createDataIter(this.data);
    this.recomputeMQM();
    this.addSegmentAggregations();
    this.setSelectOptions();
    this.show();
  }

  /**
   * Opens and reads the data file(s) picked by the user and calls setData().
   */
  openFiles() {
    this.hideViewer();
    this.clearFilters();
    const errors = document.getElementById('marot-errors');
    errors.innerHTML = '';
    const filesElt = document.getElementById('marot-file');
    const numFiles = filesElt.files.length;
    if (numFiles <= 0) {
      document.body.style.cursor = 'auto';
      errors.innerHTML = 'No files were selected';
      return;
    }
    let erroneousFile = '';
    try {
      const filesData = [];
      const fileNames = [];
      let filesRead = 0;
      for (let i = 0; i < numFiles; i++) {
        filesData.push('');
        const f = filesElt.files[i];
        fileNames.push(f.name);
        erroneousFile = f.name;
        const fr = new FileReader();
        fr.onload = (evt) => {
          erroneousFile = f.name;
          filesData[i] = fr.result;
          filesRead++;
          if (filesRead == numFiles) {
            if (typeof marotDataConverter == 'function') {
              for (let i = 0; i < filesData.length; i++) {
                filesData[i] = marotDataConverter(fileNames[i], filesData[i]);
              }
            }
            this.setData(filesData);
          }
        };
        fr.readAsText(f);
      }
      /**
       * Set the file field to empty so that re-picking the same file *will*
       * actually reload it.
       */
      filesElt.value = '';
    } catch (err) {
      let errString = err +
          (errnoeousFile ? ' (file with error: ' + erroneousFile + ')' : '');
      errors.innerHTML = errString;
      filesElt.value = '';
    }
  }

  /**
   * Fetches marot data from the given URLs and calls this.setData().
   * If the marotURLMaker() function exists, then it is applied to each URL
   * first, to get a possibly modified URL.
   * @param {!Array<string>} urls
   */
  fetchURLs(urls) {
    const errors = document.getElementById('marot-errors');
    errors.innerHTML = 'Loading metrics data from ' + urls.length +
                       ' URL(s)...';
    this.hideViewer();
    const cleanURLs = [];
    for (let url of urls) {
      if (typeof marotURLMaker == 'function') {
        url = marotURLMaker(url);
      }
      const trimmedUrl = url.trim();
      if (trimmedUrl) cleanURLs.push(trimmedUrl);
    }
    if (cleanURLs.length == 0) {
      errors.innerHTML = 'No non-empty URLs found';
      return;
    }
    let numResponses = 0;
    const tsvData = new Array(cleanURLs.length);
    const finisher = () => {
      if (numResponses == cleanURLs.length) {
        if (typeof marotDataConverter == 'function') {
          for (let i = 0; i < tsvData.length; i++) {
            tsvData[i] = marotDataConverter(cleanURLs[i], tsvData[i]);
          }
        }
        this.setData(tsvData);
      }
    };
    for (let i = 0; i < cleanURLs.length; i++) {
      const url = cleanURLs[i];
      fetch(url, {
        mode: 'cors',
        credentials: 'include',
      })
          .then(response => response.text())
          .then(result => {
            tsvData[i] = result;
            numResponses++;
            finisher();
          })
          .catch(error => {
            errors.insertAdjacentHTML('beforeend', error);
            console.log(error);
            numResponses++;
            finisher();
          });
    }
  }

  /**
   * Returns the currently filtered marot data in TSV format. The filtered
   * data is available in the this.dataFiltered array. This function reorders
   * the columns (from "source, target, rater" to "rater, source, target")
   * before splicing them into TSV format.
   * @return {string}
   */
  getFilteredTSVData() {
    let tsvData = '';
    let lastDocSegSys = '';
    for (const rowId of this.selectedRows) {
      const row = this.data[rowId];
      const tsvOrderedRow = this.data[rowId].slice(0, this.DATA_COL_NUM_PARTS);

      /** Move "Rater" up from its position in row[] */
      tsvOrderedRow[4] = row[this.DATA_COL_RATER];
      tsvOrderedRow[5] = row[this.DATA_COL_SOURCE];
      tsvOrderedRow[6] = row[this.DATA_COL_TARGET];

      /**
       * Copy metadata, as we will clear out unnecessary/bulky fields and will
       * then JSON-encode it.
       */
      const metadata = {...row[this.DATA_COL_METADATA]};

      delete metadata.evaluation;

      const docSegSys = JSON.stringify([row[this.DATA_COL_DOC],
                                        row[this.DATA_COL_DOC_SEG_ID],
                                        row[this.DATA_COL_SYSTEM]]);
      if (lastDocSegSys != docSegSys) {
        /** This is the first row for the current doc+seg+sys. */
        lastDocSegSys = docSegSys;
        /** Retain segment metadata, but delete aggregated docseg info. */
        metadata.segment = {...metadata.segment};
        delete metadata.segment.aggrDocSeg;
      } else {
        /** Delete segment metadata, already previously included. */
        delete metadata.segment;
      }

      tsvOrderedRow[this.DATA_COL_METADATA] = JSON.stringify(metadata);

      tsvData += tsvOrderedRow.join('\t') + '\n';
    }
    return tsvData;
  }

  /**
   * Returns currently filtered scores data aggregated as specified, in TSV
   * format, with aggregation-dependent fields as follows.
   *     aggregation='system': system, score.
   *     aggregation='document': system, doc, score.
   *     aggregation='segment': system, doc, docSegId, score.
   *     aggregation='rater': system, doc, docSegId, rater, score.
   * @param {string} aggregation Should be one of:
   *     'rater', 'segment', 'document', 'system'.
   * @return {string}
   */
  getScoresTSVData(aggregation) {
    /**
     * We use a fake 10-column marot-data array (with score kept in the last
     * column) to sort the data in the right order using this.sortData().
     */
    const data = [];
    const FAKE_FIELD = '--marot-FAKE-FIELD--';
    if (aggregation == 'system') {
      for (let system in this.statsBySystem) {
        if (system == this.TOTAL) {
          continue;
        }
        const segs = this.getSegStatsAsArray(this.statsBySystem[system]);
        const aggregate = this.aggregateSegStats(segs);
        const dataRow = Array(this.DATA_COL_NUM_PARTS).fill(FAKE_FIELD);
        dataRow[this.DATA_COL_SYSTEM] = system;
        dataRow[this.DATA_COL_METADATA] = aggregate.mqmStats.score;
        data.push(dataRow);
      }
    } else if (aggregation == 'document') {
      for (let system in this.statsBySystem) {
        if (system == this.TOTAL) {
          continue;
        }
        const stats = this.statsBySystem[system];
        for (let doc in stats) {
          const docStats = stats[doc];
          const segs = this.getSegStatsAsArray({doc: docStats});
          const aggregate = this.aggregateSegStats(segs);
          const dataRow = Array(this.DATA_COL_NUM_PARTS).fill(FAKE_FIELD);
          dataRow[this.DATA_COL_SYSTEM] = system;
          dataRow[this.DATA_COL_DOC] = doc;
          dataRow[this.DATA_COL_METADATA] = aggregate.mqmStats.score;
          data.push(dataRow);
        }
      }
    } else if (aggregation == 'segment') {
      for (let system in this.statsBySystem) {
        if (system == this.TOTAL) {
          continue;
        }
        const stats = this.statsBySystem[system];
        for (let doc in stats) {
          const docStats = stats[doc];
          for (let seg in docStats) {
            const docSegStats = docStats[seg];
            const segs = this.getSegStatsAsArray({doc: {seg: docSegStats}});
            const aggregate = this.aggregateSegStats(segs);
            const dataRow = Array(this.DATA_COL_NUM_PARTS).fill(FAKE_FIELD);
            dataRow[this.DATA_COL_SYSTEM] = system;
            dataRow[this.DATA_COL_DOC] = doc;
            dataRow[this.DATA_COL_DOC_SEG_ID] = seg;
            dataRow[this.DATA_COL_METADATA] = aggregate.mqmStats.score;
            data.push(dataRow);
          }
        }
      }
    } else /* (aggregation == 'rater') */ {
      for (let rater in this.statsByRaterSystem) {
        for (let system in this.statsByRaterSystem[rater]) {
          const stats = this.statsByRaterSystem[rater][system];
          for (let doc in stats) {
            const docStats = stats[doc];
            for (let seg in docStats) {
              const docSegStats = docStats[seg];
              const segs = this.getSegStatsAsArray({doc: {seg: docSegStats}});
              const aggregate = this.aggregateSegStats(segs);
              const dataRow = Array(this.DATA_COL_NUM_PARTS).fill(FAKE_FIELD);
              dataRow[this.DATA_COL_SYSTEM] = system;
              dataRow[this.DATA_COL_DOC] = doc;
              dataRow[this.DATA_COL_DOC_SEG_ID] = seg;
              dataRow[this.DATA_COL_RATER] = rater;
              dataRow[this.DATA_COL_METADATA] = aggregate.mqmStats.score;
              data.push(dataRow);
            }
          }
        }
      }
    }
    this.sortData(data);
    /** remove FAKE_FIELD columns */
    let tsvData = '';
    for (let i = 0; i < data.length; i++) {
      const trimmedRow = [];
      for (const entry of data[i]) {
        if (entry != FAKE_FIELD) {
          trimmedRow.push(entry);
        }
      }
      tsvData += trimmedRow.join('\t') + '\n';
    }
    return tsvData;
  }

  /**
   * Saves the passed data to the passed file name.
   * @param {string} tsvData
   * @param {string} fileName
   */
  saveDataInner(tsvData, fileName) {
    const a = document.createElement("a");
    a.style.display = "none";
    document.body.appendChild(a);
    a.href = window.URL.createObjectURL(
      new Blob([tsvData], {type: "text/tab-separated-values;charset=UTF-8"})
    );
    a.setAttribute("download", fileName);
    a.click();
    window.URL.revokeObjectURL(a.href);
    document.body.removeChild(a);
    this.closeMenuEntries('');
  }

  /**
   * Saves this.tsvData or filtered or filtered+aggregated data to the file
   *     marot-data.tsv. Adds a header line when saving non-aggregated marot data,
   *     if it's not already there.
   * @param {string} saveType One of 'all', 'filtered', 'system', 'document',
   *     'segment', 'rater'
   * @param {string} fileName This is appened to any prefix entered in the
   *     marot-saved-file-prefix field.
   */
  saveData(saveType, fileName) {
    let tsvData = '';
    let addHeader = true;
    if (saveType == 'all') {
      tsvData = this.tsvData;
    } else if (saveType == 'filtered') {
      tsvData = this.getFilteredTSVData();
    } else {
      tsvData = this.getScoresTSVData(saveType);
      addHeader = false;
    }
    if (!tsvData) {
      alert('There is no data to be saved!');
      return;
    }
    if (addHeader && !tsvData.startsWith('system\tdoc\t')) {
      tsvData = 'system\tdoc\tdocSegId\tglobalSegId\t' +
                'rater\tsource\ttarget\tcategory\tseverity\tmetadata\t' +
                '# Documentation: ' +
                'https://github.com/google-research/google-research/tree/m' +
                'aster/MAROT_viewer\n' + tsvData;
    }
    const prefix = document.getElementById(
        'marot-saved-file-prefix').value.trim();
    this.saveDataInner(tsvData, prefix + fileName);
  }

  /**
   * Applies updated settings for scoring.
   */
  updateSettings() {
    if (this.parseScoreSettings()) {
      this.setUpScoreSettings();
    }
    /**
     * Need to recompute metadata.segment.metrics['MQM'] for each segment first,
     * for use in subsequent filtering.
     */
    this.recomputeMQM();
    this.show();
  }

  /**
   * Resets scoring settings to their default values.
   */
  resetSettings() {
    this.mqmWeights = JSON.parse(JSON.stringify(mqmDefaultWeights));
    this.mqmSlices = JSON.parse(JSON.stringify(mqmDefaultSlices));
    this.setUpScoreSettings();
    this.updateSettings();
  }

  /**
   * Collapse all top menu zippy panels, except the one
   * with the given id.
   * @param {string=} except
   */
  closeMenuEntries(except='') {
    const menuEntries = document.getElementsByClassName('marot-menu-entry');
    for (let i = 0; i < menuEntries.length; i++) {
      const entry = menuEntries[i];
      if (entry.id != except) {
        entry.removeAttribute('open');
      }
    }
  }

  /**
   * Hides the viewer, showing only the quote by Marot (used at start-up and
   * when there is no data to show).
   */
  hideViewer() {
    this.quote.style.display = '';
    this.viewer.style.display = 'none';
  }

  /**
   * Shows the viewer, hiding the quote by Marot.
   */
  showViewer() {
    this.quote.style.display = 'none';
    this.viewer.style.display = '';
  }

  /**
   * Returns HTML for the Marot section where systems or raters are compared.
   *
   * @param {string} sysOrRater One of 'sys' or 'rater'
   * @return {string}
   */
  getSystemOrRaterSectionHTML(sysOrRater) {
    const fullName = sysOrRater == 'sys' ? 'System' : 'Rater';
    const fullNameLC = fullName.toLowerCase();
    const versus = `${sysOrRater}-v-${sysOrRater}`;
    return `
      <details>
        <summary
            title="Click to see pairwise ${sysOrRater} difference significance tests, correlations, and segment histograms">
          <span class="marot-section">
            ${fullName} score comparisons and histograms
          </span>
        </summary>
        <p><b>${fullName} score
          <select id="marot-${versus}-tables-type"
              onchange="marot.switchCmpTable('${sysOrRater}')">
            <option value="pValue"
                selected>difference significance test p-values</option>
            <option value="rho">Pearson correlations</option>
          </select>
        </b></p>
        <div id="marot-${sysOrRater}-comparison"
            class="marot-comparison-tables-pvalue">
          <div id="marot-${versus}-tables">
          </div>
          <p class="marot-pvalue">
            P-values < ${this.PVALUE_THRESHOLD} (bolded) indicate a significant
            difference.
          </p>
          <p>
            Missing values indicate a lack of enough common segments.
            ${fullName}s above any solid line
            ${sysOrRater == 'sys' ? ' are significantly better than' :
                            ' give MQM scores signinificantly lower than'}
            those below. Dotted lines identify clusters within which no
            ${fullNameLC}'s score is significantly different from any other
            ${fullNameLC}.
            <span class="marot-warning"
                id="marot-${sysOrRater}-sigtests-msg"></span>
          </p>
        </div>
        <p><b>${fullName} and ${fullNameLC}-pair segment
            score histograms</b></p>
        <div class="marot-comparison-histograms">
          <div class="marot-${versus}-header">
            <label>
              <b>${fullName} 1:</b>
              <select id="marot-${versus}-1"
                 onchange="marot.showCmp('${sysOrRater}')"></select>
            </label>
            <span id="marot-${versus}-1-segs"></span> segment(s).
            <label>
              <b>${fullName} 2:</b>
              <select id="marot-${versus}-2"
                 onchange="marot.showCmp('${sysOrRater}')"></select>
            </label>
            <span id="marot-${versus}-2-segs"></span> segment(s)
            (<span id="marot-${versus}-xsegs"></span> common).
            The Y-axis uses a log scale.
          </div>
          <div id="marot-${versus}-plots">
          </div>
        </div>
      </details>`;
  }

  /**
   * Replaces the HTML contents of elt with the HTML needed to render the
   *     Marot viewer. If tsvDataOrURLs is not null, then it can be marot TSV-data,
   *     or a CSV list of URLs from which to fetch marot TSV-data.
   * @param {!Element} elt
   * @param {string=} tsvDataOrCsvURLs
   * @param {boolean=} loadReplaces determines whether loading new data
   *     replaces the current data or augments it, by default.
   */
  init(elt, tsvDataOrCsvURLs='', loadReplaces=true) {
    const tooltip = 'Regular expressions are used case-insensitively. ' +
        'Click on the Apply button after making changes.';
    const settings = `
      <details class="marot-settings marot-menu-entry"
          id="marot-menu-entry-settings"
          title="Change scoring weights, slices, units.">
        <summary>Settings</summary>
        <div class="marot-settings-panel">
          <div>
            Number of trials for paired one-sided approximate randomization:
            <input size="6" maxlength="6" type="text"
                id="marot-sigtests-num-trials"
                value="10000" onchange="marot.setSigtestsNumTrials()"/>
          </div>
          <div class="marot-settings-row">
             Note: Changes to the following tables of MQM weights and slices
             only take effect after clicking on <b>Apply!</b>
          </div>
          <div class="marot-settings-row" title="${tooltip}">
            Ordered list of MQM <i>weights</i> to apply to error patterns:
            <button onclick="marot.settingsAddRow('marot-settings-weights', 3)"
                >Add new row</button> as row <input type="text" maxlength="2"
                class="marot-input"
                id="marot-settings-weights-add-row" size=2 placeholder="1"/>
            <table class="marot-table marot-settings-panel">
              <thead>
                <tr>
                  <th>Weight name</th>
                  <th>Regular expression to match
                      <i>severity:category[/subcategory]</i></th>
                  <th>Weight</th>
                </tr>
              </thead>
              <tbody id="marot-settings-weights">
              </tbody>
            </table>
          </div>
          <div class="marot-settings-row" title="${tooltip}">
            Ordered list of interesting <i>slices</i> of MQM error patterns:
            <button onclick="marot.settingsAddRow('marot-settings-slices', 2)"
                >Add new row</button> as row <input type="text" maxlength="2"
                class="marot-input"
                id="marot-settings-slices-add-row" size=2 placeholder="1"/>
            <table class="marot-table marot-settings-panel">
              <thead>
                <tr>
                  <th>Slice name</th>
                  <th>Regular expression to match
                      <i>severity:category[/subcategory]</i></th>
                </tr>
              </thead>
              <tbody id="marot-settings-slices">
              </tbody>
            </table>
          </div>
          <div class="marot-settings-row">
            <button id="marot-reset-settings" title="Restore default settings"
                onclick="marot.resetSettings()">Restore defaults</button>
            <button id="marot-apply-settings"
                title="Apply weight/slice settings"
                onclick="marot.updateSettings()">Apply!</button>
          </div>
        </div>
      </details>`;

    let filePanel = `
      <table class="marot-table marot-file-menu">
        <tr id="marot-file-load" class="marot-file-menu-tr"><td>
          <div>
            <b>Load</b> marot data file(s):
            <input id="marot-file"
                accept=".tsv" onchange="marot.openFiles()"
                type="file" multiple/>
          </div>
        </td></tr>
        <tr class="marot-file-menu-option marot-file-menu-tr"><td>
          <input type="checkbox" id="marot-load-file-append"/>
          Append additional data without replacing the current data
        </td></tr>
        <tr><td></td></tr>
        <tr class="marot-file-menu-entry marot-file-menu-tr"><td>
          <div onclick="marot.saveData('all', 'marot-data.tsv')"
              title="Save full 10-column marot annotations TSV data">
            <b>Save</b> all data to [prefix]marot-data.tsv
          </div>
        </td></tr>
        <tr class="marot-file-menu-entry marot-file-menu-tr"><td>
          <div onclick="marot.saveData('filtered', 'marot-data-filtered.tsv')"
          title="Save currently filtered 10-column marot annotations TSV data">
            <b>Save</b> filtered data to [prefix]marot-data-filtered.tsv
          </div>
        </td></tr>`;

    for (const saveType of ['system', 'document', 'segment', 'rater']) {
      const fname = `marot-scores-by-${saveType}.tsv`;
      filePanel += `
          <tr class="marot-file-menu-entry marot-file-menu-tr"><td>
            <div
                onclick="marot.saveData('${saveType}', '${fname}')"
                title="Save ${saveType == 'rater' ?
                  'segment-wise ' : ''}filtered scores by ${saveType}">
              <b>Save</b> filtered scores by ${saveType} to [prefix]${fname}
            </div>
          </td></tr>`;
    }
    filePanel += `
        <tr class="marot-file-menu-option marot-file-menu-tr"><td>
          Optional prefix for saved files:
          <input size="10" class="marot-input" type="text"
              id="marot-saved-file-prefix" placeholder="prefix"/>
        </td></tr>
      </table>`;

    const file = `
      <details class="marot-file marot-menu-entry" id="marot-menu-entry-file">
        <summary>File</summary>
        <div class="marot-file-panel">
          ${filePanel}
        </div>
      </details>`;

    elt.innerHTML = `
    <div class="marot-header">
      <span target="_blank" class="marot-title"><a class="marot-link"
        href="https://github.com/google-research/google-research/tree/master/marot">Marot</a></span>
      <table class="marot-menu">
        <tr>
          <td>${settings}</td>
          <td>${file}</td>
        </tr>
      </table>
    </div>
    <div id="marot-errors"></div>
    <hr>

    <div id="marot-quote">
      <p>
        ... double louange peult venir de transmuer ung transmueur
      </p>
      <p>
        —<a target="_blank" title="Clément Marot's Wikipedia page (new tab)"
             class="marot-link"
             href="https://en.wikipedia.org/wiki/Cl%C3%A9ment_Marot">Clément Marot</a>
      </p>
    </div>

    <div id="marot-viewer">

      <table class="marot-table marot-numbers-table" id="marot-stats">
        <thead id=marot-stats-thead>
        </thead>
        <tbody id="marot-stats-tbody">
        </tbody>
      </table>

      <br>

      ${this.getSystemOrRaterSectionHTML('sys')}

      <br>

      <div id="marot-rater-related-sections">

        ${this.getSystemOrRaterSectionHTML('rater')}

        <br>

        <details>
          <summary
              title="Click to see a System x Rater matrix of scores highlighting individual system-rater scores that seem out of order">
            <span class="marot-section">
              System &times; Rater scores
            </span>
          </summary>
          <table
              title="Systems and raters are sorted using total MQM score. A highlighted entry means this rater's rating of this system is contrary to the aggregate of all raters' ratings, when compared with the previous system."
              class="marot-table marot-numbers-table" id="marot-system-x-rater">
          </table>
        </details>

        <br>

        <details>
          <summary title="Click to see error severity / category counts">
            <span class="marot-section">
              Error severities and categories
            </span>
          </summary>
          <table class="marot-table" id="marot-sevcat-stats">
            <thead>
              <tr>
                <th title="Error severity"><b>Severity</b></th>
                <th title="Error category"><b>Category</b></th>
                <th> </th>
                <th id="marot-sevcat-stats-th"
                    title="Number of occurrences"><b>Count</b></th>
              </tr>
            </thead>
            <tbody id="marot-sevcat-stats-tbody">
            </tbody>
          </table>
        </details>

        <br>

        <details>
          <summary title="Click to see user interface events and timings">
            <span class="marot-section">
              Annotation events and rater timelines
            </span>
          </summary>
          <div>
            <table class="marot-table" id="marot-events">
              <thead>
                <tr>
                  <th title="User interface event"><b>Event</b></th>
                  <th title="Number of occurrences"><b>Count</b></th>
                  <th title="Average time per occurrence"><b>Avg Time
                      (millis)</b></th>
                </tr>
              </thead>
              <tbody id="marot-events-tbody">
              </tbody>
            </table>
          </div>
          <div>
            <div class="marot-subheading"
              title="The timeline is limited to events in filtered annotations">
              <b>Rater timeline for</b>
              <select onchange="marot.raterTimelineSelect()"
                  id="marot-rater-timelines-rater"></select>
              (limited to ${this.RATER_TIMELINE_LIMIT} events)
            </div>
            <table class="marot-table" id="marot-rater-timelines">
              <thead>
                <tr>
                  <th><b>Timestamp</b></th>
                  <th><b>Event</b></th>
                  <th><b>Document</b></th>
                  <th><b>System</b></th>
                  <th><b>Segment</b></th>
                  <th><b>Side</b></th>
                  <th><b>Visible</b></th>
                </tr>
              </thead>
            </table>
          </div>
        </details>

        <br>

      </div> <!-- marot-rater-related-sections -->

      <details>
        <summary
            title="Click to see advanced filtering options and documentation">
          <span class="marot-section">
            Filters
            (<span id="marot-num-unfiltered-rows">0</span> of
             <span id="marot-num-rows">0</span> rows pass filters)
            <button
              title="Clear all column filters and JavaScript filter expression"
                onclick="marot.clearFiltersAndShow()">Clear all filters</button>
          </span>
          <span>
            <input type="checkbox" checked
              onchange="marot.show()" id="marot-only-all-systems-segments"/>
              Select only the segments for which all or no systems have scores
          </span>
        </summary>

        <ul class="marot-filters">
          <li>
            You can click on any System/Doc/ID/Rater/Category/Severity (or pick
            from the drop-down list under the column name) to set its <b>column
            filter</b> to that specific value.
          </li>
          <li>
            You can provide <b>column filter</b> regular expressions for
            filtering one or more columns, in the input fields provided under
            the column names.
          </li>
          <li>
            You can create sophisticated filters (involving multiple columns,
            for example) using a <b>JavaScript filter expression</b>:
            <br>
            <input class="marot-input" id="marot-filter-expr"
        title="Provide a JavaScript boolean filter expression (and press Enter)"
                onchange="marot.show()" type="text" size="150"/>
            <div id="marot-filter-expr-error" class="marot-filter-expr-error">
            </div>
            <br>
            <ul>
              <li>This allows you to filter using any expression
                  involving the columns. It can use the following
                  variables: <b>system</b>, <b>doc</b>, <b>globalSegId</b>,
                  <b>docSegId</b>, <b>rater</b>, <b>category</b>,
                  <b>severity</b>, <b>source</b>, <b>target</b>,
                  <b>reference</b>, <b>metadata</b>.
              </li>
              <li>
                Filter expressions also have access to three aggregated objects
                named <b>aggrDocSegSys</b> (which is simply an alias for
                metadata.segment), <b>aggrDocSeg</b>, and <b>aggrDoc</b>.
                The aggrDocSegSys dict also contains aggrDocSeg (with the key
                "aggrDocSeg"), which in turn similarly contains aggrDoc.
              </li>
              <li>
                The aggregated variable named <b>aggrDocSeg</b> is an object
                with the following properties:
                <b>aggrDocSeg.catsBySystem</b>,
                <b>aggrDocSeg.catsByRater</b>,
                <b>aggrDocSeg.sevsBySystem</b>,
                <b>aggrDocSeg.sevsByRater</b>,
                <b>aggrDocSeg.sevcatsBySystem</b>,
                <b>aggrDocSeg.sevcatsByRater</b>.
                Each of these properties is an object keyed by system or rater,
                with the values being arrays of strings.
                The "sevcats*" values look like "Minor/Fluency/Punctuation" or
                are just the same as severities if categories are empty. This
                segment-level aggregation allows you to select specific segments
                rather than just specific error ratings.
              </li>
              <li>
                System-wise metrics, including MQM, are also available in
                <b>aggrDocSeg.metrics</b>, which is an object keyed by the
                metric name and then by system name.
              </li>
              <li>
                The aggregated variable named <b>aggrDoc</b> is an object
                with the following properties that are aggregates over all
                the systems:
                <b>doc</b>, <b>thumbsUpCount</b>, <b>thumbsDownCount</b>.
              </li>
              <li>
                <b>Log metadata</b> for row to JavaScript console
                (open with Ctrl-Shift-I):
                <input class="marot-input" id="marot-metadata-row"
                  title="The metadata will be logged in the JavaScript console"
                    placeholder="row #"
                    onchange="marot.logRowMetadata()" type="text" size="6"/>
                (useful for finding available fields for filter expressions).
              </li>
              <li><b>Example</b>: docSegId > 10 || severity == 'Major'</li>
              <li><b>Example</b>: target.indexOf('thethe') &gt;= 0</li>
              <li><b>Example</b>: metadata.marked_text.length &gt;= 10</li>
              <li><b>Example</b>:
                aggrDocSeg.sevsBySystem['System-42'].includes('Major')</li>
              <li><b>Example</b>:
                JSON.stringify(aggrDocSeg.sevcatsBySystem).includes('Major/Fl')
              </li>
              <li><b>Example</b>: aggrDocSegSys.metrics['MQM'] &gt; 4 &&
                (aggrDocSegSys.metrics['BLEURT-X'] ?? 1) &lt; 0.1.</li>
              <li>
                You can add segment-level filtering clauses (AND/OR) using this
                <b>helper</b> (which uses convenient shortcut functions for
                checking that a rating exists and has/does-not-have a value):
                <div>
                  <select onchange="marot.checkClause()"
                    id="marot-clause-key"></select>
                  <select onchange="marot.checkClause()"
                    id="marot-clause-inclexcl">
                    <option value="includes">has error</option>
                    <option value="excludes">does not have error</option>
                  </select>
                  <select onchange="marot.checkClause()"
                    id="marot-clause-sev"></select>
                  <select onchange="marot.checkClause()"
                    id="marot-clause-cat"></select>
                  <button onclick="marot.addClause('&&')" disabled
                    id="marot-clause-add-and">Add AND clause</button>
                  <button onclick="marot.addClause('||')" disabled
                    id="marot-clause-add-or">Add OR clause</button>
                </div>
              </li>
            </ul>
            <br>
          </li>
          <li title="Limit this to at most a few thousand to avoid OOMs!">
            <b>Limit</b> the number of rows shown to:
            <input size="6" maxlength="6" type="text" id="marot-limit"
                value="2000" onchange="marot.setShownRowsLimit()"/>
          </li>
        </ul>
      </details>

      <br>

      <span class="marot-section">Sample segments</span>
      <span id="marot-viewing-constraints" onclick="marot.show()"></span>
      <table class="marot-table" id="marot-segments-table">
        <thead id="marot-segments-thead">
          <tr id="marot-head-row">
            <th id="marot-th-doc" title="Document name">
              Doc
              <br>
              <input class="marot-input marot-filter-re" id="marot-filter-doc"
                  title="Provide a regexp to filter (and press Enter)"
                  onchange="marot.show()" type="text" placeholder=".*"
                  size="10"/>
              <br>
              <select onchange="marot.pickFilter('doc')"
                  class="marot-select" id="marot-select-doc"></select>
            </th>
            <th id="marot-th-doc-seg" title="ID of the segment
                within its document">
              DocSeg
              <br>
              <input class="marot-input marot-filter-re"
                  id="marot-filter-doc-seg"
                  title="Provide a regexp to filter (and press Enter)"
                  onchange="marot.show()" type="text" placeholder=".*"
                  size="4"/>
              <br>
              <select onchange="marot.pickFilter('doc-seg')"
                  class="marot-select" id="marot-select-doc-seg"></select>
            </th>
            <th id="marot-th-system" title="System name">
              System
              <br>
              <input class="marot-input marot-filter-re"
                  id="marot-filter-system"
                  title="Provide a regexp to filter (and press Enter)"
                  onchange="marot.show()" type="text" placeholder=".*"
                  size="10"/>
              <br>
              <select onchange="marot.pickFilter('system')"
                  class="marot-select" id="marot-select-system"></select>
            </th>
            <th id="marot-th-source" title="Source text of segment">
              Source
              <br>
              <input class="marot-input marot-filter-re"
                  id="marot-filter-source"
                  title="Provide a regexp to filter (and press Enter)"
                  onchange="marot.show()" type="text" placeholder=".*"
                  size="10"/>
            </th>
            <th id="marot-th-target" title="Translated text of segment">
              Target
              <br>
              <input class="marot-input marot-filter-re"
                  id="marot-filter-target"
                  title="Provide a regexp to filter (and press Enter)"
                  onchange="marot.show()" type="text" placeholder=".*"
                  size="10"/>
            </th>
            <th id="marot-th-rating"
                title="Annotation, Severity, Category, Rater">
              <table>
                <tr>
                  <td>
                    Severity
                    <br>
                    <input class="marot-input marot-filter-re"
                        id="marot-filter-severity"
                        title="Provide a regexp to filter (and press Enter)"
                        onchange="marot.show()" type="text" placeholder=".*"
                        size="8"/>
                    <br>
                    <select onchange="marot.pickFilter('severity')"
                        class="marot-select" id="marot-select-severity">
                    </select>
                  </td>
                  <td>
                    Category
                    <br>
                    <input class="marot-input marot-filter-re"
                        id="marot-filter-category"
                        title="Provide a regexp to filter (and press Enter)"
                        onchange="marot.show()" type="text" placeholder=".*"
                        size="8"/>
                    <br>
                    <select onchange="marot.pickFilter('category')"
                        class="marot-select" id="marot-select-category">
                    </select>
                  </td>
                  <td>
                    Rater
                    <br>
                    <input class="marot-input marot-filter-re"
                        id="marot-filter-rater"
                        title="Provide a regexp to filter (and press Enter)"
                        onchange="marot.show()" type="text" placeholder=".*"
                        size="8"/>
                    <br>
                    <select onchange="marot.pickFilter('rater')"
                        class="marot-select" id="marot-select-rater"></select>
                  </td>
                </tr>
              </table>
            </th>
          </tr>
        </thead>
        <tbody id="marot-segments-tbody">
        </tbody>
      </table>
    </div>  <!-- marot-viewer -->
    `;
    elt.className = 'marot';
    elt.scrollIntoView();

    document.getElementById('marot-load-file-append').checked = !loadReplaces;

    const menuEntries = document.getElementsByClassName('marot-menu-entry');
    for (let i = 0; i < menuEntries.length; i++) {
      const entry = menuEntries[i];
      entry.addEventListener('click', (e) => {
        this.closeMenuEntries(entry.id);
      });
    }

    this.sigtestsSysMsg = document.getElementById('marot-sys-sigtests-msg');
    this.sigtestsRaterMsg = document.getElementById('marot-rater-sigtests-msg');

    this.quote = document.getElementById('marot-quote');
    this.viewer = document.getElementById('marot-viewer');
    this.segmentsTable = document.getElementById('marot-segments-tbody');
    this.statsTable = document.getElementById('marot-stats-tbody');
    this.sevcatStatsTable = document.getElementById('marot-sevcat-stats-tbody');
    this.eventsTable = document.getElementById('marot-events-tbody');
    this.raterRelatedSections = document.getElementById(
        'marot-rater-related-sections');

    this.resetSettings();

    this.hideViewer();

    if (tsvDataOrCsvURLs) {
      if (tsvDataOrCsvURLs.indexOf('\t') >= 0) {
        this.setData(tsvDataOrCsvURLs);
      } else {
        this.fetchURLs(tsvDataOrCsvURLs.split(','));
      }
    }
  }
}

/**
 * The global Marot object that encapsulates the tool. The application needs
 * to call marot.init(...).
 */
const marot = new Marot();
