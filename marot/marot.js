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
      *     containing score and numScoringUnits.
      */
    this.scores = {};
    /**
     * {!Object} Scoring units' scores by system/rater. Each value is an array
     *     of scores that are aligned such that elements at the n-th position of
     *     all arrays correspond to the same scoring unit. Note that some scores
     *     might be null since some systems/raters might be missing scores for
     *     some scoring units.
     */
    this.unitScores = {};
    /**
     * {!Object} Common scoring unit indices shared by a pair of systems/raters.
     *     This stores positions in unitScores.
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
     * Structures used for using aligned subparas for navigation in the examples
     * table and for creating smaller scoring units.
     */
    this.subparas = {
      /**
       * Keyed by hash(doc + ":" + seg), and then by
       * "src" / "sys-" + hash(system_name) / "ref-" + hash(reference_name).
       */
      alignmentStructs: {},
      /**
       * classMap[cls] is the Set of all other span classes that should be
       * highlighted when looking for alignment with the span in class "cls".
       */
      classMap: {},
      /**
       * Class of the "pinned" span of text (if any) when viewing alignments.
       * Clicking pins a span, and then arrow keys move the pinning up or down
       * within the segment.
       */
      pinnedSubparaCls: '',
    };

    this.subparaScoring = false;
    this.DEFAULT_SUBPARA_SENTS = 1;
    this.DEFAULT_SUBPARA_TOKENS = 100;
    this.subparaSents = this.DEFAULT_SUBPARA_SENTS;
    this.subparaTokens = this.DEFAULT_SUBPARA_TOKENS;

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
     * error counts) for one scoring unit.
     *
     *   statsBySystem[system][doc][unit]
     *   statsByRaterSystem[rater][system][doc][unit]
     *   statsByRater[rater][doc][unit]
     *      stats has a special system value ('MAROT_TOTAL') that aggregates
     *      ratings for all systems, by using faux "doc" keys that look like
     *      "[doc]:[system]". The "doc" keys in statsByRater also look like
     *      "[doc]:[system]".
     *
     * statsByRaterSystem and statsByRater treat AutoMQM metrics as the MQM
     * metric (so that we can do sigtest comparisons, ordering differences,
     * etc.)
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
     * into this.metrics[]), or 'metric-<k>-' + a name from
     * this.mqmWeightedFields[] or this.mqmSliceFields[] (for MQM and AutoMQM
     * metrics).
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
     * Subset of metricsVisible that are the MQM and AutoMQM metrics.
     */
    this.mqmMetricsVisible = [];
    /**
     * Set of MQM metric indices for which the details are toggled visible.
     */
    this.mqmDetailsShown = null;

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
     * Maximum number of segments of data that we'll consume. Human eval data is
     * generally of modest size, but automated metrics data can be arbitrary
     * large. Users should limit and curate such data. The way this limit is
     * implemented is that the first MAX_SEGMENTS distinct {doc, seg} are
     * loaded. Data for these segments continues to be loaded even from file
     * lines read after the limit is hit.
     */
    this.MAX_SEGMENTS = 10000;
    this.docsegs = new Set;
    this.tooManySegsErrorShown = false;
  }

  /**
   * Return true only for MQM-like metric names: i.e., if the metric name is MQM
   * or starts with AutoMQM.
   * @param {string} metric
   * @return {boolean}
   */
  isMQMOrAutoMQM(metric) {
    const lm = metric.toLowerCase();
    return lm.startsWith('mqm') || lm.startsWith('automqm');
  }

  /**
   * Convert a rater name to its MQM-like metric name. If the rater name begins
   * with 'automqm' (case-insensitively) and treatAutoMQMAsMQM is false, then
   * the rater name itself is returned as the metric name. Otherwise 'MQM' is
   * returned.
   * @param {string} rater
   * @param {boolean=} treatAutoMQMAsMQM Set to true if AutoMQM is to be
   *     conflated with MQM (set only for comparing per-rater scores, including
   *     AutoMQM).
   * @return {string}
   */
  mqmMetricName(rater, treatAutoMQMAsMQM=false) {
    if (!treatAutoMQMAsMQM) {
      const lr = rater.toLowerCase();
      if (lr.startsWith('automqm')) {
        return 'AutoMQM' + rater.substr(7);
      }
    }
    return 'MQM';
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
   * This function returns a "comparable" version of unit by padding it
   * with leading zeros. When unit is a non-negative "docSegId" integer
   * (reasonably bounded), then this ensures numeric ordering. For subparas as
   * scoring units, unit looks like <docSegId>.<p>, where <p> is the 4-digit
   * representation of the source subpara index. For such subpara units too,
   * padding to a fixed with leading zeros ensures the right ordering.
   *
   * @param {string} s
   * @return {string}
   */
  cmpDocUnitId(s) {
    return ('' + s).padStart(12, '0');
  }

  /**
   * This sorts 10-column Marot data by fields in the order doc, docSegId,
   *   system, rater, severity, category.
   * @param {!Array<!Array>} data The Marot-10-column data to be sorted.
   */
  sortData(data) {
    data.sort((e1, e2) => {
      let diff = 0;
      const docSegId1 = this.cmpDocUnitId(e1[this.DATA_COL_DOC_SEG_ID]);
      const docSegId2 = this.cmpDocUnitId(e2[this.DATA_COL_DOC_SEG_ID]);
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
      raterDetails: {},
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
    /** Assign each rater a distinctive color and HTML-safe key */
    for (const rater of this.dataIter.raters) {
      const raterKey = MarotUtils.javaHashKey(rater);
      const raterIntHash = parseInt(raterKey, 36);
      /** Make a random dark color using 8 hash bits each for R,G,B */
      const r = ((raterIntHash & 0xFF) >> 0);
      const g = ((raterIntHash & 0xFF00) >> 8);
      const b = ((raterIntHash & 0xFF0000) >> 16);
      const raterColor = `rgb(${r},${g},${b})`;
      this.dataIter.raterDetails[rater] = {
        key: raterKey,
        color: raterColor,
      };
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
   * Given an array of all instances of annotated text for a segment (where
   * annotations have been marked using <v>..</v> spans), generates a
   * tokenization that starts with space-based splitting, but refines it to
   * ensure that each <v> and </v> is at a token boundary. Returns the
   * tokenization as well as an array containing the marked spans encoded as
   * [start, end] token indices (both inclusive).
   *
   * The structure of the returned object is: {
   *   tokens: !Array<string>
   *   spans: !Array<Pair<number, number>>
   *   sentence_splits: !Array<!Object>
   * }
   * @param {!Array<string>} annotations
   * @return {!Object}
   */
  tokenizeLegacyAnnotations(annotations) {
    let cleanText = '';
    for (const text of annotations) {
      const noMarkers = text.replace(/<\/?v>/g, '');
      if (noMarkers.length > cleanText.length) {
        cleanText = noMarkers;
      }
    }
    const tokenization = MarotUtils.tokenizeText(cleanText);
    const tokens = tokenization.tokens;
    const sents = tokenization.sentence_splits;
    const tokenOffsets = [];
    let tokenOffset = 0;
    for (const token of tokens) {
      tokenOffsets.push(tokenOffset);
      tokenOffset += token.length;
    }
    const sentTokenOffsets = [];
    let sentTokenOffset = 0;
    for (const sent of sents) {
      sentTokenOffset += sent.num_tokens;
      sentTokenOffsets.push(sentTokenOffset);
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

        const loc = MarotUtils.binSearch(tokenOffsets, x);
        if (loc == tokenOffsets.length || tokenOffsets[loc] == x) {
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
        const sentIndex = MarotUtils.binSearch(sentTokenOffsets, toSplit + 1);
        console.assert(sentIndex >= 0);
        console.assert(sentIndex < sents.length);
        sents[sentIndex].num_tokens++;
        for (let si = sentIndex; si < sents.length; si++) {
          sentTokenOffsets[si]++;
        }
      }
      markerOffsets.push(offsets);
    }
    const spansList = [];
    for (const offsets of markerOffsets) {
      const spans = [];
      for (let i = 0; i < offsets.length; i+= 2) {
        if (i + 1 >= offsets.length) break;
        spans.push([MarotUtils.binSearch(tokenOffsets, offsets[i]),
                    MarotUtils.binSearch(tokenOffsets, offsets[i + 1]) - 1]);
      }
      spansList.push(spans);
    }
    return {
      tokens: tokens,
      spans: spansList,
      sentence_splits: sents,
    };
  }

  /**
   * Given the full range of rows for the same doc+docSegId+system, tokenizes
   * the source and target side using spaces, but refining the tokenization to
   * make each <v> and </v> fall on a token boundary. Sets
   * segment.{source,target}_tokens as well as
   *     this.data[row][this.DATA_COL_METADATA].{source,target}_spans.
   * Sets segment.{source,target}_sentence_splits naively as well.
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
    const sourceTokenization = this.tokenizeLegacyAnnotations(sources);
    segment.source_tokens = sourceTokenization.tokens;
    segment.source_sentence_splits = sourceTokenization.sentence_splits;
    const targetTokenization = this.tokenizeLegacyAnnotations(targets);
    segment.target_tokens = targetTokenization.tokens;
    segment.target_sentence_splits = targetTokenization.sentence_splits;
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
      const tokenization = MarotUtils.tokenizeText(text);
      segment.reference_tokens[refKey] = tokenization.tokens;
      segment.reference_sentence_splits[refKey] = tokenization.sentence_splits;
    }
  }

  /**
   * Count the total number of source characters on each sentence and set the
   * "num_chars" property in each sentence to that value.
   * @param {!Array<string>} tokens
   * @param {!Array<!Object>} sentence_splits
   */
  setNumChars(tokens, sentence_splits) {
    let tokenOffset = 0;
    for (const split of sentence_splits) {
      const firstToken = tokenOffset;
      tokenOffset += split.num_tokens;
      split.num_chars = 0;
      for (let t = firstToken; t < tokenOffset; t++) {
        console.assert(t >= 0 && t < tokens.length, t, tokens);
        split.num_chars += tokens[t].length;
      }
    }
  }

  /**
   * Aggregates metrics info for this.data, collecting all metrics for a
   * particular segment translation into the aggrDocSeg.metrics object in the
   * metadata.segment field. Metric info is keyed by system.
   */
  addMetricSegmentAggregations() {
    for (const doc of this.dataIter.docs) {
      for (const docSegId of this.dataIter.docSegs[doc]) {
        let aggrDocSegMetrics = {};
        for (const system of this.dataIter.docSys[doc]) {
          const range = this.dataIter.docSegSys[doc][docSegId][system].rows;
          let aggrDocSegSysMetrics = {};
          for (let rowId = range[0]; rowId < range[1]; rowId++) {
            const parts = this.data[rowId];
            const segment = parts[this.DATA_COL_METADATA].segment;
            segment.aggrDocSeg.metrics = aggrDocSegMetrics;
            if (segment.hasOwnProperty('metrics')) {
              aggrDocSegSysMetrics = {
                ...segment.metrics,
                ...aggrDocSegSysMetrics,
              };
            }
          }
          for (let metric in aggrDocSegSysMetrics) {
            if (!aggrDocSegMetrics.hasOwnProperty(metric)) {
              aggrDocSegMetrics[metric] = {};
            }
            aggrDocSegMetrics[metric][system] = aggrDocSegSysMetrics[metric];
          }
        }
      }
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
          doc: doc,
          docSegId: docSegId,
          catsBySystem: {},
          catsByRater: {},
          sevsBySystem: {},
          sevsByRater: {},
          sevcatsBySystem: {},
          sevcatsByRater: {},
          aggrDoc: aggrDoc,
        };
        for (const system of this.dataIter.docSys[doc]) {
          const range = this.dataIter.docSegSys[doc][docSegId][system].rows;
          let aggrDocSegSys = {
            doc: doc,
            docSegId: docSegId,
            system: system,
            aggrDocSeg: aggrDocSeg,
          };
          for (let rowId = range[0]; rowId < range[1]; rowId++) {
            const parts = this.data[rowId];
            const segment = parts[this.DATA_COL_METADATA].segment || {};
            aggrDocSegSys = {...segment, ...aggrDocSegSys};
            if (!aggrDocSegSys.hasOwnProperty('num_source_chars')) {
              aggrDocSegSys.num_source_chars = parts.num_source_chars;
            }
            if (!aggrDocSegSys.hasOwnProperty('num_target_chars')) {
              aggrDocSegSys.num_target_chars = parts.num_target_chars;
            }
          }
          if (!aggrDocSegSys.source_tokens ||
              aggrDocSegSys.source_tokens.length == 0) {
            this.tokenizeLegacySegment(range, aggrDocSegSys);
          }
          if (aggrDocSegSys.hasOwnProperty('target_tokens')) {
            console.assert(
                aggrDocSegSys.hasOwnProperty('target_sentence_splits'),
                aggrDocSegSys);
            this.setNumChars(aggrDocSegSys.target_tokens,
                             aggrDocSegSys.target_sentence_splits);
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
        console.assert(aggrDocSeg.hasOwnProperty('num_source_chars'),
                       aggrDocSeg);
        if (aggrDocSeg.num_source_chars == 0) {
          /**
           * Degenerate case of an empty source segment. Replace with a single
           * space to avoid degenerate cases in subpara-splitting.
           */
          aggrDocSeg.num_source_chars = 1;
          aggrDocSeg.source_tokens = [' '];
          aggrDocSeg.source_sentence_splits = [{num_tokens: 1}];
        }
        this.setNumChars(aggrDocSeg.source_tokens,
                         aggrDocSeg.source_sentence_splits);
        for (const ref in (aggrDocSeg.reference_sentence_splits ?? {})) {
          this.setNumChars(aggrDocSeg.reference_tokens[ref],
                           aggrDocSeg.reference_sentence_splits[ref]);

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
    for (const error of errorsFound) {
      this.errors.insertAdjacentHTML('beforeend', `<div>${error}</div>\n`);
    }
    return (errorsFound.length == 0) ? parsed : null;
  }

  /**
   * Parses score weights and slices from the user-edited settings tables. Sets
   * this.mqmWeights and this.mqmSlices if successful.
   * @return {boolean} True if the parsing was successful.
   */
  parseScoreSettings() {
    this.errors.innerHTML = '';
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
   * Return the name of the scoring unit, based upon the current settings, for
   * the given metric.
   * @param {string} metric
   * @return {string}
   */
  scoringUnitName(metric) {
    if (!this.isMQMOrAutoMQM(metric) || !this.subparaScoring) {
      return 'segment';
    }
    if (this.subparaSents == 1) {
      return 'sentence';
    }
    return 'subpara';
  }

  /**
   * Returns an HTML string for a '<td>' tag containing the score, including
   * hover text for showing the number of segments and source characters.
   * @param {!Object} s Score object containing the fields numScoringUnits,
   *     num_source_chars, score.
   * @param {string} metric
   * @param {string=} cls Optional span class to place the score text within.
   * @param {string=} ciSpanId Optional span id to show confidence interval.
   * @return {string}
   */
  tdForScore(s, metric, cls='', ciSpanId='') {
    const title =
        `# ${this.scoringUnitName(metric)}s: ${s.numScoringUnits}, ` +
        `# source chars: ${s.num_source_chars}`;
    let td = `<td title="${title}">`;
    if (cls) {
      td += `<span class="${cls}">`;
    }
    td += this.metricDisplay(s.score, s.numScoringUnits);
    if (cls) {
      td += '</span>';
    }
    if (ciSpanId) {
      td += ' <span class="marot-gray" id="' + ciSpanId +
            '">[-.---,-.---]</span>';
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
   * Aggregates scoring stats. This returns an object that has aggregate MQM
   * score in the "score" field and these additional properties:
   *       numSegments
   *       num_source_chars
   *       metrics['<metric-name>']:
   *          score
   *          numScoringUnits
   *          num_source_chars
   *       metric-[index in this.metrics]
   *           (repeated from metrics[...].score, as a convenient sorting key)
   *       mqmStats['<mqm-metric-name>']:
   *         * Note that this has entries for 'MQM' and 'AutoMQM*' metrics.
   *         numScoringUnits # of scoring units with MQM ratings
   *         num_source_chars # of src-chars with MQM ratings
   *         numSegRatings: # segment X rater combinations for MQM
   *         timeSpentMS
   * @param {!Array} units
   * @param {boolean=} treatAutoMQMAsMQM Set to true if AutoMQM scores should
   *     be aggregated as 'MQM' (used only for comparing a rater's scoring to
   *     an AutoMQM scoring).
   * @return {!Object}
   */
  aggregateUnitStats(units, treatAutoMQMAsMQM=false) {
    const aggregates = {
      numSegments: 0,
      num_source_chars: 0,
      metrics: {},
    };
    aggregates.mqmStats = {};
    if (!units || !units.length) {
      return aggregates;
    }
    const mqmMetrics = new Set;
    for (const unitStats of units) {
      for (const r of unitStats) {
        if (!r.rater) continue;
        mqmMetrics.add(this.mqmMetricName(r.rater, treatAutoMQMAsMQM));
      }
    }
    for (const mqm of mqmMetrics) {
      aggregates.mqmStats[mqm] = {
        numScoringUnits: 0,
        num_source_chars: 0,
        numSegRatings: 0,
        ...(this.initRaterStats(mqm)),
      };
    }
    /**
     * First compute MQM aggregates into aggregates.mqmStats[], and initialize
     * aggregates.metric[metricName] for any non-MQM metricName found in
     * unitStats.
     */
    const docAndDocSegs = new Set;
    for (const unitStats of units) {
      aggregates.num_source_chars += unitStats.num_source_chars;
      const docAndDocSeg = this.aColonB(
          unitStats.doc, this.unitIdToDocSegId(unitStats.unit));
      const isFirstSubpara = !docAndDocSegs.has(docAndDocSeg);
      docAndDocSegs.add(docAndDocSeg);

      for (const mqm of mqmMetrics) {
        const allRaterStats = this.initRaterStats(mqm);
        let numRaters = 0;
        for (const r of unitStats) {
          if (!r.rater) continue;
          if (this.mqmMetricName(r.rater, treatAutoMQMAsMQM) == mqm) {
            numRaters++;
            this.addRaterStats(allRaterStats, r);
          }
        }
        if (numRaters > 0) {
          const mqmStats = aggregates.mqmStats[mqm];
          this.avgRaterStats(allRaterStats, numRaters);
          mqmStats.numScoringUnits++;
          mqmStats.num_source_chars += unitStats.num_source_chars;
          if (isFirstSubpara) {
            /** Count segment ratings only from the first subpara */
            mqmStats.numSegRatings += numRaters;
          }
          this.addRaterStats(mqmStats, allRaterStats);
        }
      }
      if (unitStats.hasOwnProperty('metrics')) {
        for (let metric in unitStats.metrics) {
          if (this.isMQMOrAutoMQM(metric)) {
            continue;
          }
          if (!aggregates.metrics.hasOwnProperty(metric)) {
            aggregates.metrics[metric] = {
              score: 0,
              /**
               * For non-MQM metrics, this will be the number of segments
               * for which that metric is available, regardless of whether
               * the scoring units are subparas for MQM.
               */
              numScoringUnits: 0,
              num_source_chars: 0,
            };
          }
        }
      }
    }
    aggregates.numSegments = docAndDocSegs.size;

    for (const mqm of mqmMetrics) {
      const mqmStats = aggregates.mqmStats[mqm];
      this.avgRaterStats(mqmStats, mqmStats.numScoringUnits);
    }

    /** Aggregate non-MQM metrics. */
    for (let metric in aggregates.metrics) {
      const metricStats = aggregates.metrics[metric];
      const metricDocAndDocSegs = new Set;
      for (const unitStats of units) {
        if (!unitStats.hasOwnProperty('metrics') ||
            !unitStats.metrics.hasOwnProperty(metric)) {
          continue;
        }
        metricDocAndDocSegs.add(
            this.aColonB(unitStats.doc, this.unitIdToDocSegId(unitStats.unit)));
        metricStats.num_source_chars += unitStats.num_source_chars;
        metricStats.score += unitStats.metrics[metric];
      }
      metricStats.numScoringUnits = metricDocAndDocSegs.size;
      if (metricStats.numScoringUnits > 0) {
        metricStats.score /= metricStats.numScoringUnits;
      }
    }
    /** Copy MQM score into aggregate.metrics[] */
    for (const mqm of mqmMetrics) {
      const mqmStats = aggregates.mqmStats[mqm];
      if (mqmStats.numSegRatings == 0) {
        continue;
      }
      aggregates.metrics[mqm] = {
        score: mqmStats.score,
        numScoringUnits: mqmStats.numScoringUnits,
        num_source_chars: mqmStats.num_source_chars,
        numSegRatings: mqmStats.numSegRatings,
      };
    }
    /** The metric-m* keys are used to enable sorting the top table. */
    for (let metric in aggregates.metrics) {
      const metricStats = aggregates.metrics[metric];
      const metricIndex = this.metricsInfo[metric].index;
      aggregates['metric-' + metricIndex] = metricStats.score;
      if (this.isMQMOrAutoMQM(metric)) {
        const mqmStats = aggregates.mqmStats[metric];
        for (const key in mqmStats) {
          if (key.startsWith(this.MQM_WEIGHTED_PREFIX) ||
              key.startsWith(this.MQM_SLICE_PREFIX)) {
            aggregates['metric-' + metricIndex + '-' + key] = mqmStats[key];
          }
        }
      }
    }
    return aggregates;
  }

  /**
   * Creates a background Worker for running significance tests. If sigtests
   * are not applicable, then this returns without creating a Worker.
   */
  startSigtests() {
    let noop = true;
    const waitingMsg = 'Computing confidence intervals and p-values...';
    if (this.sigtestsData.data.hasOwnProperty('sys')) {
      noop = false;
      this.sigtestsSysMsg.innerHTML = waitingMsg;
    }
    if (this.sigtestsData.data.hasOwnProperty('rater')) {
      noop = false;
      this.sigtestsRaterMsg.innerHTML = waitingMsg;
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
    this.sigtestsWorker.onmessage = this.sigtestsUpdate.bind(this);
    this.sigtestsWorker.postMessage(this.sigtestsData);
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
    const baselineScores = data.unitScores[baseline];
    const itemScores = data.unitScores[item];
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
   *     of stats over scoring units.
   * @return {!Object} Returns an object keyed by metric name, where each
   *     value is a MarotSigtestsMetricData object.
   */
  prepareSigtests(sysOrRater, stats, statsAggregates) {
    /**
     * Each scoring unit is uniquely determined by the (doc, unit) pair. We
     * use `pairToPos` to track which pair goes to which position in the aligned
     * unitScores[] array.
     */
    const forRater = sysOrRater == 'rater';
    const pairToPos = {};
    let maxPos = 0;
    for (const key in stats) {
      for (const doc in stats[key]) {
        if (!pairToPos.hasOwnProperty(doc)) {
          pairToPos[doc] = {};
        }
        const unitStats = stats[key][doc];
        for (const unit in unitStats) {
          if (pairToPos[doc].hasOwnProperty(unit)) {
            continue;
          }
          pairToPos[doc][unit] = maxPos;
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

    const metricIds = forRater ? [0] : this.metricsVisible;
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
            {score: 0, numScoringUnits: 0};
      }
      const augmentedItems = data.comparables.slice();
      if (forRater) {
        for (const rater of data.comparables) {
          const notRater = this.NOT_PREFIX + rater;
          augmentedItems.push(notRater);
          data.scores[notRater] =
              statsAggregates[notRater].metrics[metric] ??
              {score: 0, numScoringUnits: 0};
        }
      }
      const unitScores = data.unitScores;
      for (const item of augmentedItems) {
        /**
         * For each item, we first compute the mapping from position to score.
         * Any missing key correponds to one missing scoring unit for this item.
         */
        const posToScore = {};
        for (const doc of Object.keys(stats[item])) {
          for (const unit of Object.keys(stats[item][doc])) {
            const unitStats = stats[item][doc][unit];
            const pos = pairToPos[doc][unit];
            /** Note the extra "[]". */
            const aggregate = this.aggregateUnitStats(
                [unitStats], forRater);
            const metricStats = aggregate.metrics[metric] ?? null;
            if (metricStats && metricStats.numScoringUnits > 0 &&
                (this.isMQMOrAutoMQM(metric) ||
                 !this.subparaScoring || unitStats.subpara == 0)) {
              let score = metricStats.score;
              if (metric != 'MQM' && this.subparaScoring) {
                /** Adjust score back to segment-level score */
                score *= unitStats.numSubparas;
              }
              posToScore[pos] = score;
            }
          }
        }
        /** Now we can compute "unitScores". */
        unitScores[item] = [];
        for (let pos = 0; pos < maxPos; pos++) {
          if (posToScore.hasOwnProperty(pos)) {
            unitScores[item].push(posToScore[pos]);
          } else {
            /** This item is missing this specific unit. */
            unitScores[item].push(null);
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
        const columns = (forRater ?
                         [this.NOT_PREFIX + baseline] : []).concat(
                             data.comparables.slice(idx + 1));
        for (const item of columns) {
          if (!commonPos[baseline].hasOwnProperty(item)) {
            commonPos[baseline][item] = [];
          }
          for (let pos = 0; pos < maxPos; pos++) {
            if ((unitScores[item][pos] != null) &&
                (unitScores[baseline][pos] != null)) {
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
        tr.className = 'marot-bottomed';
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
        tr.className = 'marot-dotted-bottomed';
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
    const m = this.metricsInfo[metric].index;
    if (update.ci) {
      /** update contains a confidence-interval */
      const row = update.row;
      const ci = update.ci;
      if (!isNaN(ci[0])) {
        const spanId = `marot-${update.sysOrRater}-ci-${m}-${row}`;
        const span = document.getElementById(spanId);
        span.innerHTML = '[' + ci[0].toFixed(3) + ',' + ci[1].toFixed(3) + ']';
      }
      return;
    }
    /** update contains a p-value */
    const pValues = this.sigtestsData.data[update.sysOrRater][metric].pValues;
    const row = update.row;
    const col = update.col;
    pValues[row][col] = update.pValue;
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
   */
  showSigtests(sysOrRater, stats, statsAggregates) {
    const metricIds = (sysOrRater == 'sys') ? this.metricsVisible : [0];
    const div = document.getElementById(
        `marot-${sysOrRater}-v-${sysOrRater}-tables`);
    div.innerHTML = '';
    const sigtestsData = this.prepareSigtests(
        sysOrRater, stats, statsAggregates);
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
        html += '<th colspan="2">All other human raters</th>';
      }
      for (const item of comparables) {
        html += `<th>${item}</th>`;
      }
      html += `</tr></thead>\n<tbody>\n`;

      /** Show Pearson correlations & significance test p-value placeholders. */
      for (const [rowIdx, baseline] of comparables.entries()) {
        /** Show metric score in the second column. */
        const s = scores[baseline];
        const ciSpanId = `marot-${sysOrRater}-ci-${m}-${rowIdx}`;
        html += `
          <tr id="marot-${sysOrRater}-sigtests-${m}-row-${rowIdx}">
            <td>${baseline}</td>
            ${this.tdForScore(s, metric, '', ciSpanId)}`;
        const othersColumn = [];
        if (sysOrRater == 'rater') {
          const notRater = this.NOT_PREFIX + baseline;
          const otherRatersScore = scores[notRater];
          html += this.tdForScore(otherRatersScore, metric, 'marot-gray');
          othersColumn.push(notRater);
        }
        const columns = othersColumn.concat(comparables);
        for (const [itemIdx, item] of columns.entries()) {
          const colIdx = itemIdx - othersColumn.length;
          /**
           * Note: p-value against "all other raters" is shown in colIdx = -1
           */
          if (rowIdx >= colIdx && colIdx != -1) {
            html += '<td></td>';
            continue;
          }
          const commonPos = data.commonPosByItemPair[baseline][item] ?? [];
          const title = 'Based on ' + commonPos.length +
                        ' common ' + this.scoringUnitName(metric) + 's';
          html += '<td title="' + title + '">';
          const s2 = scores[item];
          if (s2.numScoringUnits == 0) {
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
   * For an MQM-like metric, show detail columns in the scores table at the top
   * (Accuracy/Fluency splits, etc.).
   * @param {!Element} th The TH element for the metric in the scores table.
   * @param {number} m Metric index (0 for MQM, > 0 for AutoMQM).
   */
  showMQMDetails(th, m) {
    th.colSpan = this.mqmWeightedFields.length + this.mqmSliceFields.length +
                 (m == 0 ? 2 : 0) + 2;
    th.firstElementChild.innerHTML = '[-]';
    th.title = 'Click to show fewer details';
    const elts = document.getElementsByClassName('marot-mqm-detail-' + m);
    for (let i = 0; i < elts.length; i++) {
      elts[i].style.display = '';
    }
    this.mqmDetailsShown.add(m);
  }

  /**
   * For an MQM-like metric, hide detail columns in the scores table at the top
   * (Accuracy/Fluency splits, etc.).
   * @param {!Element} th The TH element for the metric in the scores table.
   * @param {number} m Metric index (0 for MQM, > 0 for AutoMQM).
   */
  hideMQMDetails(th, m) {
    th.colSpan = 2;
    th.firstElementChild.innerHTML = '[+]';
    th.title = 'Click to show more details';
    const elts = document.getElementsByClassName('marot-mqm-detail-' + m);
    for (let i = 0; i < elts.length; i++) {
      elts[i].style.display = 'none';
    }
    this.mqmDetailsShown.delete(m);
  }

  /**
   * For an MQM-like metric, toggle showing/hiding of detail columns in the
   * scores table at the top (Accuracy/Fluency splits, etc.).
   * @param {number} m Metric index (0 for MQM, > 0 for AutoMQM).
   */
  toggleMQMDetails(m) {
    const th = document.getElementById('marot-mqm-details-th-' + m);
    if (this.mqmDetailsShown.has(m)) {
      this.hideMQMDetails(th, m);
    } else {
      this.showMQMDetails(th, m);
    }
  }

  /**
   * Shows the table header for the marot scores table. The score weighted
   * components and slices to display should be available in
   * mqmWeightedFields and mqmSliceFields.
   */
  showScoresHeader() {
    const header = document.getElementById('marot-stats-thead');
    let html = `
        <tr><th></th>`;
    for (const m of this.metricsVisible) {
      const metric = this.metrics[m];
      html +=  `<th id="marot-metric-${m}-th">${metric}
                   per ${this.scoringUnitName(metric)}</th>`;
    }
    html += `
        <th title="Number of segments"><b>#Segments</b></th>
        <th title="Number of source characters"><b>#Source-chars</b></th>`;

    const mqmPartFields =
        this.mqmWeightedFields.map(x => this.MQM_WEIGHTED_PREFIX + x)
            .concat(this.mqmSliceFields.map(x => this.MQM_SLICE_PREFIX + x));
    for (const m of this.mqmMetricsVisible) {
      const metric = this.metrics[m];
      html += `
        <th title="Number of segment ratings"><b>#Segment ratings</b></th>
        <th title="Average length of error span"><b>Avg error span</b></th>`;
      for (let i = 0; i < mqmPartFields.length; i++) {
        const scoreKey = mqmPartFields[i];
        const scoreName = this.mqmKeyToName(scoreKey);
        const partType = (i < this.mqmWeightedFields.length) ? 'weighted' :
            'slice';
        const cls = 'marot-mqm-detail-' + m + ' marot-stats-' + partType;
        const tooltip = 'Score part: ' + scoreName + '-' + partType;
        html += `
            <th id="marot-metric-${m}-${scoreKey}-th"
                class="marot-score-th ${cls}" title="${tooltip}">
              <b>${scoreName}</b>
            </th>`;
      }
      if (metric == 'MQM') {
        html += `
            <th class="marot-mqm-detail-0"
                title="Average time (seconds) per rater per segment">
                <b>Time (s)</b></th>
            <th class="marot-mqm-detail-0"
                title="Hands-on-the-wheel test"><b>HOTW Test</b></th>`;
      }
    }
    html += '</tr>\n';
    let mqmHeader = '';
    if (this.mqmMetricsVisible.length > 0) {
      if (!this.mqmDetailsShown) {
        this.mqmDetailsShown = new Set;
        if (this.mqmMetricsVisible.length == 1) {
          this.mqmDetailsShown.add(this.mqmMetricsVisible[0]);
        }
      }
      const colsBefore = this.metricsVisible.length + 3;
      mqmHeader = `<tr><th colspan="${colsBefore}"></th>`;
      for (const m of this.mqmMetricsVisible) {
        const metric = this.metrics[m];
        mqmHeader += '<th class="marot-mqm-details-th"' +
            ` id="marot-mqm-details-th-${m}">${metric} details` +
            ' <span class="marot-details-toggle"></span></th>';
      }
      mqmHeader += '</tr>\n';
    }
    header.innerHTML = mqmHeader + html;

    for (const m of this.mqmMetricsVisible) {
      const th = document.getElementById('marot-mqm-details-th-' + m);
      th.addEventListener('click', this.toggleMQMDetails.bind(this, m));
    }

    /** Make columns clickable for sorting purposes. */
    const sortFields = [];
    for (const m of this.metricsVisible) {
      const metricField = 'metric-' + m;
      sortFields.push(metricField);
      if (this.isMQMOrAutoMQM(this.metrics[m])) {
        for (const partField of mqmPartFields) {
          sortFields.push(metricField + '-' + partField);
        }
      }
    }
    const upArrow = '<span class="marot-arrow marot-arrow-up">&#129041;</span>';
    const downArrow =
        '<span class="marot-arrow marot-arrow-down">&#129043;</span>';
    for (const field of sortFields) {
      const headerId = `marot-${field}-th`;
      const th = document.getElementById(headerId);
      console.assert(th, headerId);
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
   * @param {string=} title
   */
  showScoresSeparator(title='') {
    let haveHumanRaters = false;
    for (const m of this.mqmMetricsVisible) {
      if (m == 0) {
        haveHumanRaters = true;
        break;
      }
    }
    const MQM_COLS =
        (haveHumanRaters ? 2 : 0) +
        (this.mqmWeightedFields.length + this.mqmSliceFields.length + 2) *
        this.mqmMetricsVisible.length;
    const NUM_COLS = 3 + MQM_COLS + this.metricsVisible.length;
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
   * @param {!Object} stats
   * @param {!Object} aggregates
   */
  showScores(label, stats, aggregates) {
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
      rowHTML += this.tdForScore(s, metric);
    }
    rowHTML +=
        `<td>${aggregates.numSegments}</td>` +
        `<td>${aggregates.num_source_chars}</td>`;
    for (const m of this.mqmMetricsVisible) {
      const metric = this.metrics[m];
      const mqmStats = aggregates.mqmStats[metric];
      if (!mqmStats || mqmStats.numScoringUnits <= 0) {
        const blanks = scoreFields.length + (metric == 'MQM' ? 2 : 0);
        rowHTML += '<td>-</td><td>-</td>';
        for (let i = 0; i < blanks; i++) {
          rowHTML += `<td class="marot-mqm-detail-${m}">-</td>`;
        }
        continue;
      }
      console.assert(mqmStats.numSegRatings > 0, mqmStats);
      rowHTML +=
        `<td>${mqmStats.numSegRatings}</td>`;
      let errorSpan = 0;
      if (mqmStats.numWithErrors > 0) {
        errorSpan = mqmStats.errorSpans / mqmStats.numWithErrors;
      }
      rowHTML += `<td>${(errorSpan).toFixed(1)}</td>`;
      for (const s of scoreFields) {
        let content =
            mqmStats.hasOwnProperty(s) ? mqmStats[s].toFixed(3) : '-';
        let cls = `marot-mqm-detail-${m}`;
        const nameParts = s.split('-', 2);
        if (nameParts.length == 2) {
          cls += ' marot-stats-' + nameParts[1];
        }
        rowHTML += `<td class="${cls}">${content}</td>`;
      }
      if (metric == 'MQM') {
        rowHTML += '<td class="marot-mqm-detail-0">' +
            `${(mqmStats.timeSpentMS/1000.0).toFixed(1)}</td>`;
        const hotw = mqmStats.hotwFound + mqmStats.hotwMissed;
        if (hotw > 0) {
          const perc = ((mqmStats.hotwFound * 100.0) / hotw).toFixed(1);
          rowHTML += '<td class="marot-mqm-detail-0">' +
              `${mqmStats.hotwFound}/${hotw} (${perc}%)</td>`;
        } else {
          rowHTML += '<td class="marot-mqm-detail-0">-</td>';
        }
      }
    }
    rowHTML += '</tr>\n';
    this.statsTable.insertAdjacentHTML('beforeend', rowHTML);
  }

  /**
   * Shows the system x rater matrix of scores. The rows and columns are
   * ordered by total (human) MQM score if available.
   */
  showSystemRaterStats() {
    const table = document.getElementById('marot-system-x-rater');

    const systems = Object.keys(this.statsBySystem);
    const systemAggregates = {};
    for (const sys of systems) {
      const unitStats = this.getUnitStatsAsArray(this.statsBySystem[sys]);
      systemAggregates[sys] = this.aggregateUnitStats(unitStats);
    }

    const raters = this.excludeRaterComplements(Object.keys(this.statsByRater));
    const raterAggregates = {};
    for (const rater of raters) {
      const unitStats = this.getUnitStatsAsArray(this.statsByRater[rater]);
      raterAggregates[rater] = this.aggregateUnitStats(unitStats, true);
    }

    const SORT_FIELD = 'metric-0';
    systems.sort(
        (sys1, sys2) =>
            (systemAggregates[sys1][SORT_FIELD] ?? 0) -
            (systemAggregates[sys2][SORT_FIELD] ?? 0));
    raters.sort(
        (rater1, rater2) =>
            (raterAggregates[rater1][SORT_FIELD] ?? 0) -
            (raterAggregates[rater2][SORT_FIELD] ?? 0));

    let html = `
      <thead>
        <tr>
          <th>System</th>
          <th>All human raters</th>`;
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
     * system's rating, when compared with the aggregate over all human raters.
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
      const sysMQMStats = systemAggregates[sys].mqmStats['MQM'] ?? {};
      const allRatersScore = sysMQMStats.score ?? NaN;
      html += `
        <tr><td>${sys}</td>${this.tdForScore(sysMQMStats, 'MQM')}`;
      for (const rater of raters) {
        const unitStats = this.getUnitStatsAsArray(
            (this.statsByRaterSystem[rater] ?? {})[sys] ?? {});
        if (unitStats && unitStats.length > 0) {
          const aggregate = this.aggregateUnitStats(unitStats, true);
          const raterSysMQMStats = aggregate.mqmStats['MQM'];
          const raterSysScore = raterSysMQMStats.score;
          console.assert(raterSysMQMStats, rater, sys);
          /** Note that if any quantity is NaN, cls will be ''. */
          const cls = ((raterSysScore < lastForRater[rater] &&
                        allRatersScore > lastAllRaters) ||
                       (raterSysScore > lastForRater[rater] &&
                        allRatersScore < lastAllRaters)) ?
              'marot-out-of-order' : '';
          html += this.tdForScore(raterSysMQMStats, 'MQM', cls);
          lastForRater[rater] = raterSysScore;
        } else {
          html += '<td>-</td>';
          /**
           * Ensure that the next score for this rater is marked as out-of-order
           * or not in some reasonable manner:
           */
          lastForRater[rater] = allRatersScore;
        }
      }
      lastAllRaters = allRatersScore;
      html += `
        </tr>`;
    }
    html += `
      </tbody>`;
    table.innerHTML = html;
  }

  /**
   * Sorter function where a & b are both a [doc, unit] pair. The unit part can
   * be numeric.
   * @param {!Array<string>} a
   * @param {!Array<string>} b
   * @return {number} Comparison for sorting a & b.
   */
  docUnitsSorter(a, b) {
    if (a[0] < b[0]) return -1;
    if (a[0] > b[0]) return 1;
    const unit1 = this.cmpDocUnitId(a[1]);
    const unit2 = this.cmpDocUnitId(b[1]);
    if (unit1 < unit2) return -1;
    if (unit1 > unit2) return 1;
    return 0;
  }

  /**
   * From a stats object that's keyed on doc and then on scoring units, extracts
   * all [doc, unit] pairs into an array, sorts the array, and returns it.
   * @param {!Object} stats
   * @return {!Array}
   */
  getDocUnits(stats) {
    const docUnits = [];
    for (let doc in stats) {
      const docstats = stats[doc];
      for (let unit in docstats) {
        docUnits.push([doc, unit]);
      }
    }
    return docUnits.sort(this.docUnitsSorter.bind(this));
  }

  /**
   * Makes a convenient key that captures two values, separated by a colon.
   * @param {string} a
   * @param {string|number} b
   * @return {string}
   */
  aColonB(a, b) {
    return a + ':' + b;
  }

  /**
   * Create a scoring unit id for a subpara in a segment. The subpara index
   * is appended with a decimal point and leading zeros.
   * @param {string|number} docSegId
   * @param {string|number} subpara
   * @return {string}
   */
  subparaUnitId(docSegId, subpara) {
    return docSegId + '.' + ('' + subpara).padStart(4, '0');
  }

  /**
   * Retrieve the segment id from a a scoring unit id, which could be just a
   * segment id or it could be a segment id with a subpara index after a decimal
   * point.
   * @param {string|number} unit
   * @return {string}
   */
  unitIdToDocSegId(unit) {
    let docSegId = '' + unit;
    const lastDot = docSegId.lastIndexOf('.');
    if (lastDot >= 0) {
      docSegId = docSegId.substr(0, lastDot);
    }
    return docSegId;
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
   * Shows the system v system histograms of scoring unit differences.
   * @param {string} sysOrRater 'sys' or 'rater'
   */
  showCmp(sysOrRater) {
    const forRater = sysOrRater == 'rater';
    const versus = `${sysOrRater}-v-${sysOrRater}`;
    const select1 = document.getElementById(`marot-${versus}-1`);
    const select2 = document.getElementById(`marot-${versus}-2`);
    const sr1 = select1.value;
    let sr2 = select2.value;
    if (forRater && sr2 == this.NOT_PREFIX) {
      sr2 = this.NOT_PREFIX + sr1;
    }
    const stats = forRater ? this.statsByRater : this.statsBySystem;
    const docUnits1 = this.getDocUnits(stats[sr1] || {});
    const docUnits2 = this.getDocUnits(stats[sr2] || {});
    /**
     * Find common scoring units.
     */
    let i1 = 0;
    let i2 = 0;
    const docUnits12 = [];
    while (i1 < docUnits1.length && i2 < docUnits2.length) {
      const du1 = docUnits1[i1];
      const du2 = docUnits2[i2];
      const sort = this.docUnitsSorter(du1, du2);
      if (sort < 0) {
        i1++;
      } else if (sort > 0) {
        i2++;
      } else {
        docUnits12.push(du1);
        i1++;
        i2++;
      }
    }
    document.getElementById(`marot-${versus}-xunits`).innerHTML =
        docUnits12.length;
    document.getElementById(`marot-${versus}-1-units`).innerHTML =
        docUnits1.length;
    document.getElementById(`marot-${versus}-2-units`).innerHTML =
        docUnits2.length;

    const sameSR = sr1 == sr2;

    const metricIds = forRater ? [0] : this.metricsVisible;

    for (const m of metricIds) {
      const metric = this.metrics[m];
      const metricKey = 'metric-' + m;
      /**
       * We draw up to 3 plots for a metric: sr-1, sr-2, and their diff.
       */
      const hists = [
        {
          docUnits: docUnits1,
          hide: !sr1,
          sr: sr1,
          color: 'lightgreen',
          srCmp: '',
          colorCmp: '',
          id: `marot-${sysOrRater}1-plot-` + m,
        },
        {
          docUnits: docUnits2,
          hide: sameSR,
          sr: sr2,
          color: 'lightblue',
          srCmp: '',
          colorCmp: '',
          id: `marot-${sysOrRater}2-plot-` + m,
        },
        {
          docUnits: docUnits12,
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
        const histBuilder = new MarotHistogram(sysOrRater, m,
                                               this.scoringUnitName(metric),
                                               hist.sr, hist.color,
                                               hist.srCmp, hist.colorCmp);
        const docSegsSet = new Set;
        for (let i = 0; i < hist.docUnits.length; i++) {
          const doc = hist.docUnits[i][0];
          const unit = hist.docUnits[i][1];
          const unitStats = stats[hist.sr][doc][unit];
          const aggregate1 = this.aggregateUnitStats([unitStats], forRater);
          if (!aggregate1.hasOwnProperty(metricKey)) {
            continue;
          }
          let score = aggregate1[metricKey];
          if (hist.srCmp) {
            const aggregate2 = this.aggregateUnitStats(
                [stats[hist.srCmp][doc][unit]], forRater);
            if (!aggregate2.hasOwnProperty(metricKey)) {
              continue;
            }
            score -= aggregate2[metricKey];
          }
          if (!this.isMQMOrAutoMQM(metric)) {
            /** For non-MQM metrics, units are scored only at segment level */
            const docAndDocSeg = this.aColonB(doc, this.unitIdToDocSegId(unit));
            if (!docSegsSet.has(docAndDocSeg)) {
              docSegsSet.add(docAndDocSeg);
              if (this.subparaScoring) {
                score *= unitStats.numSubparas;
              }
            } else {
              continue;
            }
          }
          histBuilder.addScoringUnit(doc, unit, score);
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
   * "complementary raters" stats. I.e., for each rater R, for each scoring
   * unit, it creates stats for the "not:R" rater, that are the ratings from all
   * other human raters (AI raters, beginning with "AutoMQM", are not included
   * in any complementary set of raters).
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
      if (this.mqmMetricName(rater) != 'MQM') {
        /* We use only human raters in the "rater-complement" sets for raters */
        continue;
      }
      const raterStats = this.statsByRater[rater];
      for (const docsys in raterStats) {
        /**
         * Note that this.statsByRater[rater] (raterStats) is keyed by docsys
         * (which is like "doc:sys"). Each
         * this.statsByRater[rater][docsys][unit] is an array consisting of
         * exactly one element, which contains the aggregate score given by this
         * rater to this particular scoring unit in this doc:sys.
         *
         * When we build the complement ratings ('not:<r>') for a rater '<r>',
         * this.statsByRater['not:<r>'][docsys][unit] will be an array
         * consisting of all existing this.statsByRater[rater][docsys][unit][0]
         * such that rater != '<r>'.
         */
        const raterDocsysStats = raterStats[docsys];
        for (const unit in raterDocsysStats) {
          const raterDocsysUnitStats = raterDocsysStats[unit];
          console.assert(raterDocsysUnitStats.length == 1,
                         raterDocsysUnitStats);
          for (const otherRater of raters) {
            if (rater == otherRater) {
              continue;
            }
            const notOtherRater = this.NOT_PREFIX + otherRater;
            const notOtherRaterStats = this.statsByRater[notOtherRater];
            if (!notOtherRaterStats.hasOwnProperty(docsys)) {
              notOtherRaterStats[docsys] = {};
            }
            if (!notOtherRaterStats[docsys].hasOwnProperty(unit)) {
              notOtherRaterStats[docsys][unit] = [];
              notOtherRaterStats[docsys][unit].num_source_chars =
                  raterDocsysUnitStats.num_source_chars;
            }
            notOtherRaterStats[docsys][unit].push(
                raterDocsysUnitStats[0]);
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
    const statsBySysAggr = {};
    for (const system of systems) {
      const unitStats = this.getUnitStatsAsArray(this.statsBySystem[system]);
      statsBySysAggr[system] = this.aggregateUnitStats(unitStats);
    }
    const overallMQMStats = this.initRaterStats('');
    const allMQMStats = statsBySysAggr[this.TOTAL].mqmStats ?? {};
    for (const mqm in allMQMStats) {
      const mqmStats = allMQMStats[mqm];
      for (const key in mqmStats) {
        if (key.startsWith(this.MQM_WEIGHTED_PREFIX) ||
            key.startsWith(this.MQM_SLICE_PREFIX)) {
          overallMQMStats[key] = (overallMQMStats[key] ?? 0) + mqmStats[key];
        }
      }
    }
    this.mqmWeightedFields = [];
    this.mqmSliceFields = [];
    for (const key in overallMQMStats) {
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

    const statsByRaterAggr = {};
    const statsByRaterAutoMQMAggr = {};
    const ratersOrGroups = Object.keys(this.statsByRater);
    for (const rater of ratersOrGroups) {
      /** Build aggregates for all, including the 'not:' rater groups. */
      const unitStats = this.getUnitStatsAsArray(this.statsByRater[rater]);
      statsByRaterAggr[rater] = this.aggregateUnitStats(unitStats);
      statsByRaterAutoMQMAggr[rater] = this.aggregateUnitStats(unitStats, true);
    }

    const raters = this.excludeRaterComplements(ratersOrGroups);

    const indexOfTotal = systems.indexOf(this.TOTAL);
    systems.splice(indexOfTotal, 1);

    systems.sort(
        (k1, k2) => (statsBySysAggr[k1][this.sortByField] ?? 0) -
                    (statsBySysAggr[k2][this.sortByField] ?? 0));
    raters.sort(
        (k1, k2) => (statsByRaterAggr[k1][this.sortByField] ?? 0) -
                    (statsByRaterAggr[k2][this.sortByField] ?? 0));
    if (this.sortReverse) {
      systems.reverse();
      raters.reverse();
    }

    /**
     * First show the scores table header with the sorted columns from
     * this.mqmWeightedFields and this.mqmSliceFields. Then add scores rows to
     * the table: by system, and then by rater.
     */
    this.showScoresHeader();
    if (systems.length > 0) {
      this.showScoresSeparator('By system');
      for (const system of systems) {
        this.showScores(system, this.statsBySystem[system],
                        statsBySysAggr[system]);
      }
    }
    if (raters.length > 0) {
      this.showScoresSeparator('By rater');
      for (const rater of raters) {
        this.showScores(rater, this.statsByRater[rater],
                        statsByRaterAggr[rater]);
      }
      this.raterRelatedSections.style.display = '';
    } else {
      this.raterRelatedSections.style.display = 'none';
    }
    this.showScoresSeparator();

    /** Restore toggled state of MQM details */
    for (const m of this.mqmMetricsVisible) {
      const th = document.getElementById('marot-mqm-details-th-' + m);
      if (this.mqmDetailsShown.has(m)) {
        this.showMQMDetails(th, m);
      } else {
        this.hideMQMDetails(th, m);
      }
    }

    this.showSystemRaterStats();
    this.createCmpPlots('sys');
    this.createCmpPlots('rater');
    this.showSevCatStats();
    this.showEvents();
    this.showSigtests('sys', this.statsBySystem, statsBySysAggr);
    this.showSigtests('rater', this.statsByRater, statsByRaterAutoMQMAggr);
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
   * Updates the last entry in stats with an error of (category, severity). The
   * weighted score component to use is the first matching one in
   * this.mqmWeights[]. Similarly, the slice to attribute the score to is the
   * first matching one in this.mqmSlices[].
   *
   * @param {!Object} stats
   * @param {number} timeSpentMS
   * @param {string} category
   * @param {string} severity
   * @param {number} span
   */
  addErrorStats(stats, timeSpentMS, category, severity, span) {
    const statsToModify = this.arrayLast(stats);
    statsToModify.timeSpentMS += timeSpentMS;

    const lcat = category.toLowerCase().trim();
    if (lcat == 'no-error' || lcat == 'no_error') {
      return;
    }

    const lsev = severity.toLowerCase().trim();
    if (lsev == 'hotw-test' || lsev == 'hotw_test') {
      if (lcat == 'found') {
        statsToModify.hotwFound++;
      } else if (lcat == 'missed') {
        statsToModify.hotwMissed++;
      }
      return;
    }
    if (lsev == 'unrateable') {
      statsToModify.unrateable++;
      return;
    }
    if (lsev == 'neutral') {
      return;
    }

    if (span > 0) {
      /* There is a scoreable error span.  */
      statsToModify.numWithErrors++;
      statsToModify.errorSpans += span;
    }

    let score = 0;
    for (const sc of this.mqmWeights) {
      if (this.matchesMQMSplit(sc, lsev, lcat)) {
        score = sc.weight;
        statsToModify.score += score;
        const key = this.mqmKey(sc.name);
        statsToModify[key] = (statsToModify[key] ?? 0) + score;
        break;
      }
    }
    if (score > 0) {
      for (const sc of this.mqmSlices) {
        if (this.matchesMQMSplit(sc, lsev, lcat)) {
          const key = this.mqmKey(sc.name, true);
          statsToModify[key] = (statsToModify[key] ?? 0) + score;
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
   * Create the scoring unit stats keyed by doc and unit, setting some
   * properties, if it does not exist, and return it.
   * @param {!Object} statsByDocUnit
   * @param {string} doc
   * @param {string} unit
   * @param {number} srcChars
   * @param {number} subpara
   * @param {number} numSubparas
   * @return {!Array<!Object>}
   */
  createOrGetUnitStats(statsByDocUnit, doc, unit,
                       srcChars, subpara, numSubparas) {
    if (!statsByDocUnit.hasOwnProperty(doc)) {
      statsByDocUnit[doc] = {};
    }
    if (!statsByDocUnit[doc].hasOwnProperty(unit)) {
      statsByDocUnit[doc][unit] = [];
      /**
       * Attach the ids of the unit itself as properties, so that we can use
       * them when aggregating.
       */
      const stats = statsByDocUnit[doc][unit];
      stats.doc = doc;
      stats.unit = unit;
      stats.num_source_chars = srcChars;
      stats.subpara = subpara;
      stats.numSubparas = numSubparas;
    }
    return statsByDocUnit[doc][unit];
  }

  /**
   * Flattens the nested stats object into an array of segment stats.
   * @param {!Object} statsByDocAndUnit
   * @return {!Array}
   */
  getUnitStatsAsArray(statsByDocAndUnit) {
    const arr = [];
    for (const doc of Object.keys(statsByDocAndUnit)) {
      const statsByUnit = statsByDocAndUnit[doc];
      for (const unit of Object.keys(statsByUnit)) {
        arr.push(statsByUnit[unit]);
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
   * @param {string} rater
   * @return {string}
   */
  spanHTML(spanText, cls, rater) {
    const s = spanText.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const raterKey = this.dataIter.raterDetails[rater].key;
    cls += ' marot-rater-' + raterKey;
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
   * For a given rater, return HTML that shows a flag with a rater-specific
   * color.
   * @param {string} rater
   * @return {string}
   */
  raterFlagHTML(rater) {
    const raterDetails = this.dataIter.raterDetails[rater];
    return `<span class="marot-rater-flag marot-rater-${raterDetails.key}" ` +
           `style="color:${raterDetails.color}">` +
           '&#x2691;</span>';
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
    const flag = this.raterFlagHTML(rater);
    const raterDetails = this.dataIter.raterDetails[rater];
    return `<span class="marot-rater marot-rater-${raterDetails.key}">` +
           '<span class="marot-val" ' +
           `id="marot-val-${rowId}-${this.DATA_COL_RATER}">` +
            rater + '</span>' + flag + '</span>';
  }

  /**
   * Render in HTML a "Raw Error," which is the format used for storing
   * deleted errors and prior-rater's original errors in Marot metadata.
   * This format uses fields named "selected", "severity", "type", "subtype".
   * @param {!Object} e
   * @param {string} rater
   * @return {string}
   */
  rawErrorHTML(e, rater) {
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
    let html = this.spanHTML(
        e.selected, this.mqmSeverityClass(e.severity), rater);
    html += '<br>' + this.casedSeverity(e.severity) +
            '&nbsp;' + cat + (subcat ? '/' + subcat : '');
    html += this.metadataHTML(e.metadata, rater);
    return html;
  }

  /**
   * Render the metadata (timestamp, note, feedback, prior_rater, etc.)
   * associated with a rating.
   * @param {!Object} metadata
   * @param {string} rater
   * @return {string}
   */
  metadataHTML(metadata, rater) {
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
    let prior_or_this_rater = rater;
    if (metadata.prior_rater) {
      prior_or_this_rater = this.PRIOR_RATER_PREFIX + metadata.prior_rater;
    }
    if (metadata.prior_error) {
      html += '<br><details><summary>Annotation from prior rater:</summary>' +
              '<div class="marot-prior-error">' +
                this.rawErrorHTML(metadata.prior_error, prior_or_this_rater) +
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
        let deleted_error_prior_or_this_rater = prior_or_this_rater;
        if (de.metadata.prior_rater) {
          deleted_error_prior_or_this_rater =
              this.PRIOR_RATER_PREFIX + de.metadata.prior_rater;
        }
        html += this.rawErrorHTML(
            de, deleted_error_prior_or_this_rater);
        html += '</td></tr>';
      }
      html += '</table></details>\n';
    }
    return html;
  }

  /**
   * If name is longer than maxlen characters, then trim it, appending "..."
   * and wrapping it in an HTML span that shows the full name upon hovering.
   * Otherwise just return name.
   * @param {string} name
   * @param {number} maxlen
   * @return {string}
   */
  getShortNameHTML(name, maxlen) {
    if (name.length <= maxlen) {
      return name;
    }
    const shortName = name.substr(0, maxlen);
    return `<span title="${name}">${shortName}...</span>`;
  }

  /**
   * Returns the "metrics line" to display for the current scoring unit, which
   * includes MQM score as well as any available automated metrics.
   * @param {!Object} currUnitStatsBySys
   * @param {!Array<string>} metrics
   * @param {string} cls class name
   * @param {boolean} isLastUnit
   * @return {string}
   */
  getSegScoresHTML(currUnitStatsBySys, metrics, cls, isLastUnit) {
    const unitScoresParts = [];
    const segMetrics = currUnitStatsBySys.metrics;
    let haveMQM = false;
    for (const metric of metrics) {
      let metricValue = segMetrics[metric];
      const isMQM = this.isMQMOrAutoMQM(metric);
      if (!isMQM) {
        metricValue *= currUnitStatsBySys.numSubparas;
      }
      unitScoresParts.push([metric, metricValue]);
      if (isMQM) {
        haveMQM = true;
        /** segMetrics[metric] is unfiltered; recompute with filtering */
        const aggregate = this.aggregateUnitStats([currUnitStatsBySys]);
        if (aggregate.mqmStats.hasOwnProperty(metric)) {
          /** metrics is still visible with current filtering */
          const score = aggregate.mqmStats[metric].score;
          if (score != segMetrics[metric]) {
            unitScoresParts.push([metric + '-filtered', score]);
          }
        }
      }
    }
    let scoresRows = '';
    const addSep = haveMQM && this.subparaScoring && !isLastUnit;
    for (let i = 0; i < unitScoresParts.length; i++) {
      const part = unitScoresParts[i];
      const scoreCls = 'marot-unit-score' +
          ((addSep && i == unitScoresParts.length - 1) ?
           ' marot-dotted-bottomed' : '');
      scoresRows += '<tr' +
          (cls ? ' class="' + cls + '"' : '') +
          '><td>' + this.getShortNameHTML(part[0], 12) +
          ': </td><td class="' + scoreCls + '">' +
          this.metricDisplay(part[1], 1) + '</td></tr>';
    }
    return scoresRows;
  }

  /**
   * For the segment identified by docsegHashKey and typeHashKey, given its
   * sentence splits, either return the previously created subpara alignment
   * structure, or create and return it.
   *
   * @param {string} docsegHashKey
   * @param {string} typeHashKey
   * @param {!Array<!Object>} sentences
   * @return {!Object}
   */
  getAlignmentStruct(docsegHashKey, typeHashKey, sentences) {
    if (!this.subparas.alignmentStructs.hasOwnProperty(docsegHashKey)) {
      this.subparas.alignmentStructs[docsegHashKey] = {};
    }
    if (!this.subparas.alignmentStructs[docsegHashKey].hasOwnProperty(
            typeHashKey)) {
      const subparas = MarotUtils.makeSubparas(
          sentences, this.subparaSents, this.subparaTokens);
      this.subparas.alignmentStructs[docsegHashKey][typeHashKey] =
          MarotAligner.getAlignmentStructure(sentences, subparas);
    }
    return this.subparas.alignmentStructs[docsegHashKey][typeHashKey];
  }

  /**
   * For the given segment, based upon the current settings, return the scoring
   * units. If subparaScoring is set, then the units are the subparas (and an
   * aligner object is also attached as a property of the returned array),
   * otherwise there is just one scoring unit for the segment. Each returned
   * unit is an object that includes the unit id, the number of source
   * characters, the subpara index, and the total number of subparas.
   * @param {!Object} segmentMetadata
   * @return {!Array<!Object>}
   */
  getScoringUnits(segmentMetadata) {
    const doc = segmentMetadata.doc;
    const docSegId = segmentMetadata.docSegId;
    const system = segmentMetadata.system;
    const docColonSeg = this.aColonB(doc, docSegId);
    const docsegHashKey = MarotUtils.javaHashKey(docColonSeg);
    const units = [];
    if (this.subparaScoring) {
      const sourceStructure = this.getAlignmentStruct(
          docsegHashKey, 'src',
          segmentMetadata.source_sentence_splits);
      const targetStructure = this.getAlignmentStruct(
          docsegHashKey, 'sys-' + MarotUtils.javaHashKey(system),
          segmentMetadata.target_sentence_splits);
      units.aligner = new MarotAligner(sourceStructure, targetStructure);
      for (let p = 0; p < sourceStructure.subparas.length; p++) {
        const unit = this.subparaUnitId(docSegId, p);
        units.push({
          unit: unit,
          srcChars: sourceStructure.subparas[p].num_chars,
          subpara: p,
          numSubparas: sourceStructure.subparas.length,
        });
      }
    } else {
      units.push({
        unit: docSegId,
        srcChars: segmentMetadata.num_source_chars,
        subpara: 0,
        numSubparas: 1,
      });
    }
    return units;
  }

  /**
   * If subparaScoring is on, then using the span location of the first token in
   * the marked error, return the scoring unit that it belongs to. Otherwuse
   * just return docSegId.
   * @param {!Object} aligner
   * @param {string} docSegId
   * @param {!Object} metadata
   * @return {string}
   */
  getScoringUnitForError(aligner, docSegId, metadata) {
    if (!this.subparaScoring) {
      return docSegId;
    }
    let spansArray = [[-1, -1]];
    let isTarget = true;
    if (metadata.target_spans && metadata.target_spans.length > 0) {
      spansArray = metadata.target_spans;
    } else if (metadata.source_spans && metadata.source_spans.length > 0) {
      spansArray = metadata.source_spans;
      isTarget = false;
    }
    const tokenIndex = spansArray[0][0];
    let sourceSubpara = -1;
    if (tokenIndex >= 0) {
      if (isTarget) {
        sourceSubpara = aligner.tgtTokenToSrcSubpara(tokenIndex);
      } else {
        const srcTokenNumber = tokenIndex + 1;
        sourceSubpara = MarotUtils.binSearch(
            aligner.srcStructure.subparaTokens, srcTokenNumber);
      }
    } else {
      /* No error, doesn't matter which subpara we attribute it to. */
      sourceSubpara = 0;
    }
    console.assert(sourceSubpara >= 0 &&
                   sourceSubpara < aligner.srcStructure.subparas.length);
    return this.subparaUnitId(docSegId, sourceSubpara);
  }

  /**
   * Return a class name to use for wrapping a subpara's text, so that it gets
   * highlighted when navigating through the examples table.
   * @param {string} docsegHashKey
   * @param {string} typeHashKey
   * @param {number} subpara
   * @return {string}
   */
  subparaClass(docsegHashKey, typeHashKey, subpara) {
    return 'marot-subpara-' + docsegHashKey + '-' + typeHashKey + '-' + subpara;
  }

  /**
   * Return HTML from tokens and text alignment structure. This involves
   * wrapping the joined text from the tokens within a <p>..</p> tag. In
   * addition, at the end of each sentence, if the sentence ends in a paragraph
   * break, then a new paragraph is initiated, while if it ends in a line break,
   * then a <br> tag is inserted.
   *
   * Subpara renderings and sentence renderings are wrapped in spans with
   * distinctive IDs.
   *
   * @param {!Array<string>} tokens
   * @param {!Array<!Object>} alignmentStruct
   * @param {string} docsegHashKey
   * @param {string} typeHashKey
   * @return {string}
   */
  htmlFromTokens(tokens, alignmentStruct, docsegHashKey, typeHashKey) {
    let html = '<p>\n';
    for (let p = 0; p < alignmentStruct.subparas.length; p++) {
      const subpara = alignmentStruct.subparas[p];
      const subparaClass = this.subparaClass(docsegHashKey, typeHashKey, p);
      html += '<span class="marot-subpara ' + subparaClass +
              '" title="Alignments shown are only approximate. ' +
              'Click to pin/unpin a sub-para. ' +
              'Use arrow keys to move pinnned sub-para.">';
      for (let s = 0; s < subpara.sentences.length; s++) {
        const sentence = subpara.sentences[s];
        const sentClass =
            'marot-sent-' + docsegHashKey + '-' + typeHashKey +
            '-' + p + '-' + sentence.index;
        html += '<span class="' + sentClass + '">';
        html += tokens.slice(
            sentence.offset, sentence.offset + sentence.num_tokens).join('');
        if (sentence.ends_with_line_break) {
          html += '<br>\n';
        }
        html += '</span>';
      }
      html += '</span>';
      if (subpara.ends_with_para_break ||
          p == (alignmentStruct.subparas.length - 1)) {
        html += '</p>\n';
        if (p < alignmentStruct.subparas.length - 1) {
          html += '<p>\n';
        }
      }
    }
    return html;
  }

  /**
   * Updates the display to show the segment data and scores according to the
   * current filters.
   */
  show() {
    /**
     * Cancel existing Sigtest computation when a new `this.show()` is called.
     */
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
    this.mqmMetricsVisible = [];

    document.getElementById('marot-filter-expr-error').innerHTML = '';
    const allFilters = this.getAllFilters();

    /**
     * Reset subpara navigation/alignment in the examples table.
     */
    this.subparas.alignmentStructs = {};
    this.subparas.classMap = {};

    document.body.style.cursor = 'wait';
    for (const doc of this.dataIter.docs) {
      for (const docSegId of this.dataIter.docSegs[doc]) {
        for (const system of this.dataIter.docSys[doc]) {
          let firstRowId = -1;
          let segmentMetadata = null;
          let lastRater = '';
          const range = this.dataIter.docSegSys[doc][docSegId][system].rows;
          const docColonSys = this.aColonB(doc, system);
          let scoringUnits = null;

          for (let rowId = range[0]; rowId < range[1]; rowId++) {
            const parts = this.data[rowId];
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
              scoringUnits = this.getScoringUnits(segmentMetadata);

              if (!this.statsBySystem.hasOwnProperty(system)) {
                this.statsBySystem[system] = {};
              }
              for (const scoringUnit of scoringUnits) {
                const unit = scoringUnit.unit;
                const currUnitStatsBySys = this.createOrGetUnitStats(
                    this.statsBySystem[system], doc, unit, scoringUnit.srcChars,
                    scoringUnit.subpara, scoringUnit.numSubparas);
                this.createOrGetUnitStats(
                    this.statsBySystem[this.TOTAL], doc, unit,
                    scoringUnit.srcChars,
                    scoringUnit.subpara, scoringUnit.numSubparas);
                if (segmentMetadata.hasOwnProperty('metrics')) {
                  currUnitStatsBySys.metrics =
                      segmentMetadata.metricsByUnit[unit];
                  for (let metric in currUnitStatsBySys.metrics) {
                    if (this.isMQMOrAutoMQM(metric)) {
                      /** Visibility might depend on filtering */
                      continue;
                    }
                    visibleMetrics[metric] = true;
                  }
                }
              }
            }

            if (rater && (rater != lastRater)) {
              lastRater = rater;
              visibleMetrics[this.mqmMetricName(rater)] = true;

              if (!this.statsByRater.hasOwnProperty(rater)) {
                this.statsByRater[rater] = {};
              }
              if (!this.statsByRaterSystem.hasOwnProperty(rater)) {
                this.statsByRaterSystem[rater] = {};
              }
              if (!this.statsByRaterSystem[rater].hasOwnProperty(system)) {
                this.statsByRaterSystem[rater][system] = {};
              }
              for (const scoringUnit of scoringUnits) {
                const unit = scoringUnit.unit;
                const currUnitStats = this.createOrGetUnitStats(
                    this.statsBySystem[this.TOTAL], docColonSys, unit,
                    scoringUnit.srcChars, scoringUnit.subpara,
                    scoringUnit.numSubparas);
                currUnitStats.push(this.initRaterStats(rater));
                const currUnitStatsBySys = this.createOrGetUnitStats(
                    this.statsBySystem[system], doc, unit,
                    scoringUnit.srcChars, scoringUnit.subpara,
                    scoringUnit.numSubparas);
                currUnitStatsBySys.push(this.initRaterStats(rater));
                const currUnitStatsByRater = this.createOrGetUnitStats(
                    this.statsByRater[rater], docColonSys, unit,
                    scoringUnit.srcChars, scoringUnit.subpara,
                    scoringUnit.numSubparas);
                currUnitStatsByRater.push(this.initRaterStats(rater));
                const currUnitStatsByRaterSys = this.createOrGetUnitStats(
                    this.statsByRaterSystem[rater][system], doc, unit,
                    scoringUnit.srcChars, scoringUnit.subpara,
                    scoringUnit.numSubparas);
                currUnitStatsByRaterSys.push(this.initRaterStats(rater));
              }
            }
            if (rater && metadata.prior_rater) {
              const priorRater = this.PRIOR_RATER_PREFIX + metadata.prior_rater;
              if (!this.statsByRater.hasOwnProperty(priorRater)) {
                this.statsByRater[priorRater] = {};
              }
              if (!this.statsByRaterSystem.hasOwnProperty(priorRater)) {
                this.statsByRaterSystem[priorRater] = {};
              }
              if (!this.statsByRaterSystem[priorRater].hasOwnProperty(system)) {
                this.statsByRaterSystem[priorRater][system] = {};
              }
              for (const scoringUnit of scoringUnits) {
                const unit = scoringUnit.unit;
                const raterStats = this.createOrGetUnitStats(
                    this.statsByRater[priorRater], docColonSys, unit,
                    scoringUnit.srcChars, scoringUnit.subpara,
                    scoringUnit.numSubparas);
                if (raterStats.length == 0) {
                  raterStats.push(this.initRaterStats(priorRater));
                }
                const raterSysStats = this.createOrGetUnitStats(
                    this.statsByRaterSystem[priorRater][system], docColonSys,
                    unit, scoringUnit.srcChars, scoringUnit.subpara,
                    scoringUnit.numSubparas);
                if (raterSysStats.length == 0) {
                  raterSysStats.push(this.initRaterStats(priorRater));
                }
              }
            }
            if (rater) {
              /** An actual rater-annotation row, not just a metadata row */
              const span = metadata.marked_text.length;
              const timeSpentMS = this.timeSpent(metadata);
              const unit = this.getScoringUnitForError(
                  scoringUnits.aligner, docSegId, metadata);
              this.addErrorStats(
                  this.statsBySystem[this.TOTAL][docColonSys][unit],
                  timeSpentMS, category, severity, span);
              this.addErrorStats(
                  this.statsBySystem[system][doc][unit],
                  timeSpentMS, category, severity, span);
              this.addErrorStats(
                  this.statsByRater[rater][docColonSys][unit],
                  timeSpentMS, category, severity, span);
              this.addErrorStats(
                  this.statsByRaterSystem[rater][system][doc][unit],
                  timeSpentMS, category, severity, span);
              this.addSevCatStats(this.sevcatStats, system, category, severity);
              this.addEvents(this.events, metadata,
                             doc, docSegId, system, rater);
              if (metadata.prior_rater) {
                const priorRater = this.PRIOR_RATER_PREFIX +
                                   metadata.prior_rater;
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
                  const errorUnit = this.getScoringUnitForError(
                      scoringUnits.aligner, docSegId, priorError.metadata);
                  const timeSpentMS = this.timeSpent(priorError.metadata);
                  this.addErrorStats(
                      this.statsByRater[priorRater][docColonSys][errorUnit],
                      timeSpentMS, cat, priorError.severity, span);
                  this.addErrorStats(
                      this.statsByRaterSystem[priorRater][
                          system][docColonSys][errorUnit],
                      timeSpentMS, cat, priorError.severity, span);
                }
              }
            }
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
        if (this.isMQMOrAutoMQM(metric)) {
          this.mqmMetricsVisible.push(m);
        }
      }
    }
    /**
     * If the currently chosen sort-by field is a metric that is not visible,
     * then change it to be the first metric that *is* visible (if any,
     * defaulting to metric-0, which is MQM). Set the default direction
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
      viewingConstraintsDesc.innerHTML = 'View showing the ' +
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
    /**
     * Reset the index of shown subparas, and reset to make sure no subpara
     * is pinned for arrow-key-based navigation.
     */
    const shownSubparas = {};
    this.subparas.pinnedSubparaCls = '';

    let allRaterFlagsHTML = '<span class="marot-all-rater-flags">';
    for (const rater of this.dataIter.raters ?? []) {
      allRaterFlagsHTML += this.raterFlagHTML(rater);
    }
    allRaterFlagsHTML += '</span>';

    for (const doc of this.dataIter.docs) {
      for (const docSegId of this.dataIter.docSegs[doc]) {
        let shownForDocSeg = 0;
        let aggrDocSeg = null;
        const docColonSeg = this.aColonB(doc, docSegId);
        const docsegHashKey = MarotUtils.javaHashKey(docColonSeg);
        for (const system of this.dataIter.docSys[doc]) {
          let shownForDocSegSys = 0;
          let aggrDocSegSys = null;
          let firstRowId = -1;
          let ratingRowsHTML = '';
          let segmentMetadata = null;
          let scoringUnits = null;
          let sourceTokens = null;
          let sourceSents = null;
          let targetTokens = null;
          let targetSents = null;
          let lastRater = '';
          const range = this.dataIter.docSegSys[doc][docSegId][system].rows;
          const docColonSys = this.aColonB(doc, system);
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
              scoringUnits = this.getScoringUnits(segmentMetadata);
              /** Copy source/target tokens as we'll wrap them in spans. */
              sourceTokens = segmentMetadata.source_tokens.slice();
              targetTokens = segmentMetadata.target_tokens.slice();
              sourceSents = segmentMetadata.source_sentence_splits;
              targetSents = segmentMetadata.target_sentence_splits;
            }
            if (rater && (rater != lastRater)) {
              lastRater = rater;
            }
            let spanClass = '';
            if (rater) {
              /** An actual rater-annotation row, not just a metadata row */
              const raterKey = this.dataIter.raterDetails[rater].key;
              spanClass = this.mqmSeverityClass(severity) +
                          ` marot-anno marot-anno-${rowId}` +
                          ` marot-rater-${raterKey}`;
              this.markSpans(
                  sourceTokens, metadata.source_spans || [], spanClass);
              this.markSpans(
                  targetTokens, metadata.target_spans || [], spanClass);
            }

            if (viewingConstraints &&
                !viewingConstraints[docColonSeg] &&
                !viewingConstraints[this.aColonB(docColonSys, docSegId)]) {
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
              ratingRowsHTML +=
                  this.spanHTML(metadata.marked_text, spanClass, rater) +
                  '<br>';
            }
            ratingRowsHTML += this.severityHTML(rowId, severity) +
                              '&nbsp;';
            ratingRowsHTML += this.categoryHTML(rowId, category) +
                              '<br>';
            ratingRowsHTML += this.raterHTML(rowId, rater);
            ratingRowsHTML += this.metadataHTML(metadata, rater);
            ratingRowsHTML += '</div></td></tr>\n';
          }
          if (shownForDocSegSys == 0) {
            continue;
          }
          console.assert(firstRowId >= 0 &&
                         sourceTokens && targetTokens &&
                         sourceSents && targetSents, firstRowId);

          if (!shownSubparas.hasOwnProperty(docsegHashKey)) {
            shownSubparas[docsegHashKey] = {
              ref: new Set,
              sys: new Set,
            };
          }
          if (shownForDocSeg == 0 && aggrDocSeg &&
              aggrDocSeg.reference_tokens) {
            for (const ref of Object.keys(aggrDocSeg.reference_tokens)) {
              let refRowHTML = '<tr class="marot-row marot-ref-row">';
              refRowHTML += '<td><div>' + doc + '</div></td>';
              refRowHTML += '<td><div>' + docSegId + '</div></td>';
              refRowHTML += '<td><div><b>Ref</b>: ' + ref + '</div></td>';
              const sourceTokensForRef = aggrDocSeg.source_tokens || [];
              const sourceHashKey = 'src';
              const sourceAlignmentStruct = this.getAlignmentStruct(
                  docsegHashKey, sourceHashKey, sourceSents);
              refRowHTML += '<td><div>' +
                  this.htmlFromTokens(
                      sourceTokensForRef, sourceAlignmentStruct,
                      docsegHashKey, sourceHashKey) +
                  '</div></td>';
              const refTokens = aggrDocSeg.reference_tokens[ref] ?? [];
              const refSents = aggrDocSeg.reference_sentence_splits[ref] ?? [];
              const refHashKey = 'ref-' + MarotUtils.javaHashKey(ref);
              const refAlignmentStruct = this.getAlignmentStruct(
                  docsegHashKey, refHashKey, refSents);
              refRowHTML += '<td><div>' +
                  this.htmlFromTokens(refTokens, refAlignmentStruct,
                                      docsegHashKey, refHashKey) +
                  '</div></td>';
              refRowHTML += '<td></td></tr>\n';
              this.segmentsTable.insertAdjacentHTML('beforeend', refRowHTML);
              shownSubparas[docsegHashKey].ref.add(refHashKey);
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

          const sourceHashKey = 'src';
          const sourceAlignmentStruct = this.getAlignmentStruct(
              docsegHashKey, sourceHashKey, sourceSents);
          const source =
              this.htmlFromTokens(sourceTokens, sourceAlignmentStruct,
                                  docsegHashKey, sourceHashKey);

          const targetHashKey = 'sys-' + MarotUtils.javaHashKey(system);
          const targetAlignmentStruct = this.getAlignmentStruct(
              docsegHashKey, targetHashKey, targetSents);
          const target =
              this.htmlFromTokens(targetTokens, targetAlignmentStruct,
                                  docsegHashKey, targetHashKey);

          rowHTML += '<td><table class="marot-table-ratings">';
          rowHTML += '<tr><td><span class="marot-val" ';
          rowHTML += `id="marot-val-${firstRowId}-${this.DATA_COL_SYSTEM}">` +
                     this.getShortNameHTML(system, 14) +
                     '</span><br>&nbsp;</td></tr>\n';
          const metricWiseRows = {};
          let mqmRows = '';
          for (const scoringUnit of scoringUnits) {
            const unit = scoringUnit.unit;
            const subpara = scoringUnit.subpara;
            const isLastUnit = (subpara == scoringUnit.numSubparas - 1);
            const currUnitStatsBySys = this.statsBySystem[system][doc][unit];
            const mqmMetrics = [];
            const otherMetrics = [];
            for (const metric in currUnitStatsBySys.metrics) {
              if (this.isMQMOrAutoMQM(metric)) {
                mqmMetrics.push(metric);
              } else {
                otherMetrics.push(metric);
              }
            }
            if (mqmMetrics.length > 0) {
              let cls = 'marot-metric';
              if (this.subparaScoring) {
                cls += ' marot-subpara ' +
                    this.subparaClass(docsegHashKey, 'src', subpara);
              }
              mqmRows += this.getSegScoresHTML(
                  currUnitStatsBySys, mqmMetrics, cls, isLastUnit);
            }
            if (subpara == 0) {
              /** Non-MQM metrics */
              for (const metric of otherMetrics) {
                if (!metricWiseRows[metric]) {
                  metricWiseRows[metric] = '';
                }
                metricWiseRows[metric] += this.getSegScoresHTML(
                    currUnitStatsBySys, [metric], 'marot-metric', isLastUnit);
              }
            }
          }
          if (mqmRows) {
            metricWiseRows['MQM'] = mqmRows;
          }
          for (const metric in metricWiseRows) {
            rowHTML += '<tr><td><table class="marot-scores-table">';
            rowHTML += metricWiseRows[metric];
            rowHTML += '</table></td></tr>';
          }
          rowHTML += '</table></td>';

          rowHTML += '<td><div>' + source + '</div>' + allRaterFlagsHTML +
                     '</td>';
          rowHTML += '<td><div>' + target + '</div>' + allRaterFlagsHTML +
                     '</td>';

          rowHTML += '<td><table class="marot-table-ratings">' +
                     ratingRowsHTML + '</table></td>';

          this.segmentsTable.insertAdjacentHTML(
              'beforeend', `<tr class="marot-row">${rowHTML}</tr>\n`);
          shownForDocSeg += shownForDocSegSys;

          shownSubparas[docsegHashKey].sys.add(targetHashKey);
        }
        if (shownForDocSeg > 0) {
          shownCount += shownForDocSeg;
        }
      }
    }
    this.addAlignmentHighlighters(shownSubparas);
    this.addAnnotationHighlighters(shownRows);
    this.addRaterSpecificViews();
    this.addFilterListeners(shownRows);
  }

  /**
   * When the user hovers over any particular rater, we let the user see just
   * the ratings done by that rater (hiding other raters' annotations). This
   * is done in this function by adding a rater-specific CSS class to the whole
   * example segments table, and some custom CSS rules that make only that
   * rater's marked spans visible.
   */
  addRaterSpecificViews() {
    if (!this.dataIter.raters) {
      return;
    }
    const raterHighlighter = (rater, raterKey, shouldShow) => {
      const raterClass = 'marot-rater-' + raterKey;
      if (shouldShow) {
        this.segmentsTable.classList.add('marot-single-rater-view');
        this.segmentsTable.classList.add(raterClass);
        const selector = '.marot-single-rater-view.' + raterClass;
        this.customStyles.innerHTML = `
          ${selector} .marot-rater-flag.${raterClass} {
            visibility: visible;
          }
          ${selector} .marot-rater.${raterClass} {
            font-weight: bold;
          }
          ${selector} .mqm-critical.${raterClass} {
            background: rgba(127, 0, 255, 0.7);
          }
          ${selector} .mqm-major.${raterClass} {
            background: rgba(255, 192, 203, 0.7);
          }
          ${selector} .mqm-minor.${raterClass} {
            background: rgba(251, 236, 93, 0.7);
          }
          ${selector} .mqm-neutral.${raterClass} {
            background: rgba(211, 211, 211, 0.7);
          }`;
      } else {
        this.segmentsTable.classList.remove('marot-single-rater-view');
        this.segmentsTable.classList.remove(raterClass);
        this.customStyles.innerHTML = '';
      }
    };
    for (const rater of this.dataIter.raters) {
      const raterDetails = this.dataIter.raterDetails[rater];
      const elts = document.getElementsByClassName(
          'marot-rater marot-rater-' + raterDetails.key);
      if (elts.length == 0) continue;
      const onHover = (e) => {
        raterHighlighter(rater, raterDetails.key, true);
      };
      const onNonHover = (e) => {
        raterHighlighter(rater, raterDetails.key, false);
      };
      for (let i = 0; i < elts.length; i++) {
        elts[i].addEventListener('mouseover', onHover);
        elts[i].addEventListener('mouseout', onNonHover);
      }
    }
  }

  /**
   * Add cross-highlighting listeners for error spans.
   *
   * @param {!Array<number>} shownRows The array of visible row indices.
   */
  addAnnotationHighlighters(shownRows) {
    const annoHighlighter = (rowId, shouldShow) => {
      const elts = document.getElementsByClassName('marot-anno-' + rowId);
      const fontWeight = shouldShow ? 'bold' : 'inherit';
      const border = shouldShow ? '1px solid blue' : 'none';
      for (let i = 0; i < elts.length; i++) {
        const style = elts[i].style;
        style.fontWeight = fontWeight;
        style.borderTop = border;
        style.borderBottom = border;
      }
    };
    for (const rowId of shownRows) {
      const elts = document.getElementsByClassName('marot-anno-' + rowId);
      if (elts.length == 0) continue;
      const onHover = (e) => {
        annoHighlighter(rowId, true);
      };
      const onNonHover = (e) => {
        annoHighlighter(rowId, false);
      };
      for (let i = 0; i < elts.length; i++) {
        elts[i].addEventListener('mouseover', onHover);
        elts[i].addEventListener('mouseout', onNonHover);
      }
    }
  }

  /**
   * Add filter listeners (when the user clicks on a rater/system/etc. to create
   * a filter) in the examples table.
   *
   * @param {!Array<number>} shownRows The array of visible row indices.
   */
  addFilterListeners(shownRows) {
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
   * Set up the display of approximately aligned subparas in the examples table.
   *
   * @param {!Array<!Object>} shownSubparas The array of visible subparas. This
   *     is indexed by docsegHashKey, and then 'ref'/'sys', mapping to the
   *     target subparas Set (each target subpara is represented in the Set
   *     by its hash-key).
   */
  addAlignmentHighlighters(shownSubparas) {
    const initMap = (map, from) => {
      if (!map.hasOwnProperty(from)) {
        map[from] = new Set;
      }
    };
    const docsegHashKeys = Object.keys(shownSubparas);
    for (const docsegHashKey of docsegHashKeys) {
      const alignmentStructs = this.subparas.alignmentStructs[docsegHashKey];
      const srcAlignmentStruct = alignmentStructs['src'];
      const shownPara = shownSubparas[docsegHashKey];
      for (let sp = 0; sp < srcAlignmentStruct.subparas.length; sp++) {
        const srcParaClass = this.subparaClass(docsegHashKey, 'src', sp);
        initMap(this.subparas.classMap, srcParaClass);
      }
      for (const refsys of ['ref', 'sys']) {
        for (const tgtHashKey of shownPara[refsys]) {
          const tgtAlignmentStruct = alignmentStructs[tgtHashKey];
          const aligner = new MarotAligner(
              srcAlignmentStruct, tgtAlignmentStruct);
          for (let tp = 0; tp < tgtAlignmentStruct.subparas.length; tp++) {
            const tgtSubpara = tgtAlignmentStruct.subparas[tp];
            if (!tgtSubpara || tgtSubpara.num_tokens == 0 ||
                tgtSubpara.sentences.length == 0) {
              continue;
            }
            const tgtParaClass = this.subparaClass(
                docsegHashKey, tgtHashKey, tp);
            initMap(this.subparas.classMap, tgtParaClass);
            const range = aligner.tgtSubparaToSrcSubparaRange(tp);
            for (let sp = range[0]; sp <= range[1]; sp++) {
              const srcParaClass = this.subparaClass(docsegHashKey, 'src', sp);
              this.subparas.classMap[srcParaClass].add(tgtParaClass);
              this.subparas.classMap[tgtParaClass].add(srcParaClass);
            }
          }
        }
      }
      for (let p = 0; p < srcAlignmentStruct.subparas.length; p++) {
        const srcParaClass = this.subparaClass(docsegHashKey, 'src', p);
        const elts = document.getElementsByClassName(srcParaClass);
        for (let i = 0; i < elts.length; i++) {
          const elt = elts[i];
          elt.addEventListener(
              'mouseover', this.subparaNavHover.bind(this, srcParaClass));
          elt.addEventListener(
              'mouseout', this.subparaNavNonHover.bind(this, srcParaClass));
          elt.addEventListener(
              'click',
              this.subparaNavClick.bind(
                  this, srcParaClass, docsegHashKey, 'src', p));
        }
      }
      for (const refsys of ['ref', 'sys']) {
        for (const tgtHashKey of shownPara[refsys]) {
          const tgtAlignmentStruct = alignmentStructs[tgtHashKey];
          for (let p = 0; p < tgtAlignmentStruct.subparas.length; p++) {
            const tgtParaClass = this.subparaClass(
                docsegHashKey, tgtHashKey, p);
            const elts = document.getElementsByClassName(tgtParaClass);
            for (let i = 0; i < elts.length; i++) {
              const elt = elts[i];
              elt.addEventListener(
                  'mouseover', this.subparaNavHover.bind(this, tgtParaClass));
              elt.addEventListener(
                  'mouseout', this.subparaNavNonHover.bind(this, tgtParaClass));
              elt.addEventListener(
                  'click',
                  this.subparaNavClick.bind(
                      this, tgtParaClass, docsegHashKey, tgtHashKey, p));
            }
          }
        }
      }
    }
  }

  /**
   * Handler for when the user hovers over or out of the subpara with class cls.
   * The shouldShow parameter is true for showing and false for hiding the
   * highlighted alignment.
   *
   * @param {string} cls
   * @param {boolean} shouldShow
   */
  subparaNavHighlighter(cls, shouldShow) {
    const classFonter = (c, fontWeight, color) => {
      const elts = document.getElementsByClassName(c);
      for (let i = 0; i < elts.length; i++) {
        elts[i].style.fontWeight = fontWeight;
        elts[i].style.color = color;
      }
    };
    let fontWeight = shouldShow ? '600' : 'inherit';
    let color = shouldShow ? '#00008B' : 'inherit';
    classFonter(cls, fontWeight, color);
    fontWeight = 'inherit';
    color = shouldShow ? 'blue' : 'inherit';
    for (const mappedCls of this.subparas.classMap[cls]) {
      classFonter(mappedCls, fontWeight, color);
    }
  }

  /**
   * Handler for when the user hovers over the subpara with class cls.
   *
   * @param {string} cls
   */
  subparaNavHover(cls) {
    if (this.subparas.pinnedSubparaCls) {
      return;
    }
    this.subparaNavHighlighter(cls, true);
  }

  /**
   * Handler for when the user hovers away from the subpara with class cls.
   *
   * @param {string} cls
   */
  subparaNavNonHover(cls) {
    if (this.subparas.pinnedSubparaCls) {
      return;
    }
    this.subparaNavHighlighter(cls, false);
  }

  /**
   * Handler for when the user clicks on a subpara. If a subpara is already
   * pinned, then this unpins it. If no subpara is already pinned, then this
   * subpara gets pinned. The class must end in a "-<subpara-index>"
   *
   * @param {string} cls
   */
  subparaNavClick(cls) {
    if (this.subparas.pinnedSubparaCls) {
      this.subparaNavHighlighter(this.subparas.pinnedSubparaCls, false);
      this.subparas.pinnedSubparaCls = '';
      return;
    }
    this.subparas.pinnedSubparaCls = cls;
    this.subparaNavHighlighter(cls, true);
  }

  /**
   * Move the pinned subpara (if it exists) up (dir=-1) or down (dir=1).
   *
   * @param {number} dir
   */
  subparaNavMove(dir) {
    if (!this.subparas.pinnedSubparaCls) {
      return;
    }
    const lastDash = this.subparas.pinnedSubparaCls.lastIndexOf('-');
    const subpara = parseInt(
        this.subparas.pinnedSubparaCls.substr(lastDash + 1));
    const p = subpara + dir;
    const newCls = this.subparas.pinnedSubparaCls.substr(0, lastDash) +
                   '-' + p;
    if (!this.subparas.classMap.hasOwnProperty(newCls)) {
      return;
    }
    this.subparaNavHighlighter(this.subparas.pinnedSubparaCls, false);
    this.subparas.pinnedSubparaCls = newCls;
    this.subparaNavHighlighter(newCls, true);
  }

  /**
   * Handler for key-down to grab kep presses on Escape or arrow keys, to
   * unpin or move the pinned subpara.
   *
   * @param {!Event} e
   */
  subparaNavKeyDown(e) {
    if (!e.key) {
      return;
    }
    if (e.key != 'Escape' &&
        e.key != 'ArrowLeft' && e.key != 'ArrowRight' &&
        e.key != 'ArrowUp' && e.key != 'ArrowDown') {
      return;
    }
    e.preventDefault();
    if (!this.subparas.pinnedSubparaCls) {
      return;
    }
    if (e.key === 'Escape') {
      this.subparaNavHighlighter(this.subparas.pinnedSubparaCls, false);
      this.subparas.pinnedSubparaCls = '';
      return;
    }
    let dir = 1;
    if (e.key == 'ArrowLeft' || e.key == 'ArrowUp') {
      dir = -1;
    }
    this.subparaNavMove(dir);
  }

  /**
   * Recomputes MQM/AutoMQM* scores for each segment (using current weight
   * settings) and sets them in segment.metrics[] and
   * segment.metricsByUnit[unit][].
   */
  recomputeMQM() {
    const statsBySystem = {};
    for (const doc of this.dataIter.docs) {
      for (const docSegId of this.dataIter.docSegs[doc]) {
        for (const system of this.dataIter.docSys[doc]) {
          let lastRater = '';
          const range = this.dataIter.docSegSys[doc][docSegId][system].rows;
          let aggrDocSegSys = null;
          let scoringUnits = null;
          for (let rowId = range[0]; rowId < range[1]; rowId++) {
            const parts = this.data[rowId];
            const metadata = parts[this.DATA_COL_METADATA];
            if (!aggrDocSegSys) {
              aggrDocSegSys = metadata.segment;
              scoringUnits = this.getScoringUnits(aggrDocSegSys);
              aggrDocSegSys.metricsByUnit = {};
              if (!statsBySystem.hasOwnProperty(system)) {
                statsBySystem[system] = {};
              }
              const nonMQMMetrics = {
                ...aggrDocSegSys.metrics,
              };
              delete nonMQMMetrics.MQM;
              for (const metric in nonMQMMetrics) {
                nonMQMMetrics[metric] /= scoringUnits.length;
              }
              for (const scoringUnit of scoringUnits) {
                const unit = scoringUnit.unit;
                this.createOrGetUnitStats(
                    statsBySystem[system], doc, unit,
                    scoringUnit.srcChars, scoringUnit.subpara,
                    scoringUnit.numSubparas);
                aggrDocSegSys.metricsByUnit[unit] = {
                  ...nonMQMMetrics,
                };
              }
            }
            const rater = parts[this.DATA_COL_RATER];
            if (!rater) {
              continue;
            }
            if (rater != lastRater) {
              lastRater = rater;
              for (const scoringUnit of scoringUnits) {
                const unit = scoringUnit.unit;
                statsBySystem[system][doc][unit].push(
                    this.initRaterStats(rater));
              }
            }
            const category = parts[this.DATA_COL_CATEGORY];
            const severity = parts[this.DATA_COL_SEVERITY];
            const unit = this.getScoringUnitForError(
                scoringUnits.aligner, docSegId, metadata);
            /** We don't care about computing avg span/time here, pass as 0. */
            this.addErrorStats(
                statsBySystem[system][doc][unit], 0, category, severity, 0);
          }
          if (aggrDocSegSys) {
            const allUnitStats = this.getUnitStatsAsArray(
                statsBySystem[system]);
            const aggrScores = this.aggregateUnitStats(allUnitStats);
            for (const mqm in aggrScores.mqmStats) {
              const mqmStats = aggrScores.mqmStats[mqm];
              if (mqmStats.numScoringUnits == 0) {
                continue;
              }
              aggrDocSegSys.metrics[mqm] = mqmStats.score;
              for (const scoringUnit of scoringUnits) {
                const unit = scoringUnit.unit;
                const currUnitStatsBySys = statsBySystem[system][doc][unit];
                const aggrUnitScores = this.aggregateUnitStats(
                    [currUnitStatsBySys]);
                if (!aggrUnitScores.mqmStats.hasOwnProperty(mqm)) {
                  // This generally means that we have an AutoMQM rating but no
                  // human rating.
                  continue;
                }
                const aggrUnitMQMStats = aggrUnitScores.mqmStats[mqm];
                aggrDocSegSys.metricsByUnit[unit] = {
                  ...aggrDocSegSys.metricsByUnit[unit],
                };
                aggrDocSegSys.metricsByUnit[unit][mqm] = aggrUnitMQMStats.score;
              }
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
   * Make the first letter of a severity level uppercase.
   * @param {string} sev
   * @return {string}
   */
  casedSeverity(sev) {
    return sev.charAt(0).toUpperCase() + sev.substr(1);
  }

  /**
   * Sets sets the passed TSV data as the new data and parses it into this.data.
   * @param {string} tsvData
   */
  setData(tsvData) {
    this.startAddingData(1, 'inline data');
    this.addData(tsvData);
  }

  /**
   * This clears filters, closes open menus, to prepare for new data to be
   * added. If the UI setting for appending new data to existing data is not
   * set, then it resets all information derived from or associated with the
   * current data (if any).
   * @param {number} n The number of expected addData() calls (the number of
   *     files or URLs).
   * @param {string} name A name for each data item (such as 'file' or 'URL').
   */
  startAddingData(n, name) {
    this.clearFilters();
    this.closeMenuEntries('');
    this.tooManySegsErrorShown = false;
    this.addDataName = name;
    this.addDataCallsNeeded = n;
    this.addDataCallsDone = 0;
    this.status.innerHTML = 'Loading ' + name + ' 1 of ' + n;
    this.mqmDetailsShown = null;
    if (document.getElementById('marot-load-file-append').checked) {
      return;
    }
    this.docsegs.clear();
    this.data = [];
    this.tsvData = '';
    this.metrics = ['MQM'];
    this.metricsVisible = [];
    this.mqmMetricsVisible = [];
    for (let key in this.metricsInfo) {
      /** Only retain the entry for 'MQM'. */
      if (key == 'MQM') continue;
      delete this.metricsInfo[key];
    }
    this.sortByField = 'metric-0';
    this.sortReverse = false;
  }

  /**
   * Note a specific doc-seg combination. We use this to avoid loading too
   * much data: after MAX_SEGMENTS distinct doc-segs, future addData() calls
   * only add the data for previously seen doc-segs.
   * @param {string} doc
   * @param {string|number} seg
   * @return {boolean} Returns true only if this doc-seg is used.
   */
  noteDocSeg(doc, seg) {
    const docseg = this.aColonB(doc, seg);
    if (this.docsegs.size == this.MAX_SEGMENTS &&
        !this.docsegs.has(docseg)) {
      if (!this.tooManySegsErrorShown) {
        this.errors.insertAdjacentHTML(
            'beforeend',
            'Skipped data for segments after number ' + this.MAX_SEGMENTS);
        this.tooManySegsErrorShown = true;
      }
      return false;
    }
    this.docsegs.add(docseg);
    return true;
  }

  /**
   * Appends the passed data to this.tsvData and parses it into this.data.
   * @param {string|!Array<string>} tsvData
   */
  addData(tsvData) {
    this.addDataCallsDone++;
    this.status.innerHTML = 'Loading ' + this.addDataName + ' ' +
                            (this.addDataCallsDone  + 1) + ' of ' +
                            this.addDataCallsNeeded;
    if (!tsvData) {
      this.errors.insertAdjacentHTML(
          'beforeend',
          'Got empty data from ' + this.addDataName + ' ' +
          this.addDataCallsDone + ' of ' + this.addDataCallsNeeded + '<br>');
    }
    const data = tsvData.split('\n');
    for (let line of data) {
      if (!line.trim()) {
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
        this.errors.insertAdjacentHTML('beforeend',
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
      /** Move "Rater" down from its position in the TSV data. */
      const rater = parts[4];
      parts[this.DATA_COL_SOURCE] = parts[5];
      parts[this.DATA_COL_TARGET] = parts[6];
      parts[this.DATA_COL_RATER] = rater;

      /** Make severity start with an uppercase letter. */
      parts[this.DATA_COL_SEVERITY] =
          this.casedSeverity(parts[this.DATA_COL_SEVERITY]);
      /** If the system name is blank, call it "blank". */
      if (!parts[this.DATA_COL_SYSTEM].trim()) {
        parts[this.DATA_COL_SYSTEM] = 'blank';
      }
      /** If the document name is blank, name it with the fp of source text. */
      if (!parts[this.DATA_COL_DOC].trim()) {
        parts[this.DATA_COL_DOC] = 'source-fp-' +
            MarotUtils.javaHashKey(parts[this.DATA_COL_SOURCE]);
      }

      if (!this.noteDocSeg(this.aColonB(parts[this.DATA_COL_DOC],
                                        this.DATA_COL_DOC_SEG_ID))) {
        continue;
      }

      /** Note any metrics that might be in the data. */
      const metricNames = Object.keys(metadata.segment.metrics);
      if (rater) {
        metricNames.push(this.mqmMetricName(rater));
      }
      for (let metric of metricNames) {
        if (this.metricsInfo.hasOwnProperty(metric)) continue;
        this.metricsInfo[metric] = {
          index: this.metrics.length,
          lowerBetter: this.isMQMOrAutoMQM(metric),
        };
        this.metrics.push(metric);
      }
      /**
       * Count all characters, including spaces, in src/tgt length, excluding
       * the span-marking <v> and </v> tags.
       */
      parts.num_source_chars =
          parts[this.DATA_COL_SOURCE].replace(/<\/?v>/g, '').length;
      parts.num_target_chars =
          parts[this.DATA_COL_TARGET].replace(/<\/?v>/g, '').length;
      this.data.push(parts);
      this.tsvData += line + '\n';
    }
    if (this.addDataCallsDone == this.addDataCallsNeeded) {
      this.status.innerHTML = '';
      this.sortData(this.data);
      this.createDataIter(this.data);
      this.addSegmentAggregations();
      this.setSelectOptions();
      this.recomputeMQM();
      this.addMetricSegmentAggregations();
      this.show();
    }
  }

  /**
   * Opens and reads the data file(s) picked by the user and calls addData().
   */
  openFiles() {
    const filesElt = document.getElementById('marot-file');
    const numFiles = filesElt.files.length;
    if (numFiles <= 0) {
      document.body.style.cursor = 'auto';
      this.errors.innerHTML = 'No files were selected';
      return;
    }
    this.hideViewer();
    this.startAddingData(numFiles, 'file');
    let erroneousFile = '';
    try {
      for (let i = 0; i < numFiles; i++) {
        const f = filesElt.files[i];
        erroneousFile = f.name;
        const fr = new FileReader();
        fr.onload = (evt) => {
          erroneousFile = f.name;
          const fileData = (typeof marotDataConverter == 'function') ?
              marotDataConverter(f.name, fr.result) : fr.result;
          this.addData(fileData);
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
      this.errors.innerHTML = errString;
      filesElt.value = '';
    }
  }

  /**
   * Fetches marot data from the given URLs and calls this.addData().
   * If the marotURLMaker() function exists, then it is applied to each URL
   * first, to get a possibly modified URL.
   * @param {!Array<string>} urls
   */
  fetchURLs(urls) {
    const cleanURLs = [];
    for (let url of urls) {
      if (typeof marotURLMaker == 'function') {
        url = marotURLMaker(url);
      }
      const splitUrl = url.split(',');
      for (const subUrl of splitUrl) {
        const trimmedSubUrl = subUrl.trim();
        if (trimmedSubUrl) {
          cleanURLs.push(trimmedSubUrl);
        }
      }
    }
    if (cleanURLs.length == 0) {
      errors.innerHTML = 'No non-empty URLs found';
      return;
    }
    this.hideViewer();
    this.startAddingData(cleanURLs.length, 'URL');
    for (const url of cleanURLs) {
      fetch(url, {
        mode: 'cors',
        credentials: 'include',
      })
      .then(response => response.text())
      .then(result => {
        const urlData = (typeof marotDataConverter == 'function') ?
            marotDataConverter(url, result) : result;
        this.addData(urlData);
      })
      .catch(error => {
        console.log(error);
        this.addData('');
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
    /** This should not be called when subpara-scoring is enabled. */
    console.assert(!this.subparaScoring);
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
        const unitStats = this.getUnitStatsAsArray(this.statsBySystem[system]);
        const aggregate = this.aggregateUnitStats(unitStats);
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
          const unitStats = this.getUnitStatsAsArray({doc: docStats});
          const aggregate = this.aggregateUnitStats(unitStats);
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
            const unitStats = this.getUnitStatsAsArray(
                {doc: {seg: docSegStats}});
            const aggregate = this.aggregateUnitStats(unitStats);
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
            for (let unit in docStats) {
              const unitStats = this.getUnitStatsAsArray(
                  {doc: {seg: docSegStats}});
              const aggregate = this.aggregateUnitStats(unitStats, true);
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
   *     marot-data.tsv. Adds a header line when saving non-aggregated marot
   *     data, if it's not already there.
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
      if (this.subparaScoring) {
        alert('Cannot save aggregated data when subpara-scoring is enabled.');
        return;
      }
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
   * Helper to get a positive integer from an input (or reset the input to
   * the given current value).
   * @param {!Element} input
   * @param {number} current
   * @return {number}
   */
  getPosIntSetting(input, current) {
    let val = parseInt(input.value);
    if (isNaN(val) || val <= 0) {
      val = current;
    }
    input.value = val;
    return val;
  }

  /**
   * Applies updated settings for scoring.
   */
  updateSettings() {
    this.subparaScoring = this.subparaScoringInput.checked;
    this.subparaSents = this.getPosIntSetting(
        this.subparaSentsInput, this.subparaSents);
    this.subparaTokens = this.getPosIntSetting(
        this.subparaTokensInput, this.subparaTokens);
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
    this.subparaScoringInput.checked = false;
    this.subparaSentsInput.value = this.DEFAULT_SUBPARA_SENTS;
    this.subparaTokensInput.value = this.DEFAULT_SUBPARA_TOKENS;
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
            title="Click to see pairwise ${sysOrRater} difference significance tests, correlations, and scoring unit histograms">
          <span class="marot-section">
            ${fullName} score confidences, comparisons, and histograms
          </span>
        </summary>
        <p><b>${fullName} score confidence intervals and
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
            Missing values indicate a lack of enough common scoring units.
            ${fullName}s above any solid line
            ${sysOrRater == 'sys' ? ' are significantly better than' :
                            ' give MQM scores signinificantly lower than'}
            those below. Dotted lines identify clusters within which no
            ${fullNameLC}'s score is significantly better than any other
            ${fullNameLC}.
            <span class="marot-warning"
                id="marot-${sysOrRater}-sigtests-msg"></span>
          </p>
        </div>
        <p><b>${fullName} and ${fullNameLC}-pair scoring unit histograms</b></p>
        <div class="marot-comparison-histograms">
          <div class="marot-${versus}-header">
            <label>
              <b>${fullName} 1:</b>
              <select id="marot-${versus}-1"
                 onchange="marot.showCmp('${sysOrRater}')"></select>
            </label>
            <span id="marot-${versus}-1-units"></span> scoring unit(s).
            <label>
              <b>${fullName} 2:</b>
              <select id="marot-${versus}-2"
                 onchange="marot.showCmp('${sysOrRater}')"></select>
            </label>
            <span id="marot-${versus}-2-units"></span> scoring unit(s)
            (<span id="marot-${versus}-xunits"></span> common).
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
          <div class="marot-settings-row">
            <input type="checkbox" id="marot-subpara-scoring"
                onchange="marot.updateSettings()"/>
            For MQM, use sub-para scoring units instead of segments
          </div>
          <div class="marot-settings-row">
            Sub-para limits:
            sentences:
            <input size="3" type="text"
                id="marot-subpara-sents"
                value="${this.subparaSents}"
                onchange="marot.updateSettings()"/>
            tokens:
            <input size="4" type="text"
                id="marot-subpara-tokens"
                value="${this.subparaTokens}"
                onchange="marot.updateSettings()"/>
          </div>
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
    <style id="marot-custom-styles">
    </style>
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
    <div id="marot-status"></div>

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

    this.customStyles = document.getElementById('marot-custom-styles');
    this.errors = document.getElementById('marot-errors');
    this.status = document.getElementById('marot-status');
    this.quote = document.getElementById('marot-quote');
    this.viewer = document.getElementById('marot-viewer');
    this.segmentsTable = document.getElementById('marot-segments-tbody');
    this.statsTable = document.getElementById('marot-stats-tbody');
    this.sevcatStatsTable = document.getElementById('marot-sevcat-stats-tbody');
    this.eventsTable = document.getElementById('marot-events-tbody');
    this.raterRelatedSections = document.getElementById(
        'marot-rater-related-sections');

    this.subparaScoringInput = document.getElementById(
        'marot-subpara-scoring');

    this.subparaSentsInput = document.getElementById(
        'marot-subpara-sents');
    this.subparaTokensInput = document.getElementById(
        'marot-subpara-tokens');
    this.resetSettings();

    this.hideViewer();

    document.addEventListener('keydown', this.subparaNavKeyDown.bind(this));

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
