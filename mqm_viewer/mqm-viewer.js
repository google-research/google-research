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
 * This file contains the JavaScript code for MQM Viewer.
 */

/**
 * Raw data read from the data file. Each entry is an array with 10 entries,
 * in this order (slightly different from the original order in the TSV data,
 * as we keep the fields in their more natural presentation order, as used in
 * the HTML table we display):
 *
 *     0: system, 1: doc, 2: docSegId, 3: globalSegId, 4: source, 5: target,
 *     6: rater, 7: category, 8: severity, 9: metadata
 *
 * The docSegId field is the 1-based index of the segment within the doc.
 *
 * The globalSegId field is an arbitrary, application-specific segment
 * identifier. If such an identifier is not needed or available, then set this
 * field to some constant value, such as 0. It is ignored by MQM Viewer, but is
 * available for use in filter expressions.
 *
 * The last field, "metadata", is an object that includes the timestamp of
 * the rating, any note the rater may have left, and other metadata.
 *
 * There is a special severity, "HOTW-test", reserved for hands-on-the-wheel
 *   test results. These are test sentences in which a deliberate error is
 *   injected, just to help the rater stay on task. The test result is captured
 *   by category = "Found" or "Missed" and there is a note in the metadata that
 *   captures the injected error.
 */
let mqmData = [];

/**
 * A data structure that provides a convenient way to iterate over mqmData in
 * nested loops on doc, docSegId, system.
 */
let mqmDataIter = {
  docs: [],
  docSegs: {},
  docSys: {},
  docSegSys: {},
  systems: [],  /** Convenient list of all systems in the data. */
};

/**
 * mqmDataFiltered has exactly the same format as mqmData, except that it
 * is limited to the current filters in place. It contains its metadata field
 * in its JSON-encoded form.
 */
let mqmDataFiltered = [];

/** Array indices in each mqmData */
const MQM_DATA_SYSTEM = 0;
const MQM_DATA_DOC = 1;
const MQM_DATA_DOC_SEG_ID = 2;
const MQM_DATA_GLOBAL_SEG_ID = 3;
const MQM_DATA_SOURCE = 4;
const MQM_DATA_TARGET = 5;
const MQM_DATA_RATER = 6;
const MQM_DATA_CATEGORY = 7;
const MQM_DATA_SEVERITY = 8;
const MQM_DATA_METADATA = 9;
const MQM_DATA_NUM_PARTS = 10;

/** Column filter id mappings */
const mqmFilterColumns = {
  'mqm-filter-doc': MQM_DATA_DOC,
  'mqm-filter-doc-seg': MQM_DATA_DOC_SEG_ID,
  'mqm-filter-system': MQM_DATA_SYSTEM,
  'mqm-filter-source': MQM_DATA_SOURCE,
  'mqm-filter-target': MQM_DATA_TARGET,
  'mqm-filter-rater': MQM_DATA_RATER,
  'mqm-filter-category': MQM_DATA_CATEGORY,
  'mqm-filter-severity': MQM_DATA_SEVERITY,
};

/**
 * If TSV data was supplied (instead of being chosen from a file), then it is
 * saved here (for possible downloading).
 */
let mqmTSVData = '';

/**
 * The first two mqmStats* objects are keyed by [system][doc][docSegId]. Apart
 * from all the systems, an additional, special system value ('_MQM_TOTAL_') is
 * used in both, for aggregates over all systems (for this aggregate, the
 * doc key used is doc:system). mqmStatsByRater is first keyed by [rater] (and
 * then by [system][doc][docSegId]). Each keyed entry is an array of per-rater
 * stats (scores, score slices, error counts) for that segment
 * (system+doc+docSegId).
 *
 * mqmSevCatStats[severity][category][system] is the total count of
 * annotations of a specific severity+category in a specific system.
 *
 * Each mqmStats* object is recomputed for any filtering applied to the data.
 */
let mqmStats = {};
let mqmStatsByRater = {};
let mqmSevCatStats = {};

/** {!Element} HTML table body elements for various tables */
let mqmTable = null;
let mqmStatsTable = null;
let mqmSevCatStatsTable = null;
let mqmEventsTable = null;

/** Events timing info for current filtered data. **/
let mqmEvents = {};

/**
 * Max number of annotations to show in the sample of ratings shown. Note that
 * this is not a hard limit, as we include all systems + raters for any
 * document segment that pass the current filter (if any).
 */
let mqmLimit = 200;

/** Clause built by helper menus, for appending to the filter expression **/
let mqmClause = '';

/** UI elements for clause-builder. */
let mqmClauseKey;
let mqmClauseInclExcl;
let mqmClauseSev;
let mqmClauseCat;
let mqmClauseAddAnd;
let mqmClauseAddOr;

/** Selected system names for system-v-system comparison. */
let mqmSysVSys1;
let mqmSysVSys2;

/** A distinctive name used as the key for aggregate stats. */
const MQM_TOTAL = '_MQM_TOTAL_';

const MQM_PVALUE_THRESHOLD = 0.05;
const MQM_SIGTEST_TRIALS = 10000;

/**
 * An object that captures all the data needed for running signigicance
 * tests on one particular metric.
 */
function MQMMetricSigtestsData() {
  /** {boolean} */
  this.lowerBetter = false;
  /**
   * {!Array<string>} Sorted array ordered by degrading scores.
   */
  this.systems = [];
   /**
    * {!Object} Scores by system. Each score itself is an object containing
    *     score and scoreDenominator.
    */
  this.scoresBySystem = {};
  /**
   * {!Object} Segment scores by system. Each value is an array of scores that
   *     are aligned such that elements at the n-th position of all arrays
   *     correspond to the same segment. Note that some scores might be null
   *     since some systems might be missing ratings for some segments.
   */
  this.segScoresBySystem = {};
  /**
   * {!Object} Common segments shared by a pair of systems. This stores
   *     positions in segScoresBySystem.
   */
  this.commonPosBySystemPair = {};
  /** {!Array<!Array<number>>} Computed matric of p-values. */
  this.pValues = [];
}

/**
 * An object with data for computing significance tests. This data is sent to a
 * background Worker thread. See computation details in mqm-sigtests.js. The
 * object metricData[] has one entry for each metric in mqmMetricsVisible[].
 */
let mqmSigtestsData = {
  metricData: {},
  /** {number} Number of trials. */
  numTrials: MQM_SIGTEST_TRIALS,
};

/** {!Worker} A background Worker thread that computes sigtests */
let mqmSigtestsWorker = null;
/**
 * The Sigtests Worker loads its code from 'mqm-sigtests.js'. If that file is
 * not servable for some reason, then set the mqmSigtestsWorkerJS variable
 * to its contents.
 */
let mqmSigtestsWorkerJS = '';
/** {!Element} An HTML span that shows a sigtests computation status message. */
let mqmSigtestsMsg = null;

/**
 * Scoring weights. Each weight has a name and a regular expression pattern
 * for matching <severity>:<category>[/<subcategory>] (case-insensitively).
 * The weights are tried out in the sequence shown and for a given annotation,
 * the first matching weight is used. While you can interactively change these
 * for experimentation, you should set this default array to values suitable
 * for your application. The best place to do this is in your own version of
 * mqm-viewer.html.
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
 * mqmWeights and mqmSlices are set from current settings in
 * mqmParseScoreSettings() and mqmResetSettings().
 */
let mqmWeights = [];
let mqmSlices = [];

/**
 * Score aggregates include 'mqm-weighted-" and "mqm-slice-" prefixed
 * scores. The names beyond the prefixes are taken from the "name" field in
 * mqmWeights and mqmSlices.
 */
const MQM_SCORE_WEIGHTED_PREFIX = 'mqm-weighted-';
const MQM_SCORE_SLICE_PREFIX = 'mqm-slice-';

/**
 * Arrays of names of currently being displayed score components, sorted in
 * decreasing score order.
 */
let mqmScoreWeightedFields = [];
let mqmScoreSliceFields = [];

/**
 * Scoring unit. If false, segments are used for scoring. If true, scores
 * are computed per "100 source characters".
 */
let mqmCharScoring = false;

/**
 * The field to sort the score table rows by. By default, sort by
 * overall MQM score. `mqmSortReverse` indicates whether it is sorted in
 * ascending order (false, default) or descending order (true).
 *
 * The value of this is something like 'metric-<k>' (where k is an index into
 * mqmMetrics[]), or a name from mqmSoreWeightedFields[]/mqmScoreSliceFields[].
 */
let mqmSortByField = 'metric-0';
let mqmSortReverse = false;

/**
 * All metrics possibly available in the current data. The entries will be like
 * 'MQM', 'BLEURT-X', etc. 'MQM' is the always the first entry in this array.
 * {!Array<string>} Indices into mqmMetrics.
 */
let mqmMetrics = ['MQM'];
/**
 * Info about metrics.
 */
const mqmMetricsInfo = {
  'MQM': {
    index: 0,  /** index into mqmMetrics[] */
    lowerBetter: true,  /** default is false */
  },
};
/**
 * The metrics that are available for the data with the current filtering.
 * {!Array<number>} Indices into mqmMetrics.
 */
let mqmMetricsVisible = [];

/**
 * Listener for changes to the input field that specifies the limit on
 * the number of rows shown.
 */
function setMqmLimit() {
  const limitElt = document.getElementById('mqm-limit');
  const limit = limitElt.value.trim();
  if (limit > 0) {
    mqmLimit = limit;
    mqmShow();
  } else {
    limitElt.value = mqmLimit;
  }
}

/**
 * This function returns a "comparable" version of docSegId by padding it
 * with leading zeros. When docSegId is a non-negative integer (reasonably
 * bounded), then this ensures numeric ordering.
 * @param {string} s
 * @return {string}
 */
function mqmCmpDocSegId(s) {
  return ('' + s).padStart(10, '0');
}

/**
 * This sorts 10-column MQM data by fields in the order doc, docSegId, system,
 *   rater, severity, category.
 * @param {!Array<!Array>} data The MQM-10-column data to be sorted.
 */
function mqmSortData(data) {
  data.sort((e1, e2) => {
    let diff = 0;
    const docSegId1 = mqmCmpDocSegId(e1[MQM_DATA_DOC_SEG_ID]);
    const docSegId2 = mqmCmpDocSegId(e2[MQM_DATA_DOC_SEG_ID]);
    if (e1[MQM_DATA_DOC] < e2[MQM_DATA_DOC]) {
      diff = -1;
    } else if (e1[MQM_DATA_DOC] > e2[MQM_DATA_DOC]) {
      diff = 1;
    } else if (docSegId1 < docSegId2) {
      diff = -1;
    } else if (docSegId1 > docSegId2) {
      diff = 1;
    } else if (e1[MQM_DATA_SYSTEM] < e2[MQM_DATA_SYSTEM]) {
      diff = -1;
    } else if (e1[MQM_DATA_SYSTEM] > e2[MQM_DATA_SYSTEM]) {
      diff = 1;
    } else if (e1[MQM_DATA_RATER] < e2[MQM_DATA_RATER]) {
      diff = -1;
    } else if (e1[MQM_DATA_RATER] > e2[MQM_DATA_RATER]) {
      diff = 1;
    } else if (e1[MQM_DATA_SEVERITY] < e2[MQM_DATA_SEVERITY]) {
      diff = -1;
    } else if (e1[MQM_DATA_SEVERITY] > e2[MQM_DATA_SEVERITY]) {
      diff = 1;
    } else if (e1[MQM_DATA_CATEGORY] < e2[MQM_DATA_CATEGORY]) {
      diff = -1;
    } else if (e1[MQM_DATA_CATEGORY] > e2[MQM_DATA_CATEGORY]) {
      diff = 1;
    }
    return diff;
  });
}

/**
 * Sets mqmDataIter to a data structure that can be used to iterate over
 * mqmData[] rows by looping over documents, segments, and systems.
 */
function mqmCreateDataIter() {
  mqmDataIter = {
    docs: [],
    docSegs: {},
    docSys: {},
    docSegSys: {},
    systems: [],
    evaluation: {},
  };
  let lastRow = null;
  const systemsSet = new Set();
  for (let rowId = 0; rowId < mqmData.length; rowId++) {
    const parts = mqmData[rowId];
    const doc = parts[MQM_DATA_DOC];
    const docSegId = parts[MQM_DATA_DOC_SEG_ID];
    const system = parts[MQM_DATA_SYSTEM];
    systemsSet.add(system);
    const sameDoc = lastRow && (doc == lastRow[MQM_DATA_DOC]);
    const sameDocSeg = sameDoc && (docSegId == lastRow[MQM_DATA_DOC_SEG_ID]);
    const sameDocSys = sameDoc && (system == lastRow[MQM_DATA_SYSTEM]);
    if (!sameDoc) {
      mqmDataIter.docs.push(doc);
      mqmDataIter.docSegs[doc] = [];
      mqmDataIter.docSys[doc] = [];
    }
    if (!sameDocSeg) {
      console.assert(!mqmDataIter.docSegs[doc].includes(docSegId),
                     doc, docSegId);
      mqmDataIter.docSegs[doc].push(docSegId);
    }
    if (!sameDocSys && !mqmDataIter.docSys[doc].includes(system)) {
      mqmDataIter.docSys[doc].push(system);
    }
    lastRow = parts;
  }
  mqmDataIter.systems = [...systemsSet];
  /**
   * Ensure that there are entries in docSegSys for each
   * docSegId x system.
   */
  for (doc of mqmDataIter.docs) {
    mqmDataIter.docSegSys[doc] = {};
    for (docSegId of mqmDataIter.docSegs[doc]) {
      mqmDataIter.docSegSys[doc][docSegId] = {};
      for (system of mqmDataIter.systems) {
        mqmDataIter.docSegSys[doc][docSegId][system] = {
          rows: [-1, -1],
          segment: {},
        };
      }
    }
  }
  lastRow = null;
  let segment = null;
  for (let rowId = 0; rowId < mqmData.length; rowId++) {
    const parts = mqmData[rowId];
    const doc = parts[MQM_DATA_DOC];
    const docSegId = parts[MQM_DATA_DOC_SEG_ID];
    const system = parts[MQM_DATA_SYSTEM];
    const metadata = parts[MQM_DATA_METADATA];
    if (metadata.evaluation) {
      mqmDataIter.evaluation = {
        ...mqmDataIter.evaluation,
        ...metadata.evaluation
      };
    }
    const sameDoc = lastRow && (doc == lastRow[MQM_DATA_DOC]);
    const sameDocSeg = sameDoc && (docSegId == lastRow[MQM_DATA_DOC_SEG_ID]);
    const sameDocSegSys = sameDocSeg && (system == lastRow[MQM_DATA_SYSTEM]);

    if (!sameDocSegSys) {
      mqmDataIter.docSegSys[doc][docSegId][system].rows =
          [rowId, rowId + 1];
      segment = metadata.segment || {};
    } else {
      mqmDataIter.docSegSys[doc][docSegId][system].rows[1] = rowId + 1;
    }
    mqmDataIter.docSegSys[doc][docSegId][system].segment = segment;
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
function mqmAddToArray(obj, key, val) {
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
function mqmBinSearch(arr, elt) {
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
function mqmTokenizeLegacyText(annotations) {
  let cleanText = '';
  for (let text of annotations) {
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
  for (let token of tokens) {
    tokenOffsets.push(tokenOffset);
    tokenOffset += token.length;
  }

  const MARKERS = ['<v>', '</v>'];
  const markerOffsets = [];
  for (let text of annotations) {
    const offsets = [];
    let markerIdx = 0;
    let modText = text;
    let x;
    while ((x = modText.indexOf(MARKERS[markerIdx])) >= 0) {
      const marker = MARKERS[markerIdx];
      offsets.push(x);
      modText = modText.substr(0, x) + modText.substr(x + marker.length);
      markerIdx = 1 - markerIdx;

      const loc = mqmBinSearch(tokenOffsets, x);
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
  for (let offsets of markerOffsets) {
    const spans = [];
    for (let i = 0; i < offsets.length; i+= 2) {
      if (i + 1 >= offsets.length) break;
      spans.push([mqmBinSearch(tokenOffsets, offsets[i]),
                  mqmBinSearch(tokenOffsets, offsets[i + 1]) - 1]);
    }
    spansList.push(spans);
  }
  return {
    tokens: tokens,
    spans: spansList,
  };
}

/**
 * Given the full range of rows for the same doc+docSegId+system, tokenizes the
 * source and target side using spaces, but refining the tokenization to make
 * each <v> and </v> fall on a token boundary. Sets
 * segment.{source,target}_tokens as well as
 *     mqmData[row][MQM_DATA_METADATA].{source,target}_spans.
 *
 * If segment.source/target_tokens is already present in the data (as
 * will be the case with newer data), this function is a no-op.
 * If rowRange does not cover any rows, then this function is a no-op.
 * @param {!Array<number>} rowRange The start (inclusive) and limit (exclusive)
 *     rowId for the segment, in mqmData[].
 * @param {!Object} segment The segment-level aggregate data.
 */
function mqmTokenizeLegacySegment(rowRange, segment) {
  if (rowRange[0] >= rowRange[1]) {
    return;
  }
  const sources = [];
  const targets = [];
  for (let row = rowRange[0]; row < rowRange[1]; row++) {
    const parts = mqmData[row];
    sources.push(parts[MQM_DATA_SOURCE]);
    targets.push(parts[MQM_DATA_TARGET]);
  }
  const sourceTokenization = mqmTokenizeLegacyText(sources);
  segment.source_tokens = sourceTokenization.tokens;
  const targetTokenization = mqmTokenizeLegacyText(targets);
  segment.target_tokens = targetTokenization.tokens;
  for (let row = rowRange[0]; row < rowRange[1]; row++) {
    const parts = mqmData[row];
    const idx = row - rowRange[0];
    parts[MQM_DATA_METADATA].source_spans = sourceTokenization.spans[idx];
    parts[MQM_DATA_METADATA].target_spans = targetTokenization.spans[idx];
  }
}

/**
 * Aggregates mqmData, collecting all data for a particular segment translation
 *     (i.e., for a given (doc, docSegId) pair) into the aggrDocSeg object in
 *     the metadata.segment field, adding to it the following properties:
 *         {cats,sevs,sevcats}By{Rater,System}.
 *     Each of these properties is an object keyed by system or rater, with the
 *     values being arrays of strings that are categories, severities,
 *     and <sev>[/<cat>], * respectively.
 *
 *     Also added are aggrDocSeg.metrics[metric][system] values.
 * Makes sure that the metadata.segment object is common for each row from
 * the same doc+seg+sys.
 */
function mqmAddSegmentAggregations() {
  for (doc of mqmDataIter.docs) {
    const aggrDoc = {
      doc: doc,
      thumbsUpCount: 0,
      thumbsDownCount: 0,
    };
    for (docSegId of mqmDataIter.docSegs[doc]) {
      aggrDocSeg = {
        catsBySystem: {},
        catsByRater: {},
        sevsBySystem: {},
        sevsByRater: {},
        sevcatsBySystem: {},
        sevcatsByRater: {},
        metrics: {},
        aggrDoc: aggrDoc,
      };
      for (system of mqmDataIter.docSys[doc]) {
        const range = mqmDataIter.docSegSys[doc][docSegId][system].rows;
        let aggrDocSegSys = {
          aggrDocSeg: aggrDocSeg,
          metrics: {},
        };
        for (let rowId = range[0]; rowId < range[1]; rowId++) {
          const parts = mqmData[rowId];
          const segment = parts[MQM_DATA_METADATA].segment || {};
          if (segment.hasOwnProperty('metrics')) {
            aggrDocSegSys.metrics = {
              ...segment.metrics,
              ...aggrDocSegSys.metrics,
            };
          }
          aggrDocSegSys = {...segment, ...aggrDocSegSys};
        }
        for (metric in aggrDocSegSys.metrics) {
          if (!aggrDocSeg.metrics.hasOwnProperty(metric)) {
            aggrDocSeg.metrics[metric] = {};
          }
          aggrDocSeg.metrics[metric][system] = aggrDocSegSys.metrics[metric];
        }
        if (!aggrDocSegSys.source_tokens ||
            aggrDocSegSys.source_tokens.length == 0) {
          mqmTokenizeLegacySegment(range, aggrDocSegSys);
        }
        if (!aggrDocSeg.hasOwnProperty('source_tokens') &&
            aggrDocSegSys.hasOwnProperty('source_tokens')) {
          aggrDocSeg.source_tokens = aggrDocSegSys.source_tokens;
        }
        if (!aggrDocSeg.hasOwnProperty('source_sentence_tokens') &&
            aggrDocSegSys.hasOwnProperty('source_sentence_tokens')) {
          aggrDocSeg.source_sentence_tokens =
              aggrDocSegSys.source_sentence_tokens;
        }
        if (!aggrDocSeg.hasOwnProperty('starts_paragraph') &&
            aggrDocSegSys.hasOwnProperty('starts_paragraph')) {
          aggrDocSeg.starts_paragraph = aggrDocSegSys.starts_paragraph;
        }
        if (aggrDocSegSys.hasOwnProperty('references')) {
          if (!aggrDocSeg.hasOwnProperty('references')) {
            aggrDocSeg.references = {};
          }
          aggrDocSeg.references = {
            ...aggrDocSeg.references,
            ...aggrDocSegSys.references
          };
        }
        if (!aggrDocSeg.hasOwnProperty('primary_reference') &&
            aggrDocSegSys.hasOwnProperty('primary_reference')) {
          aggrDocSeg.primary_reference = aggrDocSegSys.primary_reference;
        }
        for (let rowId = range[0]; rowId < range[1]; rowId++) {
          const parts = mqmData[rowId];
          const metadata = parts[MQM_DATA_METADATA];
          metadata.segment = aggrDocSegSys;
          mqmSetMarkedText(rowId, metadata);

          const rater = parts[MQM_DATA_RATER];
          if (!rater) {
            /**
             * This row is purely for metadata, such as references and/or
             * automated metrics
             */
            continue;
          }
          const category = parts[MQM_DATA_CATEGORY];
          const severity = parts[MQM_DATA_SEVERITY];

          mqmAddToArray(aggrDocSeg.catsBySystem, system, category);
          mqmAddToArray(aggrDocSeg.catsByRater, rater, category);
          mqmAddToArray(aggrDocSeg.sevsBySystem, system, severity);
          mqmAddToArray(aggrDocSeg.sevsByRater, rater, severity);
          const sevcat = severity + (category ? '/' + category : '');
          mqmAddToArray(aggrDocSeg.sevcatsBySystem, system, sevcat);
          mqmAddToArray(aggrDocSeg.sevcatsByRater, rater, sevcat);
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
function mqmGetAllFilters() {
  const res = {};
  const filters = document.getElementsByClassName('mqm-filter-re');
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
  const filterExpr = document.getElementById('mqm-filter-expr').value.trim();
  const onlyAllSysSegs = document.getElementById(
      'mqm-only-all-systems-segments').checked;
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
function mqmIncl(obj, prop, val) {
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
function mqmExcl(obj, prop, val) {
  return obj.hasOwnProperty(prop) &&
    !obj[prop].includes(val);
}

/**
 * Clears mqmClause and disables associated buttons.
 */
function mqmClearClause() {
  mqmClause = '';
  mqmClauseKey.value = '';
  mqmClauseInclExcl.value = 'includes';
  mqmClauseSev.value = '';
  mqmClauseCat.value = '';
  mqmClauseAddAnd.disabled = true;
  mqmClauseAddOr.disabled = true;
}

/**
 * Checks if filter expression clause is fully specified, enables "add"
 *     buttons if so.
 */
function mqmCheckClause() {
  mqmClause = '';
  mqmClauseAddAnd.disabled = true;
  mqmClauseAddOr.disabled = true;
  if (!mqmClauseKey.value) return;
  if (!mqmClauseSev.value && !mqmClauseCat.value) return;

  let sevcats = 'aggrDocSeg.sevcats';
  let key = '';
  let err = mqmClauseSev.value + '/' + mqmClauseCat.value;
  if (!mqmClauseSev.value) {
    sevcats = 'aggrDocSeg.cats';
    err = mqmClauseCat.value;
  }
  if (!mqmClauseCat.value) {
    sevcats = 'aggrDocSeg.sevs';
    err = mqmClauseSev.value;
  }
  if (mqmClauseKey.value.startsWith('System: ')) {
    sevcats += 'BySystem';
    key = mqmClauseKey.value.substr(8);
  } else {
    console.assert(mqmClauseKey.value.startsWith('Rater: '),
                   mqmClauseKey.value);
    sevcats += 'ByRater';
    key = mqmClauseKey.value.substr(7);
  }
  const inclexcl = (mqmClauseInclExcl.value == 'excludes') ? 'mqmExcl' :
      'mqmIncl';
  mqmClause = `${inclexcl}(${sevcats}, "${key}", "${err}")`;
  mqmClauseAddAnd.disabled = false;
  mqmClauseAddOr.disabled = false;
}

/**
 * Adds mqmClause with and/or to the filter expression.
 * @param {string} andor
 */
function mqmAddClause(andor) {
  if (!mqmClause) return;
  const elt = document.getElementById('mqm-filter-expr');
  let expr = elt.value.trim();
  if (expr) expr += ' ' + andor + ' ';
  expr += mqmClause;
  elt.value = expr;
  mqmClearClause();
  mqmShow();
}

/**
 * Evaluates the JavaScript filterExpr on an mqmData[] row and returns true
 *     only if the filter passes.
 * @param {string} filterExpr
 * @param {!Array<string>} parts
 * @return {boolean}
 */
function mqmFilterExprPasses(filterExpr, parts) {
  if (!filterExpr.trim()) return true;
  try {
    return Function(
        '"use strict";' +
        `
    const system = arguments[MQM_DATA_SYSTEM];
    const doc = arguments[MQM_DATA_DOC];
    const docSegId = arguments[MQM_DATA_DOC_SEG_ID];
    const globalSegId = arguments[MQM_DATA_GLOBAL_SEG_ID];
    const source = arguments[MQM_DATA_SOURCE];
    const target = arguments[MQM_DATA_TARGET];
    const rater = arguments[MQM_DATA_RATER];
    const category = arguments[MQM_DATA_CATEGORY];
    const severity = arguments[MQM_DATA_SEVERITY];
    const metadata = arguments[MQM_DATA_METADATA];
    const segment = metadata.segment;
    const aggrDocSegSys = segment;
    const aggrDocSeg = aggrDocSegSys.aggrDocSeg;
    const aggrDoc = aggrDocSeg.aggrDoc;` +
        'return (' + filterExpr + ')')(
        parts[MQM_DATA_SYSTEM], parts[MQM_DATA_DOC],
        parts[MQM_DATA_DOC_SEG_ID], parts[MQM_DATA_GLOBAL_SEG_ID],
        parts[MQM_DATA_SOURCE], parts[MQM_DATA_TARGET],
        parts[MQM_DATA_RATER], parts[MQM_DATA_CATEGORY],
        parts[MQM_DATA_SEVERITY], parts[MQM_DATA_METADATA]);
  } catch (err) {
    document.getElementById('mqm-filter-expr-error').innerHTML = err;
    return false;
  }
}

/**
 * This function will return false for segments that have some metric (MQM or
 * other) only available for some of the systems, not all/none.
 * @param {!Object} metadata
 * @return {boolean}
 */
function mqmAllSystemsFilterPasses(metadata) {
  const segment = metadata.segment;
  const aggrDocSeg = segment.aggrDocSeg;
  for (let metric in aggrDocSeg.metrics) {
    const numSystemsWithMetric = Object.keys(aggrDocSeg.metrics[metric]).length;
    if (numSystemsWithMetric > 0 &&
        numSystemsWithMetric != mqmDataIter.systems.length) {
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
function mqmLogRowMetadata() {
  const rowInput = document.getElementById('mqm-view-metadata-row');
  const rowInputVal = rowInput.value.trim();
  if (!rowInputVal) return;
  const row = parseInt(rowInputVal);
  if (row < 0 || row >= mqmData.length) {
    console.log(`Row must be in the range 0-${mqmData.length - 1}`);
    rowInput.value = '';
    return;
  }
  const doc = mqmData[row][MQM_DATA_DOC];
  const docSegId = mqmData[row][MQM_DATA_DOC_SEG_ID];
  const system = mqmData[row][MQM_DATA_SYSTEM];
  const rater = mqmData[row][MQM_DATA_RATER];
  console.log('Metadata for row ' + row +
              ' - doc [' + doc + '], docSegId [' + docSegId +
              '], system [' + system + '], rater [' + rater + ']:');
  console.log(mqmData[row][MQM_DATA_METADATA]);
  console.log('Note that aggrDocSegSys is an alias for metadata.segment, ' +
              'aggrDocSeg for aggrDocSegSys.aggrDocSeg, ' +
              'and aggrDoc for aggrDocSeg.aggrDoc');
}

/**
 * In the weights/slices settings table with the given element id, add a row.
 * @param {string} id
 * @param {number} cols The number of columns to use.
 */
function mqmSettingsAddRow(id, cols) {
  let html = '<tr>';
  for (let i = 0; i < cols; i++) {
    html += `
        <td><span contenteditable="true"
                 class="mqm-settings-editable"></span></td>`;
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
function mqmSetUpScoreSettings() {
  const weightSettings = document.getElementById('mqm-settings-weights');
  weightSettings.innerHTML = '';
  for (let sc of mqmWeights) {
    sc.regex = new RegExp(sc.pattern, 'i');
    weightSettings.insertAdjacentHTML('beforeend', `
        <tr>
          <td><span contenteditable="true"
                   class="mqm-settings-editable">${sc.name}</span></td>
          <td><span contenteditable="true"
                   class="mqm-settings-editable">${sc.pattern}</span></td>
          <td><span contenteditable="true"
                   class="mqm-settings-editable">${sc.weight}</span></td>
        </tr>`);
  }
  const sliceSettings = document.getElementById('mqm-settings-slices');
  sliceSettings.innerHTML = '';
  for (let sc of mqmSlices) {
    sc.regex = new RegExp(sc.pattern, 'i');
    sliceSettings.insertAdjacentHTML('beforeend', `
        <tr>
          <td><span contenteditable="true"
                   class="mqm-settings-editable">${sc.name}</span></td>
          <td><span contenteditable="true"
                   class="mqm-settings-editable">${sc.pattern}</span></td>
        </tr>`);
  }
}

/**
 * Parses score weights/slices from the user-edited table identified by id.
 * If there are errors in parsing then they are displayed to the user in the
 * mqm-errors element and null is returned.
 * @param {string} id
 * @param {boolean} hasWeight True if this is the weights table.
 * @return {?Array<!Object>} Array of parsed weights/slices, or null if errors.
 */
function mqmParseScoreSettingsInner(id, hasWeight) {
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
  const errorsElt = document.getElementById('mqm-errors');
  for (let error of errorsFound) {
    errorsElt.insertAdjacentHTML('beforeend', `<div>${error}</div>\n`);
  }
  return (errorsFound.length == 0) ? parsed : null;
}

/**
 * Parses score weights and slices from the user-edited settings tables. Sets
 * mqmWeights and mqmSlices if successful.
 * @return {boolean} True if the parsing was successful.
 */
function mqmParseScoreSettings() {
  const errors = document.getElementById('mqm-errors');
  errors.innerHTML = '';
  const newWeights = mqmParseScoreSettingsInner('mqm-settings-weights', true);
  const newSlices = mqmParseScoreSettingsInner('mqm-settings-slices', false);
  if (!newWeights || !newSlices) {
    return false;
  }
  mqmWeights = newWeights;
  mqmSlices = newSlices;
  return true;
}

/**
 * This checks if the annotation matches the pattern in the score weight/slice
 * component.
 * @param {!Object} sc Score component, with a regex property.
 * @param {string} sev Severity of the annotation.
 * @param {string} cat Category (and optional "/" + subcat.) of the annotation.
 * @return {boolean}
 */
function mqmMatchesScoreSplit(sc, sev, cat) {
  return sc.regex.test(sev + ':' + cat);
}

/**
 * Returns a string that shows the value of the metric to three decimal places.
 * If denominator is <= 0, then returns "-".
 * @param {number} metric
 * @param {number} denominator
 * @return {string}
 */
function mqmMetricDisplay(metric, denominator) {
  return (denominator > 0) ? metric.toFixed(3) : '-';
}

/**
 * Initializes and returns a rater stats object.
 * @param {string} rater
 * @return {!Object}
 */
function mqmInitRaterStats(rater) {
  return {
    rater: rater,
    score: 0,
    scoreDenominator: 0,

    errorSpans: 0,
    numWithErrors: 0,

    hotwFound: 0,
    hotwMissed: 0,
    timeSpentMS: 0,
  };
}

/**
 * Creates the key for a score weighted component or slices.
 * @param {string} name
 * @param {boolean=} isSlice
 * @return {string}
 */
function mqmScoreKey(name, isSlice = false) {
  return (isSlice ? MQM_SCORE_SLICE_PREFIX : MQM_SCORE_WEIGHTED_PREFIX) + name;
}

/**
 * Strips the prefix from a key for a score component (previously assembled by
 * mqmScoreKey).
 * @param {string} key
 * @return {string}
 */
function mqmScoreKeyToName(key) {
  if (key.startsWith(MQM_SCORE_WEIGHTED_PREFIX)) {
    return key.substr(MQM_SCORE_WEIGHTED_PREFIX.length);
  } else if (key.startsWith(MQM_SCORE_SLICE_PREFIX)) {
    return key.substr(MQM_SCORE_SLICE_PREFIX.length);
  }
  return key;
}

/**
 * Appends stats from delta into raterStats.
 * @param {!Object} raterStats
 * @param {!Object} delta
 */
function mqmAddRaterStats(raterStats, delta) {
  raterStats.score += delta.score;
  for (sc of mqmWeights) {
    const key = mqmScoreKey(sc.name);
    if (delta[key]) {
      raterStats[key] = (raterStats[key] ?? 0) + delta[key];
    }
  }
  for (sc of mqmSlices) {
    const key = mqmScoreKey(sc.name, true);
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
function mqmAvgRaterStats(raterStats, num) {
  if (!num) return;
  raterStats.score /= num;
  raterStats.timeSpentMS /= num;
  for (sc of mqmWeights) {
    const key = mqmScoreKey(sc.name);
    if (raterStats[key]) {
      raterStats[key] /= num;
    }
  }
  for (sc of mqmSlices) {
    const key = mqmScoreKey(sc.name, true);
    if (raterStats[key]) {
      raterStats[key] /= num;
    }
  }
}

/**
 * Aggregates segment stats. This returns an object that has aggregate MQM score
 * in the "score" field and these additional properties:
 *       scoreDenominator
 *       numSegments
 *       numSrcChars
 *       numRatings
 *       metrics
 *       metric-[index in mqmMetrics]
 *           (repeated from metrics[...].score, as a convenient sorting key)
 *       timeSpentMS
 * @param {!Array} segs
 * @return {!Object}
 */
function mqmAggregateSegStats(segs) {
  const aggregates = mqmInitRaterStats('');
  aggregates.metrics = {};
  if (!segs || !segs.length) {
    aggregates.score = 0;
    aggregates.scoreDenominator = 0;
    aggregates.numSegments = 0;
    aggregates.numSrcChars = 0;
    aggregates.numRatings = 0;
    aggregates.timeSpentMS = 0;
    return aggregates;
  }
  let totalSrcLen = 0;
  /**
   * numSegRatings counts each (seg, rater) combination, where the rater rated
   *     that segment, once.
   * numRatedSegs counts segments that have been rated by at least one rater.
   */
  let numSegRatings = 0;
  let numRatedSegs = 0;
  for (let segStats of segs) {
    const allRaterStats = mqmInitRaterStats('');
    for (let r of segStats) {
      mqmAddRaterStats(allRaterStats, r);
    }
    if (segStats.length > 0) {
      mqmAvgRaterStats(allRaterStats, segStats.length);
      numSegRatings += segStats.length;
      numRatedSegs++;
      totalSrcLen += segStats.srcLen;
      mqmAddRaterStats(aggregates, allRaterStats);
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
            scoreDenominator: 0,
            numSegments: 0,
            numSrcChars: 0,
          };
        }
      }
    }
  }

  aggregates.numSegments = numRatedSegs;
  aggregates.numSrcChars = totalSrcLen;
  aggregates.scoreDenominator =
      mqmCharScoring ? (aggregates.numSrcChars / 100) : aggregates.numSegments;
  mqmAvgRaterStats(aggregates, aggregates.scoreDenominator);
  aggregates.numRatings = numSegRatings;

  for (let metric in aggregates.metrics) {
    const metricStats = aggregates.metrics[metric];
    metricStats.numSegments = 0;
    metricStats.numSrcChars = 0;
    metricStats.score = 0;
    for (let segStats of segs) {
      if (!segStats.hasOwnProperty('metrics') ||
          !segStats.metrics.hasOwnProperty(metric)) {
        continue;
      }
      metricStats.numSegments++;
      metricStats.numSrcChars += segStats.srcLen;
      metricStats.score += segStats.metrics[metric];
    }
    metricStats.scoreDenominator =
        mqmCharScoring ? (metricStats.numSrcChars / 100) :
        metricStats.numSegments;
    if (metricStats.scoreDenominator > 0) {
      metricStats.score /= metricStats.scoreDenominator;
    }
  }
  /** Copy MQM score into aggregate.metrics['MQM'] */
  if (aggregates.numRatings > 0) {
    aggregates.metrics['MQM'] = {
      score: aggregates.score,
      scoreDenominator: aggregates.scoreDenominator,
      numSegments: aggregates.numSegments,
      numSrcChars: aggregates.numSrcChars,
      numRatings: aggregates.numRatings,
    };
  }
  for (let metric in aggregates.metrics) {
    const metricStats = aggregates.metrics[metric];
    const metricIndex = mqmMetricsInfo[metric].index;
    aggregates['metric-' + metricIndex] = metricStats.score;
  }
  return aggregates;
}

/**
 * This resets the significance tests data and terminates the active sigtests
 * computation Worker if it exists.
 */
function mqmResetSigtests() {
  mqmSigtestsMsg.innerHTML = '';
  mqmSigtestsData.metricData = {};
  if (mqmSigtestsWorker) {
    mqmSigtestsWorker.terminate();
  }
  mqmSigtestsWorker = null;
}

/**
 * This prepares significance tests data, setting various fields in
 * mqmSigtestsData.
 * @param {!Object} mqmStatsBySysAggregates
 */
function mqmPrepareSigtests(mqmStatsBySysAggregates) {
  /**
   * Each segment is uniquely determined by the (doc, docSegId) pair. We use
   * `pairToPos` to track which pair goes to which position in the aligned
   * segScoresBySystem[system] array.
   */
  const pairToPos = {};
  let maxPos = 0;
  for (const doc of mqmDataIter.docs) {
    pairToPos[doc] = {};
    for (const docSegId of mqmDataIter.docSegs[doc]) {
      pairToPos[doc][docSegId] = maxPos;
      maxPos += 1;
    }
  }
  const elt = document.getElementById('mqm-sigtests-num-trials');
  mqmSigtestsData.numTrials = parseInt(elt.value);
  mqmSigtestsData.metricData = {};

  const systems = Object.keys(mqmStatsBySysAggregates);
  systems.splice(systems.indexOf(MQM_TOTAL), 1);

  for (let m of mqmMetricsVisible) {
    const metricKey = 'metric-' + m;
    const metric = mqmMetrics[m];
    const metricInfo = mqmMetricsInfo[metric];
    const data = new MQMMetricSigtestsData();
    mqmSigtestsData.metricData[metric] = data;
    data.systems = systems.slice();
    data.lowerBetter = metricInfo.lowerBetter || false;
    const signReverser = metricInfo.lowerBetter ? 1.0 : -1.0;
    data.systems.sort(
        (s1, s2) => signReverser * (
                        (mqmStatsBySysAggregates[s1][metricKey] ?? 0) -
                        (mqmStatsBySysAggregates[s2][metricKey] ?? 0)));
    for (const system of data.systems) {
      data.scoresBySystem[system] =
          mqmStatsBySysAggregates[system].metrics[metric] ??
          {score: 0, scoreDenominator: 0};
    }
    segScores = data.segScoresBySystem;
    for (const system of data.systems) {
      /**
       * For each system, we first compute the mapping from position to score.
       * Any missing key correponds to one missing segment for this system.
       */
      const posToScore = {};
      for (const doc of Object.keys(mqmStats[system])) {
        for (const docSegId of Object.keys(mqmStats[system][doc])) {
          const pos = pairToPos[doc][docSegId];
          const segs = mqmStats[system][doc][docSegId];
          /** Note the extra "[]". */
          const aggregate = mqmAggregateSegStats([segs]);
          const metricStats = aggregate.metrics[metric] ?? null;
          if (metricStats && metricStats.scoreDenominator > 0) {
            posToScore[pos] = metricStats.score;
          }
        }
      }
      /** Now we can compute "segScores". */
      segScores[system] = [];
      for (let pos = 0; pos < maxPos; pos++) {
        if (posToScore.hasOwnProperty(pos)) {
          segScores[system].push(posToScore[pos]);
        } else {
          /** This system is missing this specific segment. */
          segScores[system].push(null);
        }
      }
    }

    /** Compute common positions for each system pair in `commonPos`. */
    const commonPos = data.commonPosBySystemPair;
    for (const [idx, baseline] of data.systems.entries()) {
      if (!commonPos.hasOwnProperty(baseline)) {
        commonPos[baseline] = {};
      }
      /** We only need the upper triangle in the significance test table. */
      for (const system of data.systems.slice(idx + 1)) {
        if (!commonPos[baseline].hasOwnProperty(system)) {
          commonPos[baseline][system] = [];
        }
        for (let pos = 0; pos < maxPos; pos++) {
          if ((segScores[system][pos] != null) &&
              (segScores[baseline][pos] != null)) {
            commonPos[baseline][system].push(pos);
          }
        }
      }
    }

    /** Create pValues matrix, to be populated with updates from the Worker. */
    const numSystems = data.systems.length;
    data.pValues = Array(numSystems);
    for (let row = 0; row < numSystems; row++) {
      data.pValues[row] = Array(numSystems);
      for (let col = 0; col < numSystems; col++) {
        data.pValues[row][col] = NaN;
      }
    }
  }
}

/**
 * In the significance tests table, draw a solid line under every prefix of
 * systems that is significantly better than all subsequent systems. Draw a
 * dotted line to separate clusters within which no system is significantly
 * better than any other.
 * @param {string} metric
 */
function mqmClusterSigtests(metric) {
  const m = mqmMetricsInfo[metric].index;
  const data = mqmSigtestsData.metricData[metric];
  const numSystems = data.systems.length;
  const systemBetterThanAllAfter = Array(numSystems);
  for (let row = 0; row < numSystems; row++) {
    systemBetterThanAllAfter[row] = numSystems - 1;
    for (let col = numSystems - 1; col > row; col--) {
      const pValue = data.pValues[row][col];
      if (isNaN(pValue) || pValue >= MQM_PVALUE_THRESHOLD) {
        break;
      }
      systemBetterThanAllAfter[row] = col - 1;
    }
  }
  let maxBetterThanAllAfter = 0;  /** Max over rows 0..row */
  let dottedClusterStart = 0;
  for (let row = 0; row < numSystems - 1; row++) {
    const tr = document.getElementById('mqm-sigtests-' + m + '-row-' + row);
    maxBetterThanAllAfter = Math.max(maxBetterThanAllAfter,
                                     systemBetterThanAllAfter[row]);
    if (maxBetterThanAllAfter == row) {
      tr.className = 'mqm-bottomed-tr';
      dottedClusterStart = row + 1;
      continue;
    }
    /** Is no system in dottedClusterStart..row signif. better than row+1? */
    let noneSigBetter = true;
    for (let dottedClusterRow = dottedClusterStart;
         dottedClusterRow <= row; dottedClusterRow++) {
      const pValue = data.pValues[dottedClusterRow][row + 1];
      if (!isNaN(pValue) && pValue < MQM_PVALUE_THRESHOLD) {
        noneSigBetter = false;
        break;
      }
    }
    if (!noneSigBetter) {
      tr.className = 'mqm-dotted-bottomed-tr';
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
function mqmSigtestsUpdate(e) {
  const update = e.data;
  if (update.finished) {
    mqmResetSigtests();
    return;
  }
  const metric = update.metric;
  if (update.metricDone) {
    mqmClusterSigtests(metric);
    return;
  }
  const m = mqmMetricsInfo[metric].index;
  const span = document.getElementById(
      `mqm-sigtest-${m}-${update.row}-${update.col}`);
  span.innerText = update.pValue.toFixed(3);
  span.title = `Based on ${update.numCommonSegs} common segments.`;
  if (update.pValue < MQM_PVALUE_THRESHOLD) {
    span.className = 'mqm-sigtest-significant';
  }
  mqmSigtestsData.metricData[metric].pValues[update.row][update.col] =
      update.pValue;
}

/**
 * Shows the table for significance tests.
 * @param {!Object} mqmStatsBySysAggregates
 */
function mqmShowSigtests(mqmStatsBySysAggregates) {
  const div = document.getElementById('mqm-sigtests-tables');
  div.innerHTML = '';
  if (mqmCharScoring) {
    mqmSigtestsMsg.innerHTML = 'Not available for 100-source-chars scoring';
    return;
  }
  mqmPrepareSigtests(mqmStatsBySysAggregates);
  let firstTable = true;
  for (let m of mqmMetricsVisible) {
    const metric = mqmMetrics[m];
    const data = mqmSigtestsData.metricData[metric];
    const systems = data.systems;
    const scoresBySystem = data.scoresBySystem;

    /** Header. */
    let html = `
    ${firstTable ? '' : '<br>'}
    <table id="mqm-sigtests-${m}" class="mqm-table mqm-numbers-table">
      <thead>
        <tr>
          <th>System</th>
          <th>${mqmMetrics[m]}</th>`;
    for (const system of systems) {
      const s = scoresBySystem[system];
      if (s.scoreDenominator == 0) {
        continue;
      }
      html += `<th>${system}</th>`;
    }
    html += `</tr></thead>\n<tbody>\n`;

    /** Show significance test p-value placeholders. */
    for (const [rowIdx, baseline] of systems.entries()) {
      /** Show metric score in the second column. */
      const s = scoresBySystem[baseline];
      if (s.scoreDenominator == 0) {
        continue;
      }
      const displayScore = mqmMetricDisplay(s.score, s.scoreDenominator);
      html += `
        <tr id="mqm-sigtests-${m}-row-${rowIdx}">
          <td>${baseline}</td>
          <td>${displayScore}</td>`;
      for (const [colIdx, system] of systems.entries()) {
        const s2 = scoresBySystem[system];
        if (s2.scoreDenominator == 0) {
          continue;
        }
        const spanId = `mqm-sigtest-${m}-${rowIdx}-${colIdx}`;
        const content = rowIdx >= colIdx ? '-' : '-.---';
        html += `<td><span id="${spanId}">${content}<span></td>`;
      }
      html += `</tr>`;
    }
    html += `</tbody></table>`;
    div.insertAdjacentHTML('beforeend', html);
    firstTable = false;
  }

  mqmSigtestsMsg.innerHTML = 'Computing p-values...';
  if (mqmSigtestsWorkerJS) {
    /** Create Worker using code directly. */
    blob = new Blob([mqmSigtestsWorkerJS], {type: "text/javascript" });
    mqmSigtestsWorker = new Worker(window.URL.createObjectURL(blob));
  } else {
    /** Create Worker using code file. */
    mqmSigtestsWorker = new Worker('mqm-sigtests.js');
  }
  mqmSigtestsWorker.postMessage(mqmSigtestsData);
  mqmSigtestsWorker.onmessage = mqmSigtestsUpdate;
}

/**
 * Listener for changes to the input field that specifies the number of trials
 * for paired one-sided approximate randomization.
 */
function setMqmSigtestsNumTrials() {
  const elt = document.getElementById('mqm-sigtests-num-trials');
  const numTrials = parseInt(elt.value);
  if (numTrials <= 0 || numTrials == mqmSigtestsData.numTrials) {
    elt.value = mqmSigtestsData.numTrials;
    return;
  }
  mqmShow();
}

/**
 * Shows the table header for the MQM scores table. The score weighted
 * components and slices to display should be available in
 * mqmScoreWeightedFields and mqmScoreSliceFields.
 * @param {boolean} hasRatings set to true if there are some MQM annotations.
 */
function mqmShowScoresHeader(hasRatings) {
  const header = document.getElementById('mqm-stats-thead');
  const scoringUnit = mqmCharScoring ? '100 source chars' : 'segment';
  let html = `
      <tr>
        <th>Scores are per
            <span id="mqm-scoring-unit-display">${scoringUnit}</span></th>`;
  const metricFields = [];
  for (let m of mqmMetricsVisible) {
    const metric = mqmMetrics[m];
    html +=  `<th id="mqm-metric-${m}-th">${metric}</th>`;
    metricFields.push('metric-' + m);
  }
  if (hasRatings) {
    html += `
          <th title="Number of rated segments"><b>#Rated segments</b></th>
          <th title="Number of rated source characters">
            <b>#Rated source-chars</b>
          </th>
          <th title="Number of segment ratings"><b>#Segment ratings</b></th>`;
  }

  const mqmPartFields =
      mqmScoreWeightedFields.map(x => MQM_SCORE_WEIGHTED_PREFIX + x)
          .concat(mqmScoreSliceFields.map(x => MQM_SCORE_SLICE_PREFIX + x));
  for (let i = 0; i < mqmPartFields.length; i++) {
    const scoreKey = mqmPartFields[i];
    const scoreName = mqmScoreKeyToName(scoreKey);
    const partType = (i < mqmScoreWeightedFields.length) ? 'weighted' : 'slice';
    const cls = 'mqm-stats-' + partType;
    const tooltip = 'Score part: ' + scoreName + '-' + partType;
    html += `
        <th id="mqm-${scoreKey}-th" class="mqm-score-th ${cls}"
            title="${tooltip}">
          <b>${scoreName}</b>
        </th>`;
  }
  if (hasRatings) {
    html += `
          <th title="Average time (seconds) per rater per
segment or 100-source-chars"><b>Time (s)</b></th>
          <th title="Average length of error span"><b>Err span</b></th>
          <th title="Hands-on-the-wheel test"><b>HOTW Test</b></th>
        </tr>`;
  }
  header.innerHTML = html;

  /** Make columns clickable for sorting purposes. */

  const upArrow = '<span class="mqm-arrow mqm-arrow-up">&#129041;</span>';
  const downArrow = '<span class="mqm-arrow mqm-arrow-down">&#129043;</span>';
  for (const field of metricFields.concat(mqmPartFields)) {
    const headerId = `mqm-${field}-th`;
    const th = document.getElementById(headerId);
    th.insertAdjacentHTML('beforeend', ` ${upArrow}${downArrow}`);
    th.addEventListener('click', (e) => {
      // Click again for reversing order. Otherwise sort in ascending order.
      if (field == mqmSortByField) {
        mqmSortReverse = !mqmSortReverse;
      } else {
        mqmSortReverse = false;
      }
      mqmSortByField = field;
      mqmShow();
    });
  }
  mqmShowSortArrow();
}

/**
 * In the stats table, display a separator line, optionally followed by a row
 * that displays a title (if non-empty).
 * @param {boolean} hasRatings set to true if there are some MQM annotations.
 * @param {string=} title
 */
function mqmShowScoresSeparator(hasRatings, title='') {
  const NUM_COLS = (hasRatings ? 7 : 1) + mqmMetricsVisible.length +
                   mqmScoreWeightedFields.length +
                   mqmScoreSliceFields.length;
  mqmStatsTable.insertAdjacentHTML(
      'beforeend',
      `<tr><td colspan="${NUM_COLS}"><hr></td></tr>` +
      (title ?
      `<tr><td colspan="${NUM_COLS}"><b>${title}</b></td></tr>\n` :
      ''));
}

/**
 * Appends a row with score details for "label" (shown in the first column) from
 * the stats object to mqmStatsTable.
 * @param {string} label
 * @param {boolean} hasRatings set to true if there are some MQM annotations.
 * @param {!Object} stats
 * @param {!Object} aggregates
 */
function mqmShowScores(label, hasRatings, stats, aggregates) {
  const scoreFields =
      mqmScoreWeightedFields.map(x => MQM_SCORE_WEIGHTED_PREFIX + x).concat(
          mqmScoreSliceFields.map(x => MQM_SCORE_SLICE_PREFIX + x));
  let rowHTML = `<tr><td>${label}</td>`;
  for (let m of mqmMetricsVisible) {
    const metric = mqmMetrics[m];
    if (!aggregates.metrics.hasOwnProperty(metric)) {
      rowHTML += '<td>-</td>';
      continue;
    }
    const s = aggregates.metrics[metric];
    const title = `#Segments: ${s.numSegments}, #SrcChars: ${s.numSrcChars}`;
    rowHTML += `<td title="${title}">` +
               mqmMetricDisplay(s.score, s.scoreDenominator) +
               '</td>';
  }
  if (hasRatings) {
    rowHTML +=
      `<td>${aggregates.numSegments}</td>` +
      `<td>${aggregates.numSrcChars}</td>` +
      `<td>${aggregates.numRatings}</td>`;
    if (aggregates.scoreDenominator <= 0) {
      for (let i = 0; i < scoreFields.length + 3; i++) {
        rowHTML += '<td>-</td>';
      }
    } else {
      for (let s of scoreFields) {
        let content =
            aggregates.hasOwnProperty(s) ? aggregates[s].toFixed(3) : '-';
        const nameParts = s.split('-', 2);
        const cls = (nameParts.length == 2) ?
          ' class="mqm-stats-' + nameParts[0] + '"' :
          '';
        rowHTML += `<td${cls}>${content}</td>`;
      }
      let errorSpan = 0;
      if (aggregates.numWithErrors > 0) {
        errorSpan = aggregates.errorSpans / aggregates.numWithErrors;
      }
      rowHTML += `<td>${(aggregates.timeSpentMS/1000.0).toFixed(1)}</td>`;
      rowHTML += `<td>${(errorSpan).toFixed(1)}</td>`;
      const hotw = aggregates.hotwFound + aggregates.hotwMissed;
      if (hotw > 0) {
        const perc = ((aggregates.hotwFound * 100.0) / hotw).toFixed(1);
        rowHTML += `<td>${aggregates.hotwFound}/${hotw} (${perc}%)</td>`;
      } else {
        rowHTML += '<td>-</td>';
      }
    }
  }
  rowHTML += '</tr>\n';
  mqmStatsTable.insertAdjacentHTML('beforeend', rowHTML);
}

/**
 * Shows the system x rater matrix of scores. The rows and columns are
 * ordered by total MQM score.
 */
function mqmShowSystemRaterStats() {
  const table = document.getElementById('mqm-system-x-rater');

  const systems = Object.keys(mqmStats);
  const systemAggregates = {};
  for (let sys of systems) {
    const segs = mqmGetSegStatsAsArray(mqmStats[sys]);
    systemAggregates[sys] = mqmAggregateSegStats(segs);
  }

  const SORT_FIELD = 'metric-0';
  systems.sort(
      (sys1, sys2) =>
          systemAggregates[sys1][SORT_FIELD] -
          systemAggregates[sys2][SORT_FIELD]);

  const raters = Object.keys(mqmStatsByRater);
  const raterAggregates = {};
  for (let rater of raters) {
    const segs = mqmGetSegStatsAsArray(mqmStatsByRater[rater][MQM_TOTAL]);
    raterAggregates[rater] = mqmAggregateSegStats(segs);
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
  for (let rater of raters) {
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
  for (let rater of raters) {
    lastForRater[rater] = 0;
  }
  for (let sys of systems) {
    if (sys == MQM_TOTAL) {
      continue;
    }
    const allRatersScore = systemAggregates[sys].score;
    const allRatersScoreDisplay = mqmMetricDisplay(
        allRatersScore, systemAggregates[sys].numRatings);
    html += `
      <tr><td>${sys}</td><td>${allRatersScoreDisplay}</td>`;
    for (let rater of raters) {
      const segs = mqmGetSegStatsAsArray(
          (mqmStatsByRater[rater] ?? {})[sys] ?? {});
      if (segs && segs.length > 0) {
        const aggregate = mqmAggregateSegStats(segs);
        const cls = ((aggregate.score < lastForRater[rater] &&
                      allRatersScore > lastAllRaters) ||
                     (aggregate.score > lastForRater[rater] &&
                      allRatersScore < lastAllRaters)) ?
            ' class="mqm-out-of-order"' : '';
        const scoreDisplay = mqmMetricDisplay(
            aggregate.score, aggregate.numRatings);
        html += `
            <td><span${cls}>${scoreDisplay}</span></td>`;
        lastForRater[rater] = aggregate.score;
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
 * Sorter function where a & b are both a [doc, seg] pair. The seg part can
 * be a number (0,1,2,3,etc.) or a string, so we have to be careful to ensure
 * transitivity. We do that by assuming that the numbers are not too big, so
 * padding them with up to 10 zeros should be good enough.
 * @param {!Array<string>} a
 * @param {!Array<string>} b
 * @return {number} Comparison for sorting a & b.
 */
function mqmDocSegsSorter(a, b) {
  if (a[0] < b[0]) return -1;
  if (a[0] > b[0]) return 1;
  seg1 = mqmCmpDocSegId(a[1]);
  seg2 = mqmCmpDocSegId(b[1]);
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
function mqmGetDocSegs(stats) {
  const segs = [];
  for (let doc in stats) {
    const docstats = stats[doc];
    for (let docSeg in docstats) {
      segs.push([doc, docSeg]);
    }
  }
  return segs.sort(mqmDocSegsSorter);
}

/**
 * Makes a convenient key that captures a doc name and a docSegId.
 * @param {string} doc
 * @param {string|number} seg
 * @return {string}
 */
function mqmDocSegKey(doc, seg) {
  return doc + ':' + seg;
}

/**
 * Helper class for building a segment scores histogram. Call addSegment() on it
 * multiple times to record segment scores. Then call display().
 *
 * The rendered histogram bins one special x-axis value differently, showing a
 * slim gray bar for that value. For difference histograms (hasDiffs=true), that
 * value is 0. Otherwise, it is for the perfect possible score (>= 1 for
 * automated metrics and 0 for MQM).
 *
 * @param {number} m The index of the metric in mqmMetrics.
 * @param {string} sys The name of the system.
 * @param {string} color The color to use for system.
 * @param {string=} sysCmp if the histogram is for diffs, then the name of
 *     the system being compared against.
 * @param {string=} colorCmp if the histogram is for diffs, then the color of
 *     the system being compared against.
 */
function MQMHistBuilder(m, sys, color, sysCmp='', colorCmp='') {
  this.metricIndex = m;
  this.sys = sys;
  this.sysCmp = sysCmp;
  this.hasDiffs = sysCmp ? true : false;
  this.metric = mqmMetrics[m];
  this.color = color;
  this.colorCmp = colorCmp;

  const metricInfo = mqmMetricsInfo[this.metric];
  this.lowerBetter = metricInfo.lowerBetter || false;

  /**
   * Is there a dedicated bin for value == 0?
   */
  this.hasZeroBin = this.hasDiffs ? true : (this.lowerBetter ? true : false);

  /** @const {number} Width of a histogram bin, in score units */
  this.BIN_WIDTH = (this.metric == 'MQM') ? 0.5 : 0.05;
  this.BIN_PRECISION = (this.metric == 'MQM') ? 1 : 2;

  /** @const {number} Width of a histogram bin, in pixels */
  this.BIN_WIDTH_PIXELS = 10 + (this.BIN_PRECISION * 3);

  this.PIXELS_PER_UNIT = this.BIN_WIDTH_PIXELS / this.BIN_WIDTH;

  /** @const {number} Width of the special "zero" bin, in pixels */
  this.ZERO_BIN_WIDTH_PIXELS = 6;

  this.LOG_MULTIPLIER = 1.0 / Math.LN2;
  this.LOG_UNIT_HEIGHT_PIXELS = 25;
  this.TOP_OFFSET_PIXELS = this.hasDiffs ? 49 : 19;
  this.BOTTOM_OFFSET_PIXELS = 50;
  this.X_OFFSET_PIXELS = 50;

  this.COLOR_ZERO = 'rgba(211,211,211,0.5)';
  this.COLOR_OUTLINE = 'black';
  this.COLOR_LEGEND = 'black';
  this.COLOR_LABELS = 'black';
  this.COLOR_LINES = 'lightgray';

  /**
   * @const {!Object} Dict keyed by bin. Each bin has an array of doc-seg keys.
   *    The only non-numeric key possibly present is 'zero' (when
   *    this.hasZeroBin is true).
   */
  this.segsInBin = {};

  /**
   * @const {number} The largest bin visible on the X-axis.
   */
  this.maxBin = 0;
  /**
   * @const {number} The smallest bin visible on the X-axis.
   */
  this.minBin = 0;

  /** {number} The largest count in bin (used to determine height of plot) */
  this.maxCount = 8;

  this.totalCount = 0;
  this.sys1BetterCount = 0;
  this.sys2BetterCount = 0;
}

/**
 * Returns the bin for a particular value. We return the left end-point (except
 * for the special 'zero' bin).
 * @param {number} value
 * @return {string}
 */
MQMHistBuilder.prototype.binOf = function(value) {
  if (this.hasZeroBin && value == 0) {
    return 'zero';
  }
  const absValue = Math.abs(value);
  const absBin = Math.floor(absValue / this.BIN_WIDTH) * this.BIN_WIDTH;
  const leftVal = (value < 0) ? (0 - absBin - this.BIN_WIDTH) : absBin;
  return leftVal.toFixed(this.BIN_PRECISION);
};

/**
 * Adds a segment to the histogram, updating the appropriate bin.
 * @param {string} doc
 * @param {string|number} docSegId
 * @param {number} value The score for the first system
 */
MQMHistBuilder.prototype.addSegment = function(doc, docSegId, value) {
  const bin = this.binOf(value);
  const numericBin = (bin == 'zero') ? 0 : parseFloat(bin);
  if (numericBin < this.minBin) this.minBin = numericBin;
  if (numericBin > this.maxBin) this.maxBin = numericBin;
  const docSegKey = mqmDocSegKey(doc, docSegId);
  if (!this.segsInBin.hasOwnProperty(bin)) {
    this.segsInBin[bin] = [];
  }
  this.segsInBin[bin].push(docSegKey);
  if (this.segsInBin[bin].length > this.maxCount) {
    this.maxCount = this.segsInBin[bin].length;
  }
  this.totalCount++;
  if (this.hasDiffs && bin != 'zero') {
    const firstLower = (numericBin < 0);
    const firstBetter = (firstLower && this.lowerBetter) ||
                        (!firstLower && !this.lowerBetter);
    if (firstBetter) {
      this.sys1BetterCount++;
    } else {
      this.sys2BetterCount++;
    }
  }
};

/**
 * Creates and returns an SVG rect.
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {string} color
 * @return {!Element}
 */
MQMHistBuilder.prototype.getRect = function(x, y, w, h, color) {
  const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  rect.setAttributeNS(null, 'x', x);
  rect.setAttributeNS(null, 'y', y + this.TOP_OFFSET_PIXELS);
  rect.setAttributeNS(null, 'width', w);
  rect.setAttributeNS(null, 'height', h);
  rect.style.fill = color;
  rect.style.stroke = this.COLOR_OUTLINE;
  return rect;
};

/**
 * Creates a histogram bar with a given description, makes it clickable to
 * constrain the view to the docsegs passed.
 * @param {!Element} plot
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {string} color
 * @param {string} desc
 * @param {!Array<string>} docsegs
 */
MQMHistBuilder.prototype.makeHistBar = function(
    plot, x, y, w, h, color, desc, docsegs) {
  /**
   * Need to wrap the rect in a g (group) element to be able to show
   * the description when hovering ("title" attribute does not work with SVG
   * elements).
   */
  const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  g.setAttributeNS(null, 'class', 'mqm-sys-v-sys-hist');
  g.insertAdjacentHTML('beforeend',
                       `<title>Click to see examples of ${desc}</title>`);
  const rect = this.getRect(x, y, w, h, color);
  g.appendChild(rect);
  const viewingConstraints = {};
  for (let ds of docsegs) {
    viewingConstraints[ds] = true;
  }
  viewingConstraints.description = desc;
  viewingConstraints.color = color;
  g.addEventListener('click', (e) => {
    mqmShow(viewingConstraints);
  });
  plot.appendChild(g);
};

/**
 * Creates a line on the plot.
 * @param {!Element} plot
 * @param {number} x1
 * @param {number} y1
 * @param {number} x2
 * @param {number} y2
 * @param {string} color
 */
MQMHistBuilder.prototype.makeLine = function(
    plot, x1, y1, x2, y2, color) {
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttributeNS(null, 'x1', x1);
  line.setAttributeNS(null, 'y1', y1 + this.TOP_OFFSET_PIXELS);
  line.setAttributeNS(null, 'x2', x2);
  line.setAttributeNS(null, 'y2', y2 + this.TOP_OFFSET_PIXELS);
  line.style.stroke = color;
  plot.appendChild(line);
};

/**
 * Writes some text on the plot.
 * @param {!Element} plot
 * @param {number} x
 * @param {number} y
 * @param {string} s
 * @param {string} color
 */
MQMHistBuilder.prototype.makeText = function(plot, x, y, s, color) {
  const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  text.setAttributeNS(null, 'x', x);
  text.setAttributeNS(null, 'y', y + this.TOP_OFFSET_PIXELS);
  text.innerHTML = s;
  text.style.fill = color;
  text.style.fontSize = '10px';
  plot.appendChild(text);
};

/**
 * Returns height in pixels for a histogram bar with the given count.
 * @param {number} count
 * @return {number}
 */
MQMHistBuilder.prototype.heightInPixels = function(count) {
  if (count == 0) return 0;
  return this.LOG_UNIT_HEIGHT_PIXELS *
         ((Math.log(count) * this.LOG_MULTIPLIER) + 1);
};

/**
 * Returns the color to use for the bin's histogram rectangle.
 * @param {string} bin
 * @param {number} numericBin
 * @return {string}
 */
MQMHistBuilder.prototype.binColor = function(bin, numericBin) {
  if (bin == 'zero') {
    return this.COLOR_ZERO;
  }
  if (!this.hasDiffs) {
    return this.color;
  } else {
    const firstLower = (numericBin < 0);
    const firstBetter = (firstLower && this.lowerBetter) ||
                        (!firstLower && !this.lowerBetter);
    return firstBetter ? this.color : this.colorCmp;
  }
};

/**
 * Returns a description of the bin.
 * @param {string} bin
 * @param {number} numericBin
 * @param {number} count
 * @return {string}
 */
MQMHistBuilder.prototype.binDesc = function(bin, numericBin, count) {
  if (!this.hasDiffs) {
    if (bin == 'zero') {
      return '' + count + ' segment(s) where ' + this.sys +
             ' has ' + this.metric + ' score exactly equal to 0';
    }
    const binLeft = numericBin;
    const binRight = numericBin + this.BIN_WIDTH;
    let leftParen = (numericBin < 0) ? '(' :
                    ((numericBin == 0 && this.hasZeroBin) ? '(' : '[');
    let rightParen = (numericBin < 0) ?
                     ((binRight == 0 && this.hasZeroBin) ? ')' : ']') : ')';
    return '' + count + ' segment(s) where ' + this.sys + ' has ' +
           this.metric + ' score in ' + 'range ' + leftParen +
           this.binDisplay(binLeft) +
           ',' + this.binDisplay(binRight) + rightParen;
  } else {
    if (bin == 'zero') {
      return '' + count + ' segment(s) where ' + this.sys + ' and ' +
             this.sysCmp + ' have identical ' + this.metric + ' scores';
    }
    const firstLower = (numericBin < 0);
    const firstBetter = (firstLower && this.lowerBetter) ||
                        (!firstLower && !this.lowerBetter);
    const betterSys = firstBetter ? this.sys : this.sysCmp;
    const worseSys = firstBetter ? this.sysCmp : this.sys;
    const binLeft = numericBin;
    const binRight = numericBin + this.BIN_WIDTH;
    const absBinLeft = (numericBin < 0) ? (0 - binRight) : binLeft;
    const absBinRight = absBinLeft + this.BIN_WIDTH;
    const firstParen = (absBinLeft == 0 && this.hasZeroBin) ? '(' : '[';
    return '' + count + ' segment(s) where ' + betterSys + ' is better than ' +
           worseSys + ' with ' + this.metric + ' score diff in range ' +
           firstParen + this.binDisplay(absBinLeft) + ',' +
           this.binDisplay(absBinRight) + ')';
  }
};

/**
 * Returns the x coordinate in pixels for a particular metric value.
 * @param {number} value
 * @return {number}
 */
MQMHistBuilder.prototype.xPixels = function(value) {
  return this.X_OFFSET_PIXELS + ((value - this.minBin) * this.PIXELS_PER_UNIT);
};

/**
 * Returns a string suitable to display, for a floating-point number. Strips
 * trailing zeros and then a trailing decimal point.
 * @param {number} value
 * @return {string}
 */
MQMHistBuilder.prototype.binDisplay = function(value) {
  return value.toFixed(
      this.BIN_PRECISION).replace(/0+$/, '').replace(/\.$/, '');
};

/**
 * Displays the histogram using the data collected through prior addSegment()
 * calls.
 * @param {!Element} plot
 */
MQMHistBuilder.prototype.display = function(plot) {
  /** Create some buffer space above the plot. */
  this.maxCount += 10;

  const binKeys = Object.keys(this.segsInBin);
  /** Sort so that 'zero' bin is drawn at the end. */
  binKeys.sort((a, b) => {
    let a2 = (a == 'zero') ? Number.MAX_VALUE : a;
    let b2 = (b == 'zero') ? Number.MAX_VALUE : b;
    return a2 - b2;
  });
  const plotWidth = Math.max(
      400, (2 * this.X_OFFSET_PIXELS) +
           ((this.maxBin - this.minBin) * this.PIXELS_PER_UNIT));
  const plotHeight = this.heightInPixels(this.maxCount);
  const svgWidth = plotWidth;
  const svgHeight = plotHeight +
                    (this.TOP_OFFSET_PIXELS + this.BOTTOM_OFFSET_PIXELS);
  plot.innerHTML = '';
  plot.setAttributeNS(null, 'viewBox', `0 0 ${svgWidth} ${svgHeight}`);
  plot.setAttributeNS(null, 'width', svgWidth);
  plot.setAttributeNS(null, 'height', svgHeight);

  /* y axis labels */
  this.makeLine(plot, 0, plotHeight, plotWidth, plotHeight, this.COLOR_LINES);
  this.makeText(plot, 5, plotHeight - 2, '0', this.COLOR_LABELS);
  for (let l = 1; l <= this.maxCount; l *= 2) {
    const h = this.heightInPixels(l);
    this.makeLine(plot, 0, plotHeight - h, plotWidth, plotHeight - h,
                  this.COLOR_LINES);
    this.makeText(plot, 5, plotHeight - h - 2, '' + l, this.COLOR_LABELS);
  }

  if (this.hasDiffs) {
    /* legend, shown in the area above the plot */
    legends = [
      {
        color: this.color,
        desc: this.sys1BetterCount + ' better segments for ' + this.sys,
      },
      {
        color: this.colorCmp,
        desc: this.sys2BetterCount + ' better segments for ' + this.sysCmp,
      },
    ];
    for (let s = 0; s < legends.length; s++) {
      const legend = legends[s];
      const y = -30 + (s * (this.BIN_WIDTH_PIXELS + 10));
      const x = 25;
      plot.appendChild(this.getRect(
          x, y, this.BIN_WIDTH_PIXELS, this.BIN_WIDTH_PIXELS, legend.color));
      this.makeText(plot, x + this.BIN_WIDTH_PIXELS + 5, y + 10,
                    legend.desc, this.COLOR_LEGEND);
    }
  }

  for (let bin of binKeys) {
    const segs = this.segsInBin[bin];
    if (segs.length == 0) continue;
    const numericBin = (bin == 'zero') ? 0 : parseFloat(bin);
    let x = this.xPixels(numericBin);
    const binWidth = (bin == 'zero') ? this.ZERO_BIN_WIDTH_PIXELS :
                     this.BIN_WIDTH_PIXELS;
    if (bin == 'zero') {
      x -= (binWidth / 2.0);
    }
    const color = this.binColor(bin, numericBin);
    const desc = this.binDesc(bin, numericBin, segs.length);
    const h = this.heightInPixels(segs.length);
    this.makeHistBar(
          plot, x, plotHeight - h, binWidth, h,
          color, desc, segs);
  }

  /** Draw x-axis labels */
  const maxV = Math.max(Math.abs(this.minBin), Math.abs(this.maxBin));
  const step = 2 * this.BIN_WIDTH;
  for (let v = 0; v <= maxV + this.BIN_WIDTH; v += step) {
    if (v >= 0 && v <= this.maxBin + this.BIN_WIDTH) {
      const vDisp = this.binDisplay(v);
      const x = this.xPixels(v);
      const xDelta = 3 * vDisp.length;
      this.makeLine(plot, x, plotHeight, x, plotHeight + 8, this.COLOR_LINES);
      this.makeText(plot, x - xDelta, plotHeight + 20,
                    vDisp, this.COLOR_LABELS);
    }
    const negV = 0 - v;
    if (v == 0 || negV < this.minBin) {
      continue;
    }
    const negVDisp = this.binDisplay(negV);
    const x = this.xPixels(negV);
    const xDelta = 3 * (negVDisp.length + 1);
    this.makeLine(plot, x, plotHeight, x, plotHeight + 8, this.COLOR_LINES);
    this.makeText(plot, x - xDelta, plotHeight + 20,
                  negVDisp, this.COLOR_LABELS);
  }
  /* X-axis name */
  this.makeText(plot, this.X_OFFSET_PIXELS, plotHeight + 40,
                (this.hasDiffs ? this.metric + ' score differences' :
                 this.sys + ': ' + this.totalCount + ' segments with ' +
                 this.metric + ' scores'),
                this.COLOR_LEGEND);
};

/**
 * Creates the "system vs system" plots comparing two systems for all
 * available metrics. This sets up the menus for selecting the systems,
 * creates skeletal tables, and then calls mqmShowSysVSys() to populate the
 * tables.
 */
function mqmCreateSysVSysTables() {
  const div = document.getElementById('mqm-sys-v-sys');
  div.innerHTML = `
    <div class="mqm-sys-v-sys-header">
      <label>
        <b>System 1:</b>
        <select id="mqm-sys-v-sys-1" onchange="mqmShowSysVSys()"></select>
      </label>
      <span id="mqm-sys-v-sys-1-segs"></span> segment(s).
      <label>
        <b>System 2:</b>
        <select id="mqm-sys-v-sys-2" onchange="mqmShowSysVSys()"></select>
      </label>
      <span id="mqm-sys-v-sys-2-segs"></span> segment(s)
      (<span id="mqm-sys-v-sys-xsegs"></span> common).
      The Y-axis uses a log scale.
    </div>
  `;
  for (let m of mqmMetricsVisible) {
    const metric = mqmMetrics[m];
    const html = `
    <p id="mqm-sys-v-sys-${m}">
      <b>${metric}</b><br>
      <table>
        <tr>
        <td colspan="2">
          <svg class="mqm-sys-v-sys-plot" zoomAndPan="disable"
              id="mqm-sys-v-sys-plot-${m}">
          </svg>
        </td>
        </tr>
        <tr style="vertical-align:bottom">
        <td>
          <svg class="mqm-sys-v-sys-plot" zoomAndPan="disable"
              id="mqm-sys1-plot-${m}">
          </svg>
        </td>
        <td>
          <svg class="mqm-sys-v-sys-plot" zoomAndPan="disable"
              id="mqm-sys2-plot-${m}">
          </svg>
        </td>
        </tr>
      </table>
    </p>`;
    div.insertAdjacentHTML('beforeend', html);
  }

  /** Populate menu choices. */
  const selectSys1 = document.getElementById('mqm-sys-v-sys-1');
  const selectSys2 = document.getElementById('mqm-sys-v-sys-2');
  const systems = Object.keys(mqmStats);
  /**
   * If possible, use the previously set values.
   */
  if (mqmSysVSys1 && !mqmStats.hasOwnProperty(mqmSysVSys1)) {
    mqmSysVSys1 = '';
  }
  if (mqmSysVSys2 && !mqmStats.hasOwnProperty(mqmSysVSys2)) {
    mqmSysVSys2 = '';
  }
  if (systems.length == 1) {
    mqmSysVSys1 = systems[0];
    mqmSysVSys2 = systems[0];
  }
  for (let system of systems) {
    if (system == MQM_TOTAL) {
      continue;
    }
    if (!mqmSysVSys1) {
      mqmSysVSys1 = system;
    }
    if (!mqmSysVSys2 && system != mqmSysVSys1) {
      mqmSysVSys2 = system;
    }
    const option1 = document.createElement('option');
    option1.value = system;
    option1.innerHTML = system;
    if (system == mqmSysVSys1) {
      option1.selected = true;
    }
    selectSys1.insertAdjacentElement('beforeend', option1);
    const option2 = document.createElement('option');
    option2.value = system;
    option2.innerHTML = system;
    if (system == mqmSysVSys2) {
      option2.selected = true;
    }
    selectSys2.insertAdjacentElement('beforeend', option2);
  }
  mqmShowSysVSys();
}

/**
 * Shows the system v system histograms of segment score differences.
 */
function mqmShowSysVSys() {
  const selectSys1 = document.getElementById('mqm-sys-v-sys-1');
  const selectSys2 = document.getElementById('mqm-sys-v-sys-2');
  mqmSysVSys1 = selectSys1.value;
  mqmSysVSys2 = selectSys2.value;
  const docsegs1 = mqmGetDocSegs(mqmStats[mqmSysVSys1] || {});
  const docsegs2 = mqmGetDocSegs(mqmStats[mqmSysVSys2] || {});
  /**
   * Find common segments.
   */
  let i1 = 0;
  let i2 = 0;
  const docsegs12 = [];
  while (i1 < docsegs1.length && i2 < docsegs2.length) {
    const ds1 = docsegs1[i1];
    const ds2 = docsegs2[i2];
    const sort = mqmDocSegsSorter(ds1, ds2);
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
  document.getElementById('mqm-sys-v-sys-xsegs').innerHTML = docsegs12.length;
  document.getElementById('mqm-sys-v-sys-1-segs').innerHTML = docsegs1.length;
  document.getElementById('mqm-sys-v-sys-2-segs').innerHTML = docsegs2.length;

  const sameSys = mqmSysVSys1 == mqmSysVSys2;

  for (let m of mqmMetricsVisible) {
    const metricKey = 'metric-' + m;
    /**
     * We draw up to 3 plots for a metric: system-1, system-2, and their diff.
     */
    const hists = [
      {
        docsegs: docsegs1,
        hide: !mqmSysVSys1,
        sys: mqmSysVSys1,
        color: 'lightgreen',
        sysCmp: '',
        colorCmp: '',
        id: 'mqm-sys1-plot-' + m,
      },
      {
        docsegs: docsegs2,
        hide: sameSys,
        sys: mqmSysVSys2,
        color: 'lightblue',
        sysCmp: '',
        colorCmp: '',
        id: 'mqm-sys2-plot-' + m,
      },
      {
        docsegs: docsegs12,
        hide: sameSys,
        sys: mqmSysVSys1,
        color: 'lightgreen',
        sysCmp: mqmSysVSys2,
        colorCmp: 'lightblue',
        id: 'mqm-sys-v-sys-plot-' + m,
      },
    ];
    for (let hist of hists) {
      const histElt = document.getElementById(hist.id);
      histElt.style.display = hist.hide ? 'none' : '';
      if (hist.hide) {
        continue;
      }
      const histBuilder = new MQMHistBuilder(m, hist.sys, hist.color,
                                             hist.sysCmp, hist.colorCmp);
      for (let i = 0; i < hist.docsegs.length; i++) {
        const doc = hist.docsegs[i][0];
        const docSegId = hist.docsegs[i][1];
        const aggregate1 = mqmAggregateSegStats(
            [mqmStats[hist.sys][doc][docSegId]]);
        if (!aggregate1.hasOwnProperty(metricKey)) {
          continue;
        }
        let score = aggregate1[metricKey];
        if (hist.sysCmp) {
          const aggregate2 = mqmAggregateSegStats(
              [mqmStats[hist.sysCmp][doc][docSegId]]);
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
 *   mqmSevCatStats object) in the categories table.
 */
function mqmShowSevCatStats() {
  const stats = mqmSevCatStats;
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
  const th = document.getElementById('mqm-sevcat-stats-th');
  th.colSpan = colspan;

  systemsList.sort((sys1, sys2) => systems[sys2] - systems[sys1]);

  let rowHTML = '<tr><td></td><td></td><td></td>';
  for (let system of systemsList) {
    rowHTML += `<td><b>${system == MQM_TOTAL ? 'Total' : system}</b></td>`;
  }
  rowHTML += '</tr>\n';
  mqmSevCatStatsTable.insertAdjacentHTML('beforeend', rowHTML);

  const sevKeys = Object.keys(stats);
  sevKeys.sort();
  for (let severity of sevKeys) {
    mqmSevCatStatsTable.insertAdjacentHTML(
        'beforeend', `<tr><td colspan="${3 + colspan}"><hr></td></tr>`);
    const sevStats = stats[severity];
    const catKeys = Object.keys(sevStats);
    catKeys.sort((k1, k2) => sevStats[k2][MQM_TOTAL] - sevStats[k1][MQM_TOTAL]);
    for (let category of catKeys) {
      const row = sevStats[category];
      let rowHTML = `<tr><td>${severity}</td><td>${category}</td><td></td>`;
      for (let system of systemsList) {
        const val = row.hasOwnProperty(system) ? row[system] : '';
        rowHTML += `<td>${val ? val : ''}</td>`;
      }
      rowHTML += '</tr>\n';
      mqmSevCatStatsTable.insertAdjacentHTML('beforeend', rowHTML);
    }
  }
}

/**
 * Shows UI event counts and timespans.
 */
function mqmShowEventTimespans() {
  const sortedEvents = [];
  for (let e of Object.keys(mqmEvents.aggregates)) {
    const event = {
      'name': e,
    };
    const eventInfo = mqmEvents.aggregates[e];
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
    t1 = e1.avgTimeMS || 0;
    t2 = e2.avgTimeMS || 0;
    return t2 - t1;
  });
  for (let event of sortedEvents) {
    let rowHTML = '<tr>';
    rowHTML += '<td>' + event.name + '</td>';
    rowHTML += '<td>' + event.count + '</td>';
    let t = event.avgTimeMS;
    if (t) {
      t = Math.round(t);
    }
    rowHTML += '<td>' + t + '</td>';
    rowHTML += '</tr>\n';
    mqmEventsTable.insertAdjacentHTML('beforeend', rowHTML);
  }
}

/**
 * Max number of rows to show in a rater's event timeline.
 */
const MQM_RATER_TIMELINE_LIMIT = 200;

/**
 * Make the timeline for the currently selected rater visible, hiding others.
 */
function mqmRaterTimelineSelect() {
  const raterIndex = document.getElementById('mqm-rater-timelines-rater').value;
  const tbodyId = `mqm-rater-timeline-${raterIndex}`;
  const table = document.getElementById('mqm-rater-timelines');
  const tbodies = table.getElementsByTagName('tbody');
  for (let i = 0; i < tbodies.length; i++) {
    tbodies[i].style.display = (tbodies[i].id == tbodyId) ? '' : 'none';
  }
}

/**
 * Shows rater-wise UI event timelines.
 */
function mqmShowRaterTimelines() {
  const raters = Object.keys(mqmEvents.raters);
  const raterSelect = document.getElementById('mqm-rater-timelines-rater');
  raterSelect.innerHTML = '';
  const table = document.getElementById('mqm-rater-timelines');
  const tbodies = table.getElementsByTagName('tbody');
  for (let i = 0; i < tbodies.length; i++) {
    tbodies[i].remove();
  }
  for (let i = 0; i < raters.length; i++) {
    const rater = raters[i];
    raterSelect.insertAdjacentHTML('beforeend', `
                                   <option value="${i}">${rater}</option>`);
    const tbody = document.createElement('tbody');
    tbody.setAttribute('id', `mqm-rater-timeline-${i}`);
    table.appendChild(tbody);
    const log = mqmEvents.raters[rater];
    log.sort((e1, e2) => e1.ts - e2.ts);
    let num = 0;
    for (let e of log) {
      let rowHTML = '<tr>';
      rowHTML += '<td>' + (new Date(e.ts)).toLocaleString() + '</td>';
      rowHTML += '<td>' + e.action + '</td>';
      rowHTML += '<td>' + e.doc + '</td>';
      rowHTML += '<td>' + e.system + '</td>';
      rowHTML += '<td>' + e.docSegId + '</td>';
      rowHTML += '<td>' + (e.side == 0 ? 'Source' : 'Translation') + '</td>';
      rowHTML += '<td>' + (e.sentence + 1) + '</td>';
      rowHTML += '<td>' +
                 (e.source_not_seen ? 'Translation' : 'Source, Translation') +
                 '</td>';
      rowHTML += '</tr>\n';
      tbody.insertAdjacentHTML('beforeend', rowHTML);
      num++;
      if (num >= MQM_RATER_TIMELINE_LIMIT) {
        break;
      }
    }
    tbody.style.display = 'none';
  }
  mqmRaterTimelineSelect();
}

/**
 * Shows UI event counts, timespans, and rater timelines
 */
function mqmShowEvents() {
  mqmShowEventTimespans();
  mqmShowRaterTimelines();
}

/**
 * Shows all the stats.
 */
function mqmShowStats() {
  /**
   * Get aggregates for the stats by system, including the special '_MQM_TOTAL_'
   * system (which lets us decide which score splits have non-zero values and we
   * show score columns for only those splits).
   */
  const systems = Object.keys(mqmStats);
  const mqmStatsBySysAggregates = {};
  for (let system of systems) {
    const segs = mqmGetSegStatsAsArray(mqmStats[system]);
    mqmStatsBySysAggregates[system] = mqmAggregateSegStats(segs);
  }
  const overallStats = mqmStatsBySysAggregates[MQM_TOTAL] ?? {};
  mqmScoreWeightedFields = [];
  mqmScoreSliceFields = [];
  for (let key in overallStats) {
    if (!overallStats[key]) continue;
    if (key.startsWith(MQM_SCORE_WEIGHTED_PREFIX)) {
      mqmScoreWeightedFields.push(mqmScoreKeyToName(key));
    } else if (key.startsWith(MQM_SCORE_SLICE_PREFIX)) {
      mqmScoreSliceFields.push(mqmScoreKeyToName(key));
    }
  }
  mqmScoreWeightedFields.sort(
      (k1, k2) => (overallStats[MQM_SCORE_WEIGHTED_PREFIX + k2] ?? 0) -
          (overallStats[MQM_SCORE_WEIGHTED_PREFIX + k1] ?? 0));
  mqmScoreSliceFields.sort(
      (k1, k2) => (overallStats[MQM_SCORE_SLICE_PREFIX + k2] ?? 0) -
          (overallStats[MQM_SCORE_SLICE_PREFIX + k1] ?? 0));

  const mqmStatsByRaterAggregates = {};
  const raters = Object.keys(mqmStatsByRater);
  for (let rater of raters) {
    const segs = mqmGetSegStatsAsArray(mqmStatsByRater[rater][MQM_TOTAL]);
    mqmStatsByRaterAggregates[rater] = mqmAggregateSegStats(segs);
  }

  const indexOfTotal = systems.indexOf(MQM_TOTAL);
  systems.splice(indexOfTotal, 1);

  systems.sort(
      (k1, k2) => (mqmStatsBySysAggregates[k1][mqmSortByField] ?? 0) -
                  (mqmStatsBySysAggregates[k2][mqmSortByField] ?? 0));
  raters.sort(
      (k1, k2) => (mqmStatsByRaterAggregates[k1][mqmSortByField] ?? 0) -
                  (mqmStatsByRaterAggregates[k2][mqmSortByField] ?? 0));
  if (mqmSortReverse) {
    systems.reverse();
    raters.reverse();
  }

  /**
   * First show the scores table header with the sorted columns from
   * mqmScoreWeightedFields and mqmScoreSliceFields. Then add scores rows to
   * the table: by system, and then by rater.
   */
  const haveRaters = raters.length > 0;
  mqmShowScoresHeader(haveRaters);
  if (systems.length > 0) {
    mqmShowScoresSeparator(haveRaters, 'By system');
    for (let system of systems) {
      mqmShowScores(system, haveRaters, mqmStats[system],
                    mqmStatsBySysAggregates[system]);
    }
  }
  if (haveRaters) {
    mqmShowScoresSeparator(haveRaters, 'By rater');
    for (let rater of raters) {
      mqmShowScores(rater, haveRaters, mqmStatsByRater[rater][MQM_TOTAL],
                    mqmStatsByRaterAggregates[rater]);
    }
  }
  mqmShowScoresSeparator(haveRaters);

  mqmShowSystemRaterStats();
  mqmCreateSysVSysTables();
  mqmShowSevCatStats();
  mqmShowEvents();
  mqmShowSigtests(mqmStatsBySysAggregates);
}

/**
 * Increments the counts statsArray[severity][category][system] and
 *   statsArray[severity][category][MQM_TOTAL].
 * @param {!Object} statsArray
 * @param {string} system
 * @param {string} category
 * @param {string} severity
 */
function mqmAddSevCatStats(statsArray, system, category, severity) {
  if (!statsArray.hasOwnProperty(severity)) {
    statsArray[severity] = {};
  }
  if (!statsArray[severity].hasOwnProperty(category)) {
    statsArray[severity][category] = {};
    statsArray[severity][category][MQM_TOTAL] = 0;
  }
  if (!statsArray[severity][category].hasOwnProperty(system)) {
    statsArray[severity][category][system] = 0;
  }
  statsArray[severity][category][MQM_TOTAL]++;
  statsArray[severity][category][system]++;
}

/**
 * Returns total time spent, across various timing events in metadata.timing.
 * @param {!Object} metadata
 * @return {number}
 */
function mqmTimeSpent(metadata) {
  let timeSpentMS = 0;
  if (!metadata.timing) {
    return timeSpentMS;
  }
  for (let e in metadata.timing) {
    timeSpentMS += metadata.timing[e].timeMS;
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
function mqmAddEvents(events, metadata, doc, docSegId, system, rater) {
  if (!metadata.timing) {
    return;
  }
  for (let e of Object.keys(metadata.timing)) {
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
    for (let detail of log) {
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
 * Given a lowercase severity (lsev) & category (lcat), returns true if it is
 *   the "Non-translation" error, allowing for underscore/dash variation and
 *   a possible trailing exclamation mark. Non-translation may have been marked
 *   as a severity or as a category.
 * @param {string} lsev
 * @param {string} lcat
 * @return {boolean}
 */
function mqmIsNonTrans(lsev, lcat) {
  return lsev.startsWith('non-translation') ||
    lsev.startsWith('non_translation') ||
    lcat.startsWith('non-translation') ||
    lcat.startsWith('non_translation');
}

/**
 * Given a lowercase category (lcat), returns true if it is an accuracy error.
 * @param {string} lcat
 * @return {boolean}
 */
function mqmIsAccuracy(lcat) {
  return lcat.startsWith('accuracy') || lcat.startsWith('terminology');
}

/**
 * Updates stats with an error of (category, severity). The weighted score
 * component to use is the first matching one in mqmWeights[]. Similarly, the
 * slice to attribute the score to is the first matching one in mqmSlices[].
 * @param {!Object} stats
 * @param {number} timeSpentMS
 * @param {string} category
 * @param {string} severity
 * @param {number} span
 */
function mqmAddErrorStats(stats, timeSpentMS, category, severity, span) {
  stats.timeSpentMS += timeSpentMS;
  stats.scoreDenominator = 1;

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
  for (let sc of mqmWeights) {
    if (mqmMatchesScoreSplit(sc, lsev, lcat)) {
      score = sc.weight;
      stats.score += score;
      const key = mqmScoreKey(sc.name);
      stats[key] = (stats[key] ?? 0) + score;
      break;
    }
  }
  if (score > 0) {
    for (let sc of mqmSlices) {
      if (mqmMatchesScoreSplit(sc, lsev, lcat)) {
        const key = mqmScoreKey(sc.name, true);
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
function mqmArrayLast(a) {
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
function mqmGetSegStats(statsByDocAndDocSegId, doc, docSegId) {
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
function mqmGetSegStatsAsArray(statsByDocAndDocSegId) {
  let arr = [];
  for (let doc of Object.keys(statsByDocAndDocSegId)) {
    let statsByDocSegId = statsByDocAndDocSegId[doc];
    for (let docSegId of Object.keys(statsByDocSegId)) {
      arr.push(statsByDocSegId[docSegId]);
    }
  }
  return arr;
}

/**
 * Shows up or down arrow to show which field is used for sorting.
 */
function mqmShowSortArrow() {
  // Remove existing active arrows first.
  const active = document.querySelector('.mqm-arrow-active');
  if (active) {
    active.classList.remove('mqm-arrow-active');
  }
  // Highlight the appropriate arrow for the sorting field.
  const className = mqmSortReverse ? 'mqm-arrow-down' : 'mqm-arrow-up';
  const arrow = document.querySelector(
    `#mqm-${mqmSortByField}-th .${className}`);
  if (arrow) {
    arrow.classList.add('mqm-arrow-active');
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
function mqmGetSpan(tokens, spanBounds) {
  const parts = [];
  for (let bound of spanBounds) {
    const part = tokens.slice(bound[0], bound[1] + 1).join('');
    if (part) parts.push(part);
  }
  return parts.join('...');
}

/**
 * From a segment with spans marked using <v>..</v>, scoops out and returns
 * just the marked spans. This is the fallback for finding the span to display,
 * for legacy data where detailed tokenization info may not be available.
 * @param {string} text
 * @return {string}
 */
function mqmGetLegacySpan(text) {
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
function mqmSeverityClass(severity) {
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
 * For the annotation defined in metadata, (for row rowId in mqmData), sets
 * metadata.marked_text as the text that has been marked by the rater (or
 * sets it to the empty string). The rowId is only used for legacy formats
 * where tokenization is not available in metadata.
 * @param {number} rowId
 * @param {!Object} metadata
 */
function mqmSetMarkedText(rowId, metadata) {
  let sourceSpan = mqmGetSpan(metadata.segment.source_tokens,
                              metadata.source_spans || []);
  if (!sourceSpan) {
    const source = mqmData[rowId][MQM_DATA_SOURCE];
    sourceSpan = mqmGetLegacySpan(source);
  }
  let targetSpan = mqmGetSpan(metadata.segment.target_tokens,
                              metadata.target_spans || []);
  if (!targetSpan) {
    const target = mqmData[rowId][MQM_DATA_TARGET];
    targetSpan = mqmGetLegacySpan(target);
  }
  metadata.marked_text = sourceSpan + targetSpan;
}

/**
 * For the given severity level, return an HTML string suitable for displaying
 * it, including an identifier that includes rowId (for creating a filter upon
 * clicking).
 * @param {number} rowId
 * @param {string} severity
 * @param {!Object} metadata
 * @return {string}
 */
function mqmSeverityHTML(rowId, severity, metadata) {
  let html = '';
  html += `<span class="mqm-val" id="mqm-val-${rowId}-${MQM_DATA_SEVERITY}">` +
          severity + '</span>';
  return html;
}

/**
 * For the given annotation category, return an HTML string suitable for
 * displaying it, including an identifier that includes rowId (for creating a
 * filter upon clicking). If the metadata includes a note from the rater,
 * include it in the HTML.
 * @param {number} rowId
 * @param {string} category
 * @param {!Object} metadata
 * @return {string}
 */
function mqmCategoryHTML(rowId, category, metadata) {
  let html = '';
  html += `<span class="mqm-val" id="mqm-val-${rowId}-${MQM_DATA_CATEGORY}">` +
          category + '</span>';
  if (metadata.note) {
    /* There is a note */
    html += '<br><span class="mqm-note">' + metadata.note + '</span>';
  }
  return html;
}

/**
 * For the given rater name/id, return an HTML string suitable for displaying
 * it, including an identifier that includes rowId (for creating a filter upon
 * clicking). If the metadata includes a timestamp or feedback from the rater,
 * include that in the HTML.
 * @param {number} rowId
 * @param {string} rater
 * @param {!Object} metadata
 * @return {string}
 */
function mqmRaterHTML(rowId, rater, metadata) {
  let html = '';
  html += `<span class="mqm-val" id="mqm-val-${rowId}-${MQM_DATA_RATER}">` +
          rater + '</span>';
  if (metadata.timestamp) {
    /* There is a timestamp, but it might have been stringified */
    const timestamp = parseInt(metadata.timestamp, 10);
    html += ' <span class="mqm-timestamp">' +
            (new Date(timestamp)).toLocaleString() + '</span>';
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
      feedbackHTML += '<br><span class="mqm-note">' + notes + '</span>';
    }
    html += feedbackHTML;
  }
  return html;
}

/**
 * Returns the "metrics line" to display for the current segment, which
 * includes MQM score as well as any available automated metrics.
 * @param {!Object} currSegStatsBySys
 * @return {string}
 */
function mqmGetSegScoresHTML(currSegStatsBySys) {
  const segScoresParts = [];
  for (let metric in currSegStatsBySys.metrics) {
    const s = currSegStatsBySys.metrics[metric];
    segScoresParts.push([metric, mqmMetricDisplay(s, 1)]);
    if (metric == 'MQM') {
      const aggregate = mqmAggregateSegStats([currSegStatsBySys]);
      if (aggregate.score != s) {
        segScoresParts.push(
            ['MQM-filtered',
            mqmMetricDisplay(aggregate.score, aggregate.scoreDenominator)]);
      }
    }
  }
  if (segScoresParts.length == 0) {
    return '';
  }
  let scoresRows = '';
  for (let part of segScoresParts) {
    scoresRows += '<tr><td>' + part[0] + ':&nbsp;</td>' +
    '<td><span class="mqm-seg-score">' + part[1] + '</span></td></tr>';
  }
  return '<tr><td><table class="mqm-scores-table">' +
         scoresRows +
         '</table></td></tr>\n';
}

/**
 * Updates the display to show the segment data and scores according to the
 * current filters.
 * @param {?Object=} viewingConstraints Optional dict of doc:seg to view. When
 *     not null, only these segments are shown. When not null, this parameter
 *     object should have two additional properties:
 *       description: Shown to the user, describing the constrained view.
 *       color: A useful identifying color that highlights the description.
 */
function mqmShow(viewingConstraints=null) {
  // Cancel existing Sigtest computation when a new `mqmShow` is called.
  mqmResetSigtests();

  mqmTable.innerHTML = '';
  mqmStatsTable.innerHTML = '';
  mqmSevCatStatsTable.innerHTML = '';
  mqmEventsTable.innerHTML = '';

  mqmStats = {};
  mqmStats[MQM_TOTAL] = {};
  mqmStatsByRater = {};
  mqmSevCatStats = {};

  mqmDataFiltered = [];

  mqmEvents = {
    aggregates: {},
    raters: {},
  };
  const visibleMetrics = {};
  mqmMetricsVisible = [];

  const viewingConstraintsDesc = document.getElementById(
      'mqm-viewing-constraints');
  if (viewingConstraints) {
    viewingConstraintsDesc.innerHTML = 'View limited to ' +
        viewingConstraints.description +
        ' Click on this text to remove this constraint.';
    viewingConstraintsDesc.style.backgroundColor = viewingConstraints.color;
    viewingConstraintsDesc.style.display = '';
  } else {
    viewingConstraintsDesc.innerHTML = '';
    viewingConstraintsDesc.style.display = 'none';
  }

  document.getElementById('mqm-filter-expr-error').innerHTML = '';
  const allFilters = mqmGetAllFilters();

  let currSegStats = [];
  let currSegStatsBySys = [];
  let currSegStatsByRater = [];
  let unfilteredCount = 0;
  let shownCount = 0;
  const shownRows = [];

  document.body.style.cursor = 'wait';
  for (doc of mqmDataIter.docs) {
    for (docSegId of mqmDataIter.docSegs[doc]) {
      let shownForDocSeg = 0;
      let aggrDocSeg = null;
      for (system of mqmDataIter.docSys[doc]) {
        let shownForDocSegSys = 0;
        let firstRowId = -1;
        let ratingRowsHTML = '';
        let sourceTokens = null;
        let targetTokens = null;
        let lastRater = '';
        const range = mqmDataIter.docSegSys[doc][docSegId][system].rows;
        let aggrDocSegSys = null;
        const docColonSys = doc + ':' + system;
        for (let rowId = range[0]; rowId < range[1]; rowId++) {
          const parts = mqmData[rowId];
          let match = true;
          for (let id in allFilters.filterREs) {
            const col = mqmFilterColumns[id];
            if (allFilters.filterREs[id] &&
                !allFilters.filterREs[id].test(parts[col])) {
              match = false;
              break;
            }
          }
          if (!match) {
            continue;
          }
          if (!mqmFilterExprPasses(allFilters.filterExpr, parts)) {
            continue;
          }
          const metadata = parts[MQM_DATA_METADATA];
          if (allFilters.onlyAllSysSegs &&
              !mqmAllSystemsFilterPasses(metadata)) {
            continue;
          }

          unfilteredCount++;
          const rater = parts[MQM_DATA_RATER];
          const category = parts[MQM_DATA_CATEGORY];
          const severity = parts[MQM_DATA_SEVERITY];
          if (!aggrDocSeg && metadata.segment && metadata.segment.aggrDocSeg) {
            aggrDocSeg = metadata.segment.aggrDocSeg;
          }
          if (!aggrDocSegSys) {
            aggrDocSegSys = metadata.segment;
          }

          /**
           * Copy, as we will clear out unnecessary/bulky fields from the
           * metadata in mqmDataFiltered.
           */
          const filteredMetadata = {...metadata};
          delete filteredMetadata.evaluation;

          if (firstRowId < 0) {
            firstRowId = rowId;

            sourceTokens = (metadata.segment.source_tokens || []).slice();
            targetTokens = (metadata.segment.target_tokens || []).slice();

            currSegStats = mqmGetSegStats(
                mqmStats[MQM_TOTAL], docColonSys, docSegId);
            if (!mqmStats.hasOwnProperty(system)) {
              mqmStats[system] = {};
            }
            currSegStatsBySys =
                mqmGetSegStats(mqmStats[system], doc, docSegId);
            currSegStats.srcLen = parts.srcLen;
            currSegStatsBySys.srcLen = parts.srcLen;
            if (metadata.segment.hasOwnProperty('metrics')) {
              currSegStatsBySys.metrics = metadata.segment.metrics;
              for (let metric in currSegStatsBySys.metrics) {
                visibleMetrics[metric] = true;
              }
            }
            /**
             * Clear aggregated docseg info from filteredMetadata.segment.
             */
            filteredMetadata.segment = {...metadata.segment};
            delete filteredMetadata.segment.aggrDocSeg;
          } else {
            /**
             * We keep segment info only in the first filtered row's metadata.
             */
            delete filteredMetadata.segment;
          }

          const partsForFilteredData = parts.slice();
          partsForFilteredData[MQM_DATA_METADATA] =
              JSON.stringify(filteredMetadata);
          mqmDataFiltered.push(partsForFilteredData);

          if (rater && (rater != lastRater)) {
            lastRater = rater;
            visibleMetrics['MQM'] = true;  /** We do have some MQM scores. */

            currSegStats.push(mqmInitRaterStats(rater));
            currSegStatsBySys.push(mqmInitRaterStats(rater));
            if (!mqmStatsByRater.hasOwnProperty(rater)) {
              /** New rater. **/
              mqmStatsByRater[rater] = {};
              mqmStatsByRater[rater][MQM_TOTAL] = {};
            }
            currSegStatsByRater = mqmGetSegStats(
                mqmStatsByRater[rater][MQM_TOTAL], docColonSys, docSegId);
            currSegStatsByRater.push(mqmInitRaterStats(rater));
            currSegStatsByRater.srcLen = parts.srcLen;

            if (!mqmStatsByRater[rater].hasOwnProperty(system)) {
              mqmStatsByRater[rater][system] = {};
            }
            currSegStatsByRaterSys = mqmGetSegStats(
                mqmStatsByRater[rater][system], doc, docSegId);
            currSegStatsByRaterSys.push(mqmInitRaterStats(rater));
            currSegStatsByRaterSys.srcLen = parts.srcLen;
          }
          let spanClass = '';
          if (rater) {
            /** An actual rater-annotation row, not just a metadata row */
            spanClass = mqmSeverityClass(severity) +
                        ` mqm-anno-${shownRows.length}`;
            mqmMarkSpans(sourceTokens, metadata.source_spans || [], spanClass);
            mqmMarkSpans(targetTokens, metadata.target_spans || [], spanClass);
            const span = metadata.marked_text.length;
            const timeSpentMS = mqmTimeSpent(metadata);
            mqmAddErrorStats(mqmArrayLast(currSegStats),
                             timeSpentMS, category, severity, span);
            mqmAddErrorStats(mqmArrayLast(currSegStatsBySys),
                             timeSpentMS, category, severity, span);
            mqmAddErrorStats(mqmArrayLast(currSegStatsByRater),
                             timeSpentMS, category, severity, span);
            mqmAddErrorStats(mqmArrayLast(currSegStatsByRaterSys),
                             timeSpentMS, category, severity, span);
            mqmAddSevCatStats(mqmSevCatStats, system, category, severity);
            mqmAddEvents(mqmEvents, metadata, doc, docSegId, system, rater);
          }

          if (viewingConstraints &&
              !viewingConstraints[mqmDocSegKey(doc, docSegId)]) {
            continue;
          }
          if (shownCount >= mqmLimit) {
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
            const textSpan = metadata.marked_text.replace(
                /</g, '&lt;').replace(/>/g, '&gt;');
            ratingRowsHTML += '<span class="' + spanClass + '">[' +
                              textSpan + ']</span><br>';
          }
          ratingRowsHTML += mqmSeverityHTML(rowId, severity, metadata) +
                            '&nbsp;';
          ratingRowsHTML += mqmCategoryHTML(rowId, category, metadata) + '<br>';
          ratingRowsHTML += mqmRaterHTML(rowId, rater, metadata);
          ratingRowsHTML += '</div></td></tr>\n';
        }
        if (shownForDocSegSys == 0) {
          continue;
        }
        console.assert(firstRowId >= 0, firstRowId);

        if (shownForDocSeg == 0 && aggrDocSeg && aggrDocSeg.references) {
          for (ref of Object.keys(aggrDocSeg.references)) {
            let refRowHTML = '<tr class="mqm-row mqm-ref-row">';
            refRowHTML += '<td><div>' + doc + '</div></td>';
            refRowHTML += '<td><div>' + docSegId + '</div></td>';
            refRowHTML += '<td><div><b>Ref</b>: ' + ref + '</div></td>';
            const sourceTokens = aggrDocSeg.source_tokens || [];
            refRowHTML += '<td><div>' + sourceTokens.join('') + '</div></td>';
            refRowHTML += '<td><div>' +
                          aggrDocSeg.references[ref] +
                          '</div></td>';
            refRowHTML += '<td></td></tr>\n';
            mqmTable.insertAdjacentHTML('beforeend', refRowHTML);
          }
        }
        let rowHTML = '';
        rowHTML += '<td><div class="mqm-val" ';
        rowHTML += `id="mqm-val-${firstRowId}-${MQM_DATA_DOC}">` + doc +
                   '</div></td>';
        rowHTML += '<td><div class="mqm-val" ';
        rowHTML += `id="mqm-val-${firstRowId}-${MQM_DATA_DOC_SEG_ID}">` +
                   docSegId + '</div></td>';
        rowHTML += '<td><div class="mqm-val" ';
        rowHTML += `id="mqm-val-${firstRowId}-${MQM_DATA_SYSTEM}">` +
                   system + '</div></td>';

        const source = sourceTokens.length > 0 ? sourceTokens.join('') :
                       mqmData[firstRowId][MQM_DATA_SOURCE].replace(
                           /<\/?v>/g, '');
        const target = targetTokens.length > 0 ? targetTokens.join('') :
                       mqmData[firstRowId][MQM_DATA_TARGET].replace(
                           /<\/?v>/g, '');

        rowHTML += '<td><div>' + source + '</div></td>';
        rowHTML += '<td><div>' + target + '</div></td>';

        rowHTML += '<td><table class="mqm-table-ratings">' +
                   ratingRowsHTML + mqmGetSegScoresHTML(currSegStatsBySys) +
                   '</table></td>';

        mqmTable.insertAdjacentHTML(
            'beforeend', `<tr class="mqm-row">${rowHTML}</tr>\n`);
        shownForDocSeg += shownForDocSegSys;
      }
      if (shownForDocSeg > 0) {
        shownCount += shownForDocSeg;
      }
    }
  }
  /**
   * Update #unfiltered rows display.
   */
  document.getElementById('mqm-num-rows').innerText = mqmData.length;
  document.getElementById('mqm-num-unfiltered-rows').innerText =
      unfilteredCount;

  document.body.style.cursor = 'auto';
  /**
   * Add cross-highlighting listeners.
   */
  const annoHighlighter = (a, shouldShow) => {
    const elts = document.getElementsByClassName('mqm-anno-' + a);
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
    const elts = document.getElementsByClassName('mqm-anno-' + a);
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
  const filters = document.getElementsByClassName('mqm-filter-re');
  for (let rowId of shownRows) {
    const parts = mqmData[rowId];
    for (let i = 0; i < filters.length; i++) {
      const filter = filters[i];
      const col = mqmFilterColumns[filter.id];
      const v = document.getElementById(`mqm-val-${rowId}-${col}`);
      if (!v) continue;
      v.addEventListener('click', (e) => {
        filter.value = '^' + parts[col] + '$';
        mqmShow();
      });
    }
  }
  for (let m = 0; m < mqmMetrics.length; m++) {
    const metric = mqmMetrics[m];
    if (visibleMetrics[metric]) {
      mqmMetricsVisible.push(m);
    }
  }
  if (mqmSortByField.startsWith('metric-')) {
    /**
     * If the currently chosen sort-by field is a metric that is not visible,
     * then change it to be the first metric that *is* visible (if any,
     * defaulting to metric-0, which is MQM). Set the default direction based
     * upon whether lower numbers are better for the chosen metric.
     */
    let sortingMetric = parseInt(mqmSortByField.substr(7));
    if (!mqmMetricsVisible.includes(sortingMetric)) {
      sortingMetric = 0;
      for (let m = 0; m < mqmMetrics.length; m++) {
        const metric = mqmMetrics[m];
        if (visibleMetrics[metric]) {
          sortingMetric = m;
          break;
        }
      }
      mqmSortByField = 'metric-' + sortingMetric;
      mqmSortReverse = mqmMetricsInfo[mqmMetrics[sortingMetric]].lowerBetter ?
                       false : true;
    }
  }
  mqmShowStats();
}

/**
 * Recomputes MQM score for each segment (using current weight settings) and
 * sets it in segment.metrics['MQM'].
 */
function mqmRecomputeMQM() {
  statsBySystem = {};
  let currSegStatsBySys = [];
  for (doc of mqmDataIter.docs) {
    for (docSegId of mqmDataIter.docSegs[doc]) {
      for (system of mqmDataIter.docSys[doc]) {
        let lastRater = '';
        const range = mqmDataIter.docSegSys[doc][docSegId][system].rows;
        let aggrDocSegSys = null;
        for (let rowId = range[0]; rowId < range[1]; rowId++) {
          const parts = mqmData[rowId];
          const metadata = parts[MQM_DATA_METADATA];
          if (!aggrDocSegSys) {
            aggrDocSegSys = metadata.segment;
            if (!statsBySystem.hasOwnProperty(system)) {
              statsBySystem[system] = {};
            }
            currSegStatsBySys =
                mqmGetSegStats(statsBySystem[system], doc, docSegId);
            currSegStatsBySys.srcLen = parts.srcLen;
          }
          const rater = parts[MQM_DATA_RATER];
          if (!rater) {
            continue;
          }
          if (rater != lastRater) {
            lastRater = rater;
            currSegStatsBySys.push(mqmInitRaterStats(rater));
          }
          const category = parts[MQM_DATA_CATEGORY];
          const severity = parts[MQM_DATA_SEVERITY];
          /** We don't care about computing avg span/time here, pass as 0. */
          mqmAddErrorStats(mqmArrayLast(currSegStatsBySys),
                           0, category, severity, 0);
        }
        if (aggrDocSegSys) {
          const aggrScores = mqmAggregateSegStats([currSegStatsBySys]);
          if (aggrScores.scoreDenominator > 0) {
            aggrDocSegSys.metrics['MQM'] = aggrScores.score;
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
function mqmMarkSpans(tokens, bounds, cls) {
  for (let bound of bounds) {
    for (let i = bound[0]; i <= bound[1]; i++) {
      if (i < 0 || i >= tokens.length) {
        continue;
      }
      tokens[i] = '<span class="' + cls + '">' + tokens[i] + '</span>';
    }
  }
}

/**
 * Clears all filters, except possibly 'mqm-only-all-systems-segments'.
 * @param {boolean=} resetOnlyAllSys
 */
function mqmClearFilters(resetOnlyAllSys=false) {
  const filters = document.getElementsByClassName('mqm-filter-re');
  for (let i = 0; i < filters.length; i++) {
    filters[i].value = '';
  }
  document.getElementById('mqm-filter-expr').value = '';
  document.getElementById('mqm-filter-expr-error').innerHTML = '';
  if (resetOnlyAllSys) {
    document.getElementById('mqm-only-all-systems-segments').checked = false;
  }
}

/**
 * Clears all filters and shows stats again.
 */
function mqmClearFiltersAndShow() {
  mqmClearFilters(true);
  mqmShow();
}

/**
 * For the column named by "what", sets the filter to the currently picked
 * value from its drop-down list.
 * @param {string} what
 */
function mqmPick(what) {
  const filter = document.getElementById('mqm-filter-' + what);
  if (!filter) return;
  const sel = document.getElementById('mqm-select-' + what);
  if (!sel) return;
  filter.value = sel.value;
  mqmShow();
}

/**
 * Populates the column drop-down lists and filter-expression builder with
 * unique values.
 */
function mqmSetSelectOptions() {
  const options = {};
  for (let id in mqmFilterColumns) {
    options[id] = {};
  }
  for (let parts of mqmData) {
    for (let id in mqmFilterColumns) {
      const col = mqmFilterColumns[id];
      if (col == MQM_DATA_SOURCE || col == MQM_DATA_TARGET) continue;
      options[id][parts[col].trim()] = true;
    }
  }
  for (let id in mqmFilterColumns) {
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
  mqmClauseKey = document.getElementById('mqm-clause-key');
  let html = '<option value=""></option>\n';
  const SYSTEM_FILTER_ID = 'mqm-filter-system';
  for (let sys in options[SYSTEM_FILTER_ID]) {
    html += `<option value="System: ${sys}">System: ${sys}</option>\n`;
  }
  const RATER_FILTER_ID = 'mqm-filter-rater';
  for (let rater in options[RATER_FILTER_ID]) {
    html += `<option value="Rater: ${rater}">Rater: ${rater}</option>\n`;
  }
  mqmClauseKey.innerHTML = html;

  mqmClauseInclExcl = document.getElementById('mqm-clause-inclexcl');

  mqmClauseCat = document.getElementById('mqm-clause-cat');
  html = '<option value=""></option>\n';
  const CATEGORY_FILTER_ID = 'mqm-filter-category';
  for (let cat in options[CATEGORY_FILTER_ID]) {
    html += `<option value="${cat}">${cat}</option>\n`;
  }
  mqmClauseCat.innerHTML = html;

  mqmClauseSev = document.getElementById('mqm-clause-sev');
  html = '<option value=""></option>\n';
  const SEVERITY_FILTER_ID = 'mqm-filter-severity';
  for (let sev in options[SEVERITY_FILTER_ID]) {
    html += `<option value="${sev}">${sev}</option>\n`;
  }
  mqmClauseSev.innerHTML = html;

  mqmClauseAddAnd = document.getElementById('mqm-clause-add-and');
  mqmClauseAddOr = document.getElementById('mqm-clause-add-or');
  mqmClearClause();
}

/**
 * This resets information derived from or associated with the current data (if
 * any), preparing for new data.
 */
function mqmResetData() {
  mqmClearFilters();
  mqmData = [];
  mqmMetrics = ['MQM'];
  for (let key in mqmMetricsInfo) {
    /** Only retain the entry for 'MQM'. */
    if (key == 'MQM') continue;
    delete mqmMetricsInfo[key];
  }
  mqmMetricsVisible = [];
  mqmSortByField = 'metric-0';
  mqmSortReverse = false;
  mqmCloseMenuEntries('');
}

/**
 * Maximum number of lines of data that we'll consume. Human eval data is
 * generally of modest size, but automated metrics data can be arbitrary large.
 * Users should limit and curate such data.
 * 1000 docs * 100 lines * 10 systems * 10 raters = 10,000,000
 */
const MQM_VIEWER_MAX_DATA_LINES = 10000000;

/**
 * Sets mqmTSVData from the passed TSV data string or array of strings, and
 * parses it into mqmData. If the UI option mqm-load-file-append is checked,
 * then the new data is appended to the existing data, else it replaces it.
 * @param {string|!Array<string>} tsvData
 */
function mqmSetData(tsvData) {
  const errors = document.getElementById('mqm-errors');
  errors.innerHTML = '';
  if (Array.isArray(tsvData)) {
    let allTsvData = '';
    for (let tsvDataItem of tsvData) {
      if (!tsvDataItem) continue;
      if (allTsvData && !allTsvData.endsWith('\n')) {
        allTsvData += '\n';
      }
      allTsvData += tsvDataItem;
    }
    tsvData = allTsvData;
  }
  if (!tsvData) {
    errors.innerHTML = 'Empty data passed to mqmSetData()';
    return;
  }
  if (document.getElementById('mqm-load-file-append').checked) {
    if (mqmTSVData && !mqmTSVData.endsWith('\n')) {
      mqmTSVData += '\n';
    }
  } else {
    mqmTSVData = '';
  }
  mqmTSVData += tsvData;

  mqmResetData();
  const data = mqmTSVData.split('\n');
  for (let line of data) {
    if (mqmData.length >= MQM_VIEWER_MAX_DATA_LINES) {
      errors.insertAdjacentHTML('beforeend',
          'Skipping data lines beyond number ' + MQM_VIEWER_MAX_DATA_LINES);
      break;
    }
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
    if (parts.length < MQM_DATA_METADATA) {
      errors.insertAdjacentHTML('beforeend',
          `Could not parse: ${line.substr(0, 80)}...<br>`);
      continue;
    } else if (parts.length == MQM_DATA_METADATA) {
      /** TSV data is missing the last metadata column. Create it. */
      parts.push(metadata);
    } else {
      /**
       * The 10th column should be a JSON-encoded "metadata" object. Prior to
       * May 2022, the 10th column, when present, was just a string that was a
       * "note" from the rater, so convert that to a metadata object if needed.
       */
      try {
        metadata = JSON.parse(parts[MQM_DATA_METADATA]);
      } catch (err) {
        console.log(err);
        console.log(parts[MQM_DATA_METADATA]);
        metadata = {};
        const note = parts[MQM_DATA_METADATA].trim();
        if (note) {
          metadata['note'] = note;
        }
      }
      parts[MQM_DATA_METADATA] = metadata;
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
    if (!metadata.segment.metrics) {
      metadata.segment.metrics = {};
    }
    if (!metadata.feedback) {
      metadata.feedback = {};
    }
    if (metadata.evaluation) {
      /* Show the evaluation metadata in the log. */
      console.log('Evaluation info found in row ' + mqmData.length + ':');
      console.log(metadata.evaluation);
    }
    /** Note any metrics that might be in the data. */
    const metrics = metadata.segment.metrics;
    for (let metric in metrics) {
      if (mqmMetricsInfo.hasOwnProperty(metric)) continue;
      mqmMetricsInfo[metric] = {
        index: mqmMetrics.length,
      };
      mqmMetrics.push(metric);
    }
    /** Move "Rater" down from its position in the TSV data. */
    const temp = parts[4];
    parts[MQM_DATA_SOURCE] = parts[5];
    parts[MQM_DATA_TARGET] = parts[6];
    parts[MQM_DATA_RATER] = temp;
    parts[MQM_DATA_SEVERITY] =
        parts[MQM_DATA_SEVERITY].charAt(0).toUpperCase() +
        parts[MQM_DATA_SEVERITY].substr(1);
    /**
     * Count all characters, including spaces, in src/tgt length, excluding
     * the span-marking <v> and </v> tags.
     */
    parts.srcLen = parts[MQM_DATA_SOURCE].replace(/<\/?v>/g, '').length;
    parts.tgtLen = parts[MQM_DATA_TARGET].replace(/<\/?v>/g, '').length;
    mqmData.push(parts);
  }
  mqmSortData(mqmData);
  mqmCreateDataIter(mqmData);
  mqmRecomputeMQM();
  mqmAddSegmentAggregations();
  mqmSetSelectOptions();
  mqmShow();
}

/**
 * Opens and reads the data file(s) picked by the user and calls mqmSetData().
 */
function mqmOpenFiles() {
  document.body.style.cursor = 'wait';
  mqmClearFilters();
  const errors = document.getElementById('mqm-errors');
  errors.innerHTML = '';
  const filesElt = document.getElementById('mqm-file');
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
          if (typeof mqmDataConverter == 'function') {
            for (let i = 0; i < filesData.length; i++) {
              filesData[i] = mqmDataConverter(fileNames[i], filesData[i]);
            }
          }
          mqmSetData(filesData);
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
 * Fetches MQM data from the given URLs and calls mqmSetData().
 * If the mqmURLMaker() function exists, then it is applied to each URL
 * first, to get a possibly modified URL.
 * @param {!Array<string>} urls
 */
function mqmFetchURLs(urls) {
  const errors = document.getElementById('mqm-errors');
  errors.innerHTML = 'Loading metrics data from ' + urls.length + ' URL(s)...';
  const cleanURLs = [];
  for (let url of urls) {
    if (typeof mqmURLMaker == 'function') {
      url = mqmURLMaker(url);
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
      if (typeof mqmDataConverter == 'function') {
        for (let i = 0; i < tsvData.length; i++) {
          tsvData[i] = mqmDataConverter(cleanURLs[i], tsvData[i]);
        }
      }
      mqmSetData(tsvData);
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
 * Returns the currently filtered MQM data in TSV format. The filtered
 * data is available in the mqmDataFiltered array. This function reorders
 * the columns (from "source, target, rater" to "rater, source, target")
 * before splicing them into TSV format.
 * @return {string}
 */
function mqmGetFilteredTSVData() {
  let tsvData = '';
  for (let row of mqmDataFiltered) {
    const tsvOrderedRow = [];
    for (let i = 0; i < MQM_DATA_NUM_PARTS; i++) {
      tsvOrderedRow[i] = row[i];
    }
    /** Move "Rater" up from its position in mqmDataFiltered. */
    tsvOrderedRow[4] = row[MQM_DATA_RATER];
    tsvOrderedRow[5] = row[MQM_DATA_SOURCE];
    tsvOrderedRow[6] = row[MQM_DATA_TARGET];
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
function mqmGetScoresTSVData(aggregation) {
  /**
   * We use a fake 10-column mqm-data array (with score kept in the last
   * column) to sort the data in the right order using mqmSortData().
   */
  const data = [];
  const FAKE_FIELD = '--MQM-FAKE-FIELD--';
  if (aggregation == 'system') {
    for (let system in mqmStats) {
      if (system == MQM_TOTAL) {
        continue;
      }
      const segs = mqmGetSegStatsAsArray(mqmStats[system]);
      aggregate = mqmAggregateSegStats(segs);
      dataRow = Array(MQM_DATA_NUM_PARTS).fill(FAKE_FIELD);
      dataRow[MQM_DATA_SYSTEM] = system;
      dataRow[MQM_DATA_METADATA] = aggregate.score;
      data.push(dataRow);
    }
  } else if (aggregation == 'document') {
    for (let system in mqmStats) {
      if (system == MQM_TOTAL) {
        continue;
      }
      const stats = mqmStats[system];
      for (let doc in stats) {
        const docStats = stats[doc];
        const segs = mqmGetSegStatsAsArray({doc: docStats});
        aggregate = mqmAggregateSegStats(segs);
        dataRow = Array(MQM_DATA_NUM_PARTS).fill(FAKE_FIELD);
        dataRow[MQM_DATA_SYSTEM] = system;
        dataRow[MQM_DATA_DOC] = doc;
        dataRow[MQM_DATA_METADATA] = aggregate.score;
        data.push(dataRow);
      }
    }
  } else if (aggregation == 'segment') {
    for (let system in mqmStats) {
      if (system == MQM_TOTAL) {
        continue;
      }
      const stats = mqmStats[system];
      for (let doc in stats) {
        const docStats = stats[doc];
        for (let seg in docStats) {
          const docSegStats = docStats[seg];
          const segs = mqmGetSegStatsAsArray({doc: {seg: docSegStats}});
          aggregate = mqmAggregateSegStats(segs);
          dataRow = Array(MQM_DATA_NUM_PARTS).fill(FAKE_FIELD);
          dataRow[MQM_DATA_SYSTEM] = system;
          dataRow[MQM_DATA_DOC] = doc;
          dataRow[MQM_DATA_DOC_SEG_ID] = seg;
          dataRow[MQM_DATA_METADATA] = aggregate.score;
          data.push(dataRow);
        }
      }
    }
  } else /* (aggregation == 'rater') */ {
    for (let rater in mqmStatsByRater) {
      for (let system in mqmStatsByRater[rater]) {
        if (system == MQM_TOTAL) {
          continue;
        }
        const stats = mqmStatsByRater[rater][system];
        for (let doc in stats) {
          const docStats = stats[doc];
          for (let seg in docStats) {
            const docSegStats = docStats[seg];
            const segs = mqmGetSegStatsAsArray({doc: {seg: docSegStats}});
            aggregate = mqmAggregateSegStats(segs);
            dataRow = Array(MQM_DATA_NUM_PARTS).fill(FAKE_FIELD);
            dataRow[MQM_DATA_SYSTEM] = system;
            dataRow[MQM_DATA_DOC] = doc;
            dataRow[MQM_DATA_DOC_SEG_ID] = seg;
            dataRow[MQM_DATA_RATER] = rater;
            dataRow[MQM_DATA_METADATA] = aggregate.score;
            data.push(dataRow);
          }
        }
      }
    }
  }
  mqmSortData(data);
  /** remove FAKE_FIELD columns */
  let tsvData = '';
  for (let i = 0; i < data.length; i++) {
    const trimmedRow = [];
    for (let entry of data[i]) {
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
function mqmSaveDataInner(tsvData, fileName) {
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
  mqmCloseMenuEntries('');
}

/**
 * Saves mqmTSVData or filtered or filtered+aggregated data to the file
 *     mqm-data.tsv. Adds a header line when saving non-aggregated MQM data,
 *     if it's not already there.
 * @param {string} saveType One of 'all', 'filtered', 'system', 'document',
 *     'segment', 'rater'
 * @param {string} fileName This is appened to any prefix entered in the
 *     mqm-saved-file-prefix field.
 */
function mqmSaveData(saveType, fileName) {
  let tsvData = '';
  let addHeader = true;
  if (saveType == 'all') {
    tsvData = mqmTSVData;
  } else if (saveType == 'filtered') {
    tsvData = mqmGetFilteredTSVData();
  } else {
    tsvData = mqmGetScoresTSVData(saveType);
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
              'aster/mqm_viewer\n' + tsvData;
  }
  const prefix = document.getElementById('mqm-saved-file-prefix').value.trim();
  mqmSaveDataInner(tsvData, prefix + fileName);
}

/**
 * Applies updated settings for scoring.
 */
function mqmUpdateSettings() {
  const unit = document.getElementById('mqm-scoring-unit').value;
  mqmCharScoring = (unit == 'characters');
  const unitDisplay = document.getElementById('mqm-scoring-unit-display');
  if (unitDisplay) {
    unitDisplay.innerHTML = (mqmCharScoring ? '100 source chars' : 'segment');
  }
  if (mqmParseScoreSettings()) {
    mqmSetUpScoreSettings();
  }
  /**
   * Need to recompute metadata.segment.metrics['MQM'] for each segment first,
   * for use in subsequent filtering.
   */
  mqmRecomputeMQM();
  mqmShow();
}

/**
 * Resets scoring settings to their default values.
 */
function mqmResetSettings() {
  document.getElementById('mqm-scoring-unit').value = 'segments';
  mqmWeights = JSON.parse(JSON.stringify(mqmDefaultWeights));
  mqmSlices = JSON.parse(JSON.stringify(mqmDefaultSlices));
  mqmSetUpScoreSettings();
  mqmUpdateSettings();
}

/**
 * Collapse all top menu zippy panels, except the one
 * with the given id.
 * @param {string=} except
 */
function mqmCloseMenuEntries(except='') {
  const menuEntries = document.getElementsByClassName('mqm-menu-entry');
  for (let i = 0; i < menuEntries.length; i++) {
    const entry = menuEntries[i];
    if (entry.id != except) {
      entry.removeAttribute('open');
    }
  }
}

/**
 * Replaces the HTML contents of elt with the HTML needed to render the
 *     MQM Viewer. If tsvDataOrURLs is not null, then it can be MQM TSV-data,
 *     or a CSV list of URLs from which to fetch MQM TSV-data.
 * @param {!Element} elt
 * @param {string=} tsvDataOrCsvURLs
 * @param {boolean=} loadReplaces determines whether loading new data
 *     replaces the current data or augments it, by default.
 */
function createMQMViewer(elt, tsvDataOrCsvURLs='', loadReplaces=true) {
  const tooltip = 'Regular expressions are used case-insensitively. ' +
      'Click on the Apply button after making changes.';
  const settings = `
    <details class="mqm-settings mqm-menu-entry" id="mqm-menu-entry-settings"
        title="Change scoring weights, slices, units.">
      <summary>Settings</summary>
      <div class="mqm-settings-panel">
        <div class="mqm-settings-row">
          Scoring units:
          <select id="mqm-scoring-unit" onchange="mqmUpdateSettings()">
            <option value="segments">Segments</option>
            <option value="characters">100 source characters</option>
          </select>
        </div>
        <div class="mqm-settings-row">
           Note: Changes to the following tables of weights and slices only
           take effect after clicking on <b>Apply!</b>
        </div>
        <div class="mqm-settings-row" title="${tooltip}">
          Ordered list of <i>weights</i> to apply to error patterns:
          <button onclick="mqmSettingsAddRow('mqm-settings-weights', 3)"
              >Add new row</button> as row <input type="text" maxlength="2"
              class="mqm-input"
              id="mqm-settings-weights-add-row" size=2 placeholder="1"/>
          <table class="mqm-table mqm-settings-panel">
            <thead>
              <tr>
                <th>Weight name</th>
                <th>Regular expression to match
                    <i>severity:category[/subcategory]</i></th>
                <th>Weight</th>
              </tr>
            </thead>
            <tbody id="mqm-settings-weights">
            </tbody>
          </table>
        </div>
        <div class="mqm-settings-row" title="${tooltip}">
          Ordered list of interesting <i>slices</i> of error patterns:
          <button onclick="mqmSettingsAddRow('mqm-settings-slices', 2)"
              >Add new row</button> as row <input type="text" maxlength="2"
              class="mqm-input"
              id="mqm-settings-slices-add-row" size=2 placeholder="1"/>
          <table class="mqm-table mqm-settings-panel">
            <thead>
              <tr>
                <th>Slice name</th>
                <th>Regular expression to match
                    <i>severity:category[/subcategory]</i></th>
              </tr>
            </thead>
            <tbody id="mqm-settings-slices">
            </tbody>
          </table>
        </div>
        <div class="mqm-settings-row">
          <button id="mqm-reset-settings" title="Restore default settings"
              onclick="mqmResetSettings()">Restore defaults</button>
          <button id="mqm-apply-settings" title="Apply weight/slice settings"
              onclick="mqmUpdateSettings()">Apply!</button>
        </div>
      </div>
    </details>`;

  let filePanel = `
    <table class="mqm-table mqm-file-menu">
      <tr id="mqm-file-load" class="mqm-file-menu-tr"><td>
        <div>
          <b>Load</b> MQM data file(s):
          <input id="mqm-file"
              accept=".tsv" onchange="mqmOpenFiles()"
              type="file" multiple/>
        </div>
      </td></tr>
      <tr class="mqm-file-menu-option mqm-file-menu-tr"><td>
        <input type="checkbox" id="mqm-load-file-append"/>
        Append additional data without replacing the current data
      </td></tr>
      <tr><td></td></tr>
      <tr class="mqm-file-menu-entry mqm-file-menu-tr"><td>
        <div onclick="mqmSaveData('all', 'mqm-data.tsv')"
            title="Save full 10-column MQM annotations TSV data">
          <b>Save</b> all data to [prefix]mqm-data.tsv
        </div>
      </td></tr>
      <tr class="mqm-file-menu-entry mqm-file-menu-tr"><td>
        <div onclick="mqmSaveData('filtered', 'mqm-data-filtered.tsv')"
            title="Save currently filtered 10-column MQM annotations TSV data">
          <b>Save</b> filtered data to [prefix]mqm-data-filtered.tsv
        </div>
      </td></tr>`;

  for (let saveType of ['system', 'document', 'segment', 'rater']) {
    const fname = `mqm-scores-by-${saveType}.tsv`;
    filePanel += `
        <tr class="mqm-file-menu-entry mqm-file-menu-tr"><td>
          <div
              onclick="mqmSaveData('${saveType}', '${fname}')"
              title="Save ${saveType == 'rater' ?
                'segment-wise ' : ''}filtered scores by ${saveType}">
            <b>Save</b> filtered scores by ${saveType} to [prefix]${fname}
          </div>
        </td></tr>`;
  }
  filePanel += `
      <tr class="mqm-file-menu-option mqm-file-menu-tr"><td>
        Optional prefix for saved files:
        <input size="10" class="mqm-input" type="text"
            id="mqm-saved-file-prefix" placeholder="prefix"/>
      </td></tr>
    </table>`;

  const file = `
    <details class="mqm-file mqm-menu-entry" id="mqm-menu-entry-file">
      <summary>File</summary>
      <div class="mqm-file-panel">
        ${filePanel}
      </div>
    </details>`;

  elt.innerHTML = `
  <div class="mqm-header">
    <span class="mqm-title">MQM Viewer</span>
    <table class="mqm-menu">
      <tr>
        <td>${settings}</td>
        <td>${file}</td>
      </tr>
    </table>
  </div>
  <div id="mqm-errors"></div>
  <hr>

  <table class="mqm-table mqm-numbers-table" id="mqm-stats">
    <thead id=mqm-stats-thead>
    </thead>
    <tbody id="mqm-stats-tbody">
    </tbody>
  </table>

  <br>

  <details>
    <summary
        title="Click to see significance test results.">
      <span class="mqm-section">
        Significance tests
      </span>
    </summary>
    <div class="mqm-sigtests">
      <p>
        P-values < ${MQM_PVALUE_THRESHOLD} (bolded) indicate a significant
        difference.
        <span class="mqm-warning" id="mqm-sigtests-msg"></span>
      </p>
      <div id="mqm-sigtests-tables">
      </div>
      <p>
        Systems above any solid line are significantly better than
        those below. Dotted lines identify clusters within which no
        system is significantly better than any other system.
      </p>
      <p>
        Number of trials for paired one-sided approximate randomization:
        <input size="6" maxlength="6" type="text" id="mqm-sigtests-num-trials"
            value="10000" onchange="setMqmSigtestsNumTrials()"/>
      </p>
    <div>
  </details>

  <br>

  <details>
    <summary
        title="Click to see a System x Rater matrix of scores highlighting individual system-rater scores that seem out of order">
      <span class="mqm-section">
        System &times; Rater scores
      </span>
    </summary>
    <table
        title="Systems and raters are sorted using total MQM score. A highlighted entry means this rater's rating of this system is contrary to the aggregate of all raters' ratings, when compared with the previous system."
        class="mqm-table mqm-numbers-table" id="mqm-system-x-rater">
    </table>
  </details>

  <br>

  <details>
    <summary
        title="Click to see System-wise and comparative segment scores histograms">
      <span class="mqm-section">
        System segment scores and comparative histograms
      </span>
    </summary>
    <div class="mqm-sys-v-sys" id="mqm-sys-v-sys">
    </div>
  </details>

  <br>

  <details>
    <summary title="Click to see error severity / category counts">
      <span class="mqm-section">
        Error severities and categories
      </span>
    </summary>
    <table class="mqm-table" id="mqm-sevcat-stats">
      <thead>
        <tr>
          <th title="Error severity"><b>Severity</b></th>
          <th title="Error category"><b>Category</b></th>
          <th> </th>
          <th id="mqm-sevcat-stats-th"
              title="Number of occurrences"><b>Count</b></th>
        </tr>
      </thead>
      <tbody id="mqm-sevcat-stats-tbody">
      </tbody>
    </table>
  </details>

  <br>

  <details>
    <summary title="Click to see user interface events and timings">
      <span class="mqm-section">
        Annotation events and rater timelines
      </span>
    </summary>
    <div>
      <table class="mqm-table" id="mqm-events">
        <thead>
          <tr>
            <th title="User interface event"><b>Event</b></th>
            <th title="Number of occurrences"><b>Count</b></th>
            <th title="Average time per occurrence"><b>Avg Time
                (millis)</b></th>
          </tr>
        </thead>
        <tbody id="mqm-events-tbody">
        </tbody>
      </table>
    </div>
    <div>
      <div class="mqm-subheading"
          title="The timeline is limited to events in filtered annotations">
        <b>Rater timeline for</b>
        <select onchange="mqmRaterTimelineSelect()"
            id="mqm-rater-timelines-rater"></select>
        (limited to ${MQM_RATER_TIMELINE_LIMIT} events)
      </div>
      <table class="mqm-table" id="mqm-rater-timelines">
        <thead>
          <tr>
            <th><b>Timestamp</b></th>
            <th><b>Event</b></th>
            <th><b>Document</b></th>
            <th><b>System</b></th>
            <th><b>Segment</b></th>
            <th><b>Side</b></th>
            <th><b>Sentence</b></th>
            <th><b>Visible</b></th>
          </tr>
        </thead>
      </table>
    </div>
  </details>

  <br>

  <details>
    <summary title="Click to see advanced filtering options and documentation">
      <span class="mqm-section">
        Filters
        (<span id="mqm-num-unfiltered-rows">0</span> of
         <span id="mqm-num-rows">0</span> rows pass filters)
        <button
            title="Clear all column filters and JavaScript filter expression"
            onclick="mqmClearFiltersAndShow()">Clear all filters</button>
      </span>
      <span>
        <input type="checkbox" checked
          onchange="mqmShow()" id="mqm-only-all-systems-segments"/>
          Select only the segments for which all or no systems have scores
      </span>
    </summary>

    <ul class="mqm-filters">
      <li>
        You can click on any System/Doc/ID/Rater/Category/Severity (or pick
        from the drop-down list under the column name) to set its <b>column
        filter</b> to that specific value.
      </li>
      <li>
        You can provide <b>column filter</b> regular expressions for filtering
        one or more columns, in the input fields provided under the column
        names.
      </li>
      <li>
        You can create sophisticated filters (involving multiple columns, for
        example) using a <b>JavaScript filter expression</b>:
        <br>
        <input class="mqm-input" id="mqm-filter-expr"
        title="Provide a JavaScript boolean filter expression (and press Enter)"
            onchange="mqmShow()" type="text" size="150"/>
        <div id="mqm-filter-expr-error" class="mqm-filter-expr-error"></div>
        <br>
        <ul>
          <li>This allows you to filter using any expression
              involving the columns. It can use the following
              variables: <b>system</b>, <b>doc</b>, <b>globalSegId</b>,
              <b>docSegId</b>, <b>rater</b>, <b>category</b>, <b>severity</b>,
              <b>source</b>, <b>target</b>, <b>metadata</b>.
          </li>
          <li>
            Filter expressions also have access to three aggregated objects
            named <b>aggrDocSegSys</b> (which is simply an alias for
            metadata.segment), <b>aggrDocSeg</b>, and <b>aggrDoc</b>.
            The aggrDocSegSys dict also contains aggrDocSeg (with the key
            "aggrDocSeg"), which in turn similarly contains aggrDoc.
          </li>
          <li>
            The aggregated variable named <b>aggrDocSeg</b> is an object with
            the following properties:
            <b>aggrDocSeg.catsBySystem</b>,
            <b>aggrDocSeg.catsByRater</b>,
            <b>aggrDocSeg.sevsBySystem</b>,
            <b>aggrDocSeg.sevsByRater</b>,
            <b>aggrDocSeg.sevcatsBySystem</b>,
            <b>aggrDocSeg.sevcatsByRater</b>.
            Each of these properties is an object
            keyed by system or rater, with the values being arrays of strings.
            The "sevcats*" values look like "Minor/Fluency/Punctuation" or
            are just the same as severities if categories are empty. This
            segment-level aggregation allows you to select specific segments
            rather than just specific error ratings.
          </li>
          <li>
            System-wise metrics, including MQM, are also available in
            <b>aggrDocSeg.metrics</b>, which is an object keyed by the metric
            name and then by system name.
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
            <input class="mqm-input" id="mqm-view-metadata-row"
                title="The metadata will be logged in the JavaScript console"
                placeholder="row #"
                onchange="mqmLogRowMetadata()" type="text" size="6"/>
            (useful for finding available fields for filter expressions).
          </li>
          <li><b>Example</b>: docSegId > 10 || severity == 'Major'</li>
          <li><b>Example</b>: target.indexOf('thethe') &gt;= 0</li>
          <li><b>Example</b>: metadata.marked_text.length &gt;= 10</li>
          <li><b>Example</b>:
            aggrDocSeg.sevsBySystem['System-42'].includes('Major')</li>
          <li><b>Example</b>:
            JSON.stringify(aggrDocSeg.sevcatsBySystem).includes('Major/Fl')</li>
          <li><b>Example</b>: aggrDocSegSys.metrics['MQM'] &gt; 4 &&
            (aggrDocSegSys.metrics['BLEURT-X'] ?? 1) &lt; 0.1.</li>
          <li>
            You can add segment-level filtering clauses (AND/OR) using this
            <b>helper</b> (which uses convenient shortcut functions
            mqmIncl()/mqmExcl() for checking that a rating exists and
            has/does-not-have a value):
            <div>
              <select onchange="mqmCheckClause()"
                id="mqm-clause-key"></select>
              <select onchange="mqmCheckClause()"
                id="mqm-clause-inclexcl">
                <option value="includes">has error</option>
                <option value="excludes">does not have error</option>
              </select>
              <select onchange="mqmCheckClause()"
                id="mqm-clause-sev"></select>
              <select onchange="mqmCheckClause()"
                id="mqm-clause-cat"></select>
              <button onclick="mqmAddClause('&&')" disabled
                id="mqm-clause-add-and">Add AND clause</button>
              <button onclick="mqmAddClause('||')" disabled
                id="mqm-clause-add-or">Add OR clause</button>
            </div>
          </li>
        </ul>
        <br>
      </li>
      <li title="Limit this to at most a few thousand to avoid OOMs!">
        <b>Limit</b> the number of rows shown to:
        <input size="6" maxlength="6" type="text" id="mqm-limit" value="2000"
            onchange="setMqmLimit()"/>
      </li>
    </ul>
  </details>

  <br>

  <span class="mqm-section">Rated Segments</span>
  <span id="mqm-viewing-constraints" onclick="mqmShow()"></span>
  <table class="mqm-table" id="mqm-table">
    <thead id="mqm-thead">
      <tr id="mqm-head-row">
        <th id="mqm-th-doc" title="Document name">
          Doc
          <br>
          <input class="mqm-input mqm-filter-re" id="mqm-filter-doc"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="10"/>
          <br>
          <select onchange="mqmPick('doc')"
              class="mqm-select" id="mqm-select-doc"></select>
        </th>
        <th id="mqm-th-doc-seg" title="ID of the segment
            within its document">
          DocSeg
          <br>
          <input class="mqm-input mqm-filter-re" id="mqm-filter-doc-seg"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="4"/>
          <br>
          <select onchange="mqmPick('doc-seg')"
              class="mqm-select" id="mqm-select-doc-seg"></select>
        </th>
        <th id="mqm-th-system" title="System name">
          System
          <br>
          <input class="mqm-input mqm-filter-re" id="mqm-filter-system"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="10"/>
          <br>
          <select onchange="mqmPick('system')"
              class="mqm-select" id="mqm-select-system"></select>
        </th>
        <th id="mqm-th-source" title="Source text of segment">
          Source
          <br>
          <input class="mqm-input mqm-filter-re" id="mqm-filter-source"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="10"/>
        </th>
        <th id="mqm-th-target" title="Translated text of segment">
          Target
          <br>
          <input class="mqm-input mqm-filter-re" id="mqm-filter-target"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="10"/>
        </th>
        <th id="mqm-th-rating" title="Annotation, Severity, Category, Rater">
          <table>
            <tr>
              <td>
                Severity
                <br>
                <input class="mqm-input mqm-filter-re" id="mqm-filter-severity"
                    title="Provide a regexp to filter (and press Enter)"
                    onchange="mqmShow()" type="text" placeholder=".*" size="8"/>
                <br>
                <select onchange="mqmPick('severity')"
                    class="mqm-select" id="mqm-select-severity"></select>
              </td>
              <td>
                Category
                <br>
                <input class="mqm-input mqm-filter-re" id="mqm-filter-category"
                    title="Provide a regexp to filter (and press Enter)"
                    onchange="mqmShow()" type="text" placeholder=".*" size="8"/>
                <br>
                <select onchange="mqmPick('category')"
                    class="mqm-select" id="mqm-select-category"></select>
              </td>
              <td>
                Rater
                <br>
                <input class="mqm-input mqm-filter-re" id="mqm-filter-rater"
                    title="Provide a regexp to filter (and press Enter)"
                    onchange="mqmShow()" type="text" placeholder=".*" size="8"/>
                <br>
                <select onchange="mqmPick('rater')"
                    class="mqm-select" id="mqm-select-rater"></select>
              </td>
            </tr>
          </table>
        </th>
      </tr>
    </thead>
    <tbody id="mqm-tbody">
    </tbody>
  </table>
  `;
  elt.className = 'mqm';
  elt.scrollIntoView();

  document.getElementById('mqm-load-file-append').checked = !loadReplaces;

  const menuEntries = document.getElementsByClassName('mqm-menu-entry');
  for (let i = 0; i < menuEntries.length; i++) {
    const entry = menuEntries[i];
    entry.addEventListener('click', (e) => {
      mqmCloseMenuEntries(entry.id);
    });
  }

  mqmSigtestsMsg = document.getElementById('mqm-sigtests-msg');

  mqmTable = document.getElementById('mqm-tbody');
  mqmStatsTable = document.getElementById('mqm-stats-tbody');
  mqmSevCatStatsTable = document.getElementById('mqm-sevcat-stats-tbody');
  mqmEventsTable = document.getElementById('mqm-events-tbody');

  mqmResetSettings();

  if (tsvDataOrCsvURLs) {
    if (tsvDataOrCsvURLs.indexOf('\t') >= 0) {
      mqmSetData(tsvDataOrCsvURLs);
    } else {
      mqmFetchURLs(tsvDataOrCsvURLs.split(','));
    }
  }
}
