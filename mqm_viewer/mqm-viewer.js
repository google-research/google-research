// Copyright 2021 The Google Research Authors.
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
 * Raw data read from the data file. Each entry is an array with 9 entries,
 * in this order (slightly different from the original order in the TSV data):
 *     0: system, 1: doc, 2: doc_id, 3: seg_id, 4: source, 5: target,
 *     6: rater, 7: category, 8: severity.
 * Optionally, some entries can have a value at index 9: note.
 */
let mqmData = [];

/**
 * If TSV data was supplied (instead of being chosen from a file), then it is
 * saved here (for possible downloading).
 */
let mqmTSVData = '';

/** Stats computed for current filtered data. **/
let mqmStats = {};
let mqmStatsBySystem = {};
let mqmStatsByRater = {};
let mqmStatsByCategory = {};

/** Max number of segments to show. **/
let mqmLimit = 200;

/** Clause built by helper menus, for appending to the filter expression **/
let mqmClause = '';

/** UI elements for clause-builder.  **/
let mqmClauseKey;
let mqmClauseInclExcl;
let mqmClauseSev;
let mqmClauseCat;
let mqmClauseAddAnd;
let mqmClauseAddOr;

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
 * This sorts mqmData by fields in the order shown in the comments below.
 */
function mqmSort() {
  mqmData.sort((e1, e2) => {
    let diff = e1[3] - e2[3];  /** seg_id **/
    if (diff == 0) {
      diff = e1[2] - e2[2];  /** doc_id **/
      if (diff == 0) {
        if (e1[1] < e2[1]) {  /** doc **/
          diff = -1;
        } else if (e1[1] > e2[1]) {
          diff = 1;
        } else if (e1[0] < e2[0]) {  /** system **/
          diff = -1;
        } else if (e1[0] > e2[0]) {
          diff = 1;
        } else if (e1[6] < e2[6]) {  /** rater **/
          diff = -1;
        } else if (e1[6] > e2[6]) {
          diff = 1;
        } else if (e1[8] < e2[8]) {  /** severity **/
          diff = -1;
        } else if (e1[8] > e2[8]) {
          diff = 1;
        } else if (e1[7] < e2[7]) {  /** category **/
          diff = -1;
        } else if (e1[7] > e2[7]) {
          diff = 1;
        }
      }
    }
    return diff;
  });
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
 * Aggregates mqmData, collecting all data for a particular segment translation
 * (i.e., for a given (doc, doc_id, seg_id) triple) into a "segment" object
 * that has the following properties: {cats,sevs,sevcats}_by_{rater,system}.
 * Each of these properties is an object keyed by system or rater, with the
 * values being arrays of strings that are categories, severities,
 * and <sev>[/<cat>], * respectively.
 *
 * Appends each aggregated segment object as the last column (index 9, or 10 if
 * there is a note) to each mqmData[*] array for that segment.
 */
function mqmAddSegmentAggregations() {
  let segment = null;
  let curr_doc = '';
  let curr_doc_id = -1;
  let curr_seg_id = -1;
  let curr_start = -1;
  for (let i = 0; i < mqmData.length; i++) {
    const parts = mqmData[i];
    const system = parts[0];
    const doc = parts[1];
    const doc_id = parts[2];
    const seg_id = parts[3];
    const rater = parts[6];
    const category = parts[7];
    const severity = parts[8];
    if (curr_doc == doc && curr_doc_id == doc_id && curr_seg_id == seg_id) {
      console.assert(segment != null, i);
    } else {
      if (segment != null) {
        console.assert(curr_start >= 0, segment);
        for (let j = curr_start; j < i; j++) {
          mqmData[j].push(segment);
        }
      }
      segment = {
        'cats_by_system': {},
        'cats_by_rater': {},
        'sevs_by_system': {},
        'sevs_by_rater': {},
        'sevcats_by_system': {},
        'sevcats_by_rater': {},
      };
      curr_doc = doc;
      curr_doc_id = doc_id;
      curr_seg_id = seg_id;
      curr_start = i;
    }
    mqmAddToArray(segment.cats_by_system, system, category);
    mqmAddToArray(segment.cats_by_rater, rater, category);
    mqmAddToArray(segment.sevs_by_system, system, severity);
    mqmAddToArray(segment.sevs_by_rater, rater, severity);
    const sevcat = severity + (category ? '/' + category : '');
    mqmAddToArray(segment.sevcats_by_system, system, sevcat);
    mqmAddToArray(segment.sevcats_by_rater, rater, sevcat);
  }
  if (segment != null) {
    console.assert(curr_start >= 0, segment);
    for (let j = curr_start; j < mqmData.length; j++) {
      mqmData[j].push(segment);
    }
  }
}

/**
 * Returns an array of column filter REs.
 * @return {!Array<!RegExp>}
 */
function mqmGetFilterREs() {
  const res = [];
  const filters = document.getElementsByClassName('mqm-filter-re');
  for (let i = 0; i < filters.length; i++) {
    const filter = filters[i].value.trim();
    const selectId = filters[i].id.replace(/filter/, 'select');
    const sel = document.getElementById(selectId);
    if (sel) sel.value = filter;
    if (!filter) {
      res.push(null);
      continue;
    }
    const re = new RegExp(filter);
    res.push(re);
  }
  return res;
}

/**
 * Retains only the marked part in a segment, replacing the parts before/after
 * (if they exist) with ellipsis. Used to show just the marked parts for
 * source/target text segments when the full text has already been shown
 * previously.
 * @param {string} s
 * @return {string}
 */
function mqmOnlyKeepSpans(s) {
  const start = s.indexOf('<span');
  const end = s.lastIndexOf('</span>');
  if (start >= 0 && end >= 0) {
    let sub = s.substring(start, end + 7);
    const MAX_CTX = 10;
    if (start > 0) {
      const ctx = Math.min(MAX_CTX, start);
      sub = s.substr(start - ctx, ctx) + sub;
      if (ctx < start) sub = '&hellip;' + sub;
    }
    if (end + 7 < s.length) {
      const ctx = Math.min(MAX_CTX, s.length - (end + 7));
      sub = sub + s.substr(end + 7, ctx);
      if (end + 7 + ctx < s.length) sub = sub + '&hellip;';
    }
    return sub;
  } else {
    return '&hellip;';
  }
}

/**
 * Short-cut function (convenient in filter expressions) for:
 *   "obj has array property prop that includes val"
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
 *   "obj has array property prop that excludes val"
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
 * buttons if so.
 */
function mqmCheckClause() {
  mqmClause = '';
  mqmClauseAddAnd.disabled = true;
  mqmClauseAddOr.disabled = true;
  if (!mqmClauseKey.value) return;
  if (!mqmClauseSev.value && !mqmClauseCat.value) return;

  let sevcats = 'segment.sevcats';
  let key = '';
  let err = mqmClauseSev.value + '/' + mqmClauseCat.value;
  if (!mqmClauseSev.value) {
    sevcats = 'segment.cats';
    err = mqmClauseCat.value;
  }
  if (!mqmClauseCat.value) {
    sevcats = 'segment.sevs';
    err = mqmClauseSev.value;
  }
  if (mqmClauseKey.value.startsWith('System: ')) {
    sevcats += '_by_system';
    key = mqmClauseKey.value.substr(8);
  } else {
    console.assert(mqmClauseKey.value.startsWith('Rater: '),
                   mqmClauseKey.value);
    sevcats += '_by_rater';
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
 * Evaluates the JavaScript filter_expr on an mqmData[] row and returns true
 * only if the filter passes.
 * @param {string} filter_expr
 * @param {!Array<string>} parts
 * @return {boolean}
 */
function mqmFilterExprPasses(filter_expr, parts) {
  if (!filter_expr.trim()) return true;
  try {
  return Function('"use strict";' + `
    const system = arguments[0];
    const doc = arguments[1];
    const doc_id = arguments[2];
    const seg_id = arguments[3];
    const source = arguments[4];
    const target = arguments[5];
    const rater = arguments[6];
    const category = arguments[7];
    const severity = arguments[8];
    const segment = arguments[9];` +
    'return (' + filter_expr + ')')(parts[0], parts[2], parts[2], parts[3],
                                    parts[4], parts[5], parts[6], parts[7],
                                    parts[8], parts[parts.length - 1]);
  } catch (err) {
    document.getElementById('mqm-filter-expr-error').innerHTML = err;
    return false;
  }
}

/**
 * Initializes and returns a rater object.
 * @param {string} rater
 * @return {!Object}
 */
function mqmInitRaterStats(rater) {
  return {
    'rater': rater,
    'major': 0,
    'majorA': 0,
    'majorNT': 0,
    'majorF': 0,
    'majorUncat': 0,
    'minor': 0,
    'minorA': 0,
    'minorF': 0,
    'minorPunct': 0,
    'minorUncat': 0,
    'score': 0,
    'scoreMajor': 0,
    'scoreMinor': 0,
    'scoreAccuracy': 0,
    'scoreFluency': 0,
    'scoreUncat': 0,
  };
}

/**
 * Appends stats from delta into raterStats.
 * @param {!Object} raterStats
 * @param {!Object} delta
 */
function mqmAddRaterStats(raterStats, delta) {
  raterStats.major += delta.major;
  raterStats.majorA += delta.majorA;
  raterStats.majorNT += delta.majorNT;
  raterStats.majorF += delta.majorF;
  raterStats.majorUncat += delta.majorUncat;
  raterStats.minor += delta.minor;
  raterStats.minorA += delta.minorA;
  raterStats.minorF += delta.minorF;
  raterStats.minorPunct += delta.minorPunct;
  raterStats.minorUncat += delta.minorUncat;
  raterStats.score += delta.score;
  raterStats.scoreMajor += delta.scoreMajor;
  raterStats.scoreMinor += delta.scoreMinor;
  raterStats.scoreAccuracy += delta.scoreAccuracy;
  raterStats.scoreFluency += delta.scoreFluency;
  raterStats.scoreUncat += delta.scoreUncat;
}

/**
 * Divides all metrics in raterStats by num.
 * @param {!Object} raterStats
 * @param {number} num
 */
function mqmAvgRaterStats(raterStats, num) {
  if (!num) return;
  raterStats.score /= num;
  raterStats.scoreMajor /= num;
  raterStats.scoreMinor /= num;
  raterStats.scoreAccuracy /= num;
  raterStats.scoreFluency /= num;
  raterStats.scoreUncat /= num;
}

/**
 * Appends MQM score details from the stats object to the table with the given
 * id.
 * @param {string} id
 * @param {string} title
 * @param {!Object} stats
 */
function mqmShowSegmentStats(id, title, stats) {
  const tbody = document.getElementById(id);
  if (title) {
    tbody.insertAdjacentHTML('beforeend',
      '<tr><td colspan="11"><hr></td></tr>' +
      `<tr><td colspan="11"><b>${title}</b></td></tr>\n`);
  }
  const keys = Object.keys(stats);
  const aggregates = {};
  const ratings = {};
  for (let k of keys) {
    aggregates[k] = mqmInitRaterStats('');
    ratings[k] = 0;
    const segs = stats[k];
    if (!segs || !segs.length) {
      aggregates[k].score = Infinity;
      continue;
    }
    for (let raterStats of segs) {
      const allRaterStats = mqmInitRaterStats('');
      for (let r of raterStats) {
        r.score = (r.major - r.majorNT) * 5.0 + r.majorNT * 25.0;
        r.scoreMajor = (r.major - r.majorNT) * 5.0 + r.majorNT * 25.0;
        r.scoreMinor = (r.minor - r.minorPunct) + r.minorPunct * 0.1;
        r.score = r.scoreMajor + r.scoreMinor;
        r.scoreAccuracy = (r.majorA - r.majorNT) * 5.0 + r.majorNT * 25.0 +
          r.minorA;
        r.scoreFluency = r.majorF * 5.0 + (r.minorF - r.minorPunct) +
          r.minorPunct * 0.1;
        r.scoreUncat = r.score - (r.scoreAccuracy + r.scoreFluency);
        mqmAddRaterStats(allRaterStats, r);
      }
      mqmAvgRaterStats(allRaterStats, raterStats.length);
      ratings[k] += raterStats.length;
      mqmAddRaterStats(aggregates[k], allRaterStats);
    }
    mqmAvgRaterStats(aggregates[k], segs.length);
  }
  keys.sort((k1, k2) => aggregates[k1].score - aggregates[k2].score);
  for (let k of keys) {
    const segs = stats[k];
    let rowHTML = `<tr><td>${k}</td><td>${segs.length}</td>` +
        `<td>${ratings[k]}</td>`;
    if (!segs || !segs.length || !ratings[k]) {
      for (let i = 0; i < 8; i++) {
        rowHTML += '<td>-</td>';
      }
    } else {
      for (let s of ['score', 'scoreMajor', 'scoreMinor',
                     'scoreAccuracy', 'scoreFluency', 'scoreUncat']) {
        rowHTML += `<td>${(aggregates[k][s]).toFixed(3)}</td>`;
      }
      for (let s of ['majorNT', 'minorPunct']) {
        rowHTML += `<td>${aggregates[k][s]}</td>`;
      }
    }
    rowHTML += '</tr>\n';
    tbody.insertAdjacentHTML('beforeend', rowHTML);
  }
}

/**
 * Shows details of category-wise scores (in the stats object) in the table
 * with the given id.
 * @param {string} id
 * @param {!Object} stats
 */
function mqmShowCategoryStats(id, stats) {
  const tbody = document.getElementById(id);
  const keys = Object.keys(stats);
  keys.sort((k1, k2) => stats[k2][0] - stats[k1][0]);
  for (let k of keys) {
    const row = stats[k];
    let rowHTML = `<tr><td>${k}</td>`;
    for (let i = 0; i < 3; i++) {
      rowHTML += `<td>${row[i]}</td>`;
    }
    rowHTML += '</tr>\n';
    tbody.insertAdjacentHTML('beforeend', rowHTML);
  }
}

/**
 * Shows all the stats.
 */
function mqmShowStats() {
  mqmShowSegmentStats('mqm-stats-tbody', '', mqmStats);
  mqmShowSegmentStats('mqm-stats-tbody', 'By system', mqmStatsBySystem);
  mqmShowSegmentStats('mqm-stats-tbody', 'By rater', mqmStatsByRater);
  mqmShowCategoryStats('mqm-cat-stats-tbody', mqmStatsByCategory);
}

/**
 * Updates statsArray[category], which is an array with 3 entries:
 *   total errors, major errors, minor errors.
 * @param {!Object} statsArray
 * @param {string} category
 * @param {string} severity
 */
function mqmAddCategoryStats(statsArray, category, severity) {
  if (!statsArray.hasOwnProperty(category)) {
    statsArray[category] = [0,0,0];
  }
  statsArray[category][0]++;

  const lsev = severity.toLowerCase();
  if (lsev == 'major') {
    statsArray[category][1]++;
  } else if (lsev == 'minor') {
    statsArray[category][2]++;
  }
}

/**
 * Given a lowercase category (lcat), returns true if it is the
 * "Non-translation" category, allowing for underscore/dash variation and
 * a possible trailing exclamation mark.
 * @param {string} lcat
 * @return {boolean}
 */
function mqmIsNT(lcat) {
  return lcat.startsWith('non-translation') ||
    lcat.startsWith('non_translation');
}

/**
 * Given a lowercase category (lcat), returns true if it is an accuracy error.
 * @param {string} lcat
 * @return {boolean}
 */
function mqmIsAccuracy(lcat) {
  return lcat.startsWith('accuracy') || lcat.startsWith('terminology') ||
    mqmIsNT(lcat);
}

/**
 * Updates stats with an error of (category, severity).
 * @param {!Object} stats
 * @param {string} category
 * @param {string} severity
 */
function mqmAddErrorStats(stats, category, severity) {
  const lcat = category.toLowerCase().trim();
  if (lcat == 'no-error' || lcat == 'no_error') return;

  const lsev = severity.toLowerCase();
  if (lsev == 'major') {
    stats.major++;
    if (!lcat) {
      stats.majorUncat++;
    } else if (mqmIsAccuracy(lcat)) {
      stats.majorA++;
      if (mqmIsNT(lcat)) stats.majorNT++;
    } else {
      stats.majorF++;
    }
  } else if (lsev == 'minor') {
    stats.minor++;
    if (!lcat) {
      stats.minorUncat++;
    } else if (mqmIsAccuracy(lcat)) {
      stats.minorA++;
    } else {
      stats.minorF++;
      if (lcat.startsWith('fluency/punctuation')) stats.minorPunct++;
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
 * Updates the display to show the segment data and scores according to the
 * current filters.
 */
function mqmShow() {
  document.body.style.cursor = 'wait';

  const tbody = document.getElementById('mqm-tbody');
  tbody.innerHTML = '';
  document.getElementById('mqm-stats-tbody').innerHTML = '';
  document.getElementById('mqm-cat-stats-tbody').innerHTML = '';

  /**
   * The following mqmStats* objects are all keyed by something from:
   * ({'Total'} or {system} or {rater}). Each keyed object is an array with
   * one entry per segment. Each entry for a segment is itself an array, one
   * entry per rater. Each  entry for a rater is an object with the following
   * properties: rater, major, majorA, majorNT, majorF, minor, minorA, minorF,
   * minorPunct, score.
   */
  mqmStats = {'Total': []};
  mqmStatsBySystem = {};
  mqmStatsByRater = {};

  mqmStatsByCategory = {};

  let shown = 0;
  const filter_expr = document.getElementById('mqm-filter-expr').value.trim();
  document.getElementById('mqm-filter-expr-error').innerHTML = '';
  const filters = document.getElementsByClassName('mqm-filter-re');
  const filterREs = mqmGetFilterREs();
  let lastRow = null;
  let currSegStats = [];
  let currSegStatsBySys = [];
  let currSegStatsByRater = [];
  for (let parts of mqmData) {
    let match = true;
    for (let i = 0; i < 9; i++) {
      if (filterREs[i] && !filterREs[i].test(parts[i])) {
        match = false;
        break;
      }
    }
    if (!match) {
      continue;
    }
    if (!mqmFilterExprPasses(filter_expr, parts)) {
      continue;
    }
    const system = parts[0];
    const rater = parts[6];
    let sameAsLast = lastRow && (system == lastRow[0]) &&
      (parts[1] == lastRow[1]) && (parts[2] == lastRow[2]) &&
      (parts[3] == lastRow[3]);

    if (!sameAsLast) {
      mqmStats['Total'].push([]);
      currSegStats = mqmArrayLast(mqmStats['Total']);
      if (!mqmStatsBySystem.hasOwnProperty(system)) {
        mqmStatsBySystem[system] = [];
      }
      mqmStatsBySystem[system].push([]);
      currSegStatsBySys = mqmArrayLast(mqmStatsBySystem[system]);
    }

    if (!sameAsLast || (rater != lastRow[6])) {
      currSegStats.push(mqmInitRaterStats(rater));
      currSegStatsBySys.push(mqmInitRaterStats(rater));
      /** New rater. **/
      if (!mqmStatsByRater.hasOwnProperty(rater)) {
        mqmStatsByRater[rater] = [];
      }
      mqmStatsByRater[rater].push([]);
      currSegStatsByRater = mqmArrayLast(mqmStatsByRater[rater]);
      currSegStatsByRater.push(mqmInitRaterStats(rater));
    }
    mqmAddErrorStats(mqmArrayLast(currSegStats), parts[7], parts[8]);
    mqmAddErrorStats(mqmArrayLast(currSegStatsBySys), parts[7], parts[8]);
    mqmAddErrorStats(mqmArrayLast(currSegStatsByRater), parts[7], parts[8]);

    mqmAddCategoryStats(mqmStatsByCategory, parts[7], parts[8]);

    lastRow = parts;

    if (shown >= mqmLimit) {
      continue;
    }
    let rowHTML = '';
    for (let i = 0; i < 9; i++) {
      let val = parts[i];
      let cls = 'class="mqm-val"';
      if (i == 4 || i == 5) {
        cls = '';
        if (sameAsLast) {
          val = mqmOnlyKeepSpans(val);
        }
      }
      if (i == 7 && parts.length > 10) {
        /* There is a note */
        val += '<br><i>' + parts[9] + '</i>';
      }
      rowHTML += `<td ${cls} id="mqm-val-${shown}-${i}">` + val + '</td>\n';
    }
    tbody.insertAdjacentHTML('beforeend', `<tr>${rowHTML}</tr>\n`);
    for (let i = 0; i < 9; i++) {
      if (i == 4 || i == 5) continue;
      const v = document.getElementById(`mqm-val-${shown}-${i}`);
      v.addEventListener('click', (e) => {
        filters[i].value = '^' + parts[i] + '$';
        mqmShow();
      });
    }
    shown++;
  }
  mqmShowStats();
  document.body.style.cursor = 'auto';
}

/**
 * Replaces <v>...</v> with a span element of class cls.
 * @param {string} text
 * @param {string} cls
 * @return {string}
 */
function mqmMarkSpan(text, cls) {
  text = text.replace(/<v>/, `<span class="${cls}">`);
  text = text.replace(/<\/v>/, '</span>');
  return text;
}

/**
 * Clears all filters.
 */
function mqmClearFilters() {
  const filters = document.getElementsByClassName('mqm-filter-re');
  for (let i = 0; i < filters.length; i++) {
    filters[i].value = '';
  }
  document.getElementById('mqm-filter-expr').value = '';
  document.getElementById('mqm-filter-expr-error').innerHTML = '';
}

/**
 * Clears all filters and shows stats again.
 */
function mqmClearFiltersAndShow() {
  mqmClearFilters();
  mqmShow();
}

/**
 * For the column named by "what", sets the filter to the currently picked
 * value from its drop-down list.
 * @param {string} what
 */
function mqmPick(what) {
  const filters = document.getElementsByClassName('mqm-filter-re');
  let index = -1;
  const exp = 'mqm-filter-' + what;
  for (let i = 0; i < filters.length; i++) {
    if (filters[i].id == exp) {
      index = i;
      break;
    }
  }
  if (index < 0) return;
  const sel = document.getElementById('mqm-select-' + what);
  if (!sel) return;
  filters[index].value = sel.value;
  mqmShow();
}

/**
 * Populates the column drop-down lists and filter-expression builder with
 * unique values.
 */
function mqmSetSelectOptions() {
  const options = [{}, {}, {}, {}, null, null, {}, {}, {}];
  for (let parts of mqmData) {
    for (let i = 0; i < 9; i++) {
      if (i == 4 || i == 5) continue;
      options[i][parts[i].trim()] = true;
    }
  }
  const selects = [
    document.getElementById('mqm-select-system'),
    document.getElementById('mqm-select-doc'),
    document.getElementById('mqm-select-doc-id'),
    document.getElementById('mqm-select-seg-id'),
    null,
    null,
    document.getElementById('mqm-select-rater'),
    document.getElementById('mqm-select-category'),
    document.getElementById('mqm-select-severity'),
  ];
  for (let i = 0; i < 9; i++) {
    if (i == 4 || i == 5) continue;
    const opt = options[i];
    let html = '<option value=""></option>\n';
    for (let o in opt) {
      if (!o) continue;
      html += `<option value="^${o}$">${o}</option>\n`;
    }
    selects[i].innerHTML = html;
  }

  /**
   * Populate filter clause builder's selects:
   */
  mqmClauseKey = document.getElementById('mqm-clause-key');
  let html = '<option value=""></option>\n';
  for (let sys in options[0]) {
    html += `<option value="System: ${sys}">System: ${sys}</option>\n`;
  }
  for (let rater in options[6]) {
    html += `<option value="Rater: ${rater}">Rater: ${rater}</option>\n`;
  }
  mqmClauseKey.innerHTML = html;

  mqmClauseInclExcl = document.getElementById('mqm-clause-inclexcl');

  mqmClauseCat = document.getElementById('mqm-clause-cat');
  html = '<option value=""></option>\n';
  for (let cat in options[7]) {
    html += `<option value="${cat}">${cat}</option>\n`;
  }
  mqmClauseCat.innerHTML = html;

  mqmClauseSev = document.getElementById('mqm-clause-sev');
  html = '<option value=""></option>\n';
  for (let sev in options[8]) {
    html += `<option value="${sev}">${sev}</option>\n`;
  }
  mqmClauseSev.innerHTML = html;

  mqmClauseAddAnd = document.getElementById('mqm-clause-add-and');
  mqmClauseAddOr = document.getElementById('mqm-clause-add-or');
  mqmClearClause();
}

/**
 * Parses the passed TSV data into mqmData.
 * @param {string} tsvData
 */
function mqmParseData(tsvData) {
  mqmClearFilters();
  const errors = document.getElementById('mqm-errors');
  errors.innerHTML = '';
  mqmData = [];
  const data = tsvData.split('\n');
  let firstLine = true;
  for (let line of data) {
    if (!line.trim()) {
      continue;
    }
    if (firstLine) {
      firstLine = false;
      if (line.toLowerCase().indexOf('system\tdoc') >= 0) {
        /** Skip header line **/
        continue;
      }
    }
    const parts = line.split('\t');
    if (parts.length < 9) {
      errors.insertAdjacentHTML('beforeend', `Could not parse: ${line}`);
      continue;
    }
    /** Move "Rater" to go after source/target. */
    const temp = parts[4];
    parts[4] = parts[5];
    parts[5] = parts[6];
    parts[6] = temp;
    let spanClass = 'mqm-neutral';
    const severity = parts[8].toLowerCase();
    if (severity == 'major') spanClass = 'mqm-major';
    if (severity == 'minor') spanClass = 'mqm-minor';
    parts[4] = mqmMarkSpan(parts[4], spanClass);
    parts[5] = mqmMarkSpan(parts[5], spanClass);
    mqmData.push(parts);
  }
  mqmSort();
  mqmAddSegmentAggregations();
  mqmSetSelectOptions();
  mqmShow();
}

/**
 * Opens the data file picked by the user and loads its data into mqmData[].
 */
function mqmOpenFile() {
  document.body.style.cursor = 'wait';
  mqmClearFilters();
  const errors = document.getElementById('mqm-errors');
  errors.innerHTML = '';
  const f = document.getElementById('mqm-file').files[0];
  let fr = new FileReader();
  fr.onload = function(){
    mqmParseData(fr.result);
  };
  try {
    fr.readAsText(f);
  } catch (err) {
    errors.innerHTML = err;
  }
}

/**
 * Saves mqmTSVData to the file mqm-data.tsv.
 */
function mqmSaveData() {
  const a = document.createElement("a");
  a.style.display = "none";
  document.body.appendChild(a);
  a.href = window.URL.createObjectURL(
    new Blob([mqmTSVData], {type: "text/tab-separated-values;charset=UTF-8"})
  );
  a.setAttribute("download", "mqm-data.tsv");
  a.click();
  window.URL.revokeObjectURL(a.href);
  document.body.removeChild(a);
}

/**
 * Replaces the HTML contents of elt with the HTML needed to render the
 * MQM Viewer. If tsvData is not null, then this data is loaded, and  instead of
 * a file-open button there is a "download TSV data" button.
 * @param {!Element} elt
 * @param {?string=} tsvData
 */
function createMQMViewer(elt, tsvData=null) {
  mqmTSVData = tsvData;
  const header = mqmTSVData ? `
  <div class="mqm-header">
    <span class="mqm-title">MQM Scores</span>
    <span class="mqm-open">
      <button id="mqm-save-file" onclick="mqmSaveData()">
      Save MQM data to file "mqm-data.tsv"
      </button>
    </span>
  </div>` : `
  <div class="mqm-header">
    <span class="mqm-title">MQM Scores</span>
    <span class="mqm-open">
      <b>Open MQM data file (9-column TSV format):</b>
      <input id="mqm-file" accept=".tsv" onchange="mqmOpenFile()"
          type="file"></input>
    </span>
  </div>
  `;

  elt.innerHTML = `
  ${header}
  <div id="mqm-errors"></div>
  <hr>

  <table class="mqm-table" id="mqm-stats">
    <thead>
      <tr>
        <th></th>
        <th title="Number of segments"><b>Segments</b></th>
        <th title="Number of ratings"><b>Ratings</b></th>
        <th title="MQM score"><b>MQM score</b></th>
        <th title="Major component of MQM score"><b>MQM Major</b></th>
        <th title="Minor component of MQM score"><b>MQM Minor</b></th>
        <th title="Accuracy component of MQM score"><b>MQM Accu.</b></th>
        <th title="Fluency component of MQM score"><b>MQM Flue.</b></th>
        <th title="Uncategorized component of MQM score"><b>MQM Uncat.</b></th>
        <th title="Non-translation errors"><b># Non-trans.</b></th>
        <th title="Minor Fluency/Punctuation errors"><b># Minor punct.</b></th>
      </tr>
    </thead>
    <tbody id="mqm-stats-tbody">
    </tbody>
  </table>

  <br>

  <details>
    <summary title="Click to see error category counts">
      <span class="mqm-section">
        Error categories
      </span>
    </summary>
    <table class="mqm-table" id="mqm-cat-stats">
      <thead>
        <tr>
          <th title="Error category"><b>Category</b></th>
          <th title="Total errors count"><b>Count</b></th>
          <th title="Major errors count"><b>Major count</b></th>
          <th title="Minor errors count"><b>Minor count</b></th>
        </tr>
      </thead>
      <tbody id="mqm-cat-stats-tbody">
      </tbody>
    </table>
  </details>

  <br>

  <details>
    <summary title="Click to see advanced filtering options and documentation">
      <span class="mqm-section">
        Filters
        <button title="Clear all column filters and JavaScript filter expression"
            onclick="mqmClearFiltersAndShow()">Clear all filters</button>
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
        one or more columns, in the input fields provided under the column names.
      </li>
      <li>
        You can create sophisticated filters (involving multiple columns, for
        example) using a <b>JavaScript filter expression</b>:
        <br>
        <input class="mqm-filter-expr" id="mqm-filter-expr"
        title="Provide a JavaScript boolean filter expression (and press Enter)"
            onchange="mqmShow()" type="text" size="150">
        </input>
        <div id="mqm-filter-expr-error" class="mqm-filter-expr-error"></div>
        <br>
        <ul>
          <li>This allows you to filter using any expression
              involving the columns. It can use the following
              variables: <b>system</b>, <b>doc</b>, <b>seg_id</b>,
              <b>doc_id</b>, <b>rater</b>, <b>category</b>, <b>severity</b>,
              <b>source</b>, <b>target</b>.
          </li>
          <li>
            Filter expressions also have access to an aggregated <b>segment</b>
            variable that is an object with the following properties:
            <b>segment.cats_by_system</b>,
            <b>segment.cats_by_rater</b>,
            <b>segment.sevs_by_system</b>,
            <b>segment.sevs_by_rater</b>,
            <b>segment.sevcats_by_system</b>,
            <b>segment.sevcats_by_rater</b>.
            Each of these properties is an object
            keyed by system or rater, with the values being arrays of strings.
            The "sevcats_*" values look like "Minor/Fluency/Punctuation" or
            are just the same as severities if categories are empty. This
            segment-level aggregation allows you to select specific segments
            rather than just specific error ratings.
          </li>
          <li><b>Example</b>: seg_id > 10 || severity == 'Major'</li>
          <li><b>Example</b>: target.indexOf('thethe') >= 0</li>
          <li><b>Example</b>:
            segment.sevs_by_system['System-42'].includes('Major')</li>
          <li><b>Example</b>:
            JSON.stringify(segment.sevcats_by_system).includes('Major/Fl')</li>
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
                id="mqm-clause-add-and">Add ADD clause</button>
              <button onclick="mqmAddClause('||')" disabled
                id="mqm-clause-add-or">Add OR clause</button>
            </div>
          </li>
        </ul>
        <br>
      </li>
      <li title="Limit this to at most a few thousand to avoid OOMs!">
        <b>Limit</b> the number of rows shown to:
        <input size="6" maxlength="6" type="text" id="mqm-limit" value="1000"
            onchange="setMqmLimit()">
        </input>
      </li>
    </ul>
  </details>

  <br>

  <span class="mqm-section">Rated Segments</span>
  <table class="mqm-table" id="mqm-table">
    <thead id="mqm-thead">
      <tr>
        <th id="mqm-th-system" title="System name">
          System
          <br>
          <input class="mqm-filter-re" id="mqm-filter-system"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="10">
          </input>
          <br>
          <select onchange="mqmPick('system')"
              class="mqm-select" id="mqm-select-system"></select>
        </th>
        <th id="mqm-th-doc" title="Document name">
          Doc
          <br>
          <input class="mqm-filter-re" id="mqm-filter-doc"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="10">
          </input>
          <br>
          <select onchange="mqmPick('doc')"
              class="mqm-select" id="mqm-select-doc"></select>
        </th>
        <th id="mqm-th-doc-id" title="ID of the segment within its document">
          ID/Doc
          <br>
          <input class="mqm-filter-re" id="mqm-filter-doc-id"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="4">
          </input>
          <br>
          <select onchange="mqmPick('doc-id')"
              class="mqm-select" id="mqm-select-doc-id"></select>
        </th>
        <th id="mqm-th-seg-id" title="ID of the segment across all documents">
          ID/All
          <br>
          <input class="mqm-filter-re" id="mqm-filter-seg-id"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="4">
          </input>
          <br>
          <select onchange="mqmPick('seg-id')"
              class="mqm-select" id="mqm-select-seg-id"></select>
        </th>
        <th id="mqm-th-source" title="Source text of segment">
          Source
          <br>
          <input class="mqm-filter-re" id="mqm-filter-source"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="10">
          </input>
        </th>
        <th id="mqm-th-target" title="Translated text of segment">
          Target
          <br>
          <input class="mqm-filter-re" id="mqm-filter-target"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="10">
          </input>
        </th>
        <th id="mqm-th-rater" title="Rater who evaluated">
          Rater
          <br>
          <input class="mqm-filter-re" id="mqm-filter-rater"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="10">
          </input>
          <br>
          <select onchange="mqmPick('rater')"
              class="mqm-select" id="mqm-select-rater"></select>
        </th>
        <th id="mqm-th-category" title="Error category">
          Category
          <br>
          <input class="mqm-filter-re" id="mqm-filter-category"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="10">
          </input>
          <br>
          <select onchange="mqmPick('category')"
              class="mqm-select" id="mqm-select-category"></select>
        </th>
        <th id="mqm-th-severity" title="Error severity">
          Severity
          <br>
          <input class="mqm-filter-re" id="mqm-filter-severity"
              title="Provide a regexp to filter (and press Enter)"
              onchange="mqmShow()" type="text" placeholder=".*" size="10">
          </input>
          <br>
          <select onchange="mqmPick('severity')"
              class="mqm-select" id="mqm-select-severity"></select>
        </th>
      </tr>
    </thead>
    <tbody id="mqm-tbody">
    </tbody>
  </table>
  `;
  elt.className = 'mqm';
  if (mqmTSVData) {
    mqmParseData(mqmTSVData);
  }
}
