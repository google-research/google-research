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
 * This file contains the background Worker thread code that computes
 * significance tests for metric score rankings of system/rater pairs.
 *
 * The significance testing is done through paired one-sided approximate
 * randomization (PAR). The implemention follows sacrebleu at
 * https://github.com/mjpost/sacrebleu/blob/078c440168c6adc89ba75fe6d63f0d922d42bcfe/sacrebleu/significance.py#L112.
 *
 * This file should be loaded only after loading marot.js (as it sets a field
 * in the global "marot" object).
 */

/**
 * An object that encapsulates the significance test and confidence interval
 * computations done in a Worker thread.
 */
class MarotSigtests {
  /**
   * Samples from [0, max) for a specified number of times.
   * @param {number} max
   * @param {number} size
   * @return {!Array}
   */
  getRandomInt(max, size) {
    let samples = [];
    for (let i = 0; i < size; i++) {
      samples.push(Math.floor(Math.random() * max));
    }
    return samples;
  }

  /**
   * For the non-null values in the samples array, use bootstrap resampling to
   * find the 95% confidence interval for the mean.
   * @param {!Array<number|null>} samples
   * @param {number} numTrials
   * @return {!Array<number>}
   */
  getConfidenceInterval(samples, numTrials) {
    const nonNullSamples = [];
    for (const sample of samples) {
      if (sample === null) continue;
      nonNullSamples.push(sample);
    }
    const n = nonNullSamples.length;
    if (n < 10) {
      /** Too few samples */
      return [NaN, NaN];
    }
    const bootstrapSums = [];
    for (let i = 0; i < numTrials; i++) {
      const bootstrapIndices = this.getRandomInt(n, n);
      let sum = 0.0;
      for (const index of bootstrapIndices) {
        sum += nonNullSamples[index];
      }
      bootstrapSums.push(sum);
    }
    bootstrapSums.sort((a, b) => a - b);
    const maxLoc = numTrials - 1;
    const loc02_5 = Math.min(maxLoc, Math.round(numTrials * 0.025));
    const loc97_5 = Math.min(maxLoc, Math.round(numTrials * 0.975));
    return [bootstrapSums[loc02_5] / n, bootstrapSums[loc97_5] / n];
  }

  /**
   * Performs one trial of paired approximate randomization (PAR) for a given
   * baseline and a candidate item and returns the score difference. Returns
   * null if no common scoring units are found.
   * @param {!MarotSigtestsMetricData} data
   * @param {string} baseline
   * @param {string} item
   * @return {number}
   */
  runPAROneTrial(data, baseline, item) {
    const baselineScores = data.unitScores[baseline];
    const itemScores = data.unitScores[item];
    const commonPos = data.commonPosByItemPair[baseline][item];

    if (!commonPos) {
      return null;
    }

    /**
     * This random array indicates which shuffled item a given score should be
     * assigned to.
     */
    const permutations = this.getRandomInt(2, commonPos.length);
    let shufA = 0.0;
    let shufB = 0.0;
    for (let [idx, perm] of permutations.entries()) {
      const pos = commonPos[idx];
      if (perm == 0) {
        shufA += baselineScores[pos];
        shufB += itemScores[pos];
      } else {
        shufA += itemScores[pos];
        shufB += baselineScores[pos];
      }
    }
    shufA /= commonPos.length;
    shufB /= commonPos.length;
    return (shufA - shufB) * (data.lowerBetter ? 1.0 : -1.0);
  }

  /**
   * For each system/rater, computes the confidence interval for its mean
   * score over all scoring units, for all metrics, using bootstrap resampling.
   *
   * For each system/rater-pair, computes the significance test p-value
   * of the score difference being significant for each metric, using paired
   * one-sided approximate randomization.
   *
   * @param {!Object} sigtestsData is the object that contains various pieces
   *     of data needed.
   */
  runTests(sigtestsData) {
    for (const sysOrRater of ['sys', 'rater']) {
      if (!sigtestsData.data.hasOwnProperty(sysOrRater)) {
        continue;
      }
      const metricsData = sigtestsData.data[sysOrRater];
      for (const metric in metricsData) {
        const data = metricsData[metric];
        const comparables = data.comparables;
        const metricDoneUpdate = {
          sysOrRater: sysOrRater,
          metric: metric,
          metricDone: true,
        };
        /** First compute confidence intervals. */
        for (const [rowIdx, comparable] of comparables.entries()) {
          const update = {
            sysOrRater: sysOrRater,
            metric: metric,
            row: rowIdx,
            ci: this.getConfidenceInterval(
                data.unitScores[comparable], sigtestsData.numTrials),
          };
          postMessage(update);
        }
        /**
         * We should have at least 2 comparables and 1 trial for significance
         * testing.
         */
        if (comparables.length < 2 || sigtestsData.numTrials < 1) {
          postMessage(metricDoneUpdate);
          continue;
        }
        const scores = data.scores;
        const commonPos = data.commonPosByItemPair;
        const signMultiplier = data.lowerBetter ? 1.0 : -1.0;

        /** Score differences by item-pair. */
        const parDiffs = {};

        const log2NumTrials = Math.log2(sigtestsData.numTrials);

        for (const [rowIdx, baseline] of comparables.entries()) {
          if (!parDiffs.hasOwnProperty(baseline)) {
            parDiffs[baseline] = {};
          }
          const columns = (sysOrRater == 'rater' ?
                           ['not:' + baseline] : []).concat(comparables);
          for (const [itemIdx, item] of columns.entries()) {
            const colIdx = itemIdx - (sysOrRater == 'rater' ? 1 : 0);
            if (rowIdx >= colIdx && colIdx != -1) {
              /** We only fill in the upper triangle and all-other-raters. */
              continue;
            }
            const numCommonUnits = commonPos[baseline][item].length;
            const update = {
              sysOrRater: sysOrRater,
              metric: metric,
              row: rowIdx,
              col: colIdx,
              pValue: NaN,
              numCommonUnits: numCommonUnits,
            };

            const realDiff = (signMultiplier *
                (scores[item].score - scores[baseline].score));
            /**
             * Real score differences should be non-negative since we are
             * filling in the upper triangle. When comparing a rater against
             * their complement, this may not hold. We skip the significance
             * test in that case.
             *
             * We also skip the test if there aren't enough permutations
             * possible.
             */
            if (log2NumTrials > numCommonUnits || realDiff < 0) {
              postMessage(update);
              continue;
            }

            if (!parDiffs[baseline].hasOwnProperty(item)) {
              parDiffs[baseline][item] = [];
            }
            for (let i = 0; i < sigtestsData.numTrials; i++) {
              const diff = this.runPAROneTrial(data, baseline, item);
              /** This means no common scoring units are found. */
              if (diff == null) break;
              parDiffs[baseline][item].push(diff);
            }
            let cnt = 0;
            for (const diff of parDiffs[baseline][item]) {
              /**
               * Count how many samples of the null distribution are greater
               * than or equal to the real difference. This corresponds to
               * 'alternative="greater"' in scipy's API at
               * https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html.
               * Recall that a greater value than `realDiff` indicates a bigger
               * difference between `item` and `baseline`.
               */
              if (diff >= realDiff) {
                cnt += 1;
              }
            }
            const numTrials = parDiffs[baseline][item].length;
            update.pValue = (cnt + 1) / (numTrials + 1);
            postMessage(update);
          }
        }
        postMessage(metricDoneUpdate);
      }
      postMessage({
        sysOrRater: sysOrRater,
        sysOrRaterDone: true,
      });
    }
    postMessage({
      finished: true,
    });
  }

  /**
   * Handles the message (to carry out sigtest and confidence interval
   * computations) from the parent.
   * @param {!Event} e is the message event received from the parent thread.
   *     The e.data field is the object that contains various pieces of data
   *     needed.
   */
  messageHandler(e) {
    this.runTests(e.data);
  }
}

/**
 * The Worker thread code includes the full JS for the MarotSigtests class,
 * and then it creates a marotSigtests object with a handler for message
 * events. Upon receiving the message with MarotSigtestsMetricData from the
 * parent thread, significance test computations are kicked off.
 */
marot.sigtestsWorkerJS = MarotSigtests.toString() + `
  marotSigtests = new MarotSigtests();
  onmessage = marotSigtests.messageHandler.bind(marotSigtests);
  `;