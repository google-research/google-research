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
 * This file contains the background Worker thread code that computes
 * significance tests for metric score rankings of system pairs.
 *
 * The significance testing is done through paired one-sided approximate
 * randomization (PAR). The implemention follows sacrebleu at
 * https://github.com/mjpost/sacrebleu/blob/078c440168c6adc89ba75fe6d63f0d922d42bcfe/sacrebleu/significance.py#L112.
 *
 * This file should be loaded only after loading marot.js (as it sets a field
 * in the global "marot" object).
 */

/**
 * An object that encapsulates the significance test computations done in a
 * Worker thread.
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
   * Performs one trial of paired approximate randomization (PAR) for a given
   * baseline and a system and returns the score difference. Returns null if no
   * common segments are found.
   * @param {!MarotSigtestsData} data
   * @param {string} baseline
   * @param {string} system
   * @return {number}
   */
  runPAROneTrial(data, baseline, system) {
    const baselineScores = data.segScoresBySystem[baseline];
    const systemScores = data.segScoresBySystem[system];
    const commonPos = data.commonPosBySystemPair[baseline][system];

    if (!commonPos) {
      return null;
    }

    /**
     * This random array indicates which shuffled system a given score should be
     * assigned to.
     */
    const permutations = this.getRandomInt(2, commonPos.length);
    let shufA = 0.0;
    let shufB = 0.0;
    for (let [idx, perm] of permutations.entries()) {
      const pos = commonPos[idx];
      if (perm == 0) {
        shufA += baselineScores[pos];
        shufB += systemScores[pos];
      } else {
        shufA += systemScores[pos];
        shufB += baselineScores[pos];
      }
    }
    shufA /= commonPos.length;
    shufB /= commonPos.length;
    return (shufA - shufB) * (data.lowerBetter ? 1.0 : -1.0);
  }

  /**
   * Implements the core logic to perform paired one-sided approximate
   * randomization by incrementally conducting trials.
   * @param {!MarotSigtestsData} sigtestsData is the MarotSigtestsData object
   *     that contains various pieces of data needed for significance testing.
   */
  runPAR(sigtestsData) {
    const finishedUpdate = {
      finished: true,
    };
    for (let metric in sigtestsData.metricData) {
      const data = sigtestsData.metricData[metric];
      const systems = data.systems;
      const metricDoneUpdate = {
        metric: metric,
        metricDone: true,
      };
      /** We should have at least 2 systems and 1 trial for signif. testing. */
      if (systems.length < 2 || sigtestsData.numTrials < 1) {
        postMessage(metricDoneUpdate);
        continue;
      }
      const scoresBySystem = data.scoresBySystem;
      const commonPos = data.commonPosBySystemPair;
      const signMultiplier = data.lowerBetter ? 1.0 : -1.0;

      /** Score differences by system pair. */
      const parDiffs = {};

      const log2NumTrials = Math.log2(sigtestsData.numTrials);

      for (const [rowIdx, baseline] of systems.entries()) {
        if (!parDiffs.hasOwnProperty(baseline)) {
          parDiffs[baseline] = {};
        }
        for (const [colIdx, system] of systems.entries()) {
          if (rowIdx >= colIdx) {
            /** We only fill in the upper triangle. */
            continue;
          }
          const numCommonSegs = commonPos[baseline][system].length;
          if (log2NumTrials > numCommonSegs) {
            /** Not enough permutations possible, do not compute. */
            continue;
          }
          if (!parDiffs[baseline].hasOwnProperty(system)) {
            parDiffs[baseline][system] = [];
          }
          for (let i = 0; i < sigtestsData.numTrials; i++) {
            const diff = this.runPAROneTrial(data, baseline, system);
            /** This means no common segments are found. */
            if (diff == null) break;
            parDiffs[baseline][system].push(diff);
          }

          const realDiff = (signMultiplier *
              (scoresBySystem[system].score - scoresBySystem[baseline].score));
          /**
           * Real score differences should be non-negative since we are filling in
           * the upper triangle.
           */
          console.assert(realDiff >= 0.0, realDiff);
          let cnt = 0;
          for (const diff of parDiffs[baseline][system]) {
            /**
             * Count how many samples of the null distribution are greater than or
             * equal to the real difference. This corresponds to
             * 'alternative="greater"' in scipy's API at
             * https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html.
             * Recall that a greater value than `realDiff` indicates a bigger
             * difference between `system` and `baseline`.
             */
            if (diff >= realDiff) {
              cnt += 1;
            }
          }
          const numTrials = parDiffs[baseline][system].length;
          const p = (cnt + 1) / (numTrials + 1);
          const update = {
            metric: metric,
            row: rowIdx,
            col: colIdx,
            pValue: p,
            numCommonSegs: numCommonSegs,
          };
          /** Send this p-value to the parent thread. */
          postMessage(update);
        }
      }
      postMessage(metricDoneUpdate);
    }
    postMessage(finishedUpdate);
  }

  /**
   * Handles the message (to carry out sigtest computations) from the parent.
   * @param {!Event} e is the message event received from the parent thread.
   *     The e.data field is the MarotSigtestsData object that contains various
   *     pieces of data needed.
   */
  messageHandler(e) {
    this.runPAR(e.data);
  }
}

/**
 * The Worker thread code includes the full JS fo the MarotSigtests class,
 * and then it creates a marotSigtests object with a handler for message
 * events. Upon receiving the message with MarotSigtestsData from the parent
 * thread, significance test computations are kicked off.
 */
marot.sigtestsWorkerJS = MarotSigtests.toString() + `
  marotSigtests = new MarotSigtests();
  onmessage = marotSigtests.messageHandler.bind(marotSigtests);
  `;