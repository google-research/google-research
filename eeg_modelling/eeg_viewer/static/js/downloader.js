// Copyright 2019 The Google Research Authors.
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
 * @fileoverview Allows downloading any data from the store to a JSON file.
 */

goog.module('eeg_modelling.eeg_viewer.Downloader');

const DownloadHelper = goog.require('jslib.DownloadHelper');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const formatter = goog.require('eeg_modelling.eeg_viewer.formatter');

/**
 * @typedef {{
 *   timestamp: string,
 *   store: !Store.PartialStoreData,
 *   fileParams: !Store.PartialStoreData,
 * }}
 */
let DownloadObject;

/**
 * @typedef {{
 *   id: (number|undefined),
 *   score: (number|undefined),
 *   labelText: (string|undefined),
 *   startTime: number,
 *   duration: number,
 *   startHour: string,
 *   endHour: string,
 *   channelList: !Array<string>,
 * }}
 */
let PrettyPattern;

/**
 * @typedef {{
 *   template: ?PrettyPattern,
 *   unseen: !Array<!PrettyPattern>,
 *   seen: !Array<!PrettyPattern>,
 * }}
 */
let PrettySimilarityTrial;


class Downloader {
  constructor() {
    const store = Store.getInstance();
    // This listener callback will download the requested data.
    store.registerListener([Store.Property.DOWNLOAD_DATA], 'Downloader',
        (store) => this.handleDownloadData(store));

    this.formatPattern = this.formatPattern.bind(this);
    this.formatPatterns = this.formatPatterns.bind(this);
    this.formatTrials = this.formatTrials.bind(this);

    /** @private {!Object<!Store.Property, !Function>} */
    this.formatterFunctions_ = {
      [Store.Property.WAVE_EVENTS]: this.formatPatterns,
      [Store.Property.SIMILAR_PATTERNS_UNSEEN]: this.formatPatterns,
      [Store.Property.SIMILAR_PATTERNS_SEEN]: this.formatPatterns,
      [Store.Property.SIMILAR_PATTERN_TEMPLATE]: this.formatPattern,
      [Store.Property.SIMILAR_PATTERN_PAST_TRIALS]: this.formatTrials,
    };
  }

  /**
   * Extracts data from the store from the given properties.
   * @param {!Store.StoreData} store Store data.
   * @param {!Array<!Store.Property>} properties List of properties to extract.
   * @return {!Store.PartialStoreData}
   * @private
   */
  extractFromStore_(store, properties) {
    const /** !Store.PartialStoreData */ extract = {};

    properties.forEach((property) => {
      if (!(property in store)) {
        return;
      }

      const formatterFunction = this.formatterFunctions_[property];
      let propValue = store[property];
      if (formatterFunction) {
        propValue = formatterFunction(store, propValue);
      }

      extract[property] = propValue;
    });

    return extract;
  }

  /**
   * Downloads data as specified in the downloadData property to a JSON file.
   * @param {!Store.StoreData} store Store data.
   */
  handleDownloadData(store) {
    if (!store.downloadData) {
      return;
    }

    const { properties, name, timestamp } = store.downloadData;
    const /** !DownloadObject */ data = {
      timestamp: timestamp || '',
      store: this.extractFromStore_(store, properties),
      fileParams: this.extractFromStore_(store, Store.FileRequestProperties),
    };

    let fileId = store.sstableKey || store.edfPath || store.tfExFilePath || '';
    if (fileId) {
      const splitted = fileId.split('/');
      fileId = splitted[splitted.length - 1];
    }
    const filename = `${fileId}-${name}.json`;

    const formattedData = JSON.stringify(data, null, 2);
    const contentType = 'application/json; encoding=UTF-8';
    DownloadHelper.download(formattedData, filename, contentType);
  }

  /**
   * Formats a list of similarity trials.
   * @param {!Store.StoreData} store Store data.
   * @param {!Array<!Store.SimilarityTrial>} trials Array of trials to format.
   * @return {!Array<!PrettySimilarityTrial>} Formatted trials.
   */
  formatTrials(store, trials) {
    return trials.map((trial) => ({
      template: this.formatPattern(store, trial.template),
      unseen: this.formatPatterns(store, trial.unseen),
      seen: this.formatPatterns(store, trial.seen),
    }));
  }

  /**
   * Formats a list of patterns to be saved to file.
   * @param {!Store.StoreData} store Store data.
   * @param {!Array<!Store.Annotation|!Store.SimilarPattern>} patternList A list
   *     of patterns to format.
   * @return {!Array<!PrettyPattern>}
   */
  formatPatterns(store, patternList) {
    return patternList.map((pattern) => this.formatPattern(store, pattern));
  }

  /**
   * Formats a pattern to be saved to file.
   * @param {!Store.StoreData} store Store data.
   * @param {?Store.Annotation|?Store.SimilarPattern} pattern A pattern to
   *     format.
   * @return {?PrettyPattern}
   */
  formatPattern(store, pattern) {
    if (!pattern) {
      return null;
    }
    const absStartTime = store.absStart + pattern.startTime;
    const absEndTime = absStartTime + pattern.duration;
    return /** @type {!PrettyPattern} */ (Object.assign({}, pattern, {
      startHour: formatter.formatTime(absStartTime, true),
      endHour: formatter.formatTime(absEndTime, true),
    }));
  }
}

goog.addSingletonGetter(Downloader);

exports = Downloader;
