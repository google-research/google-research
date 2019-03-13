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

goog.module('eeg_modelling.eeg_viewer.WindowLocation');

const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');

/** @const {!Array<!Store.Property>} */
const LIST_PARAMS = [Store.Property.CHANNEL_IDS];
/** @const {!Array<!Store.Property>} */
const NUM_PARAMS = [Store.Property.CHUNK_START, Store.Property.CHUNK_DURATION,
      Store.Property.LOW_CUT, Store.Property.HIGH_CUT, Store.Property.NOTCH];
/** @const {!Array<!Store.Property>} */
const FILE_PARAMS = [Store.Property.TFEX_SSTABLE_PATH,
      Store.Property.PREDICTION_SSTABLE_PATH, Store.Property.SSTABLE_KEY,
      Store.Property.EDF_PATH, Store.Property.TFEX_FILE_PATH,
      Store.Property.PREDICTION_FILE_PATH];
/** @const {!Array<!Store.Property>} */
const ALL_PARAMS = [...NUM_PARAMS, ...FILE_PARAMS, ...LIST_PARAMS];

class WindowLocation {

  constructor() {
    const store = Store.getInstance();
    // This listener callback will update the request parameters in the URL
    // hash and replace the current web history state.
    store.registerListener(NUM_PARAMS.concat(LIST_PARAMS),
        'WindowLocation', (store) => this.handleRequestParams(false, store));
    // This listener callback will update the request parameters in the URL hash
    // and add a new element to web history state.
    store.registerListener(FILE_PARAMS,
        'WindowLocation', (store) => this.handleRequestParams(true, store));
  }

  /**
   * Parses the URL fragment into a QueryData object.
   * @return {!Dispatcher.FragmentData} Dictionary of query data fields to
   * update store with.
   */
  parseFragment() {
    const fragmentData = {};
    const hash = location.hash.substring(1);
    if (hash) {
      const assignments = hash.split('&');
      assignments.forEach((assignment) => {
        const elements = assignment.split('=');
        if (elements.length < 2) {
          Dispatcher.getInstance().sendAction({
            actionType: Dispatcher.ActionType.WINDOW_LOCATION_ERROR,
            data: {
              errorMessage: 'Bad Fragment',
            },
          });
        }
        fragmentData[elements[0]] = decodeURIComponent(
            elements.slice(1).join('='));
      });
    }

    const actionData = {};
    ALL_PARAMS.forEach((storeParam) => {
      const param = storeParam.toLowerCase();
      const valid = (param in fragmentData &&
          (fragmentData[param] == 0 || fragmentData[param]));
      actionData[param] = valid ? fragmentData[param] : null;
    });
    return /** @type {!Dispatcher.FragmentData} */ (actionData);
  }

  /**
   * Creates a URL event that makes a data request.
   */
  makeDataRequest() {
    const fragmentData = this.parseFragment();
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.WINDOW_LOCATION_PENDING_REQUEST,
      data: fragmentData,
    });
  }

  /**
   * Converts store chunk data into a fragment string and sets the fragment.
   * Also handles browser history state based on whether the file parameters in
   * the request have changed.
   * @param {boolean} fileInputDirty Whether or not the update was caused by a
   * change in file parameters.
   * @param {!Store.StoreData} store Store chunk data to use to update fragment.
   */
  handleRequestParams(fileInputDirty, store) {
    if (!store.chunkGraphData) {
      return;
    }
    const assignments = ALL_PARAMS.map((param) => {
      const key = param.toLowerCase();
      let value = store[param];
      if (FILE_PARAMS.includes(param)) {
        value = value ? value : '';
      } else if (LIST_PARAMS.includes(param)) {
        value = value ? value.join(',') : '';
      }
      return `${key}=${value}`;
    });
    const url = '#' + assignments.join('&');
    if (fileInputDirty) {
      history.pushState({}, '', url);
    } else {
      history.replaceState({}, '', url);
    }
  }
}

goog.addSingletonGetter(WindowLocation);

exports = WindowLocation;
