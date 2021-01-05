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

goog.module('eeg_modelling.eeg_viewer.WindowLocation');

const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');

class WindowLocation {

  constructor() {
    const store = Store.getInstance();
    // This listener callback will update the request parameters in the URL
    // hash and update the web history state accordingly.
    store.registerListener(
        Store.RequestProperties, 'WindowLocation',
        (store, changedProperties) =>
            this.handleRequestParams(store, changedProperties));
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
            actionType: Dispatcher.ActionType.ERROR,
            data: {
              message: 'Bad Fragment',
            },
          });
        }
        fragmentData[elements[0]] = decodeURIComponent(
            elements.slice(1).join('='));
      });
    }

    const actionData = {};
    Store.RequestProperties.forEach((storeParam) => {
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
   * @param {!Store.StoreData} store Store chunk data to use to update fragment.
   * @param {!Array<!Store.Property>} changedProperties List of the properties
   *     that changed because of the last action.
   */
  handleRequestParams(store, changedProperties) {
    if (!store.chunkGraphData) {
      return;
    }

    const fileParamDirty = Store.FileRequestProperties.some(
        (param) => changedProperties.includes(param));
    const assignments = Store.RequestProperties.map((param) => {
      const key = param.toLowerCase();
      let value = store[param];
      if (Store.FileRequestProperties.includes(param)) {
        value = value ? value : '';
      } else if (Store.ListRequestProperties.includes(param)) {
        value = value ? value.join(',') : '';
      }
      return `${key}=${value}`;
    });
    const url = '#' + assignments.join('&');
    if (fileParamDirty) {
      history.pushState({}, '', url);
    } else {
      history.replaceState({}, '', url);
    }
  }
}

goog.addSingletonGetter(WindowLocation);

exports = WindowLocation;
