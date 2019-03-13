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

goog.module('eeg_modelling.eeg_viewer.Error');

const Store = goog.require('eeg_modelling.eeg_viewer.Store');


/**
 * @typedef {{
 *   MaterialSnackbar: {
 *     showSnackbar: function({message: string, timeout: number}),
 *   },
 * }}
 */
let MaterialSnackbarElement;


class Error {

  constructor() {
    const store = Store.getInstance();
    // This listener callback will display a toast error message.
    store.registerListener([Store.Property.ERROR_MESSAGE],
        'Error', (store) => this.handleError(store));
    // This listener callback will display a toast warning message.
    store.registerListener([Store.Property.WARNING_MESSAGE],
        'Error', (store) => this.handleWarning(store));
  }

  handleError(store) {
    this.display(store.errorMessage);
  }

  handleWarning(store) {
    this.display(store.warningMessage);
  }

  /**
   * Displays an error message in a snackbar
   */
  display(data) {
    const text = data ? data.message : 'Unknown Error';
    const notificationSnackbar = /** @type {!MaterialSnackbarElement} */ (document.querySelector('#notification-snackbar'));
    notificationSnackbar.MaterialSnackbar.showSnackbar({
      message: text,
      timeout: 5000,
    });
  }

}

goog.addSingletonGetter(Error);

exports = Error;
