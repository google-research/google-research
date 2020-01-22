// Copyright 2020 The Google Research Authors.
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
 * @fileoverview Allows uploading data from a JSON file to the store.
 */

goog.module('eeg_modelling.eeg_viewer.Uploader');

const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const Downloader = goog.require('eeg_modelling.eeg_viewer.Downloader');
const HtmlSanitizer = goog.require('goog.html.sanitizer.HtmlSanitizer');
const SafeHtml = goog.require('goog.html.SafeHtml');
const utils = goog.require('eeg_modelling.eeg_viewer.utils');


class Uploader {
  constructor() {
    /** @private {string} */
    this.containerId_ = 'uploader-modal';
    /** @private {string} */
    this.fileInputId_ = 'uploader-file-input';
    /** @private {string} */
    this.textDisplayId_ = 'uploader-text-display';
  }

  /**
   * Dispatches an error with a given message.
   * @param {string} message Message to send in the error.
   * @private
   */
  dispatchError_(message) {
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.ERROR,
      data: {
        message,
      },
    });
  }

  /**
   * Retrieves the file selected and sends an action to upload it.
   * @return {!Promise} Promise that resolves once the file was read.
   */
  upload() {
    const input = utils.getInputElement(this.fileInputId_);
    if (input.files.length === 0) {
      this.dispatchError_('No file selected');
      return Promise.resolve();
    }

    const file = input.files[0];

    const reader = new FileReader();
    return new Promise((resolve) => {
      reader.onload = () => {
        // The file text is sanitized before parsing to prevent code injections.
        // Note that the string can't be escaped to prevent injections, since
        // that would escape the quotes and break the JSON.

        const sanitizer = new HtmlSanitizer();

        let fileData;
        const text = /** @type {string} */ (reader.result);
        const safeText = SafeHtml.unwrap(sanitizer.sanitize(text));

        try {
          fileData = JSON.parse(safeText);
        } catch (err) {
          this.dispatchError_('Can\'t parse file as JSON');
          resolve();
          return;
        }

        fileData = /** @type {!Downloader.DownloadObject} */ (fileData);

        if (!fileData.store || !fileData.timestamp || !fileData.fileParams) {
          this.dispatchError_('JSON file has missing attributes');
          resolve();
          return;
        }

        Dispatcher.getInstance().sendAction({
          actionType: Dispatcher.ActionType.IMPORT_STORE,
          data: fileData.store,
        });

        this.closeMenu();

        resolve();
      };
      reader.readAsText(file);
    });
  }

  /**
   * Sets the text to be displayed.
   * @param {?string} value Text to be displayed.
   * @private
   */
  setTextDisplay_(value) {
    const textDisplay = utils.getInputElement(this.textDisplayId_);
    textDisplay.value = value || 'No file chosen';
  }

  /**
   * Handles a change in the hidden file input, which will update the text
   * displayed.
   */
  handleFileChange() {
    const input = utils.getInputElement(this.fileInputId_);
    if (input.files.length === 0) {
      this.setTextDisplay_(null);
      return;
    }
    this.setTextDisplay_(input.files[0].name);
  }

  /**
   * Closes the modal menu.
   */
  closeMenu() {
    const fileInput = utils.getInputElement(this.fileInputId_);
    fileInput.value = null;
    this.setTextDisplay_(null);

    utils.hideElement(this.containerId_);
  }

  /**
   * Opens the modal menu.
   */
  openMenu() {
    utils.showElement(this.containerId_);
  }
}

goog.addSingletonGetter(Uploader);

exports = Uploader;
