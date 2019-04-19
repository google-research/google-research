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
 * @fileoverview Handles the form to create a new Wave Event.
 */

goog.module('eeg_modelling.eeg_viewer.WaveEventForm');

const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const formatter = goog.require('eeg_modelling.eeg_viewer.formatter');

/** @const {number} default width of the form. */
const defaultFormWidth = 330;

/** @const {number} default height of the form. */
const defaultFormHeight = 391;

class WaveEventForm {

  constructor() {
    /** @private @const {string} */
    this.formId_ = 'wave-event-form';
    /** @private @const {string} */
    this.startTimeId_ = 'wave-event-start-time';
    /** @private @const {string} */
    this.endTimeId_ = 'wave-event-end-time';

    /** @private {?number} */
    this.startTime_ = null;
    /** @private {?number} */
    this.endTime_ = null;

    const store = Store.getInstance();
    // This handler will register the click in the chart and update the
    // wave event being created.
    store.registerListener(
        [
          Store.Property.GRAPH_POINT_CLICK,
        ],
        'WaveEventForm',
        (store) => this.handleGraphPointClick(store));

  }

  /**
   * Returns a cast HTML Input element.
   * @param {string} id The HTML id of the element.
   * @return {!HTMLInputElement} The input element.
   * @private
   */
  getInputElement_(id) {
    return /** @type {!HTMLInputElement} */ (document.getElementById(id));
  }

  /**
   * Sets the wave events form position considering where was the click.
   * Tries to position the form directly left to the click.
   * If not possible, tries below the click.
   * If not possible, move it above the click.
   * @param {!HTMLElement} waveEventForm Container element of the form.
   * @param {number} xPos left position of the click, relative to the viewport.
   * @param {number} yPos top position of the click, relative to the viewport.
   * @private
   */
  setWaveEventFormPosition_(waveEventForm, xPos, yPos) {
    // If the form is hidden the offsetHeight and offsetWidth are 0, so the
    // default values are needed to calculate the position.
    const formWidth = waveEventForm.offsetWidth || defaultFormWidth;
    const formHeight = waveEventForm.offsetHeight || defaultFormHeight;
    let left = xPos - formWidth - 20;
    let top = yPos;
    let movedLeft = false;
    if (left < 0) {
      left = xPos + 10;
      top = yPos + 80;
      movedLeft = true;
    }

    const verticalLimit = window.innerHeight - formHeight - 100;
    if (top > verticalLimit) {
      const verticalMovement = movedLeft ? 200 : 20;
      top = yPos - formHeight - verticalMovement;
    }

    waveEventForm.style.left = `${left}px`;
    waveEventForm.style.top = `${top}px`;
  }

  /**
   * Handles a click in a point value in the graph.
   * @param {!Store.StoreData} store Store data.
   */
  handleGraphPointClick(store) {
    const { timeValue, xPos, yPos } = store.graphPointClick;

    const waveEventForm = /** @type {!HTMLElement} */ (
        document.getElementById(this.formId_));
    const startTimeInput = this.getInputElement_(this.startTimeId_);
    const endTimeInput = this.getInputElement_(this.endTimeId_);

    const prettyTime = formatter.formatTime(store.absStart + timeValue, true);

    const isFirstClick = this.startTime_ == null;
    const isSecondClick = !isFirstClick && this.endTime_ == null;

    if (isFirstClick) {
      startTimeInput.value = prettyTime;
      endTimeInput.value = '';

      this.setWaveEventFormPosition_(waveEventForm, xPos, yPos);
      waveEventForm.classList.remove('hidden');

      this.startTime_ = timeValue;
    } else if (
        isSecondClick && timeValue > /** @type {number} */ (this.startTime_)) {
      endTimeInput.value = prettyTime;

      this.endTime_ = timeValue;
    }
  }

  /**
   * Selects a wave event type in the form, by setting the dropdown text in the
   * UI.
   * @param {string} type Type selected.
   */
  selectType(type) {
    const dropdown = document.getElementById('wave-event-type-dropdown-text');
    dropdown.textContent = type;
  }

  /**
   * Closes the wave event form and clears the clicks previously made.
   */
  close() {
    const waveEventForm = document.getElementById(this.formId_);
    const startTimeInput = this.getInputElement_(this.startTimeId_);
    const endTimeInput = this.getInputElement_(this.endTimeId_);

    startTimeInput.value = '';
    endTimeInput.value = '';

    this.startTime_ = null;
    this.endTime_ = null;

    waveEventForm.classList.add('hidden');
  }

  /**
   * Saves the wave event determined by the clicks made before.
   */
  save() {
    if (this.startTime_ == null) {
      return;
    }
    const startTime = this.startTime_;
    const endTime = this.endTime_ == null ? startTime : this.endTime_;

    if (endTime < startTime) {
      return;
    }

    const labelText =
        document.getElementById('wave-event-type-dropdown-text').innerHTML;

    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.ADD_WAVE_EVENT,
      data: {
        labelText,
        startTime,
        duration: endTime - startTime,
      },
    });
    this.close();
  }
}

goog.addSingletonGetter(WaveEventForm);

exports = WaveEventForm;

