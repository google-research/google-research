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
const montages = goog.require('eeg_modelling.eeg_viewer.montages');
const utils = goog.require('eeg_modelling.eeg_viewer.utils');

/** @const {number} default width of the form. */
const defaultFormWidth = 330;

/** @const {number} default height of the form. */
const defaultFormHeight = 487;

/**
 * Possible inputs in the form.
 * This enum is used to define what the form is expecting in the next click.
 * @enum {string}
 */
const InputType = {
  START_TIME: 'startTime',
  END_TIME: 'endTime',
  CHANNEL: 'channel',
};

class WaveEventForm {

  constructor() {
    /** @private @const {string} */
    this.formId_ = 'wave-event-form';
    /** @private @const {string} */
    this.startTimeId_ = 'wave-event-start-time';
    /** @private @const {string} */
    this.endTimeId_ = 'wave-event-end-time';
    /** @private @const {string} */
    this.channelsContainerId_ = 'wave-event-channels';
    /** @private @const {string} */
    this.checkboxesContainerId_ = 'wave-event-channels-checkboxes';

    /** @private {?number} */
    this.startTime_ = null;
    /** @private {?number} */
    this.endTime_ = null;
    /** @private {!InputType} */
    this.waitingFor_ = InputType.START_TIME;

    /** @private {!Set<string>} set of channel names selected */
    this.selectedChannels_ = new Set();
    /** @private {!Array<string>} Array of all the channel names */
    this.allChannels_ = [];

    const store = Store.getInstance();
    // This handler will register the click in the chart and update the
    // wave event being created.
    store.registerListener(
        [
          Store.Property.GRAPH_POINT_CLICK,
        ],
        'WaveEventForm',
        (store) => this.handleGraphPointClick(store));
    // This handler will create the checkboxes in the form.
    store.registerListener(
        [Store.Property.INDEX_CHANNEL_MAP, Store.Property.CHANNEL_IDS],
        'WaveEventForm', (store) => this.handleChannelNames(store));
  }


  /**
   * Sets the duration in the form UI.
   * @private
   */
  setDurationUI_() {
    const startTime = this.startTime_ == null ? 0 : this.startTime_;
    const endTime = this.endTime_ == null ? startTime : this.endTime_;
    const duration = formatter.formatDuration(endTime - startTime);
    this.getInputElement_('wave-event-duration').value = duration;
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
    let movedLeft = true;
    if (left < 0) {
      left = xPos + 10;
      top = yPos + 80;
      movedLeft = false;
    }

    const verticalLimit = window.innerHeight - formHeight - 100;
    if (top > verticalLimit) {
      const verticalMovement = movedLeft ? 20 : 200;
      top = yPos - formHeight - verticalMovement;
      top = Math.max(0, top);
    }

    waveEventForm.style.left = `${left}px`;
    waveEventForm.style.top = `${top}px`;
  }

  /**
   * Handles a click in a point value in the graph.
   * @param {!Store.StoreData} store Store data.
   */
  handleGraphPointClick(store) {
    const { timeValue, channelName, xPos, yPos } = store.graphPointClick;

    const waveEventForm = /** @type {!HTMLElement} */ (
        document.getElementById(this.formId_));
    const startTimeInput = this.getInputElement_(this.startTimeId_);
    const endTimeInput = this.getInputElement_(this.endTimeId_);

    const prettyTime = formatter.formatTime(store.absStart + timeValue, true);

    if (this.waitingFor_ === InputType.END_TIME &&
        this.startTime_ != null && timeValue < this.startTime_) {
      this.waitingFor_ = InputType.START_TIME;
    }

    const channelLabelElement = /** @type {!Element} */ (document.querySelector(
        `label[for="${this.getChannelCheckboxId_(channelName)}"]`));

    const markChannelSelected = () => {
      this.selectedChannels_.add(channelName);
      utils.toggleMDLCheckbox(channelLabelElement, true);
    };

    const markChannelUnselected = () => {
      this.selectedChannels_.delete(channelName);
      utils.toggleMDLCheckbox(channelLabelElement, false);
    };

    if (this.waitingFor_ === InputType.START_TIME) {
      startTimeInput.value = prettyTime;

      if (this.endTime_ != null && this.endTime_ < timeValue) {
        endTimeInput.value = '';
        this.endTime_ = null;
      }

      this.setWaveEventFormPosition_(waveEventForm, xPos, yPos);
      waveEventForm.classList.remove('hidden');

      this.startTime_ = timeValue;
      markChannelSelected();

      this.setDurationUI_();

      this.waitFor_(InputType.END_TIME);
    } else if (this.waitingFor_ === InputType.END_TIME) {
      endTimeInput.value = prettyTime;

      this.endTime_ = timeValue;

      this.setDurationUI_();

      this.waitFor_(InputType.CHANNEL);
    } else if (this.waitingFor_ === InputType.CHANNEL) {
      if (this.selectedChannels_.has(channelName)) {
        markChannelUnselected();
      } else {
        markChannelSelected();
      }
    }
    this.emitDraftChange();
  }

  /**
   * Selects a wave event type in the form, by setting the dropdown text in the
   * UI.
   * @param {string} type Type selected.
   */
  selectType(type) {
    const dropdown = document.getElementById('wave-event-type-dropdown-text');
    dropdown.textContent = type;
    this.emitDraftChange();
  }

  /**
   * Closes the wave event form and clears the clicks previously made.
   */
  close(clearDraft = true) {
    const waveEventForm = document.getElementById(this.formId_);
    const startTimeInput = this.getInputElement_(this.startTimeId_);
    const endTimeInput = this.getInputElement_(this.endTimeId_);

    startTimeInput.value = '';
    endTimeInput.value = '';

    this.toggleAllChannelCheckboxes_(false);

    this.startTime_ = null;
    this.endTime_ = null;
    this.waitingFor_ = InputType.START_TIME;

    this.selectedChannels_.clear();

    waveEventForm.classList.add('hidden');

    if (clearDraft) {
      Dispatcher.getInstance().sendAction({
        actionType: Dispatcher.ActionType.UPDATE_WAVE_EVENT_DRAFT,
        data: null,
      });
    }
  }

  /**
   * Returns the WaveEvent draft by accessing the elements in the view.
   * If the draft is not valid (i.e no startTime defined or any inconsistency),
   * returns null.
   * @return {?Store.Annotation} Wave event being created by the user.
   */
  getDraft() {
    if (this.startTime_ == null) {
      return null;
    }
    const startTime = this.startTime_;
    const endTime = this.endTime_ == null ? startTime : this.endTime_;

    if (endTime < startTime) {
      return null;
    }

    const labelText =
        document.getElementById('wave-event-type-dropdown-text').innerHTML;

    return {
      labelText,
      startTime,
      duration: endTime - startTime,
      channelList: Array.from(this.selectedChannels_),
    };
  }

  /**
   * Emits a change in the WaveEvent being created, by dispatching an action
   * that will update it in the store.
   * Call this method after any change coming from the UI.
   *
   * Note that both the store and this object keep a copy of the waveEventDraft.
   * The information kept here is considered the source of truth.
   */
  emitDraftChange() {
    const waveEventDraft = this.getDraft();
    if (!waveEventDraft) {
      return;
    }

    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.UPDATE_WAVE_EVENT_DRAFT,
      data: waveEventDraft,
    });
  }

  /**
   * Saves the wave event determined by the clicks made before.
   */
  save() {
    const waveEventDraft = this.getDraft();
    if (!waveEventDraft) {
      return;
    }

    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.ADD_WAVE_EVENT,
      data: waveEventDraft,
    });
    this.close(false);
  }

  /**
   * Returns a HTML id to use in the checkbox for a given channel.
   * @param {string} channelName Channel to get the id.
   * @return {string} HTML checkbox id.
   * @private
   */
  getChannelCheckboxId_(channelName) {
    return `wave-event-channel-${channelName}`;
  }

  /**
   * Check or uncheck every checkbox inside the form.
   * @param {boolean} checked Indicates if it should check or uncheck.
   * @private
   */
  toggleAllChannelCheckboxes_(checked) {
    document.querySelectorAll(`#${this.channelsContainerId_} label`)
        .forEach((label) => {
          utils.toggleMDLCheckbox(label, checked);
        });
  }

  /**
   * Handles a click in the "All" checkbox.
   * @param {!Event} event Event triggered by the checkbox.
   */
  toggleAllChannels(event) {
    const target = /** @type {!HTMLInputElement} */ (event.target);
    if (target.checked) {
      this.allChannels_.forEach((channelName) => {
        this.selectedChannels_.add(channelName);
      });
    } else {
      this.selectedChannels_.clear();
    }
    this.toggleAllChannelCheckboxes_(target.checked);
    this.emitDraftChange();
  }

  /**
   * Handles a change in the channel configuration, which updates the checkboxes
   * in the form and makes a copy of the channel names to use later.
   * @param {!Store.StoreData} store Store data.
   */
  handleChannelNames(store) {
    if (!store.channelIds || !store.indexChannelMap) {
      return;
    }

    this.allChannels_ =
        montages.channelIndexesToNames(store.channelIds, store.indexChannelMap);

    const channelsContainer =
        document.getElementById(this.channelsContainerId_);

    let checkboxesContainer = /** @type {!Element} */ (
        document.getElementById(this.checkboxesContainerId_));
    if (checkboxesContainer) {
      checkboxesContainer.remove();
    }

    checkboxesContainer = document.createElement('div');
    checkboxesContainer.id = this.checkboxesContainerId_;
    checkboxesContainer.onclick = () => {
      this.waitFor_(InputType.CHANNEL);
    };
    channelsContainer.appendChild(checkboxesContainer);

    this.allChannels_.forEach((channelName) => {
      const checkboxId = this.getChannelCheckboxId_(channelName);

      utils.addMDLCheckbox(
          checkboxesContainer, checkboxId, channelName, (checked) => {
            if (checked) {
              this.selectedChannels_.add(channelName);
            } else {
              this.selectedChannels_.delete(channelName);
            }
            this.emitDraftChange();
          });
    });
  }

  /**
   * Sets the visual focus in the selected input, and sets the waitingFor_
   * field, which will make the form wait for that type of input.
   * The focus state is not set using HTML focus(), but a different class named
   * force-focus. This is to allow the user clicking anywhere else in the UI,
   * without losing focus from the input selected here.
   * @param {!InputType} inputType Type to select.
   * @private
   */
  waitFor_(inputType) {
    const startTimeInput = this.getInputElement_(this.startTimeId_);
    const endTimeInput = this.getInputElement_(this.endTimeId_);
    const channelsContainer =
        document.getElementById(this.channelsContainerId_);

    switch(inputType) {
      case InputType.START_TIME:
        startTimeInput.classList.add('force-focus');
        endTimeInput.classList.remove('force-focus');
        channelsContainer.classList.remove('force-focus');
        break;
      case InputType.END_TIME:
        startTimeInput.classList.remove('force-focus');
        endTimeInput.classList.add('force-focus');
        channelsContainer.classList.remove('force-focus');
        break;
      case InputType.CHANNEL:
        startTimeInput.classList.remove('force-focus');
        endTimeInput.classList.remove('force-focus');
        channelsContainer.classList.add('force-focus');
        break;
      default:
        return;
    }
    this.waitingFor_ = inputType;
  }

  /**
   * Handles a click in an input, which will make the form wait for a click in
   * that input.
   * @param {!InputType} inputType Type to select.
   */
  clickInput(inputType) {
    this.waitFor_(inputType);
  }
}

goog.addSingletonGetter(WaveEventForm);

exports = WaveEventForm;

