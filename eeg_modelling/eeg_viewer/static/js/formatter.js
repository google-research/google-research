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

goog.module('eeg_modelling.eeg_viewer.formatter');

/**
 * Formats seconds in HH:MM:SS format.
 * @param {number} sec Seconds since epoch.
 * @param {boolean=} show_ms Whether to display ms.
 * @return {string} HH:MM:SS format time.
 */
function formatTime(sec, show_ms = false) {
  const d = new Date(Math.floor(sec * 1000));
  const hours = String(d.getUTCHours()).padStart(2, '0');
  const minutes = String(d.getUTCMinutes()).padStart(2, '0');
  const seconds = String(d.getUTCSeconds()).padStart(2, '0');
  if (show_ms) {
    const ms = String(d.getUTCMilliseconds()).padStart(3, '0');
    return `${hours}:${minutes}:${seconds}.${ms}`;
  }
  return `${hours}:${minutes}:${seconds}`;
}

/**
 * Formats seconds in date string plus HH:MM:SS format.
 * @param {number} sec Seconds since epoch.
 * @return {string} Date string and HH:MM:SS time.
 */
function formatDateAndTime(sec) {
  const d = new Date(Math.floor(sec * 1000));
  return d.toUTCString();
}

/**
 * Formats an amount of seconds.
 * @param {number} sec duration in seconds.
 * @return {string} Duration in seconds or milliseconds, depending on the size.
 */
function formatDuration(sec) {
  if (sec <= 0.1) {
    const milliseconds = Math.floor(sec * 1000);
    return `${milliseconds} ms`;
  }
  return `${sec.toFixed(1)} s`;
}

exports = {
  formatTime,
  formatDateAndTime,
  formatDuration,
};
