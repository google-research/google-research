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

/**
 * @fileoverview Provides functions to handle channel names and montages.
 */

goog.module('eeg_modelling.eeg_viewer.montages');

const JspbMap = goog.require('jspb.Map');
const googMemoize = goog.require('goog.memoize');

// Montages are a standard format for clinicians to view EEG signals in.
// Each list element is a standard (A-B), and each standard consists of the
// difference between the signals from two leads (A, B) placed on the scalp.
// Some leads are referred to by multiple codes (C|D), but only one will exist
// in one file.
//
// See this website for more info about EEG montages.
// http://eegatlas-online.com/index.php/en/montages
const MONTAGES = {};
MONTAGES['longitudinal bipolar'] = [
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP1-F7', 'F7-T3|T7', 'T3|T7-T5|P7', 'T5|P7-O1',
    'FP2-F8', 'F8-T4|T8', 'T4|T8-T|T8', 'T6|P8-O2',
    'FZ-CZ', 'CZ-PZ',
    'EKG1-EKG2'
];
MONTAGES['transverse bipolar'] = [
    'F7-FP1', 'FP1-FP2', 'FP2-F8',
    'F7-F3', 'F3-FZ', 'FZ-F4', 'F4-F8',
    'A1-T3|T7', 'T3|T7-C3', 'C3-CZ', 'CZ-C4', 'C4-T4|T8', 'T4|T8-A2',
    'T5|P7-P3', 'P3-PZ', 'PZ-P4', 'P4-T6|P8',
    'T5|P7-O1', 'O1-O2', 'O2-T6|P8',
    'EKG1-EKG2'
];
MONTAGES['referential'] = [
    'FP1-CZ', 'F3-CZ', 'C3-CZ', 'P3-CZ', 'O1-CZ',
    'FP2-CZ', 'F4-CZ', 'C4-CZ', 'P4-CZ', 'O2-CZ',
    'F7-CZ', 'T3|T7-CZ', 'T5|P7-CZ',
    'F8-CZ', 'T4|T8-CZ', 'T6|P8-CZ',
    'FZ-CZ', 'PZ-CZ',
    'A1-CZ', 'A2-CZ',
    'EKG1-EKG2'
];
MONTAGES['circumferential'] = [
    'O1-T5|P7', 'T5|P7-T3|T7', 'T3|T7-F7', 'F7-FP1', 'FP1-FP2', 'FP2-F8',
    'F8-T4|T8', 'T4|T8-T6|P8', 'T6|P8-O2',
    'FP1-F7', 'F7-A1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FZ-CZ', 'CZ-PZ',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-A2',
    'EKG1-EKG2'
];
MONTAGES['reverse circumferential'] = [
    'FP1-F7', 'F7-T3|T7', 'T3|T7-T5|P7', 'T5|P7-O1', 'O1-O2', 'O2-T6|P8',
    'T6|P8-T4|T8', 'T4|T8-F8', 'F8-FP2',
    'FP1-F7', 'F7-A1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FZ-CZ', 'CZ-PZ',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-A2',
    'EKG1-EKG2'
];

/**
 * @typedef {{
 *   indexStrList: !Array<string>,
 *   missingChannelList: ?Array<string>,
 * }}
 */
let MontageInfo;


/**
 * Return the list of available montages.
 * @return {!Object<string, !Array<string>>}
 */
function getMontages() {
  return MONTAGES;
}


/**
 * Reverses a channel-index map: receives a indexToChannel and returns a
 * channelToIndex map.
 * @param {?JspbMap<string, string>} indexToChannelMap Map index to channel.
 * @return {!Map<string, number>} The index info for a pre-programmed montage.
 */
const getChannelToIndexMap = googMemoize((indexToChannelMap) => {
  if (!indexToChannelMap) {
    return new Map();
  }
  return new Map(indexToChannelMap.getEntryList().map(x => [x[1], x[0]]));
});


/**
 * Creates the index-centric info for a pre-programmed montage. This
 * includes a list of channel indexes strings for the montage and an optional
 * list of channel names for which there were no associated indices.
 *
 * Note that any list of channel names can be passed as a montage, it doesn't
 * need to be a well known montage.
 * @param {?JspbMap<string, string>} indexToChannelMap Map channel to index.
 * @param {!Array<string>} montage The list of channel names.
 * @return {!MontageInfo} The index info for a pre-programmed montage.
 */
function createMontageInfo(indexToChannelMap, montage) {
  const channelToIndexMap = getChannelToIndexMap(indexToChannelMap);

  const missingChannelList = [];

  // Given a singular channel string that may have multiple pseudonyns like A
  // or B, 'A|B', return the index associated with either A or B if one exists
  // or return null.
  const getChannelIndexFromNameOptions = (nameWithOptions) => {
    const validNames = nameWithOptions.split('|')
        .filter(name => channelToIndexMap.has(name));
    const validIndices = validNames.map(
        (name) => channelToIndexMap.get(name));
    if (validIndices.length > 0) {
      return validIndices[0];
    } else {
      missingChannelList.push(nameWithOptions);
      return null;
    }
  };

  // Given a channel that may be bipolar (consist of the difference between
  // two singular channels) A-B|C, return the index string that results from
  // joining the results of getting the index of each set of channel options
  // (A, B|C) if all results are non-null, otherwise return null.
  const getBipolarChannelIndexString = (bipolarName) => {
    const singularChannels = bipolarName.split('-');
    const indices = singularChannels
        .map((channelName) => getChannelIndexFromNameOptions(channelName))
        .filter(index => index != null);
    if (indices.length == singularChannels.length) {
      return indices.join('-');
    } else {
      return null;
    }
  };

  const indexStrList = montage
      .map((channelStr) => getBipolarChannelIndexString(channelStr))
      .filter(channel => channel != null);

  return {
    indexStrList: indexStrList,
    missingChannelList: missingChannelList,
  };
}

/**
 * Transform a list of channel indexes to channel names.
 * @param {!Array<string>} channelIds List of channel indexes.
 * @param {!JspbMap<string, string>} indexChannelMap Map index to channel.
 * @return {!Array<string>} List of channel names.
 */
function channelIndexesToNames(channelIds, indexChannelMap) {
  return channelIds.map((channelId) => {
    const singularChannelsIds = channelId.split('-');
    const singularChannelsNames =
        singularChannelsIds.map((channelId) => indexChannelMap.get(channelId));
    return singularChannelsNames.join('-');
  });
}

exports = {
  MontageInfo,
  getMontages,
  createMontageInfo,
  channelIndexesToNames,
};
