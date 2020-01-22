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

goog.module('eeg_modelling.eeg_viewer.Predictions.tests');
goog.setTestOnly();

const ChunkScoreData = goog.require('proto.eeg_modelling.protos.PredictionMetadata.ChunkScoreData');
const MockControl = goog.require('goog.testing.MockControl');
const Predictions = goog.require('eeg_modelling.eeg_viewer.Predictions');
const ScoreData = goog.require('proto.eeg_modelling.protos.PredictionMetadata.ChunkScoreData.ScoreData');
const dom = goog.require('goog.dom');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;
let storeData;

const mockMDLComponentHandler = {
  upgradeElement: () => {},
};

testSuite({

  setUp() {
    mockControl = new MockControl();

    window.componentHandler = mockMDLComponentHandler;

    const scoreDataSZ = new ScoreData();
    scoreDataSZ.setPredictedValue(3);
    scoreDataSZ.setActualValue(1);
    scoreDataSZ.setPredictionProbability(0.6);

    const scoreDataED = new ScoreData();
    scoreDataED.setPredictedValue(87);
    scoreDataED.setActualValue(0);
    scoreDataED.setPredictionProbability(0.5);

    const scoreDataNoSZ = new ScoreData();
    scoreDataNoSZ.setPredictedValue(-3);
    scoreDataNoSZ.setActualValue(0);
    scoreDataNoSZ.setPredictionProbability(0.2);

    const scoreDataNoED = new ScoreData();
    scoreDataNoED.setPredictedValue(-7);
    scoreDataNoED.setActualValue(0);
    scoreDataNoED.setPredictionProbability(0.1);

    const chunkScoreData1 = new ChunkScoreData();
    chunkScoreData1.setDuration(96);
    chunkScoreData1.setStartTime(3743); // '01:02:23'
    chunkScoreData1.getScoreDataMap().set('SZ', scoreDataSZ);
    chunkScoreData1.getScoreDataMap().set('ED', scoreDataED);
    const chunkScoreData2 = new ChunkScoreData();
    chunkScoreData2.setDuration(96);
    chunkScoreData2.setStartTime(64); // '00:01:04'
    chunkScoreData2.getScoreDataMap().set('SZ', scoreDataNoSZ);
    chunkScoreData2.getScoreDataMap().set('ED', scoreDataNoED);
    storeData = {
      absStart: 0,
      chunkDuration: 43,
      chunkScores: [chunkScoreData1, chunkScoreData2],
      chunkStart: 50,
      label: 'SZ',
      numSecs: 100,
      predictionMode: 'None',
    };

    const predictionTable = document.createElement('table');
    predictionTable.classList.add('prediction');
    document.body.append(predictionTable);

    const tableHeader = document.createElement('thead');
    const headerRow = document.createElement('tr');
    tableHeader.appendChild(headerRow);
    predictionTable.appendChild(tableHeader);
    const addHeader = (text) => {
      const element = document.createElement('th');
      element.textContent = text;
      headerRow.appendChild(element);
    };
    addHeader('Start');
    addHeader('Type');
    addHeader('Pred');

    const labelSelector = document.createElement('div');
    labelSelector.id = 'label-dropdown';
    const labelValue = document.createElement('div');
    labelSelector.appendChild(labelValue);
    document.body.appendChild(labelSelector);

    const modeSelector = document.createElement('div');
    modeSelector.id = 'mode-dropdown';
    const modeValue = document.createElement('div');
    modeSelector.appendChild(modeValue);
    document.body.appendChild(modeSelector);

    const noPredictionsText = document.createElement('div');
    noPredictionsText.id = 'no-predictions-text';
    document.body.appendChild(noPredictionsText);

    const filterButton = document.createElement('button');
    filterButton.id = 'predictions-filter-button';
    document.body.appendChild(filterButton);
  },

  testHandleChunkScores() {
    Predictions.getInstance().handleChunkScores(storeData);

    assertEquals(
        '01:02:23',
        document.querySelector('tr[prediction="pos"][label="SZ"] td')
            .innerHTML);
    assertEquals(
        '00:01:04',
        document.querySelector('tr[prediction="neg"][label="SZ"] td')
            .innerHTML);

    // Ordered by time
    assertEquals('00:01:04',
        document.querySelector('tbody tr td').innerHTML);
  },

  tearDown() {
    dom.removeChildren(dom.getDocument().body);
    mockControl.$tearDown();
  },

});
