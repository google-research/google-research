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


testSuite({

  setUp() {
    mockControl = new MockControl();

    const scoreData1SZ = new ScoreData();
    scoreData1SZ.setPredictedValue(3);
    scoreData1SZ.setActualValue(1);

    const scoreData1ED = new ScoreData();
    scoreData1ED.setPredictedValue(87);
    scoreData1ED.setActualValue(0);

    const scoreData2SZ = new ScoreData();
    scoreData2SZ.setPredictedValue(78);
    scoreData2SZ.setActualValue(0);

    const scoreData2ED = new ScoreData();
    scoreData2ED.setPredictedValue(54);
    scoreData2ED.setActualValue(0);

    const chunkScoreData1 =  new ChunkScoreData();
    chunkScoreData1.setDuration(96);
    chunkScoreData1.setStartTime(3743);
    chunkScoreData1.getScoreDataMap().set('SZ', scoreData1SZ);
    chunkScoreData1.getScoreDataMap().set('ED', scoreData1ED);
    const chunkScoreData2 = new ChunkScoreData();
    chunkScoreData2.setDuration(96);
    chunkScoreData2.setStartTime(64);
    chunkScoreData2.getScoreDataMap().set('SZ', scoreData2SZ);
    chunkScoreData2.getScoreDataMap().set('ED', scoreData2ED);
    storeData = {
      absStart: 0,
      chunkDuration: 43,
      chunkScores: [chunkScoreData1, chunkScoreData2],
      chunkStart: 50,
      fileInputDirty: true,
      label: 'SZ',
      numSecs: 100,
      predictionMode: 'None',
      requestStatus: 'RETURNED',
      annotations: [{start_time: 0, label: 'Douglas Adams'}],
    };

    const predictionTable = document.createElement('table');
    predictionTable.classList.add('prediction');
    document.body.append(predictionTable);
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
    const predictionItem = document.createElement('li');
    predictionItem.classList.add('prediction');
    document.body.appendChild(predictionItem);
    const truePos = document.createElement('button');
    truePos.id = 'true-pos';
    document.body.appendChild(truePos);
    const trueNeg = document.createElement('button');
    trueNeg.id = 'true-neg';
    document.body.appendChild(trueNeg);
    const modeButton = document.createElement('button');
    modeButton.id = 'mode-button';
    document.body.appendChild(modeButton);
  },

  testHandleChunkScores() {
    Predictions.getInstance().handleChunkScores(storeData);

    assertEquals('00:01:04',
        document.querySelector('#prediction-row-64 .time').innerHTML);
    assertEquals('78',
        document.querySelector('#prediction-row-64 .SZ.predicted').innerHTML);
    assertEquals('0',
        document.querySelector('#prediction-row-64 .SZ.actual').innerHTML);
    assertEquals('54',
        document.querySelector('#prediction-row-64 .ED.predicted').innerHTML);
    assertEquals('0',
        document.querySelector('#prediction-row-64 .ED.actual').innerHTML);

    const firstValueRow = document.querySelector('#prediction-row-64');
    const secondValueRow = document.querySelector('#prediction-row-3743');
    assertTrue(firstValueRow.classList.contains('in-viewport'));
    assertTrue(secondValueRow.classList.contains('ir-true-pos'));
  },

  tearDown() {
    dom.removeChildren(dom.getDocument().body);
    mockControl.$tearDown();
  },

});
