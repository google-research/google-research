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

goog.module('eeg_modelling.eeg_viewer.Stores.tests');
goog.setTestOnly();

const ChannelDataId = goog.require('proto.eeg_modelling.protos.ChannelDataId');
const DataResponse = goog.require('proto.eeg_modelling.protos.DataResponse');
const Label = goog.require('proto.eeg_modelling.protos.WaveformMetadata.Label');
const MockControl = goog.require('goog.testing.MockControl');
const SingleChannel = goog.require('proto.eeg_modelling.protos.ChannelDataId.SingleChannel');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const WaveformChunk = goog.require('proto.eeg_modelling.protos.WaveformChunk');
const WaveformMetadata = goog.require('proto.eeg_modelling.protos.WaveformMetadata');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;

const store = Store.getInstance();

testSuite({

  setUp() {
    mockControl = new MockControl();
    store.storeData.chunkStart = 10;
    store.storeData.chunkDuration = 10;
    store.storeData.tfExSSTablePath = null;
    store.storeData.edfPath = null;
    store.storeData.predictionSSTablePath = null;
    store.storeData.tfExFilePath = null;
    store.storeData.predictionFilePath = null;
    store.storeData.sstableKey = null;
    store.storeData.channelIds = null;
    store.storeData.fileInputDirty = false;
    store.storeData.queryDataDirty = false;
    store.storeData.seriesHeight = 100;
  },

  testHandleResponseOk() {
    const waveformChunk = new WaveformChunk();
    waveformChunk.setWaveformDatatable(JSON.stringify({data: 'fake data'}));
    waveformChunk.setChannelDataIdsList(['0', '1', '1', '2', '3', '5'].map((x) => {
      const singleChannel = new SingleChannel();
      singleChannel.setIndex(x);

      const channelDataId = new ChannelDataId();
      channelDataId.setSingleChannel(singleChannel);
      return channelDataId;
    }));

    const label = new Label();
    label.setLabelText('note');
    label.setStartTime(42);

    const waveformMeta = new WaveformMetadata();
    waveformMeta.setAbsStart(0);
    waveformMeta.setLabelsList([label]);
    waveformMeta.setFileType('EEG');
    waveformMeta.setNumSecs(50);
    waveformMeta.setPatientId('test id');
    waveformMeta.setSstableKey('grievous');
    waveformMeta.getChannelDictMap()['0'] = 'shorthand';

    const mockData = new DataResponse();
    mockData.setWaveformChunk(waveformChunk);
    mockData.setWaveformMetadata(waveformMeta);

    const newStoreData = store.handleRequestResponseOk(mockData);
    assertObjectEquals({ data: 'fake data' }, newStoreData.chunkGraphData);
    assertArrayEquals(
        ['0', '1', '1', '2', '3', '5'], newStoreData.channelIds);
    assertEquals(0, newStoreData.absStart);
    assertEquals('note', newStoreData.annotations[0].labelText);
    assertEquals(42.0, newStoreData.annotations[0].startTime);
    assertEquals('EEG', newStoreData.fileType);
    assertEquals(50, newStoreData.numSecs);
    assertEquals('test id', newStoreData.patientId);
    assertEquals('grievous', newStoreData.sstableKey);
  },

  testHandleWindowLocationPendingRequest() {
    const mockFragmentData = {
      tfexsstablepath: '/hello/there',
      predictionsstablepath: 'general/kenobi',
      tfexfilepath: '/good/bye',
      predictionfilepath: 'hello/world',
      sstablekey: 'grievous',
      chunkstart: '42',
      chunkduration: '3',
      channelids: '0,1,1,2,3,5',
    };

    const newStoreData =
        store.handleWindowLocationPendingRequest(mockFragmentData);
    assertEquals('/hello/there', newStoreData.tfExSSTablePath);
    assertEquals('general/kenobi', newStoreData.predictionSSTablePath);
    assertEquals('/good/bye', newStoreData.tfExFilePath);
    assertEquals('hello/world', newStoreData.predictionFilePath);
    assertEquals('grievous', newStoreData.sstableKey);
    assertEquals(42, newStoreData.chunkStart);
    assertEquals(3, newStoreData.chunkDuration);
    assertArrayEquals(['0','1','1','2','3','5'], newStoreData.channelIds);
  },

  testHandleRequestResponseError() {
    const mockData = {
      message: 'Bad Request',
    };
    const newStoreData = store.handleRequestResponseError(mockData);
    assertEquals('Bad Request', newStoreData.error.message);
  },

  testHandleToolBarNextChunk() {
    const newStoreData = store.handleToolBarNextChunk();

    assertEquals(20, newStoreData.chunkStart);
  },

  testHandleToolBarNextSec() {
    const newStoreData = store.handleToolBarNextSec();

    assertEquals(11 , newStoreData.chunkStart);
  },

  testHandleToolBarActionPrevChunk() {
    const newStoreData = store.handleToolBarPrevChunk();

    assertEquals(0, newStoreData.chunkStart);
  },

  testHandleToolBarActionPrevSec() {
    const newStoreData = store.handleToolBarPrevSec();

    assertEquals(9, newStoreData.chunkStart);
  },

  testHandleToolBarZoom() {
    const mockData = {
      selectedValue: 42,
    };

    const newStoreData = store.handleToolBarZoom(mockData);
    assertEquals(42, newStoreData.chunkDuration);
  },

  testHandleToolBarGridlines() {
    const mockData = {
      selectedValue: 0.2,
    };

    const newStoreData = store.handleToolBarGridlines(mockData);
    assertEquals(0.2, newStoreData.timeScale);
  },

  testHandlePredictionModeSelection() {
    const mockData = {
      selectedValue: 'None',
    };

    const newStoreData = store.handlePredictionModeSelection(mockData);
    assertEquals(Store.PredictionMode.NONE, newStoreData.predictionMode);
  },

  testHandlePredictionLabelSelection() {
    const mockData = {
      selectedValue: 'you were my brother',
    };

    const newStoreData = store.handlePredictionLabelSelection(mockData);
    assertEquals('you were my brother', newStoreData.label);
  },

  testHandlePredictionChunkRequest() {
    const mockData = {
      time: 3,
    };

    const newStoreData = store.handlePredictionChunkRequest(mockData);
    assertEquals(3, newStoreData.chunkStart);
  },

  testHandleAnnotationSelection() {
    const mockData = {
      time: 5,
    };

    const newStoreData = store.handleAnnotationSelection(mockData);
    assertEquals(0, newStoreData.chunkStart);
  },

  testHandleMenuFileLoad() {
    const mockData = {
      tfExSSTablePath: '/hello/there',
      predictionSSTablePath: null,
      tfExFilePath: null,
      predictionFilePath: null,
      edfPath: '/edf/file',
      sstableKey: 'grievous',
      channelIds: ['0', '1', '1', '2', '3', '5'],
    };

    const newStoreData = store.handleMenuFileLoad(mockData);

    assertEquals('/hello/there', newStoreData.tfExSSTablePath);
    assertNull(newStoreData.predictionSSTablePath);
    assertEquals('grievous', newStoreData.sstableKey);
    assertNull(newStoreData.tfExFilePath);
    assertNull(newStoreData.predictionFilePath);
  },

  tearDown() {
    mockControl.$tearDown();
  },

});
