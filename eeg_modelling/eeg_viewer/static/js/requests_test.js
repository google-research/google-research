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

goog.module('eeg_modelling.eeg_viewer.Requests.tests');
goog.setTestOnly();

const BipolarChannel = goog.require('proto.eeg_modelling.protos.ChannelDataId.BipolarChannel');
const ChannelDataId = goog.require('proto.eeg_modelling.protos.ChannelDataId');
const DataResponse = goog.require('proto.eeg_modelling.protos.DataResponse');
const MockControl = goog.require('goog.testing.MockControl');
const Requests = goog.require('eeg_modelling.eeg_viewer.Requests');
const TestingXhrIo = goog.require('goog.testing.net.XhrIo');
const WaveformChunk = goog.require('proto.eeg_modelling.protos.WaveformChunk');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;
let mockXhrIo;
let requests;


testSuite({

  setUp() {
    mockControl = new MockControl();
    requests = Requests.getInstance();
    // Initialize mocks for XhrIo requests.
    mockXhrIo = new TestingXhrIo();
    const mockGetXhrIo = mockControl.createMethodMock(requests, 'getXhrIo');
    mockGetXhrIo().$returns(mockXhrIo);
  },

  async testCreatePromise() {
    const chunkData = {
      tfExSSTablePath: 'hello there',
      predictionSSTablePath: 'general kenobi',
      sstableKey: '*cough*',
      edfPath: 'edf file',
      chunkStart: 32,
      chunkDuration: 3,
      channelIds: ['0', '0-1'],
      lowCut: 1.6,
      highCut: 70,
      notch: 60,
    };


    const bipolarChannel = new BipolarChannel();
    bipolarChannel.setIndex(0);
    bipolarChannel.setReferentialIndex(1);

    const channelDataId = new ChannelDataId();
    channelDataId.setBipolarChannel(bipolarChannel);

    const waveformChunk = new WaveformChunk();
    waveformChunk.setWaveformDatatable('waveform data');
    waveformChunk.setChannelDataIdsList([channelDataId]);

    const response = new DataResponse();
    response.setWaveformChunk(waveformChunk);


    mockControl.$replayAll();

    const promise = requests.createDataResponsePromise(chunkData);
    mockXhrIo.simulateResponse(200, response.serializeBinary());
    const promiseResponse = await promise;
    const responseData = promiseResponse.getWaveformChunk();

    mockControl.$verifyAll();

    assertEquals('waveform data', responseData.getWaveformDatatable());
    assertArrayEquals(['0-1'], responseData.getChannelDataIdsList().map((x) => {
      if (x.hasBipolarChannel()) {
        return (
            x.getBipolarChannel().getIndex() + '-' +
            x.getBipolarChannel().getReferentialIndex());
      } else {
        return String(x.getSingleChannel().getIndex());
      }
    }));
  },

  tearDown() {
    TestingXhrIo.cleanup();
    mockControl.$tearDown();
  },

});
