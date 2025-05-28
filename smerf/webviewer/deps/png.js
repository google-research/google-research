// Copyright 2025 The Google Research Authors.
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

/*
 * MIT LICENSE
 * Copyright (c) 2011 Devon Govett
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
 * to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * This implementation slightly deviates from that of
 * https://unpkg.com/png-js@1.0.0/png.js in that references to "document"
 * and "window" have been removed.
 *
 */

let PNG = (function() {
  let APNG_DISPOSE_OP_NONE = 0;
  let APNG_DISPOSE_OP_BACKGROUND = 1;
  let APNG_DISPOSE_OP_PREVIOUS = 2;
  let APNG_BLEND_OP_SOURCE = 0;
  let APNG_BLEND_OP_OVER = 1;

  class PNG {
    static load(url, canvas, callback) {
      if (typeof canvas === 'function') {
        callback = canvas;
      }

      const xhr = new XMLHttpRequest();
      xhr.open('GET', url, true);
      xhr.responseType = 'arraybuffer';
      xhr.onload = () => {
        const data = new Uint8Array(xhr.response || xhr.mozResponseArrayBuffer);
        const png = new PNG(data);
        if (typeof (canvas && canvas.getContext) === 'function') {
          png.render(canvas);
        }
        return typeof callback === 'function' ? callback(png) : undefined;
      };

      return xhr.send(null);
    }

    constructor(data1) {
      let i;
      this.data = data1;
      this.pos = 8; // Skip the default header

      this.palette = [];
      this.imgData = [];
      this.transparency = {};
      this.animation = null;
      this.text = {};
      let frame = null;

      while (true) {
        var data;
        let chunkSize = this.readUInt32();
        let section = '';
        for (i = 0; i < 4; i++) {
          section += String.fromCharCode(this.data[this.pos++]);
        }

        switch (section) {
          case 'IHDR':
            // we can grab  interesting values from here (like width, height, etc)
            this.width = this.readUInt32();
            this.height = this.readUInt32();
            this.bits = this.data[this.pos++];
            this.colorType = this.data[this.pos++];
            this.compressionMethod = this.data[this.pos++];
            this.filterMethod = this.data[this.pos++];
            this.interlaceMethod = this.data[this.pos++];
            break;

          case 'acTL':
            // we have an animated PNG
            this.animation = {
              numFrames: this.readUInt32(),
              numPlays: this.readUInt32() || Infinity,
              frames: []
            };
            break;

          case 'PLTE':
            this.palette = this.read(chunkSize);
            break;

          case 'fcTL':
            if (frame) {
              this.animation.frames.push(frame);
            }

            this.pos += 4; // skip sequence number
            frame = {
              width: this.readUInt32(),
              height: this.readUInt32(),
              xOffset: this.readUInt32(),
              yOffset: this.readUInt32()
            };

            var delayNum = this.readUInt16();
            var delayDen = this.readUInt16() || 100;
            frame.delay = (1000 * delayNum) / delayDen;

            frame.disposeOp = this.data[this.pos++];
            frame.blendOp = this.data[this.pos++];
            frame.data = [];
            break;

          case 'IDAT':
          case 'fdAT':
            if (section === 'fdAT') {
              this.pos += 4; // skip sequence number
              chunkSize -= 4;
            }

            data = (frame && frame.data) || this.imgData;
            for (i = 0; i < chunkSize; i++) {
              data.push(this.data[this.pos++]);
            }
            break;

          case 'tRNS':
            // This chunk can only occur once and it must occur after the
            // PLTE chunk and before the IDAT chunk.
            this.transparency = {};
            switch (this.colorType) {
              case 3:
                // Indexed color, RGB. Each byte in this chunk is an alpha for
                // the palette index in the PLTE ("palette") chunk up until the
                // last non-opaque entry. Set up an array, stretching over all
                // palette entries which will be 0 (opaque) or 1 (transparent).
                this.transparency.indexed = this.read(chunkSize);
                var short = 255 - this.transparency.indexed.length;
                if (short > 0) {
                  for (i = 0; i < short; i++) {
                    this.transparency.indexed.push(255);
                  }
                }
                break;
              case 0:
                // Greyscale. Corresponding to entries in the PLTE chunk.
                // Grey is two bytes, range 0 .. (2 ^ bit-depth) - 1
                this.transparency.grayscale = this.read(chunkSize)[0];
                break;
              case 2:
                // True color with proper alpha channel.
                this.transparency.rgb = this.read(chunkSize);
                break;
            }
            break;

          case 'tEXt':
            var text = this.read(chunkSize);
            var index = text.indexOf(0);
            var key = String.fromCharCode.apply(String, text.slice(0, index));
            this.text[key] = String.fromCharCode.apply(
              String,
              text.slice(index + 1)
            );
            break;

          case 'IEND':
            if (frame) {
              this.animation.frames.push(frame);
            }

            // we've got everything we need!
            switch (this.colorType) {
              case 0:
              case 3:
              case 4:
                this.colors = 1;
                break;
              case 2:
              case 6:
                this.colors = 3;
                break;
            }

            this.hasAlphaChannel = [4, 6].includes(this.colorType);
            var colors = this.colors + (this.hasAlphaChannel ? 1 : 0);
            this.pixelBitlength = this.bits * colors;

            switch (this.colors) {
              case 1:
                this.colorSpace = 'DeviceGray';
                break;
              case 3:
                this.colorSpace = 'DeviceRGB';
                break;
            }

            this.imgData = new Uint8Array(this.imgData);
            return;
            break;

          default:
            // unknown (or unimportant) section, skip it
            this.pos += chunkSize;
        }

        this.pos += 4; // Skip the CRC

        if (this.pos > this.data.length) {
          throw new Error('Incomplete or corrupt PNG file');
        }
      }
    }

    read(bytes) {
      const result = new Array(bytes);
      for (let i = 0; i < bytes; i++) {
        result[i] = this.data[this.pos++];
      }
      return result;
    }

    readUInt32() {
      const b1 = this.data[this.pos++] << 24;
      const b2 = this.data[this.pos++] << 16;
      const b3 = this.data[this.pos++] << 8;
      const b4 = this.data[this.pos++];
      return b1 | b2 | b3 | b4;
    }

    readUInt16() {
      const b1 = this.data[this.pos++] << 8;
      const b2 = this.data[this.pos++];
      return b1 | b2;
    }

    decodePixels(data) {
      if (data == null) {
        data = this.imgData;
      }
      if (data.length === 0) {
        return new Uint8Array(0);
      }

      data = new FlateStream(data);
      data = data.getBytes();

      const { width, height } = this;
      const pixelBytes = this.pixelBitlength / 8;

      const pixels = new Uint8Array(width * height * pixelBytes);
      const { length } = data;
      let pos = 0;

      function pass(x0, y0, dx, dy, singlePass = false) {
        const w = Math.ceil((width - x0) / dx);
        const h = Math.ceil((height - y0) / dy);
        const scanlineLength = pixelBytes * w;
        const buffer = singlePass ? pixels : new Uint8Array(scanlineLength * h);
        let row = 0;
        let c = 0;
        while (row < h && pos < length) {
          var byte, col, i, left, upper;
          switch (data[pos++]) {
            case 0: // None
              for (i = 0; i < scanlineLength; i++) {
                buffer[c++] = data[pos++];
              }
              break;

            case 1: // Sub
              for (i = 0; i < scanlineLength; i++) {
                byte = data[pos++];
                left = i < pixelBytes ? 0 : buffer[c - pixelBytes];
                buffer[c++] = (byte + left) % 256;
              }
              break;

            case 2: // Up
              for (i = 0; i < scanlineLength; i++) {
                byte = data[pos++];
                col = (i - (i % pixelBytes)) / pixelBytes;
                upper =
                  row &&
                  buffer[
                    (row - 1) * scanlineLength +
                      col * pixelBytes +
                      (i % pixelBytes)
                  ];
                buffer[c++] = (upper + byte) % 256;
              }
              break;

            case 3: // Average
              for (i = 0; i < scanlineLength; i++) {
                byte = data[pos++];
                col = (i - (i % pixelBytes)) / pixelBytes;
                left = i < pixelBytes ? 0 : buffer[c - pixelBytes];
                upper =
                  row &&
                  buffer[
                    (row - 1) * scanlineLength +
                      col * pixelBytes +
                      (i % pixelBytes)
                  ];
                buffer[c++] = (byte + Math.floor((left + upper) / 2)) % 256;
              }
              break;

            case 4: // Paeth
              for (i = 0; i < scanlineLength; i++) {
                var paeth, upperLeft;
                byte = data[pos++];
                col = (i - (i % pixelBytes)) / pixelBytes;
                left = i < pixelBytes ? 0 : buffer[c - pixelBytes];

                if (row === 0) {
                  upper = upperLeft = 0;
                } else {
                  upper =
                    buffer[
                      (row - 1) * scanlineLength +
                        col * pixelBytes +
                        (i % pixelBytes)
                    ];
                  upperLeft =
                    col &&
                    buffer[
                      (row - 1) * scanlineLength +
                        (col - 1) * pixelBytes +
                        (i % pixelBytes)
                    ];
                }

                const p = left + upper - upperLeft;
                const pa = Math.abs(p - left);
                const pb = Math.abs(p - upper);
                const pc = Math.abs(p - upperLeft);

                if (pa <= pb && pa <= pc) {
                  paeth = left;
                } else if (pb <= pc) {
                  paeth = upper;
                } else {
                  paeth = upperLeft;
                }

                buffer[c++] = (byte + paeth) % 256;
              }
              break;

            default:
              throw new Error(`Invalid filter algorithm: ${data[pos - 1]}`);
          }

          if (!singlePass) {
            let pixelsPos = ((y0 + row * dy) * width + x0) * pixelBytes;
            let bufferPos = row * scanlineLength;
            for (i = 0; i < w; i++) {
              for (let j = 0; j < pixelBytes; j++)
                pixels[pixelsPos++] = buffer[bufferPos++];
              pixelsPos += (dx - 1) * pixelBytes;
            }
          }

          row++;
        }
      }

      if (this.interlaceMethod === 1) {
        /*
          1 6 4 6 2 6 4 6
          7 7 7 7 7 7 7 7
          5 6 5 6 5 6 5 6
          7 7 7 7 7 7 7 7
          3 6 4 6 3 6 4 6
          7 7 7 7 7 7 7 7
          5 6 5 6 5 6 5 6
          7 7 7 7 7 7 7 7
        */
        pass(0, 0, 8, 8); // 1
        pass(4, 0, 8, 8); // 2
        pass(0, 4, 4, 8); // 3
        pass(2, 0, 4, 4); // 4
        pass(0, 2, 2, 4); // 5
        pass(1, 0, 2, 2); // 6
        pass(0, 1, 1, 2); // 7
      } else {
        pass(0, 0, 1, 1, true);
      }

      return pixels;
    }

    decodePalette() {
      const { palette } = this;
      const { length } = palette;
      const transparency = this.transparency.indexed || [];
      const ret = new Uint8Array((transparency.length || 0) + length);
      let pos = 0;
      let c = 0;

      for (let i = 0; i < length; i += 3) {
        var left;
        ret[pos++] = palette[i];
        ret[pos++] = palette[i + 1];
        ret[pos++] = palette[i + 2];
        ret[pos++] = (left = transparency[c++]) != null ? left : 255;
      }

      return ret;
    }

    copyToImageData(imageData, pixels) {
      let j, k;
      let { colors } = this;
      let palette = null;
      let alpha = this.hasAlphaChannel;

      if (this.palette.length) {
        palette =
          this._decodedPalette || (this._decodedPalette = this.decodePalette());
        colors = 4;
        alpha = true;
      }

      const data = imageData.data || imageData;
      const { length } = data;
      const input = palette || pixels;
      let i = (j = 0);

      if (colors === 1) {
        while (i < length) {
          k = palette ? pixels[i / 4] * 4 : j;
          const v = input[k++];
          data[i++] = v;
          data[i++] = v;
          data[i++] = v;
          data[i++] = alpha ? input[k++] : 255;
          j = k;
        }
      } else {
        while (i < length) {
          k = palette ? pixels[i / 4] * 4 : j;
          data[i++] = input[k++];
          data[i++] = input[k++];
          data[i++] = input[k++];
          data[i++] = alpha ? input[k++] : 255;
          j = k;
        }
      }
    }

    decode() {
      const ret = new Uint8Array(this.width * this.height * 4);
      this.copyToImageData(ret, this.decodePixels());
      return ret;
    }

    renderFrame(ctx, number) {
      const { frames } = this.animation;
      const frame = frames[number];
      const prev = frames[number - 1];

      // if we're on the first frame, clear the canvas
      if (number === 0) {
        ctx.clearRect(0, 0, this.width, this.height);
      }

      // check the previous frame's dispose operation
      if ((prev && prev.disposeOp) === APNG_DISPOSE_OP_BACKGROUND) {
        ctx.clearRect(prev.xOffset, prev.yOffset, prev.width, prev.height);
      } else if ((prev && prev.disposeOp) === APNG_DISPOSE_OP_PREVIOUS) {
        ctx.putImageData(prev.imageData, prev.xOffset, prev.yOffset);
      }

      // APNG_BLEND_OP_SOURCE overwrites the previous data
      if (frame.blendOp === APNG_BLEND_OP_SOURCE) {
        ctx.clearRect(frame.xOffset, frame.yOffset, frame.width, frame.height);
      }

      // draw the current frame
      return ctx.drawImage(frame.image, frame.xOffset, frame.yOffset);
    }

    animate(ctx) {
      let frameNumber = 0;
      const { numFrames, frames, numPlays } = this.animation;

      const doFrame = () => {
        const f = frameNumber++ % numFrames;
        const frame = frames[f];
        this.renderFrame(ctx, f);

        if (numFrames > 1 && frameNumber / numFrames < numPlays) {
          this.animation._timeout = setTimeout(doFrame, frame.delay);
        }
      };

      doFrame();
    }

    stopAnimation() {
      return clearTimeout(this.animation && this.animation._timeout);
    }

    render(canvas) {
      // if this canvas was displaying another image before,
      // stop the animation on it
      if (canvas._png) {
        canvas._png.stopAnimation();
      }

      canvas._png = this;
      canvas.width = this.width;
      canvas.height = this.height;
      const ctx = canvas.getContext('2d');

      if (this.animation) {
        this.decodeFrames(ctx);
        return this.animate(ctx);
      } else {
        const data = ctx.createImageData(this.width, this.height);
        this.copyToImageData(data, this.decodePixels());
        return ctx.putImageData(data, 0, 0);
      }
    }
  }
  return PNG;
})();
