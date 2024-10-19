// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.
const OFFSET: u32 = 42;
const WG: u32 = 256;

const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 1u << BITS_PER_PASS;
const HISTOGRAM_SIZE: u32 = WG * BIN_COUNT;
const ELEMENTS_PER_THREAD: u32 = 4;

const BLOCK_SIZE = WG * ELEMENTS_PER_THREAD;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}
