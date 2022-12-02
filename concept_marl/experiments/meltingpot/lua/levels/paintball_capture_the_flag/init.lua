--[[ Copyright 2020 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

-- Entry point lua file for the capture_the_flag level.
-- Extends meltingpot/lua/levels/paintball_capture_the_flag/init.lua

local meltingpot = 'meltingpot.lua.'

local mp_modules = meltingpot .. 'modules.'
local paintball = meltingpot .. 'levels.paintball.'

local api_factory = require(mp_modules .. 'api_factory')
local simulation = require(mp_modules .. 'base_simulation')

-- Required to be able to use the components in the level
local component_library = require(mp_modules .. 'component_library')
local avatar_library = require(mp_modules .. 'avatar_library')
-- Next require the general paintball game components.
local shared_components = require(paintball .. 'shared_components')
-- Finally add the local components for this game, overriding any previously
-- loaded having the same name.
local components = require 'components'

return api_factory.apiFactory{
    Simulation = simulation.BaseSimulation,
    settings = {
        -- Scale each sprite to a square of size `spriteSize` X `spriteSize`.
        spriteSize = 8,
        -- Terminate the episode after this many frames.
        maxEpisodeLengthFrames = 1000,
        -- Settings to pass to simulation.lua.
        simulation = {},
    }
}