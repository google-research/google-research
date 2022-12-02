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

-- Extends meltingpot/lua/levels/clean_up/components.lua

local args = require 'common.args'
local class = require 'common.class'
local helpers = require 'common.helpers'
local log = require 'common.log'
local random = require 'system.random'
local tensor = require 'system.tensor'
local set = require 'common.set'
local events = require 'system.events'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')


--[[ AppleGrow is a component on each potential apple object that notifies scene
components when it has been spawned or cleaned.

Arguments:
`maxAppleGrowthRate` (float in [0, 1]): base rate of apple growth, to be
multiplied by a value determined as a function of the amount of dirt in the
river. Less dirt means higher apple regrowth rate.
`thresholdDepletion` (float in [0, 1]): Once dirt fraction exceeds this value
then the apple regrowth rate drops to 0.0.
`thresholdRestoration` (float in [0, 1]): Once dirt fraction decreases below
this value then apple regrowth rate saturates at its maximum value, which is
maxAppleGrowthRate.
]]
local AppleGrow = class.Class(component.Component)

function AppleGrow:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AppleGrow')},
      {'maxAppleGrowthRate', args.ge(0.0), args.le(1.0)},
      {'thresholdDepletion', args.ge(0.0), args.le(1.0)},
      {'thresholdRestoration', args.ge(0.0), args.le(1.0)},
  })
  AppleGrow.Base.__init__(self, kwargs)

  self._config.maxAppleGrowthRate = kwargs.maxAppleGrowthRate
  self._config.thresholdDepletion = kwargs.thresholdDepletion
  self._config.thresholdRestoration = kwargs.thresholdRestoration
end

function AppleGrow:postStart()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  self._riverMonitor = sceneObject:getComponent('RiverMonitor')
end

function AppleGrow:update()
  local dirtCount = self._riverMonitor:getDirtCount()
  local cleanCount = self._riverMonitor:getCleanCount()
  local dirtFraction = dirtCount / (dirtCount + cleanCount)

  local depletion = self._config.thresholdDepletion
  local restoration = self._config.thresholdRestoration
  local interpolation = (dirtFraction - depletion) / (restoration - depletion)
  -- By setting `thresholdRestoration` > 0.0 it would be possible to push
  -- the interpolation factor above 1.0, but we disallow that.
  interpolation = math.min(interpolation, 1.0)

  local probability = self._config.maxAppleGrowthRate * interpolation
  if random:uniformReal(0.0, 1.0) < probability then
    self.gameObject:setState('apple')
  end
end


--[[ The DirtTracker is a component on each river object that notifies scene
components when it has been spawned or cleaned.

Arguments:
`activeState` (string): Name of the active state, typically = 'dirt'.
`inactiveState` (string): Name of the inactive state, typically = 'dirtWait'.
]]
local DirtTracker = class.Class(component.Component)

function DirtTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DirtTracker')},
      {'activeState', args.default('dirt'), args.stringType},
      {'inactiveState', args.default('dirtWait'), args.stringType},
  })
  DirtTracker.Base.__init__(self, kwargs)
  self._activeState = kwargs.activeState
  self._inactiveState = kwargs.inactiveState
end

function DirtTracker:postStart()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  self._riverMonitor = sceneObject:getComponent('RiverMonitor')
  self._dirtSpawner = sceneObject:getComponent('DirtSpawner')

  -- If starting in inactive state, must register with the dirt spawner and
  -- river monitor.
  if self.gameObject:getState() == self._inactiveState then
    self._dirtSpawner:addPieceToPotential(self.gameObject:getPiece())
    self._riverMonitor:incrementCleanCount()
  elseif self.gameObject:getState() == self._activeState then
    self._riverMonitor:incrementDirtCount()
  end
end

function DirtTracker:onStateChange(oldState)
  local newState = self.gameObject:getState()
  if oldState == self._inactiveState and newState == self._activeState then
    self._riverMonitor:incrementDirtCount()
    self._riverMonitor:decrementCleanCount()
    self._dirtSpawner:removePieceFromPotential(self.gameObject:getPiece())
  elseif oldState == self._activeState and newState == self._inactiveState then
    self._riverMonitor:decrementDirtCount()
    self._riverMonitor:incrementCleanCount()
    self._dirtSpawner:addPieceToPotential(self.gameObject:getPiece())
  end
end


local DirtCleaning = class.Class(component.Component)

function DirtCleaning:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DirtCleaning')},
  })
  DirtCleaning.Base.__init__(self, kwargs)
end

function DirtCleaning:onHit(hittingGameObject, hitName)
  if self.gameObject:getState() == 'dirt' and hitName == 'cleanHit' then
    self.gameObject:setState('dirtWait')
    -- Trigger role-specific logic if applicable.
    if hittingGameObject:hasComponent('Taste') then
      hittingGameObject:getComponent('Taste'):cleaned()
    end
    local avatar = hittingGameObject:getComponent('Avatar')
    events:add('player_cleaned', 'dict',
               'player_index', avatar:getIndex()) -- int
    -- return `true` to prevent the beam from passing through a hit dirt.
    return true
  end
end


--[[ The Cleaner component provides a beam that can be used to clean dirt.

Arguments:
`cooldownTime` (int): Minimum time (frames) between cleaning beam shots.
`beamLength` (int): Max length of the cleaning beam.
`beamRadius` (int): Maximum distance from center to left/right of the cleaning
beam. The maximum width is 2*beamRadius+1.
]]
local Cleaner = class.Class(component.Component)

function Cleaner:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Cleaner')},
      {'cooldownTime', args.positive},
      {'beamLength', args.positive},
      {'beamRadius', args.positive},
  })
  Cleaner.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius
end

function Cleaner:addHits(worldConfig)
  worldConfig.hits['cleanHit'] = {
      layer = 'beamClean',
      sprite = 'BeamClean',
  }
  component.insertIfNotPresent(worldConfig.renderOrder, 'beamClean')
end

function Cleaner:addSprites(tileSet)
  -- This color is light blue.
  tileSet:addColor('BeamClean', {99, 223, 242, 175})
end

function Cleaner:registerUpdaters(updaterRegistry)
  local aliveState = self:getAliveState()
  local waitState = self:getWaitState()

  local clean = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if self.gameObject:getState() == aliveState then
      if self._config.cooldownTime >= 0 then
        if self._coolingTimer > 0 then
          self._coolingTimer = self._coolingTimer - 1
        else
          if actions['fireClean'] == 1 then
            self._coolingTimer = self._config.cooldownTime
            self.gameObject:hitBeam(
                'cleanHit', self._config.beamLength, self._config.beamRadius)
          end
        end
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = clean,
      priority = 140,
  }
end

function Cleaner:reset()
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
end

function Cleaner:getAliveState()
  return self.gameObject:getComponent('Avatar'):getAliveState()
end

function Cleaner:getWaitState()
  return self.gameObject:getComponent('Avatar'):getWaitState()
end


--[[ The RiverMonitor is a scene component that tracks the state of the river.

Other components such as dirt spawners and loggers can pull data from it.
]]
local RiverMonitor = class.Class(component.Component)

function RiverMonitor:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('RiverMonitor')},
  })
  RiverMonitor.Base.__init__(self, kwargs)
end

function RiverMonitor:reset()
  self._dirtCount = 0
  self._cleanCount = 0
end

function RiverMonitor:incrementDirtCount()
  self._dirtCount = self._dirtCount + 1
end

function RiverMonitor:decrementDirtCount()
  self._dirtCount = self._dirtCount - 1
end

function RiverMonitor:incrementCleanCount()
  self._cleanCount = self._cleanCount + 1
end

function RiverMonitor:decrementCleanCount()
  self._cleanCount = self._cleanCount - 1
end

function RiverMonitor:getDirtCount()
  return self._dirtCount
end

function RiverMonitor:getCleanCount()
  return self._cleanCount
end


--[[ The DirtSpawner is a scene component that spawns dirt at a fixed rate.

Arguments:
`dirtSpawnProbability` (float in [0, 1]): Probability of spawning one dirt on
each frame.
]]
local DirtSpawner = class.Class(component.Component)

function DirtSpawner:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DirtSpawner')},
      -- Probability per step of one dirt cell spawning in the river.
      {'dirtSpawnProbability', args.ge(0.0), args.le(1.0)},
      -- Number of steps to wait after the start of each episode before spawning
      -- dirt in the river.
      {'delayStartOfDirtSpawning', args.default(0), args.numberType},
  })
  DirtSpawner.Base.__init__(self, kwargs)
  self._config.delayStartOfDirtSpawning = kwargs.delayStartOfDirtSpawning
  self._dirtSpawnProbability = kwargs.dirtSpawnProbability
  self._potentialDirts = set.Set{}
end

function DirtSpawner:reset()
  self._potentialDirts = set.Set{}
  self._timeStep = 1
end

function DirtSpawner:update()
  if self._timeStep > self._config.delayStartOfDirtSpawning then
    if random:uniformReal(0.0, 1.0) < self._dirtSpawnProbability then
      local piece = random:choice(set.toSortedList(self._potentialDirts))
      if piece then
        self.gameObject.simulation:getGameObjectFromPiece(piece):setState(
          'dirt')
      end
    end
  end
  self._timeStep = self._timeStep + 1
end

function DirtSpawner:removePieceFromPotential(piece)
  self._potentialDirts[piece] = nil
end

function DirtSpawner:addPieceToPotential(piece)
  self._potentialDirts[piece] = true
end


-- An object that is edible switches state when an avatar touches it, and
-- provides a reward. It can be used in combination to the FixedRateRegrow.
local Edible = class.Class(component.Component)

function Edible:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Edible')},
      {'liveState', args.stringType},
      {'waitState', args.stringType},
      {'rewardForEating', args.numberType},
  })
  Edible.Base.__init__(self, kwargs)

  self._config.liveState = kwargs.liveState
  self._config.waitState = kwargs.waitState
  self._config.rewardForEating = kwargs.rewardForEating
end

function Edible:reset()
  self._waitState = self._config.waitState
  self._liveState = self._config.liveState
end

function Edible:setWaitState(newWaitState)
  self._waitState = newWaitState
end

function Edible:getWaitState()
  return self._waitState
end

function Edible:setLiveState(newLiveState)
  self._liveState = newLiveState
end

function Edible:getLiveState()
  return self._liveState
end

function Edible:onEnter(enteringGameObject, contactName)
  if contactName == 'avatar' then
    if self.gameObject:getState() == self._liveState then
      -- Reward the player who ate the edible.
      local avatarComponent = enteringGameObject:getComponent('Avatar')
      -- Trigger role-specific logic if applicable.
      if enteringGameObject:hasComponent('Taste') then
        enteringGameObject:getComponent('Taste'):consumed(
          self._config.rewardForEating)
      else
        avatarComponent:addReward(self._config.rewardForEating)
      end
      events:add('edible_consumed', 'dict',
                 'player_index', avatarComponent:getIndex())  -- int
      -- Change the edible to its wait (disabled) state.
      self.gameObject:setState(self._waitState)
    end
  end
end


--[[ The Taste component assigns specific roles to agents. Not used in defaults.
]]
local Taste = class.Class(component.Component)

function Taste:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Taste')},
      {'role', args.default('free'), args.oneOf('free', 'cleaner', 'consumer', 'cleaner_consumer')},
      {'cleanRewardAmount', args.default(0.1), args.numberType},
      {'eatRewardAmount', args.default(1), args.numberType},
  })
  Taste.Base.__init__(self, kwargs)
  self._config.role = kwargs.role
  self._config.cleanRewardAmount = kwargs.cleanRewardAmount
  self._config.eatRewardAmount = kwargs.eatRewardAmount
end

function Taste:cleaned()
  if self._config.role == 'cleaner' or self._config.role == 'cleaner_consumer' then
    self.gameObject:getComponent('Avatar'):addReward(self._config.cleanRewardAmount)
  end
  if self._config.role == 'consumer' then
    self.gameObject:getComponent('Avatar'):addReward(0.0)
  end
end

function Taste:consumed(edibleDefaultReward)
  if self._config.role == 'cleaner' then
    self.gameObject:getComponent('Avatar'):addReward(0.0)
  elseif self._config.role == 'consumer' or self._config.role == 'cleaner_consumer' then
    self.gameObject:getComponent('Avatar'):addReward(self._config.eatRewardAmount)
  else
    self.gameObject:getComponent('Avatar'):addReward(edibleDefaultReward)
  end
end


local GlobalStateTracker = class.Class(component.Component)

function GlobalStateTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GlobalStateTracker')},
      {'numPlayers', args.numberType},
      {'numClosestObjects', args.numberType},
  })
  GlobalStateTracker.Base.__init__(self, kwargs)

  self._config.numPlayers = kwargs.numPlayers
  self._config.orientationMap = {N = 0, E = 1, S = 2, W = 3}
  self._config.numClosestObjects = kwargs.numClosestObjects
end

function GlobalStateTracker:reset()
  -- avatar positions/orientations
  self.playerPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numPlayers, 2):fill(0)
  self.playerOrientations = tensor.Int32Tensor(self._config.numPlayers, self._config.numPlayers):fill(0)

  -- closest objects
  self.closestApplePositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numClosestObjects, 2):fill(0)
  self.closestPollutionPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numClosestObjects, 2):fill(0)
end

function GlobalStateTracker:postStart()
  -- init avatar positions/orientations
  for agentIdx = 1, self._config.numPlayers do
    -- process this agent's observations of other agents.
    -- e.g. could extract agent's position here and use in loop below
    -- to compute relative positions.
    for otherIdx = 1, self._config.numPlayers do
      local otherAvatarObj = self.gameObject.simulation:getAvatarFromIndex(otherIdx)
      local otherPlayerPosition = otherAvatarObj:getPosition()
      local otherPlayerOrientation = otherAvatarObj:getOrientation()
      self.playerPositions(agentIdx, otherIdx):val(otherPlayerPosition)
      self.playerOrientations(agentIdx, otherIdx):val(self._config.orientationMap[otherPlayerOrientation])
    end
  end
end

function compareDists(a,b)
  return a[2] < b[2]
end

function GlobalStateTracker:registerUpdaters(updaterRegistry)
  local updateStateInfo = function()
    -- update avatar
    self.closestApplePositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numClosestObjects, 2):fill(-1)
    self.closestPollutionPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numClosestObjects, 2):fill(-1)

    for agentIdx = 1, self._config.numPlayers do
      local avatarObj = self.gameObject.simulation:getAvatarFromIndex(agentIdx)
      local playerPosition = avatarObj:getPosition()

      -- position/orientation of other agents
      for otherIdx = 1, self._config.numPlayers do
        local otherAvatarObj = self.gameObject.simulation:getAvatarFromIndex(otherIdx)
        local otherPlayerPosition = otherAvatarObj:getPosition()
        local otherPlayerOrientation = otherAvatarObj:getOrientation()
        self.playerPositions(agentIdx, otherIdx):val(otherPlayerPosition)
        self.playerOrientations(agentIdx, otherIdx):val(self._config.orientationMap[otherPlayerOrientation])
      end

      -- get apple patch info
      local appleObjs = self.gameObject.simulation:getGameObjectsByName('potentialApple')
      local numApples = table.getn(appleObjs)
      local appleDists = {}
      local applePositions = {}
      for appleIdx, appleObj in pairs(appleObjs) do
        local applePosition = appleObj:getPosition()
        local appleState = appleObj:getState()
        local dist
        if appleState == 'apple' then
          dist = math.abs(applePosition[1] - playerPosition[1]) + math.abs(applePosition[2] - playerPosition[2])
        else
          dist = 1000.0 
        end
        table.insert(appleDists, {appleIdx, dist})
        table.insert(applePositions, applePosition)
      end

      -- return num_closest apples
      table.sort(appleDists, compareDists)
      for i=1,self._config.numClosestObjects do
        local appleDist = appleDists[i]

        if appleDist[2] ~= 1000.0 then
          self.closestApplePositions(agentIdx, i):val(applePositions[appleDist[1]])
        end
      end

      -- get pollution info
      local dirtObjs = self.gameObject.simulation:getGameObjectsByName('DirtContainer')
      local numDirts = table.getn(dirtObjs)
      local dirtDists = {}
      local dirtPositions = {}
      for dirtIdx, dirtObj in pairs(dirtObjs) do
        local dirtPosition = dirtObj:getPosition()
        local dirtState = dirtObj:getState()
        local dist
        if dirtState == 'dirt' then
          dist = math.abs(dirtPosition[1] - playerPosition[1]) + math.abs(dirtPosition[2] - playerPosition[2])
        else
          dist = 1000.0 
        end
        table.insert(dirtDists, {dirtIdx, dist})
        table.insert(dirtPositions, dirtPosition)
      end

      -- return num_closest dirts
      table.sort(dirtDists, compareDists)
      for i=1,self._config.numClosestObjects do
        local dirtDist = dirtDists[i]
        
        if dirtDist[2] ~= 1000.0 then
          self.closestPollutionPositions(agentIdx, i):val(dirtPositions[dirtDist[1]])
        end
      end

    end
  end

  updaterRegistry:registerUpdater{
    updateFn = updateStateInfo,
    priority = 100,
  }
end

function GlobalStateTracker:getPlayerPositions()
  return self.playerPositions
end

function GlobalStateTracker:getPlayerOrientations()
  return self.playerOrientations
end

local allComponents = {
    -- Non-avatar components.
    AppleGrow = AppleGrow,
    DirtTracker = DirtTracker,
    DirtCleaning = DirtCleaning,
    Edible = Edible,

    -- Avatar components.
    Cleaner = Cleaner,
    Taste = Taste,

    -- Scene components.
    RiverMonitor = RiverMonitor,
    DirtSpawner = DirtSpawner,
    GlobalStateTracker = GlobalStateTracker,
}

component_registry.registerAllComponents(allComponents)

return allComponents
