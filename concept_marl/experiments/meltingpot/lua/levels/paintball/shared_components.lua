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

local args = require 'common.args'
local class = require 'common.class'
local helpers = require 'common.helpers'
local log = require 'common.log'
local set = require 'common.set'
local events = require 'system.events'
local random = require 'system.random'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')


local AllBeamBlocker = class.Class(component.Component)

function AllBeamBlocker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AllBeamBlocker')},
  })
  AllBeamBlocker.Base.__init__(self, kwargs)
end

function AllBeamBlocker:onHit(hittingGameObject, hitName)
  -- no beams pass through.
  return true
end


local Destroyable = class.Class(component.Component)

function Destroyable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Destroyable')},
      {'hitNames', args.tableType},
      {'initialHealth', args.ge(0)},
      {'damagedHealthLevel', args.positive},
  })
  Destroyable.Base.__init__(self, kwargs)
  self._config.hitNames = set.Set(kwargs.hitNames)
  self._config.initialHealth = kwargs.initialHealth
  self._config.damagedHealthLevel = kwargs.damagedHealthLevel
end

function Destroyable:reset()
  self._health = self._config.initialHealth
end

function Destroyable:onHit(hittingGameObject, hitName)
  if self._config.hitNames[hitName] then
    self._health = self._health - 1
    if self._health == self._config.damagedHealthLevel then
      self.gameObject:setState('damaged')
    end
    if self._health <= 0 then
      self.gameObject:setState('destroyed')
      -- beams do pass through when they destroy the object.
      return false
    end
  end
  -- no beams pass through.
  return true
end


local Ground = class.Class(component.Component)

function Ground:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Ground')},
      {'teamNames', args.tableType},
  })
  Ground.Base.__init__(self, kwargs)
  self._teamNames = set.Set(kwargs.teamNames)
  self._teamNamesArray = kwargs.teamNames
end

function Ground:registerUpdaters(updaterRegistry)
  local transformComponent = self.gameObject:getComponent('Transform')
  local function preventMovementOverOpposingTeamColor()
    -- Get avatar standing on ground or nil.
    local avatarObject = transformComponent:queryPosition('upperPhysical')
    if avatarObject then
      local groundTeam = self.gameObject:getState()
      local avatarTeam = avatarObject:getComponent('TeamMember'):getTeam()
      if groundTeam ~= avatarTeam then
        -- Prevent avatars on ground claimed by opponent team from moving.
        avatarObject:getComponent('Avatar'):disallowMovement()
      else
        -- Avatars on ground claimed by own team can move.
        avatarObject:getComponent('Avatar'):allowMovement()
      end
    end
  end
  updaterRegistry:registerUpdater{
      updateFn = preventMovementOverOpposingTeamColor,
      -- Priority must be greater than avatar movement priority, which is 150.
      priority = 175,
      -- Note that teamNames should not include the "clean" (unclaimed) state.
      states = self._teamNamesArray
  }
end

function Ground:onHit(hittingGameObject, hittingTeam)
  -- Assume teamNames are identical to color states e.g. red, blue etc.
  if self._teamNames[hittingTeam] then
    self.gameObject:setState(hittingTeam)
    -- Beams always pass through.
    return false
  end
end


--[[ The `ColorZapper` component endows an avatar with the ability to fire a
colored beam.
]]
local ColorZapper = class.Class(component.Component)

function ColorZapper:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ColorZapper')},
      {'team', args.stringType},
      {'color', args.tableType},
      {'cooldownTime', args.numberType},
      {'beamLength', args.numberType},
      {'beamRadius', args.numberType},
      {'secondaryBeamCooldownTime', args.numberType},
      {'secondaryBeamLength', args.numberType},
      {'secondaryBeamRadius', args.numberType},
      {'aliveStates', args.tableType},
      {'groundLayer', args.default('alternateLogic'), args.stringType},
  })
  ColorZapper.Base.__init__(self, kwargs)

  self._config.team = kwargs.team
  self._config.color = kwargs.color
  self._config.cooldownTime = kwargs.cooldownTime
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius
  self._config.secondaryBeamCooldownTime = kwargs.secondaryBeamCooldownTime
  self._config.secondaryBeamLength = kwargs.secondaryBeamLength
  self._config.secondaryBeamRadius = kwargs.secondaryBeamRadius
  self._config.aliveStates = set.Set(kwargs.aliveStates)

  -- Ground layer is used to make the beam strike the location underneath the
  -- avatar in addition to the usual projected beam shape. This is needed so
  -- avatars can color the ground beneath their own feet in order to escape
  -- being stuck on their opposing team's color.
  self._config.groundLayer = kwargs.groundLayer

  self._config.beamLayer = 'beamZap_' .. self._config.team
  self._config.beamSprite = 'BeamZap_' .. self._config.team
end

function ColorZapper:addHits(worldConfig)
  worldConfig.hits[self._config.team] = {
      layer = self._config.beamLayer,
      sprite = self._config.beamSprite,
  }
  table.insert(worldConfig.renderOrder, self._config.beamLayer)
end

function ColorZapper:addSprites(tileSet)
  tileSet:addColor(self._config.beamSprite, self._config.color)
end

function ColorZapper:positionsAreEqual(tableA, tableB)
  if tableA[1] == tableB[1] and tableA[2] == tableB[2] then
    return true
  end
  return false
end

function ColorZapper:registerUpdaters(updaterRegistry)
  local transformComponent = self.gameObject:getComponent('Transform')
  local zap = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if self._config.aliveStates[self.gameObject:getState()] then
      if self._config.cooldownTime >= 0 then
        if self._coolingTimer > 0 then
          self._coolingTimer = self._coolingTimer - 1
        else
          if actions['fireZap'] == 1 then
            -- A short-range beam with a wide area of effect.
            self._coolingTimer = self._config.cooldownTime
            self.gameObject:hitBeam(
                self._config.team,
                self._config.beamLength,
                self._config.beamRadius
            )
            -- Also mark the avatar's current location.
            local groundObject = transformComponent:queryPosition(
              self._config.groundLayer)
            if groundObject and groundObject:hasComponent('Ground') then
              groundObject:getComponent('Ground'):onHit(self.gameObject,
                                                        self._config.team)
            end
          elseif actions['fireZap'] == 2 and
              self:positionsAreEqual(self._previousPosition,
                                     transformComponent:getPosition()) then
            -- A longer range beam with a thin area of effect.
            -- This beam can only be used if the player did not change its
            -- position on the previous frame (must stand still for one frame
            -- before it can be used).
            -- This beam takes twice as long to cool down after use before any
            -- beam can be used again (all beams share a cooling timer).
            self._coolingTimer = self._config.secondaryBeamCooldownTime
            self.gameObject:hitBeam(
                self._config.team,
                self._config.secondaryBeamLength,
                self._config.secondaryBeamRadius
            )
            -- Note: long-range zaps do not color the avatar's current location.
          end
        end
      end
      self._previousPosition = transformComponent:getPosition()
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = zap,
      priority = 140,
  }
end

function ColorZapper:reset()
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
  self._previousPosition = nil
end

function ColorZapper:readyToShoot()
  local normalizedTimeTillReady = self._coolingTimer / self._config.cooldownTime
  return 1 - normalizedTimeTillReady
end


--[[ The `ZappedByColor` component makes it so avatars can be hit by colored
beams, and will respawn after a time.
]]
local ZappedByColor = class.Class(component.Component)

function ZappedByColor:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ZappedByColor')},
      {'team', args.stringType},
      {'allTeamNames', args.tableType},
      {'framesTillRespawn', args.numberType},
      {'penaltyForBeingZapped', args.numberType},
      {'rewardForZapping', args.numberType},
      {'healthRegenerationRate', args.ge(0.0), args.le(1.0)},
      {'maxHealthOnGround', args.positive},
      {'maxHealthOnOwnColor', args.positive},
      {'maxHealthOnEnemyColor', args.positive},
      {'groundLayer', args.default('alternateLogic'), args.stringType},
  })
  ZappedByColor.Base.__init__(self, kwargs)

  self._config.team = kwargs.team
  self._config.allTeamNames = set.Set(kwargs.allTeamNames)
  self._config.framesTillRespawn = kwargs.framesTillRespawn
  self._config.penaltyForBeingZapped = kwargs.penaltyForBeingZapped
  self._config.rewardForZapping = kwargs.rewardForZapping
  self._config.healthRegenerationRate = kwargs.healthRegenerationRate
  self._config.maxHealthOnGround = kwargs.maxHealthOnGround
  self._config.maxHealthOnOwnColor = kwargs.maxHealthOnOwnColor
  self._config.maxHealthOnEnemyColor = kwargs.maxHealthOnEnemyColor
  self._config.groundLayer = kwargs.groundLayer
end

function ZappedByColor:_getColorHere()
  local maybeGameObject = self.gameObject:getComponent(
      'Transform'):queryPosition(self._config.groundLayer)
  if maybeGameObject and maybeGameObject:hasComponent('Ground') then
    return maybeGameObject:getState()
  end
  return nil
end

function ZappedByColor:registerUpdaters(updaterRegistry)
  local aliveState = self:getAliveState()
  local waitState = self:getWaitState()

  local respawn = function()
    local spawnGroup = self.gameObject:getComponent('Avatar'):getSpawnGroup()
    self.gameObject:teleportToGroup(spawnGroup, aliveState)
    self.playerRespawnedThisStep = true
    self._health = self._config.maxHealthOnGround
    self.gameObject:getComponent('Avatar'):allowMovement()
  end

  local function _incrementHealthIfBelowThreshold(threshold)
    if self._health < threshold then
      self._health = math.min(self._health + 1, threshold)
    end
  end

  local regenerateHealth = function()
    local colorAtThisPosition = self:_getColorHere()
    if self._config.allTeamNames[colorAtThisPosition] then
      if colorAtThisPosition == self._config.team then
        -- Only modify health if it is lower than max for own color.
        _incrementHealthIfBelowThreshold(self._config.maxHealthOnOwnColor)
      else
        -- Only modify health if it is lower than its max for enemy color.
        _incrementHealthIfBelowThreshold(self._config.maxHealthOnEnemyColor)
      end
    else
      -- Only modify health if it is lower than its max for ground location.
      _incrementHealthIfBelowThreshold(self._config.maxHealthOnGround)
    end
  end

  local function updateHealthState()
    local state = self.gameObject:getState()
    if state ~= self:getWaitState() and
        self._health ~= self._healthLastFrame then
      self.gameObject:setState('health' .. tostring(self._health))
    end

    self._healthLastFrame = self._health

    self.playerRespawnedThisStep = false
    self.zapperIndex = nil
  end

  updaterRegistry:registerUpdater{
      updateFn = respawn,
      priority = 135,
      state = waitState,
      startFrame = self._config.framesTillRespawn
  }

  updaterRegistry:registerUpdater{
      updateFn = regenerateHealth,
      priority = 2,
      probability = self._config.healthRegenerationRate
  }

  updaterRegistry:registerUpdater{
        updateFn = updateHealthState,
        priority = 1,
    }
end

function ZappedByColor:onStateChange()
  self._respawnTimer = self._config.framesTillRespawn
end

function ZappedByColor:reset()
  self.playerRespawnedThisStep = false
  self._health = self._config.maxHealthOnGround
  self._healthLastFrame = self._config.maxHealthOnGround
end

function ZappedByColor:onHit(hittingGameObject, hitName)
  if self._config.allTeamNames[hitName] and hitName ~= self._config.team then
    self._health = self._health - 1
    if self._health <= 0 then
      local zappedAvatar = self.gameObject:getComponent('Avatar')
      local zappedIndex = zappedAvatar:getIndex()
      local zapperAvatar = hittingGameObject:getComponent('Avatar')
      local zapperIndex = zapperAvatar:getIndex()
      if self.playerZapMatrix then
        self.playerZapMatrix(zappedIndex, zapperIndex):add(1)
      end
      events:add('zap', 'dict',
                 'source', zapperAvatar:getIndex(),  -- int
                 'target', zappedAvatar:getIndex())  -- int
      zappedAvatar:addReward(self._config.penaltyForBeingZapped)
      if hittingGameObject:hasComponent('Taste') then
        hittingGameObject:getComponent('Taste'):zap(zappedAvatar.gameObject)
      else
        zapperAvatar:addReward(self._config.rewardForZapping)
      end
      -- Remove the player who was hit (temporarily).
      self.gameObject:setState(self:getWaitState())
      -- Temporarily store the index of the zapper avatar in state so it can
      -- be observed elsewhere.
      self.zapperIndex = zapperIndex
      -- return `true` to prevent the beam from passing through a hit player.
      return true
    end
  end
end

function ZappedByColor:start()
  local scene = self.gameObject.simulation:getSceneObject()
  self.playerZapMatrix = nil
  if scene:hasComponent("GlobalMetricHolder") then
    self.playerZapMatrix = scene:getComponent(
        "GlobalMetricHolder").playerZapMatrix
  end
end

function ZappedByColor:getAliveState()
  return self.gameObject:getComponent('Avatar'):getAliveState()
end

function ZappedByColor:getWaitState()
  return self.gameObject:getComponent('Avatar'):getWaitState()
end


local allComponents = {
    -- Object components.
    AllBeamBlocker = AllBeamBlocker,
    Destroyable = Destroyable,
    Ground = Ground,

    -- Avatar components.
    ColorZapper = ColorZapper,
    ZappedByColor = ZappedByColor,
}

component_registry.registerAllComponents(allComponents)

return allComponents