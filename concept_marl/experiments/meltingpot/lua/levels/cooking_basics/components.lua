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

-- Extends meltingpot/lua/levels/collaborative_cooking/components.lua

local args = require 'common.args'
local class = require 'common.class'
local helpers = require 'common.helpers'
local set = require 'common.set'
local log = require 'common.log'
local events = require 'system.events'
local random = require 'system.random'
local tensor = require 'system.tensor'
local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

--[[ The `InteractBeam` component endows an avatar with the ability to fire a
1 length, 0 radius beam to interact with other objects that they are facing.
]]
local InteractBeam = class.Class(component.Component)

function InteractBeam:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('InteractBeam')},
      {'cooldownTime', args.numberType},
      {'shapes', args.tableType},
      {'palettes', args.tableType},
  })
  InteractBeam.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.shape = kwargs.shapes[1]
  self._config.palette = kwargs.palettes[1]

  self._inventoryObject = nil
end

function InteractBeam:awake()
  self.hitAndSpriteName = 'interact_' .. self.gameObject:getUniqueState()
end

function InteractBeam:getHitName()
  return self.hitAndSpriteName
end

function InteractBeam:setInventoryObject(obj)
  self._inventoryObject = obj
end

function InteractBeam:getAvatarsInventory()
  return self._inventoryObject
end

function InteractBeam:addHits(worldConfig)
  worldConfig.hits[self.hitAndSpriteName] = {
      layer = self.hitAndSpriteName,
      sprite = self.hitAndSpriteName,
  }
  table.insert(worldConfig.renderOrder, self.hitAndSpriteName)
end

function InteractBeam:addSprites(tileSet)
  tileSet:addShape(self.hitAndSpriteName,
                   {palette = self._config.palette,
                    text = self._config.shape,
                    noRotate = true})
end

function InteractBeam:registerUpdaters(updaterRegistry)
  local interact = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if self._config.cooldownTime >= 0 then
      if self._coolingTimer > 0 then
        self._coolingTimer = self._coolingTimer - 1
      else
        if actions['interact'] == 1 then
          self._coolingTimer = self._config.cooldownTime
          self.gameObject:hitBeam(self.hitAndSpriteName, 1, 0)
        end
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = interact,
      priority = 140,
  }
end

--[[ Reset gets called just before start regardless of whether we have a new
environment instance or one that was reset by calling reset() in python.]]
function InteractBeam:reset()
  local kwargs = self._kwargs
  self._playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
end


--[[ A container component which can temporarily/permanently hold items through
an attached inventory (e.g. received from an avatar or given to an avatar), and
then give that interacting avatar a reward.
]]
local Container = class.Class(component.Component)

function Container:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Container')},
      {'startingItem', args.default('empty')},
      {'infinite', args.default(false)},
      {'reward', args.default(0)},
  })
  Container.Base.__init__(self, kwargs)

  self._config.startingItem = kwargs.startingItem
  self._config.infinite = kwargs.infinite
  self._config.reward = kwargs.reward
  self._inventory = nil
  self._usedThisStep = false
end

function Container:onHit(hittingGameObject, hitName)
  -- Assume nothing will send a hit that doesn't also have InteractBeam.
  local hitName =
      hittingGameObject:getComponent('InteractBeam'):getHitName()
  if hitName == hitName and not self._usedThisStep then
    self._usedThisStep = true
    -- Local variables.
    local avatar = hittingGameObject:getComponent('Avatar')
    local avatarsInventory = (
      hittingGameObject:getComponent('InteractBeam'):getAvatarsInventory()
    )
    local avatarsHeldItem = avatarsInventory:getHeldItem()
    local containersHeldItem = self._inventory:getHeldItem()
    -- Move item between contain and avatar if only one is empty.
    if containersHeldItem ~= 'empty' and avatarsHeldItem == 'empty' then
      -- Pick up item from container to avatar.
      avatarsInventory:setHeldItem(containersHeldItem)
      if not self._config.infinite then
        self._inventory:setHeldItem("empty")
      end
    elseif containersHeldItem == 'empty' and avatarsHeldItem ~= 'empty' then
      -- Move item from avatar to container.
      avatarsInventory:setHeldItem('empty')
      self._inventory:setHeldItem(avatarsHeldItem)
    end
  end
end

function Container:attachInventory(inventory)
  self._inventory = inventory
  inventory:setHeldItem(self._config.startingItem)
end

function Container:registerUpdaters(updaterRegistry)
  local tick = function()
      self._usedThisStep = false
  end

  updaterRegistry:registerUpdater{
      updateFn = tick,
      priority = 140,
  }
end


--[[ An inventory component which holds an item and visualises it. Can be
connected to a player's avatar or placed over another game object.
]]
local Inventory = class.Class(component.Component)

function Inventory:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Inventory')},
      -- `playerIndex` (int): player index for the avatar to connect to.
      {'playerIndex', args.numberType},
      {'emptyState', args.stringType},
      {'waitState', args.stringType},
  })
  Inventory.Base.__init__(self, kwargs)

  self._kwargs = kwargs
  self._config.layerName = 'inventoryLayer'
  self._emptyState = kwargs.emptyState
  self._waitState = kwargs.waitState
end

--[[ Reset gets called just before start regardless of whether we have a new
environment instance or one that was reset by calling reset() in python.]]
function Inventory:reset()
  local kwargs = self._kwargs
  self._playerIndex = kwargs.playerIndex
end

function Inventory:_placeAtCorrectLocation()
  -- It is essential to set the state before teleporting it.
  -- This is because pieces with no assigned layer have no position, and thus
  -- cannot be teleported.
  local avatarComponent = self._avatarObject:getComponent('Avatar')
  avatarComponent:disconnect(self.gameObject)
  self.gameObject:setState(self._emptyState)
  self.gameObject._grid:teleport(self.gameObject:getPiece(),
                                 self._avatarObject:getPosition())
  avatarComponent:connect(self.gameObject)
  self.gameObject:setOrientation(self._avatarObject:getOrientation())
end

--[[ postStart is called from the avatar manager, after start has been
called on all other game objects, even avatars.]]
function Inventory:postStart()
  local sim = self.gameObject.simulation
  if self._playerIndex ~= -1 then
    -- Store a reference to the connected avatar object.
    self._avatarObject = sim:getAvatarFromIndex(self._playerIndex)
    -- Teleport the hand to its correct location.
    self:_placeAtCorrectLocation()
    -- Get the avatar game object and connect to it.
    local avatarComponent = self._avatarObject:getComponent('Avatar')
    avatarComponent:connect(self.gameObject)
    self._avatarObject:getComponent('InteractBeam'):setInventoryObject(self)
  else
    self.gameObject:setState(self._emptyState)
    local underlyingObjects = (
      self.gameObject:getComponent('Transform'):queryDisc("upperPhysical", 0)
    )
    for _, underlyingObject in pairs(underlyingObjects) do
      underlyingObject:getComponent('Container'):attachInventory(self)
    end
  end
end

function Inventory:getHeldItem()
  local state = self.gameObject:getState()
  if string.match(state, "_offset") then
    return string.sub(state, 1, -8)
  else
    return state
  end
end

function Inventory:setHeldItem(item)
  if self._playerIndex ~= -1 then
    self.gameObject:setState(item .. '_offset')
  else
    self.gameObject:setState(item)
  end
end

function Inventory:getAliveState()
  return self._aliveState
end

function Inventory:getWaitState()
  return self._waitState
end

function Inventory:getPlayerIndex()
  return self._playerIndex
end


--[[ A receiver component which accepts items from the avatar and gives the
avatar a reward (or all avatars if globalReward=True).
]]
local Receiver = class.Class(component.Component)

function Receiver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Receiver')},
      {'acceptedItems', args.default('onion')},
      {'reward', args.default(0)},
      {'globalReward', args.default(false)},
  })
  Receiver.Base.__init__(self, kwargs)

  self._kwargs = kwargs
  self._config.acceptedItems = kwargs.acceptedItems
  self._config.reward = kwargs.reward
  self._config.globalReward = kwargs.globalReward
  self._itemStorage = nil
end

function Receiver:onHit(hittingGameObject, hitName)
  -- Assume nothing will send a hit that doesn't also have InteractBeam.
  local hitName =
      hittingGameObject:getComponent('InteractBeam'):getHitName()
  if hitName == hitName then
    -- Local variables.
    local avatar = hittingGameObject:getComponent('Avatar')
    local avatarsInventory = (
      hittingGameObject:getComponent('InteractBeam'):getAvatarsInventory()
    )
    local avatarsHeldItem = avatarsInventory:getHeldItem()
    -- If receiver accepts held item, move it from avatar to receiver & reward.
    if avatarsHeldItem == self._config.acceptedItems then
      if self._config.globalReward then
        local allAvatars = self.gameObject._grid:groupShuffledWithCount(
          random, 'players',
          self.gameObject._grid:groupCount('players'))
        for i, avatarPiece in ipairs(allAvatars) do
          self.gameObject.simulation:getGameObjectFromPiece(
            avatarPiece):getComponent('Avatar'):addReward(self._config.reward)
        end
      else
        avatar:addReward(self._config.reward)
      end
      avatarsInventory:setHeldItem('empty')
      -- Record that the receiver took the item offered by the player.
      events:add('receiver_accepted_item', 'dict',
                 'player_index', avatar:getIndex(),  -- int
                 'receiver', self.name,  -- string
                 'item', avatarsHeldItem) -- string
    end
  end
end


--[[ A cooking pot component which accepts up to three items, visualises them,
cooks them for a number of updates, and then changes to a cooked state after
'cookingTime' updates. This cooked item can then be collected by the avatar.
]]
local CookingPot = class.Class(component.Component)

function CookingPot:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('CookingPot')},
      {'acceptedItems', args.default({'onion', 'tomato'})},
      {'yieldedItems', args.default('soup')},
      {'reward', args.default(0)},
      {'cookingTime', args.default(20)},
      {'customStateNames', args.default({}), args.tableType},
  })
  CookingPot.Base.__init__(self, kwargs)

  self._kwargs = kwargs
  self._config.acceptedItems = set.Set(kwargs.acceptedItems)
  self._config.reward = kwargs.reward
  self._config.cookingTime = kwargs.cookingTime
  self._config.stateNames = kwargs.customStateNames
end

--[[ Reset gets called just before start regardless of whether we have a new
environment instance or one that was reset by calling reset() in python.]]
function CookingPot:reset()
  self._containedItems = {}
  self._currentCookingTime = 0
  self._cooked = false
end

function CookingPot:getCookingTime()
  return self._currentCookingTime
end

function CookingPot:isCooked()
  return self._cooked
end

function CookingPot:onHit(hittingGameObject, hitName)
  -- Assume nothing will send a hit that doesn't also have InteractBeam.
  local hitName =
      hittingGameObject:getComponent('InteractBeam'):getHitName()
  -- When the avatar interacts with the cooking pot, move items if applicable.
  if hitName == hitName then
    -- Local variables.
    local avatar = hittingGameObject:getComponent('Avatar')
    local avatarsInventory = (
      hittingGameObject:getComponent('InteractBeam'):getAvatarsInventory()
    )
    local avatarsHeldItem = avatarsInventory:getHeldItem()
    local itemsInPot = #self._containedItems
    if self._config.acceptedItems[avatarsHeldItem] and itemsInPot < 3 then
      -- Drop onion from avatar to cooking pot.
      table.insert(self._containedItems, avatarsHeldItem)
      avatar:addReward(self._config.reward)
      avatarsInventory:setHeldItem('empty')
      -- Record that the avatar dropped an item into the pot.
      events:add('item_dropped_into_pot', 'dict',
                 'player_index', avatar:getIndex(),  -- int
                 'pot', self.name,  -- string
                 'item', avatarsHeldItem) -- string
    elseif avatarsHeldItem == 'dish' and self._cooked then
      -- Collect soup from cooking pot.
      local cookedItem = 'soup'
      avatar:addReward(self._config.reward)
      avatarsInventory:setHeldItem(cookedItem)
      self._containedItems = {}
      self._cooked = false
      self._currentCookingTime = 0
      -- Record that the avatar collected a cooked product from the pot.
      events:add('cooked_food_collected_from_pot', 'dict',
                 'player_index', avatar:getIndex(),  -- int
                 'pot', self.name,  -- string
                 'cooked_item', cookedItem) -- string
    end
    -- Update state based on items on in the pot.
    if not self._cooked then
      local newStateName = ''
      local lookingFor
      if #self._containedItems == 0 then
        lookingFor = 'empty_empty_empty'
      elseif #self._containedItems == 1 then
        lookingFor = self._containedItems[1] .. '_empty_empty'
      elseif #self._containedItems == 2 then
        lookingFor = (self._containedItems[1] .. '_' .. self._containedItems[2]
          .. '_empty')
      elseif #self._containedItems == 3 then
        lookingFor = (self._containedItems[1] .. '_' .. self._containedItems[2]
          .. '_' .. self._containedItems[3])
      end
      -- Find the appropriate next state given the items in the pot.
      for _, stateName in ipairs(self._config.stateNames) do
        if string.match(stateName, lookingFor) then
          newStateName = stateName
          break
        end
      end
      self.gameObject:setState(newStateName)
    end
  end
end

function CookingPot:registerUpdaters(updaterRegistry)
  -- If the cooking pot contains 3 items, increment cooking time until cooked.
  local tickPotFn = function()
    if #self._containedItems == 3 and self._cooked == false then
      if self._currentCookingTime == self._config.cookingTime then
        self._cooked = true
        local newStateName = ''
        for _, stateName in ipairs(self._config.stateNames) do
          if string.match(stateName, 'cooked') then
            newStateName = stateName
            break
          end
        end
        self.gameObject:setState(newStateName)
      end
      self._currentCookingTime = self._currentCookingTime + 1
    end
  end

  updaterRegistry:registerUpdater{
    updateFn = tickPotFn,
    priority = 140,
  }
end


--[[ A component which visualises a loading bar based on the underlying
state of the cooking pot it is attached to.]]
local LoadingBarVisualiser = class.Class(component.Component)

function LoadingBarVisualiser:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('LoadingBarVisualiser')},
      {'totalTime', args.default(10)},
      {'customStateNames', args.default({}), args.tableType},
  })
  self.Base.__init__(self, kwargs)

  self._config.stateNames = kwargs.customStateNames
  self._config.totalTime = kwargs.totalTime
  self._barIntervals = self._config.totalTime / 10  -- Assumes 10 visual ticks.
end

function LoadingBarVisualiser:registerUpdaters(updaterRegistry)
  local tickLoadingBarFn = function()
    local underlyingObjects = (
      self.gameObject:getComponent('Transform'):queryDisc("upperPhysical", 0)
      )
    local cookingTime = 0
    local cooked = false
    for _, underlyingObject in pairs(underlyingObjects) do
      cookingTime = underlyingObject:getComponent('CookingPot'):getCookingTime()
      cooked = underlyingObject:getComponent('CookingPot'):isCooked()
    end

    local idx = math.floor(cookingTime / self._barIntervals) + 1
    local newStateName = self._config.stateNames[idx]
    self.gameObject:setState(newStateName)
  end

  updaterRegistry:registerUpdater{
    updateFn = tickLoadingBarFn,
    priority = 140,
  }
end

local GlobalStateTracker = class.Class(component.Component)

function GlobalStateTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GlobalStateTracker')},
      {'numPlayers', args.numberType},
      {'numTomatoes', args.numberType},
      {'numDishes', args.numberType},
      {'numCookingPots', args.numberType},
      {'numDeliveryLocations', args.numberType},
      {'numNavigationLocations', args.numberType},
      {'numLooseObjects', args.numberType},
      {'numCounters', args.numberType},
  })
  GlobalStateTracker.Base.__init__(self, kwargs)

  self._config.numPlayers = kwargs.numPlayers
  self._config.orientationMap = {N = 0, E = 1, S = 2, W = 3}
  self._config.numTomatoes = kwargs.numTomatoes
  self._config.numDishes = kwargs.numDishes
  self._config.numCookingPots = kwargs.numCookingPots
  self._config.numDeliveryLocations = kwargs.numDeliveryLocations
  self._config.numNavigationLocations = kwargs.numNavigationLocations
  self._config.numLooseObjects = kwargs.numLooseObjects
  self._config.numCounters = kwargs.numCounters
end

function GlobalStateTracker:reset()
  -- avatar positions/orientations
  self.playerPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numPlayers, 2):fill(0)
  self.playerOrientations = tensor.Int32Tensor(self._config.numPlayers, self._config.numPlayers):fill(0)
  self.playerHasTomatos = tensor.Int32Tensor(self._config.numPlayers, self._config.numPlayers):fill(0)
  self.playerHasDishes = tensor.Int32Tensor(self._config.numPlayers, self._config.numPlayers):fill(0)
  self.playerHasSoups = tensor.Int32Tensor(self._config.numPlayers, self._config.numPlayers):fill(0)

  -- tomato positions
  self.tomatoPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numTomatoes, 2):fill(0)

  -- dish positions
  self.dishPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numDishes, 2):fill(0)

  -- cooking pot positions
  self.potPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numCookingPots, 2):fill(0)
  self.potTomatoCounts = tensor.Int32Tensor(self._config.numPlayers, self._config.numCookingPots):fill(0)
  self.potCookingProgress = tensor.DoubleTensor(self._config.numPlayers, self._config.numCookingPots):fill(0.0)

  -- delivery locations
  self.deliveryPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numDeliveryLocations, 2):fill(0)

  -- navigation locations
  self.navigationPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numNavigationLocations, 2):fill(0)

  -- loose objects (tomatoes, dishes, soups)
  -- keeps track of up to numLooseObjects of each object type
  -- (-1, -1) is placeholder value if no loose objects exist this time-step
  self.looseTomatoPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numLooseObjects, 2):fill(-1)
  self.looseDishPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numLooseObjects, 2):fill(-1)
  self.looseSoupPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numLooseObjects, 2):fill(-1)
  
  -- objects on counters (total # of each object type on counters)
  -- binary and not size limited like loose objects 
  self.counterTomatoes = tensor.Int32Tensor(self._config.numPlayers, self._config.numCounters):fill(0)
  self.counterDishes = tensor.Int32Tensor(self._config.numPlayers, self._config.numCounters):fill(0)
  self.counterSoups = tensor.Int32Tensor(self._config.numPlayers, self._config.numCounters):fill(0)
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

    -- init tomato positions
    local tomatoObjs = self.gameObject.simulation:getGameObjectsByName('tomato_dispenser')
    for tomatoIdx, tomatoObj in pairs(tomatoObjs) do
      local tomatoPosition = tomatoObj:getPosition()
      self.tomatoPositions(agentIdx, tomatoIdx):val(tomatoPosition)
    end

    -- init dish positions
    local dishObjs = self.gameObject.simulation:getGameObjectsByName('dish_dispenser')
    for dishIdx, dishObj in pairs(dishObjs) do
      local dishPosition = dishObj:getPosition()
      self.dishPositions(agentIdx, dishIdx):val(dishPosition)
    end

    -- init cooking pot positions
    local potObjs = self.gameObject.simulation:getGameObjectsByName('cooking_pot')
    for potIdx, potObj in pairs(potObjs) do
      local potPosition = potObj:getPosition()
      self.potPositions(agentIdx, potIdx):val(potPosition)
    end

    -- init delivery location positions
    local deliveryObjs = self.gameObject.simulation:getGameObjectsByName('delivery_location')
    for deliveryIdx, deliveryObj in pairs(deliveryObjs) do
      local deliveryPosition = deliveryObj:getPosition()
      self.deliveryPositions(agentIdx, deliveryIdx):val(deliveryPosition)
    end

    -- init navigation location positions
    local navigationObjs = self.gameObject.simulation:getGameObjectsByName('navigation_location')
    for navIdx, navigationObj in pairs(navigationObjs) do
      local navigationPosition = navigationObj:getPosition()
      self.navigationPositions(agentIdx, navIdx):val(navigationPosition)
    end
  end
end

function GlobalStateTracker:registerUpdaters(updaterRegistry)
  local updateStateInfo = function()
    self.looseTomatoPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numLooseObjects, 2):fill(-1)
    self.looseDishPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numLooseObjects, 2):fill(-1)
    self.looseSoupPositions = tensor.Int32Tensor(self._config.numPlayers, self._config.numLooseObjects, 2):fill(-1)

    -- update avatar
    for agentIdx = 1, self._config.numPlayers do
      local avatarObject = self.gameObject.simulation:getAvatarFromIndex(agentIdx)

      -- position/orientation
      for otherIdx = 1, self._config.numPlayers do
        -- position/orientation of other agents
        local otherAvatarObj = self.gameObject.simulation:getAvatarFromIndex(otherIdx)
        local otherPlayerPosition = otherAvatarObj:getPosition()
        local otherPlayerOrientation = otherAvatarObj:getOrientation()
        self.playerPositions(agentIdx, otherIdx):val(otherPlayerPosition)
        self.playerOrientations(agentIdx, otherIdx):val(self._config.orientationMap[otherPlayerOrientation])

        -- inventory of other agents
        local comps = otherAvatarObj:getComponent('Avatar'):getConnectedObjects()
        local inventoryItem = nil
        for _, comp in pairs(comps) do
          if comp.name == 'inventory' then
            inventoryItem = comp:getComponent('Inventory'):getHeldItem()
          end
        end

        if inventoryItem == 'tomato' then
          self.playerHasTomatos(agentIdx, otherIdx):val(1)
        else
          self.playerHasTomatos(agentIdx, otherIdx):val(0)
        end

        if inventoryItem == 'dish' then
          self.playerHasDishes(agentIdx, otherIdx):val(1)
        else
          self.playerHasDishes(agentIdx, otherIdx):val(0)
        end
        
        if inventoryItem == 'soup' then
          self.playerHasSoups(agentIdx, otherIdx):val(1)
        else
          self.playerHasSoups(agentIdx, otherIdx):val(0)
        end

      end

      -- update tomato positions
      local tomatoObjs = self.gameObject.simulation:getGameObjectsByName('tomato_dispenser')
      for tomatoIdx, tomatoObj in pairs(tomatoObjs) do
        local tomatoPosition = tomatoObj:getPosition()
        self.tomatoPositions(agentIdx, tomatoIdx):val(tomatoPosition)
      end

      -- update dish positions
      local dishObjs = self.gameObject.simulation:getGameObjectsByName('dish_dispenser')
      for dishIdx, dishObj in pairs(dishObjs) do
        local dishPosition = dishObj:getPosition()
        self.dishPositions(agentIdx, dishIdx):val(dishPosition)
      end

      -- update cooking pot positions
      local potObjs = self.gameObject.simulation:getGameObjectsByName('cooking_pot')
      for potIdx, potObj in pairs(potObjs) do
        -- position
        local potPosition = potObj:getPosition()
        self.potPositions(agentIdx, potIdx):val(potPosition)

        -- number of tomatoes in pot
        self.potTomatoCounts(agentIdx, potIdx):val(#potObj:getComponent('CookingPot')._containedItems)

        -- cooking progress
        local cookingTime = potObj:getComponent('CookingPot'):getCookingTime()
        local cookingProgress = math.max(0.0, (cookingTime-1)/20)
        self.potCookingProgress(agentIdx, potIdx):val(cookingProgress)
      end

      -- update delivery location positions
      local deliveryObjs = self.gameObject.simulation:getGameObjectsByName('delivery_location')
      for deliveryIdx, deliveryObj in pairs(deliveryObjs) do
        local deliveryPosition = deliveryObj:getPosition()
        self.deliveryPositions(agentIdx, deliveryIdx):val(deliveryPosition)
      end

      -- update navigation location positions
      local navigationObjs = self.gameObject.simulation:getGameObjectsByName('navigation_location')
      for navIdx, navigationObj in pairs(navigationObjs) do
        local navigationPosition = navigationObj:getPosition()
        self.navigationPositions(agentIdx, navIdx):val(navigationPosition)
      end

      -- update loose tomatoes/dishes/soups
      local counterObjs = self.gameObject.simulation:getGameObjectsByName('counter')
      local tomatoIdx = 1
      local dishIdx = 1
      local soupIdx = 1
      for counterIdx, counterObj in pairs(counterObjs) do
        local inventory = counterObj:getComponent('Container')._inventory
        local inventoryItem = inventory:getHeldItem()
        local counterPos = counterObj:getPosition()

        if inventoryItem == 'tomato' then
          self.counterTomatoes(agentIdx, counterIdx):val(1)

          -- add to tomato slot (if available)
          if tomatoIdx < self._config.numLooseObjects + 1 then
            self.looseTomatoPositions(agentIdx, tomatoIdx):val(counterPos)
            tomatoIdx = tomatoIdx + 1
          end
        else
          self.counterTomatoes(agentIdx, counterIdx):val(0)
        end

        if inventoryItem == 'dish' then
          self.counterDishes(agentIdx, counterIdx):val(1)

          -- add to dish slot (if available)
          if dishIdx < self._config.numLooseObjects + 1 then
            self.looseDishPositions(agentIdx, dishIdx):val(counterPos)
            dishIdx = dishIdx + 1
          end
        else
          self.counterDishes(agentIdx, counterIdx):val(0)
        end
        
        if inventoryItem == 'soup' then
          self.counterSoups(agentIdx, counterIdx):val(1)

          -- add to soup slot (if available)
          if soupIdx < self._config.numLooseObjects + 1 then
            self.looseSoupPositions(agentIdx, soupIdx):val(counterPos)
            soupIdx = soupIdx + 1
          end
        else
          self.counterSoups(agentIdx, counterIdx):val(0)
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

function GlobalStateTracker:getTomatoPositions()
  return self.tomatoPositions
end

function GlobalStateTracker:getDishPositions()
  return self.dishPositions
end

function GlobalStateTracker:getCookingPotPositions()
  return self.potPositions
end

function GlobalStateTracker:getDeliveryPositions()
  return self.deliveryPositions
end

function GlobalStateTracker:getNavigationPositions()
  return self.navigationPositions
end

function GlobalStateTracker:getPlayerTomatoInventories()
  return self.playerHasTomatos
end

function GlobalStateTracker:getPlayerTomatoInventories()
  return self.playerHasTomatos
end

function GlobalStateTracker:getPlayerDishInventories()
  return self.playerHasDishes
end

function GlobalStateTracker:getPlayerSoupInventories()
  return self.playerHasSoups
end

function GlobalStateTracker:getCookingPotTomatoCounts()
  return self.potTomatoCounts
end

local allComponents = {
  Inventory = Inventory,
  InteractBeam = InteractBeam,
  Container = Container,
  Receiver = Receiver,
  CookingPot = CookingPot,
  LoadingBarVisualiser = LoadingBarVisualiser,

  -- Scene components.
  GlobalStateTracker = GlobalStateTracker,
}

component_registry.registerAllComponents(allComponents)

return allComponents
