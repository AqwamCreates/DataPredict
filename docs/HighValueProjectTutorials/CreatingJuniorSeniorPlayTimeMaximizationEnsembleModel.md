# Creating Junior-Senior Play Time Maximization Ensemble Model

## High-Level Explanation

* Our Tabular (Junior) and Deep (Senior) Play Time Maximization Models will gather states and updates at the same time.

* Should the junior chooses "ConsultSenior" action, the senior will have a look at the states more closely and produce a more fine-grained action prediction.

* The junior can choose to be more independent by setting the previous "ConsultSenior" action to whatever the senior's action have chosen.

## Code

### Feature Vector, States List Classes List Design

```lua

-- This one is for our senior (deep) model.

local function getPlayerDataVector(Player)

  -- We have five features with one "bias".

  return {
    {
        1,
        numberOfCurrencyAmount,
        numberOfItemsAmount,
        timePlayedInCurrentSession,
        timePlayedInAllSessions,
        healthAmount
    }
  }

end

local SeniorClassesList = {

  "NoEvent",
  "FreeGiftEvent",
  "ResourceMultiplierEvent",
  "QuestEvent",
  "ItemSpawnEvent",
  "BossSpawnEvent",
  "LimitedTimeQuestEvent",
  "LimitedTimeItemSpawnEvent",
  "LimitedTimeBossSpawnEvent",

}

-- This one is for our junior (tabular) model.

local StatesList = {

  "Idle",
  "AwayFromKeyboard",
  "Exploring",
  "GatheringItems",
  "GoingToQuestLocation",
  "PerformingQuest",
  "AttackingEnemies",
  "AttackingResourceEntities",
  "Retreating",
  "Rewarded",
  "Left",
  "UnknownDisconnect",

}

local JuniorClassesList = table.copy(SeniorClassesList)

table.insert(JuniorClassesList, "ConsultSenior")

```

### Constructing Simple Play Time Maximization Model

```lua 

-- You can use Tabular SARSA here for safer learning. However, because our model is simple, it should be already be safe
local NeuralNetwork = DataPredict.Models.TabularQLearning.new({maximumNumberOfIterations = 1})

NeuralNetwork:setStatesList(StatesList)

NeuralNetwork:setClassesList(ClassesList)

-- This code shown below checks if we already have trained the models previously.

if (JuniorPlayTimeMaximizationModelParameters) then NeuralNetwork:setModelParameters(JuniorPlayTimeMaximizationModelParameters) end

local JuniorPlayTimeMaximizationModel = DataPredict.QuickSetups.CategoricalPolicy.new()

-- Inserting our Tabular Reinforcement Learning Model here.

JuniorPlayTimeMaximizationModel:setModel(SimplePlayTimeMaximizationModel)

```

### Constructing Deep Play Time Maximization Model

```lua 

local NeuralNetwork = DataPredict.Models.NeuralNetwork.new({maximumNumberOfIterations = 1})

NeuralNetwork:setClassesList(ClassesList)

NeuralNetwork:addLayer(5, true) -- Five features and one bias.

NeuralNetwork:addLayer(#PlayTimeMaximizationModelClassesList, false) -- No bias.

-- This code shown below checks if we already have trained the models previously.

if (SeniorPlayTimeMaximizationModelParameters) then NeuralNetwork:setModelParameters(SeniorPlayTimeMaximizationModelParameters) end

-- You can use deep Q-Learning here for faster learning. However, for more "safer" model, stick with deep SARSA.

local DeepReinforcementLearningModel = DataPredict.Model.DeepStateActionRewardStateAction.new()

-- Inserting our Neural Network here.

DeepReinforcementLearningModel:setModel(NeuralNetwork)

local SeniorPlayTimeMaximizationModel = DataPredict.QuickSetups.CategoricalPolicy.new()

-- Inserting our Deep Reinforcement Learning Model here.

SeniorPlayTimeMaximizationModel:setModel(DeepReinforcementLearningModel)

```

### Player Data Collection

```lua

local playerDataMatrix = {}
  
local recordedTimeArray = {}
  
local snapshotIndex = 1

local function getPlayerDataArray()

  return {1, numberOfCurrencyAmount, numberOfItemsAmount, timePlayedInCurrentSession, timePlayedInAllSessions, healthAmount}

end

local function getPlayerState()

-- Return whatever your player's current state based on external logic

end

```

### On Player Join

```lua

-- The switch here is for how often you want the junior to be reliant on the senior.

local isJuniorShouldBeIndepdendent = true

local eventFunctionDictionary = {

  ["NoEvent"] = nil,
  ["ResourceMultiplierEvent"] = resourceMultiplierEvent,
  ["QuestEvent"] = questEvent,
  ["ItemSpawnEvent"] = itemSpawnEvent,
  ["BossSpawnEvent"] = bossSpawnEvent,
  ["LimitedTimeQuestEvent"] = limitedTimeQuestEvent,
  ["LimitedTimeItemSpawnEvent"] = limitedTimeItemSpawnEvent,
  ["LimitedTimeBossSpawnEvent"] = limitedTimeBossSpawnEvent,

}

local function run(Player)

    local isPlayerInServer = true

    local rewardValue = 0

    local playerState

    local playerDataArray

    local playerDataVector

    local isSeniorConsulted

    local eventName

    local eventFunction

    while isPlayerInServer do

        playerState = getPlayerState(Player)

        playerDataArray = getPlayerDataArray(Player)

        playerDataVector = {playerDataArray}

        if (isJuniorShouldBeIndepdendent) and (isSeniorConsulted) then JuniorPlayTimeMaximizationModel.previousAction = eventName end

        eventName = JuniorPlayTimeMaximizationModel:reinforce(playerState, rewardValue)

        isSeniorConsulted = (eventName == "ConsultSenior")

        if (isSeniorConsulted) then

           eventName = SeniorPlayTimeMaximizationModel:reinforce(playerDataVector, rewardValue)

        else

            -- This is because we only use senior when consulted, and hence need to force setting previous action value.

             SeniorPlayTimeMaximizationModel.previousAction = eventName 

        end

        eventFunction = eventFunctionDictionary[eventName]

        if (eventFunction) then eventFunction() end

        task.wait(30)

        isPlayerInServer = checkIfPlayerIsInServer(Player)

        rewardValue = (isPlayerInServer and 20) or -100

    end

    playerState = getPlayerState(Player)

    playerDataArray = getPlayerDataArray(Player)

    playerDataVector = {playerDataArray}

    if (isJuniorShouldBeIndepdendent) and (isSeniorConsulted) then JuniorPlayTimeMaximizationModel.previousAction = eventName end

    SeniorPlayTimeMaximizationModel.previousAction = eventName 

    JuniorPlayTimeMaximizationModel:reinforce(playerState, rewardValue)

    SeniorPlayTimeMaximizationModel:reinforce(playerDataVector, rewardValue)

end

```

### On Player Leave

```lua

-- We then need to get our model parameters. If you only kept the quick setup and discarded the rest, don't worry!

--]]

local JuniorModelParameters = JuniorPlayTimeMaximizationModel:getModel():getModelParameters()

local SeniorModelParameters = SeniorPlayTimeMaximizationModel:getModel():getModelParameters()

```

That's all for today! See you later!
