# Creating Junior-Senior Play Time Maximization Ensemble Model

# Creating Layered Deep Play Time Maximization Ensemble Model

## High-Level Explanation

| First Layer                         | Final Layer     |
|-------------------------------------|-----------------|
| Deep Play Time Maximization Model   | None            |
| Simple Play Time Maximization Model |                 |

* Our deep and tabular Play Time Maximization Model

## Code

### Feature Vector, States List Classes List Design

```lua

-- This one is for our junior (tabular) model.

local StatesList = {

  "",
  "Idle",
  "Active",
  "QuestEvent",
  "ItemSpawnEvent",
  "BossSpawnEvent",
  "LimitedTimeQuestEvent",
  "LeftAfterReward",
  "Left",

}

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

-- This is for both our senior and junior models.

local PlClassesList = {

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

    local eventName

    local eventFunction

    while isPlayerInServer do

        playerState = getPlayerState(Player)

        playerDataArray = getPlayerDataArray(Player)

        snapshotData(playerDataArray)

        playerDataVector = {playerDataArray}

        predictedTimeToLeave = TimeToLeavePredictionModel:predict(playerDataArray)[1][1]

        predictedProbabilityToLeave = ProbabilityToLeavePredictionModel:predict(playerDataArray)[1][1]

        activatePlayTimeMaximization = (predictedProbabilityToLeave >= 0.5) or (predictedTimeToLeave <= 5)

        eventName = PlayTimeMaximizationModel:reinforce(playerDataVector, rewardValue)

        eventFunction = eventFunctionDictionary[eventName]

        if (eventFunction) then eventFunction() end


        task.wait(30)

        isPlayerInServer = checkIfPlayerIsInServer(Player)

        if (activatePlayTimeMaximization) then

          -- Player leaving the game is more of a "rarer" and "extremely undesirable" event, therefore a very large negative value is used.

          rewardValue = (isPlayerInServer and 20) or -100

        end

    end

end

```

### On Player Leave

```lua

We then need to get our model parameters. If you only kept the quick setup and discarded the rest, don't worry!

--]]

local JuniorModelParameters = JuniorPlayTimeMaximizationModel:getModelParameters()

local NeuralNetwork = DeepReinforcementLearningModel:getModel()

-- Notice that we must get it from the Neural Network model.

ModelParameters = NeuralNetwork:getModelParameters()

```

That's all for today! See you later!
