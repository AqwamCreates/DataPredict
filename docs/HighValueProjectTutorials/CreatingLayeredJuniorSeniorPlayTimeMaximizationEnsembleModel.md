# Creating Gated Junior-Senior Play Time Maximization Ensemble Model

## High-Level Explanation

| First Layer                           | Final Layer                         |
|---------------------------------------|-------------------------------------|
| Probability-To-Leave Prediction Model | Deep Play Time Maximization Model   |
| Time-To-Leave Prediction Model        | Simple Play Time Maximization Model |

### First Layer

* Should the probability-to-leave be greater than 50% or "time-to-leave" is less than 5 seconds, it activates the "Play Time Maximization Model". For the latter metric, even if the "Probability-To-Leave Prediction Model" says the player is unlikely to leave, we still have a chance that the player will leave in near term within a short period of time and the effects of player leaving is generally permanent.

* Once "Play Time Maximization Model" chooses an event that it thinks it will increase play time, it will wait for the event's outcome based on "time-to-leave" value before receiving the rewards and update it.

* Unlike using "Play Time Maximization Model" by itself, introducing "probability-to-leave" value as a trigger allows a more controlled exploration for "Play Time Maximization Model" as the low "probability-to-leave" value and high "time-to-leave" gets ignored. As a result, a more risky intervention is only applied when players are likely to leave.

* The first-layer model provides a strong signal about player state. Feeding that state into the final layer means the "Play Time Maximization Model" learns in contextually meaningful situations, which improves its long-term performance.

* The "Time-To-Leave Prediction Model" is in the same layer as "Probability-To-Leave Prediction Model" because we want it to constantly update on how long the player will stay. If we were to put it between the first and final layer, the updates will be too sparse to predict accurate wait times for "Play Time Maximization Model".

### Final Layer

* Our Tabular (Junior) and Deep (Senior) Play Time Maximization Models will gather states and updates at the same time.

* The junior model tends to learn very fast due to its tabular nature. Meanwhile, The senior model will learn complex patterns between states and actions.

* Should the junior chooses "ConsultSenior" action, the senior will have a look at the states more closely and produce a more fine-grained action prediction.

* The junior can choose to be more independent by setting the previous "ConsultSenior" action to whatever the senior's action have chosen.

* If the junior is set to be independent, the junior will rely on senior less over time. This is because of no reward is being received through the "ConsultSenior" action and its associated states.

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

local PlayerStatesList = {

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

### Creating Time-To-Leave And Probability-To-Leave Prediction Models

```lua

local TimeToLeavePredictionModel = DataPredict.Models.LinearRegression.new({maximumNumberOfIterations = 100, learningRate = 0.01})

local ProbabilityToLeavePredictionModel = DataPredict.Models.LogisticRegression.new({maximumNumberOfIterations = 100, learningRate = 0.01})

-- The code shown below checks if we already have trained the models previously.

if (TimeToLeavePredictionModelParameters) then TimeToLeavePredictionModel:setModelParameters(TimeToLeavePredictionModelParameters) end

if (ProbabilityToLeavePredictionModelParameters) then ProbabilityToLeavePredictionModel:setModelParameters(ProbabilityToLeavePredictionModelParameters) end

```

### Constructing Junior Play Time Maximization Model

```lua 

--[[

  You can use Tabular SARSA here for safer learning. However, because our model is simple, it should be already be safe.

  So, it is better to speed up our learning using Tabular Q-Learning.

--]]

local TabularReinforcementLearning = DataPredict.Models.TabularQLearning.new({maximumNumberOfIterations = 1})

NeuralNetwork:setStatesList(PlayerStatesList)

NeuralNetwork:setClassesList(JuniorClassesList)

-- This code shown below checks if we already have trained the models previously.

if (JuniorPlayTimeMaximizationModelParameters) then NeuralNetwork:setModelParameters(JuniorPlayTimeMaximizationModelParameters) end

local JuniorPlayTimeMaximizationModel = DataPredict.QuickSetups.CategoricalPolicy.new()

-- Inserting our Tabular Reinforcement Learning Model here.

JuniorPlayTimeMaximizationModel:setModel(TabularReinforcementLearning)

```

### Constructing Senior Play Time Maximization Model

```lua 

local NeuralNetwork = DataPredict.Models.NeuralNetwork.new({maximumNumberOfIterations = 1})

NeuralNetwork:setClassesList(SeniorClassesList)

NeuralNetwork:addLayer(5, true) -- Five features and one bias.

NeuralNetwork:addLayer(#SeniorClassesList, false) -- No bias.

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
  
local function snapshotData(playerDataArray)
  
  playerDataMatrix[snapshotIndex] = getPlayerDataArray()
  
  recordedTimeArray[snapshotIndex] = os.time()
  
  snapshotIndex = snapshotIndex + 1

end

```

### On Player Join

```lua

-- The switch here is for how often you want the junior to be reliant on the senior.

local isJuniorShouldBeIndependent = true

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

    local playerDataArray

    local playerState

    local playerDataVector

    local predictedTimeToLeave

    local predictedProbabilityToLeave

    local activatePlayTimeMaximization

    local isSeniorConsulted

    local juniorEventName

    local seniorEventName

    local finalEventName

    local eventFunction

    while isPlayerInServer do

        playerDataArray = getPlayerDataArray(Player)

        playerState = getPlayerState(Player)

        snapshotData(playerDataArray)

        playerDataVector = {playerDataArray}

        predictedTimeToLeave = TimeToLeavePredictionModel:predict(playerDataArray)[1][1]

        predictedProbabilityToLeave = ProbabilityToLeavePredictionModel:predict(playerDataArray)[1][1]

        activatePlayTimeMaximization = (predictedProbabilityToLeave >= 0.5) or (predictedTimeToLeave <= 5)

        if (activatePlayTimeMaximization) then

            juniorEventName = JuniorPlayTimeMaximizationModel:reinforce(playerState, rewardValue)

            seniorEventName = SeniorPlayTimeMaximizationModel:reinforce(playerDataVector, rewardValue)

            isSeniorConsulted = (eventName == "ConsultSenior")

            if (isSeniorConsulted) then

                finalEventName = seniorEventName

                if (isJuniorShouldBeIndependent) then JuniorPlayTimeMaximizationModel.previousAction = seniorEventName end

            else

            finalEventName = juniorEventName

            SeniorPlayTimeMaximizationModel.previousAction = juniorEventName 

        end

        task.wait(predictedTimeToLeave)

        isPlayerInServer = checkIfPlayerIsInServer(Player)

        if (activatePlayTimeMaximization) then

          -- Player leaving the game is more of a "rarer" and "extremely undesirable" event, therefore a very large negative value is used.

          rewardValue = (isPlayerInServer and 20) or -100

        end

    end

    playerDataArray = getPlayerDataArray(Player)

    playerDataVector = {playerDataArray}

    playerState = getPlayerState(Player)

    playerDataVector = {playerDataArray}

    JuniorPlayTimeMaximizationModel:reinforce(playerState, rewardValue)

    SeniorPlayTimeMaximizationModel:reinforce(playerDataVector, rewardValue)


end

```

### On Player Leave

```lua

local timeToLeaveVector = {}

local probabilityToLeaveVector = {}

for i = 1, snapshotIndex, 1 do

  local timeToLeave = os.time() - recordedTime[i]

  -- To ensure that this does not result in negative probabilityToLeave value if we're using sigmoid function for our logistic regression.

  local clampedTimeToLeave = math.max(timeToLeave, 1)

  local probabilityToLeave = 1 - (1 / clampedTimeToLeave)

  timeToLeaveVector[i] = {timeToLeave}

  probabilityToLeaveVector[i] = {probabilityToLeave}

end

TimeToLeavePredictionModel:train(playerDataMatrix, timeToLeaveVector)

ProbabilityToLeavePredictionModel:train(playerDataMatrix, probabilityToLeaveVector)

-- Just getting our model parameters to save them

local TimeToLeavePredictionModelParameters = TimeToLeavePredictionModel:getModelParameters()

local ProbabilityToLeavePredictionModelParameters = ProbabilityToLeavePredictionModel:getModelParameters()

local ModelParameters = JuniorPlayTimeMaximizationModel:getModel():getModelParameters()

local ModelParameters = SeniorPlayTimeMaximizationModel:getModel():getModelParameters()

```

That's all for today! See you later!
