# Creating Play Time Maximization Ensemble Model

## High-Level Explanation

| First Layer                           | Final Layer                  |
|---------------------------------------|------------------------------|
| Time-To-Leave Prediction Model        | Play Time Maximization Model |
| Probability-To-Leave Prediction Model |                              |

* Should the probability-to-leave be greater than 50%, it activates the "Play Time Maximization Model".

* Once "Play Time Maximization Model" chooses an event that it thinks it will increase play time, it will wait for the event's outcome based on "time-to-leave" value before receiving the rewards and update it.

* Unlike using "Play Time Maximization Model" by itself, introducing "probability-to-leave" value as a trigger allows a more controlled exploration for "Play Time Maximization Model" as the lower "probability-to-leave" value gets ignored. As a result, a more risky intervention is only applied when players are likely to leave.

* The first-layer model provides a strong signal about player state. Feeding that state into the final layer means the "Play Time Maximization Model" learns in contextually meaningful situations, which improves its long-term performance.

* The "Time-To-Leave Prediction Model" is in the same layer as "Probability-To-Leave Prediction Model" because we want it to constantly update on how long the player will stay. If we were to put it between the first and final layer, the updates will be too sparse to predict accurate wait times for "Play Time Maximization Model".

## Code

### Feature Vector And Classes List Design

```lua

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

local PlayTimeMaximizationModelClassesList = {

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

### Creating Time-To-Leave And Probability-To-Leave Prediction Models

```lua

local TimeToLeavePredictionModel = DataPredict.Models.LinearRegression.new({maximumNumberOfIterations = 100, learningRate = 0.3})

local ProbabilityToLeavePredictionModel = DataPredict.Models.LogisticRegression.new({maximumNumberOfIterations = 100, learningRate = 0.3})

-- The code shown below checks if we already have trained the models previously.

if (TimeToLeavePredictionModelParameters) then TimeToLeavePredictionModel:setModelParameters(TimeToLeavePredictionModelParameters) end

if (ProbabilityToLeavePredictionModelParameters) then ProbabilityToLeavePredictionModel:setModelParameters(ProbabilityToLeavePredictionModelParameters) end

```

### Constructing Play Time Maximization Model

```lua 

local NeuralNetwork = DataPredict.Model.NeuralNetwork.new({maximumNumberOfIterations = 1})

NeuralNetwork:setClassesList(ClassesList)

NeuralNetwork:addLayer(5, true) -- Five features and one bias.

NeuralNetwork:addLayer(#PlayTimeMaximizationModelClassesList, false) -- No bias.

-- This code shown below checks if we already have trained the models previously.

if (PlayTimeMaximizationModelParameters) then NeuralNetwork:setModelParameters(PlayTimeMaximizationModelParameters) end

-- You can use deep Q-Learning here for faster learning. However, for more "safer" model, stick with deep SARSA.

local DeepReinforcementLearningModel = DataPredict.Model.DeepStateActionRewardStateAction.new()

-- Inserting our Neural Network here.

DeepReinforcementLearningModel:setModel(NeuralNetwork)

local PlayTimeMaximizationModel = DataPredict.QuickSetups.CategoricalPolicy.new()

-- Inserting our Deep Reinforcement Learning Model here.

PlayTimeMaximizationModel:setModel(DeepReinforcementLearningModel)

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
  
 playerDataMatrix[snapshotIndex] = playerDataArray
  
  recordedTimeArray[snapshotIndex] = os.time()
  
  snapshotIndex = snapshotIndex + 1

end

```

### While Player On Server

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

  probabilityToLeaveVector[i] = {clampedTimeToLeave}

end

TimeToLeavePredictionModel:train(playerDataMatrix, timeToLeaveVector)

ProbabilityToLeavePredictionModel:train(playerDataMatrix, probabilityToLeaveVector)

-- Just getting our model parameters to save them

TimeToLeavePredictionModelParameters = TimeToLeavePredictionModel:getModelParameters(true)

ProbabilityToLeavePredictionModelParameters = ProbabilityToLeavePredictionModel:getModelParameters(true)

--[[ 

We then need to get our Neural Network model from the "Play Time Maximization Model". If you only kept the quick setup and discarded the rest, don't worry!

We can just do getModel() twice to get our Neural Network model.

--]]

local DeepReinforcementLearningModel =  PlayTimeMaximizationModel:getModel()

local NeuralNetwork = DeepReinforcementLearningModel:getModel()

-- Notice that we must get it from the Neural Network model.

ModelParameters = NeuralNetwork:getModelParameters()

-- Notice that we must set it to the Neural Network model too.

NeuralNetwork:setModelParameters(ModelParameters)

```
