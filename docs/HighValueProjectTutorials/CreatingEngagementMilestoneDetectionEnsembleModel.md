# Creating Engagement Milestone Detection Ensemble Model

## High-Level Explanation

| First Layer                           | Final Layer                    |
|---------------------------------------|--------------------------------|
| Probability-To-Leave Prediction Model | Left-Too-Early Detection Model |
| Time-To-Leave Prediction Model        |                                |

* Should the probability-to-leave be greater than 50% or "time-to-leave" is less than 5 seconds, it activates the "Left-Too-Early Detection Model". For the latter metric, even if the "Probability-To-Leave Prediction Model" says the player is unlikely to leave, we still have a chance that the player will leave in near term within a short period of time and the effects of player leaving is generally permanent.

* The "Time-To-Leave Prediction Model" is in the same layer as "Probability-To-Leave Prediction Model" because we want it to constantly produce update on how long the player will stay. If we were to put it between the first and final layer, the updates will be too sparse to predict accurate wait times for "Play Time Maximization Model".

## Code

### Feature Vector

```lua

local function getPlayerDataVectors(Player)

  local playerDataVectorWithBias = {

    {
        1,
        numberOfCurrencyAmount,
        numberOfItemsAmount,
        timePlayedInCurrentSession,
        timePlayedInAllSessions,
        healthAmount
    }
  }

  local playerDataVectorWithoutBias = {

    {
        numberOfCurrencyAmount,
        numberOfItemsAmount,
        timePlayedInCurrentSession,
        timePlayedInAllSessions,
        healthAmount
    }
  }

  return playerDataVectorWithBias, playerDataVectorWithoutBias

end

```

### Creating Time-To-Leave And Probability-To-Leave Prediction Models

```lua

local TimeToLeavePredictionModel = DataPredict.Models.LinearRegression.new({maximumNumberOfIterations = 100, learningRate = 0.01})

local ProbabilityToLeavePredictionModel = DataPredict.Models.LogisticRegression.new({maximumNumberOfIterations = 100, learningRate = 0.01})

local LeftToEarlyPredictionModel = DataPredict.Models.SupportVectorMachine.new({maximumNumberOfIterations = 100, beta = 1, kernelFunction = "RadialBasisFunction"})

-- The code shown below checks if we already have trained the models previously.

if (TimeToLeavePredictionModelParameters) then TimeToLeavePredictionModel:setModelParameters(TimeToLeavePredictionModelParameters) end

if (ProbabilityToLeavePredictionModelParameters) then ProbabilityToLeavePredictionModel:setModelParameters(ProbabilityToLeavePredictionModelParameters) end

if (LeftToEarlyPredictionModelParameters) then LeftToEarlyPredictionModel:setModelParameters(LeftToEarlyPredictionModelParameters) end

```

### Constructing Play Time Maximization Model

### Player Data Collection

```lua

local playerDataMatrixWithBias = {}

local playerDataMatrixWithoutBias = {}

local recordedTimeArray = {}
  
local snapshotIndex = 1

local function getPlayerDataArrays()

   local playerDataArrayWithBias = {

      1,
      numberOfCurrencyAmount,
      numberOfItemsAmount,
      timePlayedInCurrentSession,
      timePlayedInAllSessions,
      healthAmount

  }

  local playerDataArrayWithoutBias = {

      numberOfCurrencyAmount,
      numberOfItemsAmount,
      timePlayedInCurrentSession,
      timePlayedInAllSessions,
      healthAmount

  }

  return playerDataVectorWithBias, playerDataVectorWithoutBias

end
  
local function snapshotData(playerDataArray)

  playerDataMatrixWithBias[snapshotIndex], playerDataMatrixWithoutBias[snapshotIndex] = getPlayerDataArrays()
  
  recordedTimeArray[snapshotIndex] = os.time()
  
  snapshotIndex = snapshotIndex + 1

end

```

### On Player Join

```lua

local function run(Player)

    local isPlayerInServer = true

    local rewardValue = 0

    local playerDataArrayWithBias

    local playerDataArrayWithoutBias

    local playerDataVectorWithBias

    local playerDataVectorWithoutBias

    local predictedTimeToLeave

    local predictedProbabilityToLeave

    local activateLeftToEarlyPredictionModel

    local stayProbability

    while isPlayerInServer do

        playerDataArrayWithBias, playerDataArrayWithoutBias = getPlayerDataArrays()

        snapshotData(playerDataArray)

        playerDataVectorWithBias = {playerDataArrayWithBias}

        predictedTimeToLeave = TimeToLeavePredictionModel:predict(playerDataVectorWithBias)[1][1]

        predictedProbabilityToLeave = ProbabilityToLeavePredictionModel:predict(playerDataVectorWithBias)[1][1]

        activateLeftToEarlyPredictionModel = (predictedProbabilityToLeave >= 0.5) or (predictedTimeToLeave <= 5)

        if (activateLeftToEarlyPredictionModel) then

          playerDataVectorWithoutBias = {playerDataArrayWithoutBias}

          probabilityToStay = LeftToEarlyPredictionModel:predict(playerDataVectorWithoutBias)[1][1]

          -- Let's reward the player for staying much more longer than our models' predictions.

          if (probabilityToStay <= 0.1) then givePlayerReward(Player) end 

        end

        task.wait(predictedTimeToLeave)

        isPlayerInServer = checkIfPlayerIsInServer(Player)

    end

end

```

### On Player Leave

```lua

local timeToLeaveVector = {}

local probabilityToLeaveVector = {}

local probabilityToStayVector = {}

for i = 1, snapshotIndex, 1 do

  local timeToLeave = os.time() - recordedTime[i]

  local probabilityToLeave = 1 / timeToLeave

  local probabilityToStay = 1 - probabilityToLeave

  timeToLeaveVector[i] = {timeToLeave}

  probabilityToLeaveVector[i] = {probabilityToLeave}

  probabilityToStayVector[i] = {probabilityToStay}

end

TimeToLeavePredictionModel:train(playerDataMatrix, timeToLeaveVector)

ProbabilityToLeavePredictionModel:train(playerDataMatrix, probabilityToLeaveVector)

LeftToEarlyPredictionModel:train(playerDataMatrix, probabilityToStayVector)

-- Just getting our model parameters to save them

TimeToLeavePredictionModelParameters = TimeToLeavePredictionModel:getModelParameters(true)

ProbabilityToLeavePredictionModelParameters = ProbabilityToLeavePredictionModel:getModelParameters(true)

LeftToEarlyPredictionModelParameters = LeftToEarlyPredictionModel:getModelParameters(true)

```
